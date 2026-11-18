"""Dựng trang thử mù A/B ảnh rõ (phase 14 bước 7, Deferred 16): ``results/ab/ab-test.html`` + ``img/`` + khoá.

AX::

    python -m xla.eval.ab_page build --pairs 55 --seed 20261118 --json
    # exit 0 dựng xong; 1 dữ liệu sai / bundle sai / đã có trang (thêm --force); 2 sai cách gọi

Cấu trúc ``--pairs N`` = ``(N − repeats)/2`` cặp mạng-vs-gốc + cùng số cặp mạng-vs-cổ-điển + ``--repeats`` (mặc định 5)
cặp lặp lại để đo độ nhất quán. Khung: tập test của ``split.json``, rải đều theo tầng (bin độ cao, có/không người theo
nhãn test đã duyệt), mỗi khung dùng một lần; hai so sánh nhận tầng xen kẽ nên cân nhau. Trái/phải ngẫu nhiên theo
seed; cặp lặp đổi bên so với lần đầu và cách lần đầu ≥ ``MIN_REPEAT_GAP`` lượt.

Hiển thị: mọi biến thể cùng một cách — cắt bỏ cột ``[0, crop_left)`` (dải mép trái) rồi JPEG q95 4:4:4 — nên không
lộ biến thể qua dải mép hay kiểu nén. Tên ảnh là mã hex ngẫu nhiên; biến thể chỉ có trong ``ab_key.json``. Trang HTML
tự đủ, mở thẳng bằng file://, không mạng, không thư viện ngoài; tiến độ lưu ``localStorage``; nút Xuất tải JSON cho
``python -m xla.eval.ab_score``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xla.camera.split import alt_bin
from xla.eval.ab_score import EXPORT_FORMAT, KEY_FORMAT
from xla.eval.detection import EvalInputError
from xla.eval.enhance_eval import (
    FRAMES_REL,
    LOCK_TIMEOUT_S,
    SPLIT_REL,
    EvalError,
    Fn,
    UsageError,
    add_variant_args,
    build_variants,
    limit_cpu_threads,
    maybe_gpu,
)
from xla.paths import repo_root

COMPARISONS = {"net_vs_raw": ("net", "raw"), "net_vs_classic": ("net", "classic")}
OUT_REL = Path("results/ab")
LABELS_REL = Path("data/real/labels")
MIN_REPEAT_GAP = 10
JPEG_QUALITY = 95
CODE_BYTES = 6


class AbPageError(Exception):
    """Dữ liệu sai / đã có trang → exit 1."""


# ---------------------------------------------------------------- chọn khung và xếp cặp


def select_frames(cands: list[dict[str, Any]], n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """``n`` khung khác nhau, lần lượt từng tầng (bin độ cao, có người) theo thứ tự ngẫu nhiên — rải đều các tầng."""
    if len(cands) < n:
        raise AbPageError(f"cần {n} khung test khác nhau, chỉ có {len(cands)}")
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for c in cands:
        strata.setdefault((c["alt_bin"], c["has_person"]), []).append(c)
    keys = sorted(strata)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    pools = {k: [strata[k][i] for i in rng.permutation(len(strata[k]))] for k in keys}
    out: list[dict[str, Any]] = []
    while len(out) < n:
        for k in keys:
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


def plan_pairs(frames: list[dict[str, Any]], n_repeats: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Xếp cặp gốc, xáo thứ tự, chèn cặp lặp (đổi bên, cách ≥ MIN_REPEAT_GAP).

    So sánh xen kẽ **trong từng tầng** (mỗi tầng lệch nhau ≤ 1 cặp giữa hai so sánh); khung lẻ của các tầng lẻ chia
    luân phiên cho hai so sánh nên tổng mỗi so sánh bằng nhau (số tầng lẻ luôn chẵn khi tổng khung chẵn)."""
    names = list(COMPARISONS)
    if len(frames) % len(names):
        raise AbPageError(f"số khung gốc {len(frames)} phải chia hết cho {len(names)}")
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for f in frames:
        groups.setdefault((f["alt_bin"], f["has_person"]), []).append(f)
    odd = [k for k in groups if len(groups[k]) % 2]
    odd = [odd[i] for i in rng.permutation(len(odd))]
    first = {k: names[i % len(names)] for i, k in enumerate(odd)}
    comp_of: dict[str, str] = {}
    for k, members in groups.items():
        lead = first.get(k, names[int(rng.integers(len(names)))])
        order2 = [lead] + [n for n in names if n != lead]
        for j, f in enumerate(members):
            comp_of[f["frame"]] = order2[j % len(order2)]
    orig = []
    for f in frames:
        comp = comp_of[f["frame"]]
        a, b = COMPARISONS[comp]
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        orig.append({"comparison": comp, "frame": f["frame"], "alt_m": f["alt_m"], "alt_bin": f["alt_bin"],
                     "has_person": f["has_person"], "block": f["block"], "left": left, "right": right,
                     "repeat_of": None})
    order = [orig[i] for i in rng.permutation(len(orig))]
    eligible = list(range(max(0, len(order) - MIN_REPEAT_GAP)))
    if n_repeats > len(eligible):
        raise AbPageError(f"{n_repeats} cặp lặp cần ≥ {n_repeats + MIN_REPEAT_GAP} cặp gốc")
    chosen: list[int] = []
    for _ in range(200):  # cần cả hai so sánh khi có ≥ 2 cặp lặp; tất định theo rng
        chosen = sorted(int(i) for i in rng.choice(eligible, n_repeats, replace=False)) if n_repeats else []
        if n_repeats < len(names) or len({order[i]["comparison"] for i in chosen}) == len(names):
            break
    seq = list(order)
    for pos in chosen:
        src = order[pos]
        cur = next(i for i, p in enumerate(seq) if p is src)
        at = int(rng.integers(cur + MIN_REPEAT_GAP + 1, len(seq) + 1))
        seq.insert(at, src | {"left": src["right"], "right": src["left"], "repeat_of": src})
    for i, p in enumerate(seq):
        p["id"] = f"{i + 1:02d}"
    for p in seq:
        if p["repeat_of"] is not None:
            p["repeat_of"] = p["repeat_of"]["id"]
    return seq


def display_image(img: np.ndarray, crop_left: int) -> bytes:
    """Ảnh hiển thị: cắt bỏ dải mép ``[0, crop_left)``, JPEG q95 4:4:4 — giống nhau cho mọi biến thể."""
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[:, crop_left:]),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                            cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if not ok:
        raise AbPageError("không mã hoá được JPEG")
    return buf.tobytes()


def key_id_of(pairs: list[dict[str, Any]]) -> str:
    fields = ("id", "comparison", "frame", "left", "right", "left_img", "right_img", "repeat_of")
    canon = json.dumps([{k: p[k] for k in fields} for p in pairs], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def build(cands: list[dict[str, Any]], load: Callable[[str], np.ndarray], fns: dict[str, Fn], out_dir: Path, *,
          n_pairs: int, n_repeats: int, seed: int, crop_left: int, variants_info: dict[str, Any],
          force: bool = False) -> dict[str, Any]:
    """Dựng trang + ảnh + khoá vào ``out_dir``; trả tóm tắt. ``cands``: ``{"frame","alt_m","alt_bin","has_person","block"}``."""
    n_orig = n_pairs - n_repeats
    if n_repeats < 0 or n_orig <= 0 or n_orig % len(COMPARISONS):
        raise UsageError(f"--pairs {n_pairs} − --repeats {n_repeats} phải là số dương chia hết cho {len(COMPARISONS)}")
    img_dir = out_dir / "img"
    html_p, key_p = out_dir / "ab-test.html", out_dir / "ab_key.json"
    old_imgs = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    old_kid = json.loads(key_p.read_text(encoding="utf-8")).get("key_id") if key_p.is_file() else None
    has_result = (out_dir / "ab_result.json").is_file()
    if (html_p.exists() or key_p.exists() or old_imgs) and not force:
        raise AbPageError(f"đã có trang thử ở {out_dir}; thêm --force để dựng lại (kết quả chấm cũ sẽ lệch khoá mới)")
    rng = np.random.default_rng(seed)
    pairs = plan_pairs(select_frames(cands, n_orig, rng), n_repeats, rng)
    codes: set[str] = set()

    def new_code() -> str:
        while True:
            c = rng.bytes(CODE_BYTES).hex()
            if c not in codes:
                codes.add(c)
                return c

    images: dict[str, bytes] = {}
    by_id = {p["id"]: p for p in pairs}
    for p in pairs:
        if p["repeat_of"] is not None:
            continue
        img = load(p["frame"])
        for side in ("left", "right"):
            name = f"img/{new_code()}.jpg"
            images[name] = display_image(fns[p[side]](img), crop_left)
            p[f"{side}_img"] = name
    for p in pairs:
        if p["repeat_of"] is not None:
            src = by_id[p["repeat_of"]]
            p["left_img"], p["right_img"] = src["right_img"], src["left_img"]
    for f in old_imgs:
        f.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)
    for name, data in images.items():
        (out_dir / name).write_bytes(data)
    kid = key_id_of(pairs)
    strata: dict[str, int] = {}
    for p in pairs:
        if p["repeat_of"] is None:
            k = f"{p['comparison']}|{p['alt_bin']}|{'nguoi' if p['has_person'] else 'khong_nguoi'}"
            strata[k] = strata.get(k, 0) + 1
    key = {"format": KEY_FORMAT, "key_id": kid, "seed": seed, "created": datetime.now(UTC).isoformat(timespec="seconds"),
           "crop_left": crop_left, "display": f"cột [{crop_left}, 640) của mọi biến thể, JPEG q{JPEG_QUALITY} 4:4:4",
           "min_repeat_gap": MIN_REPEAT_GAP, "variants": variants_info, "n_pairs": len(pairs), "strata": strata,
           "images_sha256": {n: hashlib.sha256(d).hexdigest() for n, d in sorted(images.items())}, "pairs": pairs}
    key_p.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    public = [{"id": p["id"], "l": p["left_img"], "r": p["right_img"]} for p in pairs]
    html_p.write_text(render_html(public, kid), encoding="utf-8", newline="\n")
    return {"out_dir": str(out_dir), "html": str(html_p), "key": str(key_p), "key_id": kid, "n_pairs": len(pairs),
            "by_comparison": {c: sum(p["comparison"] == c and p["repeat_of"] is None for p in pairs)
                              for c in COMPARISONS}, "n_repeats": n_repeats, "n_images": len(images),
            "strata": strata, "warning": ("ab_result.json cũ không còn khớp khoá mới"
                                                   if has_result and old_kid != kid else None)}


# ---------------------------------------------------------------- HTML

_HTML = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thử mù A/B — ảnh nào rõ hơn?</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body { font: 16px/1.4 system-ui, "Segoe UI", sans-serif; background: #f4f4f5; color: #18181b;
  display: flex; flex-direction: column; }
header { display: flex; align-items: center; gap: 16px; padding: 10px 16px; background: #fff;
  border-bottom: 1px solid #e4e4e7; }
header h1 { font-size: 18px; margin: 0; white-space: nowrap; }
.progress { flex: 1; display: flex; align-items: center; gap: 10px; min-width: 0; }
.bar { flex: 1; height: 10px; background: #e4e4e7; border-radius: 5px; overflow: hidden; }
.bar > div { height: 100%; width: 0; background: #2563eb; transition: width .15s; }
#count { font-variant-numeric: tabular-nums; white-space: nowrap; font-weight: 600; }
button { font: inherit; padding: 7px 14px; border-radius: 8px; border: 1px solid #d4d4d8; background: #fff;
  cursor: pointer; }
button:hover { background: #f4f4f5; }
button.primary { background: #2563eb; border-color: #2563eb; color: #fff; }
button.primary:hover { background: #1d4ed8; }
main { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 12px 16px; min-height: 0; }
figure { margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 8px; min-height: 0; cursor: pointer; border-radius: 10px; padding: 6px; border: 4px solid transparent; }
figure:hover { background: #e4e4e7; }
figure.picked { border-color: #2563eb; background: #dbeafe; }
figure img { width: 100%; height: auto; max-height: calc(100vh - 170px); object-fit: contain; display: block;
  background: #000; }
figcaption { font-weight: 600; }
kbd { border: 1px solid #a1a1aa; border-bottom-width: 2px; border-radius: 4px; padding: 0 6px; background: #fff;
  font: 14px ui-monospace, monospace; }
footer { padding: 8px 16px; background: #fff; border-top: 1px solid #e4e4e7; font-size: 15px; }
#warn { display: none; padding: 8px 16px; background: #fef3c7; color: #92400e; }
#done { display: none; flex: 1; align-items: center; justify-content: center; flex-direction: column; gap: 14px;
  font-size: 20px; }
#done button { font-size: 18px; padding: 10px 22px; }
</style>
</head>
<body>
<header>
  <h1>Ảnh nào rõ hơn?</h1>
  <div class="progress"><div class="bar"><div id="bar"></div></div><span id="count"></span></div>
  <button id="back" title="Phím Backspace">Quay lại</button>
  <button id="export" class="primary">Xuất kết quả</button>
  <button id="reset">Làm lại</button>
</header>
<div id="warn"></div>
<main id="pair">
  <figure id="L"><img id="imgL" alt="Ảnh trái"><figcaption><kbd>&larr;</kbd> Trái rõ hơn</figcaption></figure>
  <figure id="R"><img id="imgR" alt="Ảnh phải"><figcaption>Phải rõ hơn <kbd>&rarr;</kbd></figcaption></figure>
</main>
<div id="done">
  <div>Đã chấm xong <b id="doneN"></b> cặp. Cảm ơn!</div>
  <button class="primary" id="export2">Xuất kết quả (tải file JSON)</button>
  <div style="font-size:15px">Bấm Quay lại nếu muốn sửa lựa chọn.</div>
</div>
<footer>Chọn ảnh <b>rõ hơn, dễ nhìn hơn</b> (thấy người, vật, mặt đất rõ hơn).
Bấm <kbd>&larr;</kbd> nếu ảnh trái rõ hơn, <kbd>&rarr;</kbd> nếu ảnh phải rõ hơn, hoặc bấm vào ảnh.
Không có lựa chọn hoà: phân vân thì chọn ảnh dễ nhìn hơn. <kbd>Backspace</kbd> = quay lại cặp trước.</footer>
<script>
"use strict";
const PAIRS = __PAIRS__;
const KEY_ID = "__KEY_ID__";
const FORMAT = "__FORMAT__";
const STORE = "xla-ab-" + KEY_ID;
const $ = (id) => document.getElementById(id);
let state = { key_id: KEY_ID, idx: 0, choices: {} };
let busy = false;
let shownAt = performance.now();

function warn(msg) { const w = $("warn"); w.textContent = msg; w.style.display = "block"; }
function load() {
  try {
    const raw = localStorage.getItem(STORE);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (s && s.key_id === KEY_ID && s.choices && typeof s.idx === "number") state = s;
  } catch (e) {
    warn("Trình duyệt không đọc được tiến độ đã lưu (" + e + "). Đừng đóng tab trước khi bấm Xuất kết quả.");
  }
}
function save() {
  try { localStorage.setItem(STORE, JSON.stringify(state)); }
  catch (e) { warn("Không lưu được tiến độ (" + e + "). Đừng đóng tab trước khi bấm Xuất kết quả."); }
}
function answered() { return PAIRS.filter((p) => state.choices[p.id]).length; }
function render() {
  const n = answered();
  $("count").textContent = "Cặp " + Math.min(state.idx + 1, PAIRS.length) + " / " + PAIRS.length +
    " · đã chấm " + n;
  $("bar").style.width = (100 * n / PAIRS.length) + "%";
  const done = state.idx >= PAIRS.length;
  $("pair").style.display = done ? "none" : "grid";
  $("done").style.display = done ? "flex" : "none";
  $("doneN").textContent = n + " / " + PAIRS.length;
  $("back").disabled = state.idx === 0;
  $("L").classList.remove("picked");
  $("R").classList.remove("picked");
  if (done) return;
  const p = PAIRS[state.idx];
  $("imgL").src = p.l;
  $("imgR").src = p.r;
  const c = state.choices[p.id];
  if (c) $(c.c).classList.add("picked");
  shownAt = performance.now();
  const nx = PAIRS[state.idx + 1];
  if (nx) { new Image().src = nx.l; new Image().src = nx.r; }
}
function choose(side) {
  if (busy || state.idx >= PAIRS.length) return;
  const p = PAIRS[state.idx];
  state.choices[p.id] = { c: side, ms: Math.round(performance.now() - shownAt), at: new Date().toISOString() };
  busy = true;
  $(side).classList.add("picked");
  state.idx += 1;
  save();
  setTimeout(() => { busy = false; render(); }, 150);
}
function back() {
  if (busy || state.idx === 0) return;
  state.idx -= 1;
  save();
  render();
}
function exportResult() {
  const n = answered();
  if (n < PAIRS.length && !confirm("Mới chấm " + n + "/" + PAIRS.length + " cặp. Vẫn xuất?")) return;
  const out = {
    format: FORMAT, key_id: KEY_ID, n_pairs: PAIRS.length, n_answered: n, complete: n === PAIRS.length,
    exported_at: new Date().toISOString(),
    choices: PAIRS.filter((p) => state.choices[p.id]).map((p) => ({
      id: p.id, choice: state.choices[p.id].c === "L" ? "left" : "right",
      ms: state.choices[p.id].ms, at: state.choices[p.id].at })),
  };
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ab_result-" + KEY_ID + ".json";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
function reset() {
  if (!confirm("Xoá mọi lựa chọn và chấm lại từ đầu?")) return;
  state = { key_id: KEY_ID, idx: 0, choices: {} };
  save();
  render();
}
document.addEventListener("keydown", (e) => {
  if (e.repeat || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); choose("L"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); choose("R"); }
  else if (e.key === "Backspace") { e.preventDefault(); back(); }
});
$("L").addEventListener("click", () => choose("L"));
$("R").addEventListener("click", () => choose("R"));
$("back").addEventListener("click", back);
$("export").addEventListener("click", exportResult);
$("export2").addEventListener("click", exportResult);
$("reset").addEventListener("click", reset);
for (const id of ["imgL", "imgR"]) {
  $(id).addEventListener("error", (e) => warn("Không mở được ảnh " + e.target.getAttribute("src") +
    " — thư mục img/ phải nằm cạnh file HTML."));
}
load();
if (state.idx > PAIRS.length) state.idx = PAIRS.length;
render();
</script>
</body>
</html>
"""


def render_html(public_pairs: list[dict[str, str]], key_id: str) -> str:
    """Trang tự đủ: chỉ có id cặp + đường dẫn ảnh tương đối, không có tên biến thể."""
    data = json.dumps(public_pairs, ensure_ascii=False, separators=(",", ":"))
    return _HTML.replace("__PAIRS__", data).replace("__KEY_ID__", key_id).replace("__FORMAT__", EXPORT_FORMAT)


# ---------------------------------------------------------------- CLI


def test_candidates(root: Path, split_json: Path, labels_dir: Path) -> list[dict[str, Any]]:
    from xla.eval.detection import read_yolo_label

    rows = json.loads(split_json.read_text(encoding="utf-8"))["frames"]
    base = labels_dir / "test" if (labels_dir / "test").is_dir() else labels_dir
    out = []
    for r in rows:
        if r["split"] != "test" or r.get("alt_m") is None:
            continue
        lab = base / (Path(r["frame"]).stem + ".txt")
        if not lab.is_file():
            raise AbPageError(f"thiếu nhãn test {lab}")
        out.append({"frame": r["frame"], "alt_m": float(r["alt_m"]), "alt_bin": alt_bin(float(r["alt_m"])),
                    "has_person": len(read_yolo_label(lab)) > 0, "block": int(r["block"])})
    return out


def cmd_build(a: argparse.Namespace) -> dict[str, Any]:
    limit_cpu_threads()
    root = repo_root()
    split_json = a.split_json or root / SPLIT_REL
    if not split_json.is_file():
        raise AbPageError(f"không có {split_json}")
    cands = test_candidates(root, split_json, a.labels_dir or root / LABELS_REL)
    vs = build_variants(a, root)
    frames_dir = a.frames_dir or root / FRAMES_REL

    def load(frame: str) -> np.ndarray:
        from xla.contract import decode_frame

        p = frames_dir / frame
        if not p.is_file():
            raise AbPageError(f"thiếu khung {p}")
        return decode_frame(p)

    with maybe_gpu(vs.uses_gpu, "ab_page", a.lock_timeout) as lock:
        res = build(cands, load, vs.fns, a.out_dir or root / OUT_REL, n_pairs=a.pairs, n_repeats=a.repeats,
                    seed=a.seed, crop_left=vs.crop_left, variants_info=vs.info, force=a.force)
    return res | {"gpu": lock}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.eval.ab_page", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="dựng ab-test.html + img/ + ab_key.json")
    b.add_argument("--pairs", type=int, required=True, help="tổng số lượt, vd. 55 = 25 + 25 + 5 lặp")
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--seed", type=int, required=True)
    add_variant_args(b)
    b.add_argument("--split-json", type=Path, default=None, help=f"mặc định {SPLIT_REL}")
    b.add_argument("--frames-dir", type=Path, default=None, help=f"mặc định {FRAMES_REL}")
    b.add_argument("--labels-dir", type=Path, default=None, help=f"mặc định {LABELS_REL} (dùng thư mục con test/)")
    b.add_argument("--out-dir", type=Path, default=None, help=f"mặc định {OUT_REL}")
    b.add_argument("--force", action="store_true", help="dựng lại, xoá ảnh cũ trong img/")
    b.add_argument("--lock-timeout", type=float, default=LOCK_TIMEOUT_S)
    b.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = cmd_build(a)
    except UsageError as e:
        print(f"LỖI cách gọi: {e}", file=sys.stderr)
        return 2
    except (AbPageError, EvalError, EvalInputError, TimeoutError, ValueError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res, ensure_ascii=False) if a.json else json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
