"""Phase 07 bước 4: D1 / D2 / D2+Real-SR so với khung **test** thật (chỉ ảnh, không nhãn), cộng hàng tham chiếu thật–thật.

Mỗi hàng gọi ``xla.eval.realism.compute_realism`` (thống kê mức thấp, C2ST, KID-DINOv2) với cùng tập thật (test), cùng
số patch, cùng seed. Tập tổng hợp lấy ``--n-images`` ảnh (train + val) chọn có seed; tên ảnh trùng nhau giữa các tập nên
D1/D2/D2+Real-SR dùng CÙNG đơn vị nguồn. Hàng ``real_dev`` thay tập tổng hợp bằng khung dev thật (cùng số ảnh, cùng seed):
nền "cùng camera, khác cảnh" để đọc các hàng kia (realism smoke: hai nửa dev cho C2ST 0,77, KID 1,33).

CLI (AX):
    python -m xla.degrade.realism_compare --smoke --json          # 1 lần KID 1 000 patch (D2 vs test), đo giờ GPU
    python -m xla.degrade.realism_compare --json                   # đủ: ghi results/realism/d1_vs_d2.json
Exit: 0 đo xong; 1 lỗi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from xla import paths
from xla.eval.patches import list_images, split_images, subsample

SEED = 20261118
OUT_REL = Path("results") / "realism" / "d1_vs_d2.json"
SET_ROWS = {"d1": "D1 (Real-ESRGAN ngẫu nhiên)", "d2": "D2 (bản sao camera, đã hiệu chỉnh)",
            "d2sr": "D2 + Real-SR (nhiễu từ khung dev)", "real_dev": "thật–thật: khung dev"}


def _split_file() -> Path:
    return paths.repo_root() / "data" / "real" / "split.json"


def synth_paths(name: str, n_images: int, seed: int) -> list[Path]:
    if name == "real_dev":
        return subsample(split_images(_split_file(), "dev"), n_images, seed + 1)
    return subsample(list_images(paths.data_root() / name), n_images, seed + 1)


def _manifest_sha(name: str) -> str | None:
    p = paths.data_root() / name / "manifest.jsonl"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def run_row(name: str, real: list[Path], n_images: int, n_patches: int, seed: int, tiers: tuple, device: str) -> dict:
    from xla.eval.realism import compute_realism
    from xla.gpu_lock import gpu_lock

    synth = synth_paths(name, n_images, seed)
    t0 = time.perf_counter()
    res = compute_realism(real, synth, tiers=tiers, device=device, cache_dir=paths.cache_dir() / "realism",
                          lock=lambda: gpu_lock(f"degrade:realism:{name}"), seed=seed, n_patches=n_patches)
    t = res["timing_s"]
    gpu_s = round(t.get("c2st", 0) - t.get("c2st_load", 0) + t.get("kid_features", 0), 2)
    row = {"row": name, "what": SET_ROWS[name], "n_real": len(real), "n_synth": len(synth), "seed": seed,
           "n_patches": n_patches, "manifest_sha256": _manifest_sha(name), "gpu_seconds": gpu_s,
           "seconds_total": round(time.perf_counter() - t0, 1), "synth_examples": [p.name for p in synth[:3]]}
    if "stats" in res:
        row["stats"] = {"total": res["stats"]["total"], "dqt_match": res["stats"]["dqt"]["match"],
                        "items": {k: round(v["value"], 4) for k, v in res["stats"]["items"].items()}}
    if "c2st" in res:
        c = res["c2st"]
        row["c2st"] = {"mean": c["accuracy"], "ci_low": c["ci95"][0], "ci_high": c["ci95"][1],
                       "n_patches": n_patches}
    if "kid" in res:
        k = res["kid"]
        row["kid"] = {"mean": k["mean"], "se": k["se"], "ci_low": k["ci95"][0], "ci_high": k["ci95"][1],
                      "n_patches": n_patches, "n_repeats": len(k["subsets"]), "subset_size": k["subset_size"],
                      "seed": seed, "encoder": k.get("encoder")}
    row["timing_s"] = t
    row["vram_peak_mb"] = res.get("vram_peak_mb")
    row["gpu"] = res.get("gpu")
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xla.degrade.realism_compare", description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="d1,d2,d2sr,real_dev")
    ap.add_argument("--n-images", type=int, default=600, help="số ảnh tổng hợp (và khung dev) mỗi hàng")
    ap.add_argument("--n-patches", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="chỉ KID cho D2, ghi giờ GPU vào khối smoke")
    ap.add_argument("--out", type=Path, default=paths.repo_root() / OUT_REL)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        real = split_images(_split_file(), "test")
        out = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}
        if a.smoke:
            row = run_row("d2", real, a.n_images, a.n_patches, a.seed, ("kid",), a.device)
            out["smoke"] = {"what": "1 lần KID (10 tập con) D2 vs test, đặc trưng chưa cache",
                            "n_patches": a.n_patches, "gpu_seconds": row["gpu_seconds"],
                            "seconds_total": row["seconds_total"], "kid": row["kid"], "gpu": row["gpu"],
                            "measured": time.strftime("%Y-%m-%d %H:%M")}
            result = out["smoke"]
        else:
            old = {r["row"]: r for r in out.get("rows", [])}
            rows = []
            for name in [r for r in a.rows.split(",") if r]:
                prev = old.get(name)
                if (prev and prev["seed"] == a.seed and prev["n_patches"] == a.n_patches
                        and prev.get("n_images_requested") == a.n_images and prev["manifest_sha256"] == _manifest_sha(name)):
                    rows.append(prev)                      # chịu ngắt: hàng đã đo cùng tham số + cùng tập thì giữ
                    continue
                row = run_row(name, real, a.n_images, a.n_patches, a.seed, ("stats", "c2st", "kid"), a.device)
                row["n_images_requested"] = a.n_images
                rows.append(row)
                out["rows"] = rows + [r for k, r in old.items() if k not in {x["row"] for x in rows}]
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
            out.update({
                "phase": "07", "measured": time.strftime("%Y-%m-%d"),
                "real": {"split": "test", "n_frames": len(real), "split_file": "data/real/split.json",
                         "split_file_sha256": hashlib.sha256(_split_file().read_bytes()).hexdigest(),
                         "labels_used": False},
                "params": {"n_images_synth": a.n_images, "n_patches": a.n_patches, "seed": a.seed,
                           "kid_repeats": 10, "ci": "95 % bootstrap"},
                "rows": rows,
                "gpu_seconds_total": round(sum(r["gpu_seconds"] for r in rows), 1),
                "env": {"python": platform.python_version(), "platform": platform.platform()},
            })
            result = {r["row"]: {"stats": r["stats"]["total"], "c2st": [r["c2st"]["mean"], r["c2st"]["ci_low"],
                                                                         r["c2st"]["ci_high"]],
                                 "kid": [r["kid"]["mean"], r["kid"]["ci_low"], r["kid"]["ci_high"]]} for r in rows}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    except Exception as e:  # noqa: BLE001 — CLI: lỗi nào cũng exit 1, in nguyên văn
        print(json.dumps({"error": type(e).__name__, "detail": str(e)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=None if a.json else 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
