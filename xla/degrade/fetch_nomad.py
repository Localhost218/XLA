"""Tải một tập con NOMAD (WACV 2024, CC BY 4.0) từ Google Drive, hạ mẫu ngay để vừa ổ D.

Vì sao tập con: mỗi zip 10 diễn viên nặng 38 GB (10 zip = 380 GB), ổ D chỉ còn ~60 GB. Tải từng ảnh qua thư mục
`images/ActorXXX/ActorXXX_aNN/` của Drive, lấy một khung trên `stride`, chỉ các khoảng cách gần (mặc định 10 m, 30 m:
người đủ to để thu nhỏ về dải 21–139 px của camera), hạ mẫu `INTER_AREA` hệ số `downscale` rồi lưu JPEG q95, xoá bản
gốc. sha256 của bản gốc + hệ số hạ mẫu ghi vào `fetch_manifest.jsonl` để tái lập và để `sources.py` quy đổi hộp nhãn.

    python -m xla.degrade.fetch_nomad --out D:/DevCache/datasets/src/nomad --actors 1-50 --dist a10,a30 --stride 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT_FOLDER = "1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg"
IMAGES_FOLDER = "14oxtWhFOoRRHIKBYFMSU945cTeUrT2Zr"
ANNOTATION_FILES = {  # tên → id file Drive (thư mục annotations/)
    "annotations.json": "1XQ8hqMxytGg6wiQXJXiZFfT5S6rYI3hY",
    "metadata.json": "1QHeWq5ZU2H4aGcupFsZJv8wjFrK9Bo5U",
}
LIST_URL = "https://drive.google.com/embeddedfolderview?id={}"
DL_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ENTRY_RE = re.compile(r'(file/d|folders)/([A-Za-z0-9_-]{20,}).*?flip-entry-title">([^<]*)', re.DOTALL)
JPEG_SAVE_QUALITY = 95
RETRIES = 3


def _get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_drive(folder_id: str) -> list[tuple[str, str, bool]]:
    """(id, tên, là_thư_mục) của một thư mục Drive công khai."""
    html = _get(LIST_URL.format(folder_id)).decode("utf-8", "replace")
    out = []
    for kind, fid, name in ENTRY_RE.findall(html):
        out.append((fid, name.strip(), kind == "folders"))
    return out


def download_file(fid: str) -> bytes:
    last: Exception | None = None
    for i in range(RETRIES):
        try:
            data = _get(DL_URL.format(fid))
            if data[:2] != b"\xff\xd8" and not data.lstrip().startswith((b"{", b"[")):
                raise RuntimeError(f"Drive không trả file (có thể giới hạn lượt tải): {data[:120]!r}")
            return data
        except Exception as e:  # noqa: BLE001 - thử lại rồi ném lỗi rõ ràng
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"tải {fid} hỏng sau {RETRIES} lần: {last}")


def parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def _process(fid: str, name: str, dst: Path, downscale: float) -> dict:
    data = download_file(fid)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"không giải mã được {name}")
    h, w = img.shape[:2]
    small = cv2.resize(img, (round(w / downscale), round(h / downscale)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_SAVE_QUALITY])
    if not ok:
        raise RuntimeError(f"không mã hoá được {name}")
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(dst)
    return {"name": name, "drive_id": fid, "orig_sha256": hashlib.sha256(data).hexdigest(),
            "orig_bytes": len(data), "orig_wh": [w, h], "saved_wh": [small.shape[1], small.shape[0]],
            "downscale": downscale, "rel": dst.as_posix()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--actors", default="1-50")
    ap.add_argument("--dist", default="a10,a30")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--downscale", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-files", type=int, default=0, help="0 = không giới hạn (chạy thử: 1)")
    a = ap.parse_args(argv)

    out: Path = a.out
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for name, fid in ANNOTATION_FILES.items():
        p = out / "annotations" / name
        if not p.exists():
            p.write_bytes(download_file(fid))
    manifest = out / "fetch_manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(line)["name"] for line in manifest.read_text("utf-8").splitlines() if line.strip()}

    wanted_actors = {f"Actor{n:03d}" for n in parse_range(a.actors)}
    dists = set(a.dist.split(","))
    jobs: list[tuple[str, str, Path]] = []
    for aid, aname, is_dir in list_drive(IMAGES_FOLDER):
        if not is_dir or aname not in wanted_actors:
            continue
        for did, dname, is_dir2 in list_drive(aid):
            if not is_dir2 or dname.split("_")[-1] not in dists:
                continue
            files = sorted((n, f) for f, n, d in list_drive(did) if not d and n.lower().endswith(".jpg"))
            for n, f in files[:: a.stride]:
                if n not in done:
                    jobs.append((f, n, out / "images" / aname / dname / n))
    if a.max_files:
        jobs = jobs[: a.max_files]
    print(f"[fetch_nomad] {len(jobs)} file cần tải (đã có {len(done)})", flush=True)

    lock = threading.Lock()
    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, manifest.open("a", encoding="utf-8", newline="\n") as mf:
        futs = {ex.submit(_process, f, n, p, a.downscale): n for f, n, p in jobs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    mf.flush()
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - ghi lỗi từng file, đi tiếp
                n_err += 1
                print(f"[fetch_nomad] LỖI {futs[fut]}: {e}", file=sys.stderr, flush=True)
            if (n_ok + n_err) % 100 == 0:
                print(f"[fetch_nomad] {n_ok + n_err}/{len(jobs)} ok={n_ok} err={n_err} {time.time() - t0:.0f}s", flush=True)
    print(f"[fetch_nomad] xong ok={n_ok} err={n_err} {time.time() - t0:.0f}s", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
