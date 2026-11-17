"""
Web server for the XLA UAV Image & Video Restoration Pipeline.
Provides drag-and-drop upload, automatic pipeline execution, live video progress,
quantitative metrics, and before/after comparison.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.uav_pipeline import (  # noqa: E402
    compute_entropy,
    compute_laplacian_var,
    compute_pink_cast_index,
    run_full_pipeline,
    run_stage1_preprocess,
    run_stage2_contrast,
    upscale_to_hd,
)

WEB_DIR = Path(__file__).resolve().parent
JOBS_DIR = ROOT_DIR / "results" / "web_jobs"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
VIDEOS_DIR = ROOT_DIR / "data" / "raw_videos"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="XLA UAV Restoration Web", version="1.0.0")

# In-memory job registry for async video (and image) status tracking
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _set_job(target_job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        if target_job_id not in _JOBS:
            _JOBS[target_job_id] = {}
        _JOBS[target_job_id].update(updates)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload.bin")
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    return cleaned or "upload.bin"


def _compute_image_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    return {
        "pci": round(compute_pink_cast_index(img_bgr), 2),
        "entropy": round(compute_entropy(img_bgr), 3),
        "laplacian_var": round(compute_laplacian_var(img_bgr), 1),
        "width": int(img_bgr.shape[1]),
        "height": int(img_bgr.shape[0]),
    }


def _process_image_sync(
    input_path: Path,
    job_dir: Path,
    job_id: str,
    image_mode: str,
) -> Dict[str, Any]:
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Không thể đọc file ảnh đầu vào. Vui lòng kiểm tra định dạng.")

    upscale_hd = image_mode in ("hd_720p", "full_hd_1080p")
    hd_height = 1080 if image_mode == "full_hd_1080p" else 720

    res = run_full_pipeline(
        img_bgr,
        return_intermediates=True,
        upscale_hd=upscale_hd,
        hd_height=hd_height,
    )

    out_img = res["output"]
    in_png = job_dir / "input.png"
    s1_png = job_dir / "stage1.png"
    s2_png = job_dir / "stage2.png"
    s3_png = job_dir / "stage3.png"
    out_png = job_dir / "output.png"

    cv2.imwrite(str(in_png), img_bgr)
    cv2.imwrite(str(s1_png), res["stage1"])
    cv2.imwrite(str(s2_png), res["stage2"])
    cv2.imwrite(str(s3_png), res["output_native"])
    cv2.imwrite(str(out_png), out_img)

    metrics_before = _compute_image_metrics(img_bgr)
    metrics_after = _compute_image_metrics(out_img)

    return {
        "job_id": job_id,
        "media_type": "image",
        "status": "done",
        "mode": image_mode,
        "files": {
            "input": f"/api/files/{job_id}/input.png",
            "stage1": f"/api/files/{job_id}/stage1.png",
            "stage2": f"/api/files/{job_id}/stage2.png",
            "stage3": f"/api/files/{job_id}/stage3.png",
            "output": f"/api/files/{job_id}/output.png",
        },
        "metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "runtime_ms": round(float(res["runtime_ms"]), 1),
            "stage1_ms": round(float(res["stage1_ms"]), 1),
            "stage2_ms": round(float(res["stage2_ms"]), 1),
            "stage3_ms": round(float(res["stage3_ms"]), 1),
        },
    }


def _transcode_h264(src_path: Path, dst_path: Path) -> None:
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(src_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if src_path.exists():
            src_path.unlink()
    except Exception:
        os.replace(str(src_path), str(dst_path))


def _process_video_worker(
    job_id: str,
    input_path: Path,
    job_dir: Path,
    video_mode: str,
    max_frames: int,
) -> None:
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            _set_job(job_id, status="error", error="Không thể mở file video đầu vào.")
            return

        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frames = (
            total_src_frames
            if max_frames <= 0
            else (min(total_src_frames, max_frames) if total_src_frames > 0 else max_frames)
        )

        if video_mode in ("side_by_side_1080p", "full_hd_1080p"):
            out_w, out_h = 1920, 1080
        else:
            out_w, out_h = 1280, 720

        _set_job(
            job_id,
            status="processing",
            frames_done=0,
            total_frames=target_frames,
            out_width=out_w,
            out_height=out_h,
        )

        tmp_mp4v = job_dir / "output_tmp_mp4v.mp4"
        out_mp4 = job_dir / "output.mp4"
        writer = cv2.VideoWriter(
            str(tmp_mp4v),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_vid,
            (out_w, out_h),
        )

        frames_done = 0
        total_ms = 0.0
        first_in = None
        first_out = None

        while max_frames <= 0 or frames_done < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            proc_in = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            s1, _ = run_stage1_preprocess(proc_in)
            s2, _, _ = run_stage2_contrast(s1)

            if video_mode == "full_hd_1080p":
                out_frame = upscale_to_hd(s2, target_height=1080, target_width=1920)
                restored_for_metrics = out_frame
            elif video_mode == "hd_720p":
                out_frame = upscale_to_hd(s2, target_height=720, target_width=1280)
                restored_for_metrics = out_frame
            elif video_mode == "side_by_side_1080p":
                restored_fhd = upscale_to_hd(s2, target_height=1080, target_width=1920)
                in_fhd = cv2.resize(proc_in, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                out_frame = np.hstack([in_fhd[:, :960], restored_fhd[:, 960:]])
                cv2.line(out_frame, (960, 0), (960, 1080), (0, 255, 180), 4)
                cv2.putText(
                    out_frame,
                    "INPUT: ESP32-CAM (Noise + Blur + Pink)",
                    (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (70, 70, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out_frame,
                    "OUTPUT: Wavelet+NLM + RL + ESPCN Full HD 1080p",
                    (984, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (70, 255, 100),
                    2,
                    cv2.LINE_AA,
                )
                restored_for_metrics = restored_fhd
            else:
                restored_hd = upscale_to_hd(s2, target_height=720, target_width=1280)
                in_hd = cv2.resize(proc_in, (1280, 720), interpolation=cv2.INTER_NEAREST)
                out_frame = np.hstack([in_hd[:, :640], restored_hd[:, 640:]])
                cv2.line(out_frame, (640, 0), (640, 720), (0, 255, 180), 3)
                cv2.putText(
                    out_frame,
                    "Input: ESP32-CAM (OV2640)",
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (70, 70, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out_frame,
                    "Output: Wavelet+NLM + RL + ESPCN HD 720p",
                    (658, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (70, 255, 100),
                    2,
                    cv2.LINE_AA,
                )
                restored_for_metrics = restored_hd

            dt_ms = (time.perf_counter() - t0) * 1000.0
            total_ms += dt_ms

            if first_in is None:
                first_in = proc_in
                first_out = restored_for_metrics

            writer.write(out_frame)
            frames_done += 1
            _set_job(
                job_id,
                frames_done=frames_done,
                avg_ms=round(total_ms / frames_done, 1),
            )

        cap.release()
        writer.release()

        if frames_done == 0:
            _set_job(job_id, status="error", error="Video không có frame hợp lệ.")
            return

        _set_job(job_id, status="encoding")
        _transcode_h264(tmp_mp4v, out_mp4)

        metrics = {}
        if first_in is not None and first_out is not None:
            metrics = {
                "before": _compute_image_metrics(first_in),
                "after": _compute_image_metrics(first_out),
                "frames": frames_done,
                "avg_ms": round(total_ms / frames_done, 1),
                "fps": round(1000.0 / max(total_ms / frames_done, 1e-3), 2),
            }

        _set_job(
            job_id,
            status="done",
            files={"output": f"/api/files/{job_id}/output.mp4"},
            metrics=metrics,
        )
    except Exception as exc:
        _set_job(job_id, status="error", error=str(exc))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/samples")
async def list_samples() -> JSONResponse:
    images = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                images.append({"name": p.name, "path": f"data/samples/{p.name}", "type": "image"})
    videos = []
    if VIDEOS_DIR.exists():
        for p in sorted(VIDEOS_DIR.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append({"name": p.name, "path": f"data/raw_videos/{p.name}", "type": "video"})
    return JSONResponse({"images": images, "videos": videos})


@app.post("/api/process")
async def process_media(
    request: Request,
    image_mode: str = Query("hd_720p"),
    video_mode: str = Query("side_by_side_720p"),
    max_frames: int = Query(90),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = json.loads(await request.body())
        rel_path = payload.get("sample_path", "")
        sample_file = (ROOT_DIR / rel_path).resolve()
        if not str(sample_file).startswith(str((ROOT_DIR / "data").resolve())) or not sample_file.exists():
            raise HTTPException(status_code=400, detail="Sample file không hợp lệ.")
        filename = sample_file.name
        input_path = job_dir / f"raw_{_safe_filename(filename)}"
        input_path.write_bytes(sample_file.read_bytes())
    else:
        raw_name = unquote(request.headers.get("x-filename", "upload.bin"))
        filename = _safe_filename(raw_name)
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Không nhận được dữ liệu file.")
        input_path = job_dir / f"raw_{filename}"
        input_path.write_bytes(body)

    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        try:
            result = await asyncio.to_thread(
                _process_image_sync, input_path, job_dir, job_id, image_mode
            )
            _set_job(job_id, **result)
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    if ext in VIDEO_EXTS:
        _set_job(
            job_id,
            job_id=job_id,
            media_type="video",
            status="queued",
            mode=video_mode,
            frames_done=0,
            total_frames=max_frames,
        )
        t = threading.Thread(
            target=_process_video_worker,
            args=(job_id, input_path, job_dir, video_mode, max_frames),
            daemon=True,
        )
        t.start()
        return JSONResponse({"job_id": job_id, "media_type": "video", "status": "queued"})

    raise HTTPException(
        status_code=400,
        detail=f"Định dạng '{ext}' chưa được hỗ trợ. Hãy dùng ảnh (.jpg, .png, .webp) hoặc video (.mp4, .avi, .mov).",
    )


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job.")
    return JSONResponse(job)


@app.get("/api/files/{job_id}/{filename}")
async def get_job_file(job_id: str, filename: str) -> FileResponse:
    safe_job = _safe_filename(job_id)
    safe_name = _safe_filename(filename)
    file_path = (JOBS_DIR / safe_job / safe_name).resolve()
    if not str(file_path).startswith(str(JOBS_DIR.resolve())) or not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại.")
    return FileResponse(str(file_path))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web.server:app", host="127.0.0.1", port=port, reload=False)
