"""
Benchmark "Trước vs Sau cải thiện" + sinh bảng/hình/notebook/video cho pipeline ESP32-CAM UAV.

  - TRƯỚC  = src/uav_pipeline_v1.py (bản gốc đóng băng từ commit a900626)
  - SAU    = src/uav_pipeline.py    (v2: khử hồng bảo toàn texture, AWB có kiểm soát, khử nhiễu thích nghi,
                                    Stage 2 Retinex/veil/dehaze thích nghi, ESPCN trên Y, làm nét chống halo)

Nhóm đánh giá (tách riêng): (A) 30 cặp synthetic gốc; (B) tập augmented theo kịch bản (data_augmentation.py);
(C) 60 ảnh data/real_uav/ (không tham chiếu) + ca đặc biệt (ảnh tối 051..056, ảnh hồng nặng 001/013/032).

Chạy:  python -m src.build_notebooks_and_results [--skip-video] [--workers N]
"""

import argparse
import base64
import glob
import io
import os
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.uav_pipeline as P2          # noqa: E402  (SAU)
import src.uav_pipeline_v1 as P1       # noqa: E402  (TRƯỚC)
from src.data_augmentation import generate_augmented_benchmark, SCENARIOS  # noqa: E402

TABLES = f"{ROOT}/results/tables"
FIGS = f"{ROOT}/results/figures"
HD_DIR = f"{ROOT}/results/hd_outputs"
VID_DIR = f"{ROOT}/results/video"
NB_DIR = f"{ROOT}/notebooks"

M_INPUT, M_HE_RGB, M_HE_LAB = "1. Ảnh đầu vào (suy giảm)", "2. Đối chứng 1: Global HE (RGB)", "3. Đối chứng 2: Global HE (LAB)"
M_V1 = "4. TRƯỚC cải thiện: Pipeline v1 (VGA)"
M_S1, M_S12 = "5. SAU: Giai đoạn 1 (khử hồng + AWB + khử nhiễu thích nghi)", "6. SAU: Giai đoạn 1+2 (Retinex/veil/dehaze + CLAHE)"
M_V2 = "7. SAU cải thiện: Pipeline v2 (3 giai đoạn - VGA)"
M_HD, M_FHD = "8. SAU: Pipeline v2 + ESPCN HD 720p", "9. SAU: Pipeline v2 + ESPCN Full HD 1080p"
BEFORE_AFTER = [M_INPUT, M_V1, M_V2]


# ----------------------------------------------------------------------------- metrics
def _metrics(im, ref=None):
    d = {
        "Entropy (bits)": P2.compute_entropy(im),
        "Laplacian_Var": P2.compute_laplacian_var(im),
        "Pink_Cast (%)": P2.compute_pink_cast_index(im),
        "L_mean": P2.compute_l_mean(im),
        "Crush0 (%)": P2.compute_crush_ratio(im),
        "Clip255 (%)": P2.compute_clip_ratio(im),
        "Resolution": f"{im.shape[1]}x{im.shape[0]}",
    }
    if ref is not None:
        if ref.shape[:2] != im.shape[:2]:
            ref = cv2.resize(ref, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        d.update({"PSNR (dB)": P2.compute_psnr(ref, im), "SSIM": P2.compute_ssim(ref, im),
                  "DeltaE_ab": P2.compute_delta_e(ref, im)})
    return d


def _init_worker():
    cv2.setNumThreads(2)


def _eval_one(task):
    """Đánh giá 1 ảnh: trả về list record (mỗi method một dòng)."""
    group, scenario, name, deg_path, gt_path, with_hd = task
    cv2.setNumThreads(2)
    deg = cv2.imread(deg_path)
    gt = cv2.imread(gt_path) if gt_path else None
    cands = [(M_INPUT, deg, 0.0), ]
    t = time.perf_counter(); he1 = P2.baseline_global_he_rgb(deg); cands.append((M_HE_RGB, he1, (time.perf_counter() - t) * 1e3))
    t = time.perf_counter(); he2 = P2.baseline_global_he_lab(deg); cands.append((M_HE_LAB, he2, (time.perf_counter() - t) * 1e3))
    r1 = P1.run_full_pipeline(deg)
    cands.append((M_V1, r1["output"], r1["runtime_ms"]))
    r2 = P2.run_full_pipeline(deg, return_intermediates=True, hd_heights=(720, 1080) if with_hd else None)
    if with_hd:
        cands.append((M_S1, r2["stage1"], r2["stage1_ms"]))
        cands.append((M_S12, r2["stage2"], r2["stage1_ms"] + r2["stage2_ms"]))
    cands.append((M_V2, r2["output"], r2["runtime_ms"]))
    if with_hd:
        base = r2["stage1_ms"] + r2["stage2_ms"]
        cands.append((M_HD, r2["output_720p"], base + r2["hd_ms"][720]))
        cands.append((M_FHD, r2["output_1080p"], base + r2["hd_ms"][1080]))
    out = []
    for m, im, ms in cands:
        rec = {"Group": group, "Scenario": scenario, "Image": name, "Method": m, "Runtime (ms)": ms}
        rec.update(_metrics(im, gt))
        if m == M_V2:
            rec["_info"] = {k: round(float(v), 3) for k, v in r2["scene_info"].items() if isinstance(v, (int, float))}
        out.append(rec)
    return out


def _run_tasks(tasks, workers):
    recs = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(_eval_one, tasks), 1):
            recs.extend(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"   ... {i}/{len(tasks)}", flush=True)
    df = pd.DataFrame(recs)
    if "_info" in df:
        df = df.drop(columns=["_info"])
    return df


def _summ(df, by, cols):
    present = [c for c in cols if c in df.columns]
    return df.groupby(by, sort=False)[present].mean().reset_index()


# ----------------------------------------------------------------------------- figures
def fig_to_b64(fig, save_path=None):
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _wrap(t, w=34):
    return "\n".join(textwrap.wrap(t, w))


def _stat(im):
    return f"L̄={P2.compute_l_mean(im):.1f} | đen≤3={P2.compute_crush_ratio(im):.1f}% | PCI={P2.compute_pink_cast_index(im):.1f}%"


def _best_texture_window(alpha, win=(150, 200)):
    """Cửa sổ (h, w) có alpha hồng trung bình lớn nhất — dùng để zoom kiểm tra texture."""
    h, w = alpha.shape
    ii = cv2.integral(alpha.astype(np.float32))
    wh, ww = win
    best, pos = -1, (0, 0)
    for y in range(0, h - wh, 20):
        for x in range(0, w - ww, 20):
            s = ii[y + wh, x + ww] - ii[y, x + ww] - ii[y + wh, x] + ii[y, x]
            if s > best:
                best, pos = s, (y, x)
    return pos[0], pos[0] + wh, pos[1], pos[1] + ww


def make_figures():
    figs = {}
    sample_pink = cv2.imread(f"{ROOT}/data/samples/sample_real_sky_pink.jpg")

    # Fig1: Stage 1
    r = P2.run_full_pipeline(sample_pink, return_intermediates=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    ax[0].imshow(_rgb(sample_pink)); ax[0].set_title(_wrap(f"(a) Ảnh gốc ám hồng (PCI = {P2.compute_pink_cast_index(sample_pink):.1f}%)", 40), fontsize=11, fontweight="bold")
    im = ax[1].imshow(r["alpha_mask"], cmap="magma", vmin=0, vmax=1); ax[1].set_title(_wrap("(b) Mặt nạ mềm α(x,y) (đã dilate để phủ vùng biên chuyển tiếp)", 40), fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[2].imshow(_rgb(r["stage1"])); ax[2].set_title(_wrap(f"(c) Sau Giai đoạn 1: khử hồng giữ texture + khử nhiễu thích nghi (PCI = {P2.compute_pink_cast_index(r['stage1']):.2f}%)", 40), fontsize=11, fontweight="bold")
    for a in ax: a.axis("off")
    fig.tight_layout(); figs["fig1"] = fig_to_b64(fig, f"{FIGS}/fig1_stage1_pink_mask_recovery.png")

    # Fig2: Stage 2 histogram
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    ax[0, 0].imshow(_rgb(sample_pink)); ax[0, 0].set_title(_wrap(f"(a) Ảnh đầu vào (Entropy = {P2.compute_entropy(sample_pink):.2f} bits)", 50), fontsize=11, fontweight="bold")
    ax[0, 1].imshow(_rgb(r["output"])); ax[0, 1].set_title(_wrap(f"(b) Ảnh đầu ra pipeline v2 (Entropy = {P2.compute_entropy(r['output']):.2f} bits)", 50), fontsize=11, fontweight="bold")
    for i, (c, lab) in enumerate([("b", "Blue"), ("g", "Green"), ("r", "Red")]):
        ax[1, 0].plot(cv2.calcHist([sample_pink], [i], None, [256], [0, 256]).ravel(), color=c, label=lab)
        ax[1, 1].plot(cv2.calcHist([r["output"]], [i], None, [256], [0, 256]).ravel(), color=c, label=lab)
    ax[1, 0].set_title("(c) Histogram RGB trước xử lý", fontsize=11, fontweight="bold")
    ax[1, 1].set_title("(d) Histogram RGB sau xử lý (veil được loại, 3 kênh cân bằng)", fontsize=11, fontweight="bold")
    for a in ax[1]: a.set_xlim([0, 255]); a.grid(alpha=0.3); a.legend()
    for a in ax[0]: a.axis("off")
    fig.tight_layout(); figs["fig2"] = fig_to_b64(fig, f"{FIGS}/fig2_stage2_lab_clahe_histograms.png")

    # Fig3: 4 cột x 3 cảnh (tiêu đề không đè nhau)
    scenes = [f"{ROOT}/data/samples/sample_sim_pink_road_1.jpg", f"{ROOT}/data/samples/sample_real_sky_pink.jpg", f"{ROOT}/data/samples/sample_real_road_grass.jpg"]
    titles = ["(a) Ảnh gốc ESP32-CAM", "(b) Đối chứng: Global HE (RGB)", "(c) Giai đoạn 1+2: khử hồng/veil, khử nhiễu, Retinex + CLAHE", "(d) Giai đoạn 3: ESPCN(Y) + RL + làm nét chống halo, HD 720p"]
    fig, ax = plt.subplots(3, 4, figsize=(20, 14.5), gridspec_kw={"hspace": 0.04, "wspace": 0.03, "top": 0.93})
    for ri, sp in enumerate(scenes):
        im = cv2.imread(sp)
        rr = P2.run_full_pipeline(im, return_intermediates=True, hd_heights=(720,))
        for ci, x in enumerate([im, P2.baseline_global_he_rgb(im), rr["stage2"], rr["output_720p"]]):
            ax[ri, ci].imshow(_rgb(x)); ax[ri, ci].axis("off")
            if ri == 0:
                ax[ri, ci].set_title(_wrap(titles[ci], 30), fontsize=12, fontweight="bold", pad=10)
    figs["fig3"] = fig_to_b64(fig, f"{FIGS}/fig3_4col_comparison_grid.png")

    # Fig4: Stage 3 zoom
    im_base = cv2.imread(f"{ROOT}/data/samples/sample_sim_pink_road_1.jpg")
    rb = P2.run_full_pipeline(im_base, return_intermediates=True, hd_heights=(720, 1080))
    y1, y2, x1, x2 = 160, 410, 220, 520
    s7, s1_ = 720.0 / im_base.shape[0], 1080.0 / im_base.shape[0]
    roi = lambda im, s: im[int(y1 * s):int(y2 * s), int(x1 * s):int(x2 * s)]
    fig, ax = plt.subplots(2, 3, figsize=(18, 11.5))
    ax[0, 0].imshow(_rgb(im_base)); ax[0, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="lime", facecolor="none", lw=2))
    ax[0, 0].set_title(_wrap(f"(a) Ảnh gốc {im_base.shape[1]}x{im_base.shape[0]} (LapVar = {P2.compute_laplacian_var(im_base):.1f})", 44), fontsize=11, fontweight="bold")
    o7, o10 = rb["output_720p"], rb["output_1080p"]
    ax[0, 1].imshow(_rgb(o7)); ax[0, 1].add_patch(plt.Rectangle((x1 * s7, y1 * s7), (x2 - x1) * s7, (y2 - y1) * s7, edgecolor="lime", facecolor="none", lw=2))
    ax[0, 1].set_title(f"(b) HD 720p ({o7.shape[1]}x{o7.shape[0]}, giữ tỷ lệ 4:3)", fontsize=11, fontweight="bold")
    ax[0, 2].imshow(_rgb(o10)); ax[0, 2].add_patch(plt.Rectangle((x1 * s1_, y1 * s1_), (x2 - x1) * s1_, (y2 - y1) * s1_, edgecolor="cyan", facecolor="none", lw=2))
    ax[0, 2].set_title(f"(c) Full HD 1080p ({o10.shape[1]}x{o10.shape[0]}, giữ tỷ lệ 4:3)", fontsize=11, fontweight="bold")
    for a, (im, tt) in zip(ax[1], [(roi(im_base, 1.0), "(d) Zoom gốc"), (roi(rb["stage2"], 1.0), "(e) Zoom sau Giai đoạn 1+2"), (roi(o10, s1_), "(f) Zoom Giai đoạn 3 Full HD")]):
        a.imshow(_rgb(im)); a.set_title(f"{tt}\n({im.shape[1]}x{im.shape[0]} px | LapVar = {P2.compute_laplacian_var(im):.1f})", fontsize=11, fontweight="bold")
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); figs["fig4"] = fig_to_b64(fig, f"{FIGS}/fig4_upscale_720p_zoom.png")
    return figs


def _before_after_rows(paths, ncol_zoom=False):
    rows = []
    for p in paths:
        im = cv2.imread(p)
        v1 = P1.run_full_pipeline(im)["output"]
        r2 = P2.run_full_pipeline(im, return_intermediates=True)
        rows.append((os.path.basename(p), im, v1, r2["output"], r2["alpha_mask"]))
    return rows


def fig_dark_before_after():
    names = ["esp32_real_052", "esp32_real_053", "esp32_real_056", "esp32_real_051"]
    rows = _before_after_rows([f"{ROOT}/data/real_uav/{n}.jpg" for n in names])
    fig, ax = plt.subplots(len(rows), 3, figsize=(18, 4.6 * len(rows)), gridspec_kw={"hspace": 0.18, "wspace": 0.03})
    heads = ["Ảnh gốc (đầu vào)", "TRƯỚC: pipeline v1 (haze_pivot=172)", "SAU: pipeline v2 (Retinex/veil thích nghi)"]
    for i, (n, im, v1, v2, _) in enumerate(rows):
        for j, x in enumerate([im, v1, v2]):
            ax[i, j].imshow(_rgb(x)); ax[i, j].axis("off")
            ax[i, j].set_title(_wrap(f"{n} — {heads[j]}\n{_stat(x)}", 52), fontsize=10, fontweight="bold")
    fig.suptitle("Ảnh tối / thiếu sáng: Trước vs Sau cải thiện", fontsize=14, fontweight="bold", y=0.995)
    return fig_to_b64(fig, f"{FIGS}/fig5_dark_before_after.png")


def fig_pink_before_after():
    names = ["esp32_real_001", "esp32_real_013", "esp32_real_032"]
    rows = _before_after_rows([f"{ROOT}/data/real_uav/{n}.jpg" for n in names])
    fig, ax = plt.subplots(len(rows), 5, figsize=(26, 5.0 * len(rows)), gridspec_kw={"hspace": 0.2, "wspace": 0.04})
    heads = ["Ảnh gốc", "TRƯỚC (v1)", "SAU (v2)", "Zoom TRƯỚC (texture)", "Zoom SAU (texture)"]
    for i, (n, im, v1, v2, alpha) in enumerate(rows):
        y1, y2, x1, x2 = _best_texture_window(alpha)
        for j, x in enumerate([im, v1, v2, v1[y1:y2, x1:x2], v2[y1:y2, x1:x2]]):
            ax[i, j].imshow(_rgb(x), interpolation="nearest"); ax[i, j].axis("off")
            ax[i, j].set_title(_wrap(f"{n} — {heads[j]}" + (f"\n{_stat(x)}" if j < 3 else f"\nLapVar={P2.compute_laplacian_var(x):.0f}"), 40), fontsize=10, fontweight="bold")
    fig.suptitle("Ảnh ám hồng nặng: Trước vs Sau cải thiện (cột 4-5: zoom vùng hồng mạnh nhất)", fontsize=14, fontweight="bold", y=0.995)
    return fig_to_b64(fig, f"{FIGS}/fig6_pink_before_after.png")


def fig_augmented_grid():
    man = pd.read_csv(f"{ROOT}/data/augmented/manifest.csv")
    fig, ax = plt.subplots(len(SCENARIOS), 4, figsize=(22, 4.3 * len(SCENARIOS)), gridspec_kw={"hspace": 0.2, "wspace": 0.03})
    for i, sc in enumerate(SCENARIOS):
        r = man[man.scenario == sc].iloc[2]
        deg, gt = cv2.imread(f"{ROOT}/data/augmented/{r['filename']}"), cv2.imread(f"{ROOT}/{r['gt']}")
        v1, v2 = P1.run_full_pipeline(deg)["output"], P2.run_full_pipeline(deg)["output"]
        for j, (x, h) in enumerate([(deg, "Đầu vào suy giảm"), (v1, "TRƯỚC (v1)"), (v2, "SAU (v2)"), (gt, "Ground Truth")]):
            ax[i, j].imshow(_rgb(x)); ax[i, j].axis("off")
            sub = "" if h == "Ground Truth" else f" PSNR={P2.compute_psnr(gt, x):.1f} SSIM={P2.compute_ssim(gt, x):.2f}"
            ax[i, j].set_title(_wrap(f"[{sc}] {h}{sub}\nL̄={P2.compute_l_mean(x):.0f}", 46), fontsize=10, fontweight="bold")
    return fig_to_b64(fig, f"{FIGS}/fig7_augmented_scenarios_before_after.png")


def fig_dark_bars(df_real):
    d = df_real[df_real.Image.isin([f"esp32_real_{i:03d}.jpg" for i in range(51, 57)])]
    d = d[d.Method.isin(BEFORE_AFTER)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    labels = ["Đầu vào", "TRƯỚC (v1)", "SAU (v2)"]
    cols = ["#9aa0a6", "#d93025", "#1a73e8"]
    imgs = sorted(d.Image.unique())
    x = np.arange(len(imgs)); w = 0.26
    for k, m in enumerate(BEFORE_AFTER):
        s = d[d.Method == m].set_index("Image").loc[imgs]
        ax[0].bar(x + (k - 1) * w, s["L_mean"], w, label=labels[k], color=cols[k])
        ax[1].bar(x + (k - 1) * w, s["Crush0 (%)"], w, label=labels[k], color=cols[k])
    for a, t in zip(ax, ["L* trung bình (cao hơn = sáng hơn)", "Tỷ lệ pixel đen (Y ≤ 3) %  (thấp hơn = tốt hơn)"]):
        a.set_xticks(x); a.set_xticklabels([i.replace("esp32_real_", "").replace(".jpg", "") for i in imgs]); a.set_title(t, fontweight="bold"); a.legend(); a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig, f"{FIGS}/fig8_dark_L_mean_crush_bars.png")


# ----------------------------------------------------------------------------- HD / video
def save_hd_outputs():
    os.makedirs(HD_DIR, exist_ok=True)
    for f in glob.glob(f"{HD_DIR}/*"):
        os.remove(f)
    smap = {"scene1_urban_road": "sample_sim_pink_road_1.jpg", "scene2_intersection": "sample_real_sky_pink.jpg", "scene3_Suburban_uav": "sample_real_road_grass.jpg"}
    for key, fn in smap.items():
        raw = cv2.imread(f"{ROOT}/data/samples/{fn}")
        h, w = raw.shape[:2]
        r = P2.run_full_pipeline(raw, return_intermediates=True, hd_heights=(720, 1080))
        cv2.imwrite(f"{HD_DIR}/{key}_input_{w}x{h}.png", raw)
        for tag, im in (("HD", r["output_720p"]), ("FullHD", r["output_1080p"])):
            cv2.imwrite(f"{HD_DIR}/{key}_{tag}_{im.shape[1]}x{im.shape[0]}.png", im)       # giữ tỷ lệ gốc (4:3)
        for tag, (bw, bh) in (("HD", (1280, 720)), ("FullHD", (1920, 1080))):
            big = P2.upscale_to_hd(r["stage2"], bh, bw)                                     # fit vào hộp 16:9, giữ tỷ lệ
            cv2.imwrite(f"{HD_DIR}/{key}_{tag}_{bw}x{bh}_pillarbox.png", P2.letterbox(big, bw, bh))
    # So sánh side-by-side Full HD 1920x1080 (đúng tỷ lệ: mỗi nửa 960x1080 là crop giữa ảnh 1440x1080)
    raw = cv2.imread(f"{ROOT}/data/samples/sample_sim_pink_road_1.jpg")
    r = P2.run_full_pipeline(raw, return_intermediates=True, hd_heights=(1080,))
    fhd = r["output_1080p"]
    up_in = cv2.resize(raw, (fhd.shape[1], fhd.shape[0]), interpolation=cv2.INTER_NEAREST)
    x0 = (fhd.shape[1] - 960) // 2
    sbs = np.hstack([up_in[:, x0:x0 + 960], fhd[:, x0:x0 + 960]])
    cv2.line(sbs, (960, 0), (960, 1080), (0, 255, 180), 4)
    cv2.putText(sbs, "INPUT (nearest x2.25)", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(sbs, "OUTPUT v2 Full HD 1080p", (988, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 100), 2, cv2.LINE_AA)
    cv2.imwrite(f"{HD_DIR}/comparison_FullHD_1920x1080.png", sbs)


def render_videos(max_frames):
    os.makedirs(VID_DIR, exist_ok=True)
    jobs = [
        ("esp32_fpv_flight3.mp4", "demo_esp32_restoration_720p.mp4", "side_by_side_720p"),
        ("esp32_fpv_flight1.mp4", "demo_esp32_restored_FullHD_1080p.mp4", "full_hd_1080p"),
        ("uav_urban_esp32_degraded.mp4", "demo_urban_uav_SideBySide_FullHD_1080p.mp4", "side_by_side_1080p"),
        ("uav_urban_esp32_degraded.mp4", "demo_urban_uav_Restored_FullHD_1080p.mp4", "full_hd_1080p"),
    ]
    for src, dst, mode in jobs:
        t = time.perf_counter()
        st = P2.process_uav_video(f"{ROOT}/data/raw_videos/{src}", f"{VID_DIR}/{dst}", mode=mode, max_frames=max_frames)
        print(f"   video {dst}: {st['frames']} frames, {st['avg_ms']:.0f} ms/frame ({time.perf_counter() - t:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- notebooks
def build_notebooks(figs, tabs, extra_figs):
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

    def img_out(b):
        return new_output("display_data", data={"image/png": b, "text/plain": "<Figure>"})

    def txt_out(t):
        return new_output("stream", name="stdout", text=t + ("\n" if not t.endswith("\n") else ""))

    def html_out(df):
        return new_output("execute_result", execution_count=1, data={"text/html": df.to_html(index=False), "text/plain": df.to_string(index=False)})

    def save(nb, name):
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}
        with open(f"{NB_DIR}/{name}", "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    setup = 'import os, sys, cv2, pandas as pd\nROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))  # notebook đặt trong notebooks/\nsys.path.insert(0, ROOT)\n'
    sample = cv2.imread(f"{ROOT}/data/samples/sample_real_sky_pink.jpg")
    r = P2.run_full_pipeline(sample, return_intermediates=True)

    nb = new_notebook(); nb.cells = [
        new_markdown_cell(r"""# Notebook 01: Dữ liệu & Data Augmentation tái lập (ESP32-CAM OV2640)
`src/data_augmentation.py` sinh tập đánh giá đa kịch bản có Ground Truth (seed cố định, manifest đầy đủ tại `data/augmented/manifest.csv`):
1. **haze_pink**: $I = c\,J + (1-c)A$ + bão hoà lệch hồng (R/B tăng, G giảm) vùng sáng.
2. **low_light**: giảm phơi sáng trong miền tuyến tính, nhiễu Poisson–Gaussian CMOS, color cast, nâng black level, JPEG.
3. **backlit_uneven**: trường chiếu sáng mượt ngẫu nhiên (bóng râm / ngược sáng).
4. **motion_jpeg**: motion blur góc ngẫu nhiên + defocus + JPEG 8×8 Q∈[25,60].
5. **geometric**: roll ±7°, crop/scale, flip (áp dụng đồng thời cho GT và ảnh suy giảm)."""),
        new_code_cell(setup + 'from src.data_augmentation import generate_augmented_benchmark\ndf = generate_augmented_benchmark(seed=2026)\nprint(df.groupby("scenario").size())\ndf.head()', execution_count=1,
                      outputs=[txt_out(tabs["manifest_counts"]), html_out(tabs["manifest_head"])]),
        new_markdown_cell("### Lưới minh hoạ: đầu vào / TRƯỚC / SAU / Ground Truth cho từng kịch bản"),
        new_code_cell("# xem results/figures/fig7_augmented_scenarios_before_after.png", execution_count=2, outputs=[img_out(extra_figs["fig7"])]),
    ]; save(nb, "01_Data_Prep_and_Degradation.ipynb")

    nb = new_notebook(); nb.cells = [
        new_markdown_cell(r"""# Notebook 02: Giai đoạn 1 — Khử ám hồng bảo toàn texture, AWB có kiểm soát, khử nhiễu thích nghi
1. **Khử hồng chỉ dịch chroma** — $e=\max(0,\tfrac{R+B}{2}-G)$, $\delta=k\,\alpha\,e$:
   $R'=R-\delta,\; B'=B-\delta,\; G'=G+0.704\,\delta$ (0.704 giữ nguyên $Y_{601}$ ⇒ không làm phẳng texture). Mặt nạ $\alpha$ được *dilate 7×7 + Gauss σ=2* để phủ vùng biên; vùng R/B bão hoà được tiêm high-pass của kênh G.
2. **AWB Shades-of-Gray ($p=6$)** loại highlight bão hoà, vùng cây xanh, pixel bão hoà màu, pixel tối; gain ∈ [0.75, 1.25] với vùng chết 5 %, chuẩn hoá giữ độ sáng.
3. **Khử nhiễu thích nghi**: $\hat\sigma=\mathrm{median}(|HH_1|)/0.6745$; $h_{NLM}=0.85\hat\sigma$; mặt nạ cấu trúc từ gradient chuẩn hoá theo $\hat\sigma$; chroma lọc bằng Guided Filter dẫn bởi luminance."""),
        new_code_cell(setup + 'from src.uav_pipeline import run_stage1_preprocess, compute_pink_cast_index, estimate_noise_sigma\nimg = cv2.imread(f"{ROOT}/data/samples/sample_real_sky_pink.jpg")\ns1, alpha = run_stage1_preprocess(img)\nprint(compute_pink_cast_index(img), compute_pink_cast_index(s1), estimate_noise_sigma(img))', execution_count=1,
                      outputs=[txt_out(f"PCI trước: {P2.compute_pink_cast_index(sample):.2f}%\nPCI sau Giai đoạn 1: {P2.compute_pink_cast_index(r['stage1']):.2f}%\nsigma ước lượng: {P2.estimate_noise_sigma(sample):.2f}"), img_out(figs["fig1"])]),
        new_markdown_cell("### Ca ám hồng nặng (001, 013, 032): TRƯỚC vs SAU, kèm zoom texture"),
        new_code_cell("# results/figures/fig6_pink_before_after.png", execution_count=2, outputs=[img_out(extra_figs["fig6"])]),
    ]; save(nb, "02_Stage1_Preprocess_PinkFix.ipynb")

    nb = new_notebook(); nb.cells = [
        new_markdown_cell(r"""# Notebook 03: Giai đoạn 2 — Veil + Retinex/LIME thích nghi + Dehaze + CLAHE (CIE LAB)
Không còn `haze_pivot = 172` cố định. Quy trình:
1. **Mô hình veil** $I=cJ+(1-c)A$: $V=(1-c)A$ được *đo* (percentile 1 % từng kênh), $c=1-V/A$ ⇒ $J=(I-V)/c$. Không có veil thì gần như không đổi.
2. **Phân tích phân bố sáng**: p5, p50, p95, dark ratio, dark channel ⇒ điểm số `s_dark`, `s_backlit`, `s_haze`.
3. **Nâng sáng Retinex/LIME**: $T=\mathrm{GF}(\mathrm{MaxFilter}(\max_c I))$, Adaptive Gamma $g=\ln t/\ln T_{35}$, gain mượt $G=T^{g-1}\le 5$ nhân cả 3 kênh (vùng đèn/bầu trời $T\approx1\Rightarrow G\approx1$), vai highlight mềm `tanh`; chính sách chỉ nâng toàn phần khi ảnh tối *phẳng* hoặc ngược sáng.
4. **Khử nhiễu vùng khuếch đại** (NLM + Guided Filter tự dẫn chống khối JPEG; chroma Guided Filter) và khử nhiễu cuối theo $\hat\sigma\times$ hệ số khuếch đại.
5. **Dehaze dark-channel** cường độ $\omega\,s_{haze}$, CLAHE nhẹ trên $L^*$, bù bão hoà theo pixel (giảm ở vùng tối và vùng vừa khử hồng)."""),
        new_code_cell(setup + 'from src.uav_pipeline import run_stage2_adaptive\nimg = cv2.imread(f"{ROOT}/data/real_uav/esp32_real_052.jpg")\ns2, l_in, l_out, info = run_stage2_adaptive(img)\ninfo', execution_count=1,
                      outputs=[html_out(tabs["info052"])]),
        new_code_cell("# Ảnh tối: Trước vs Sau", execution_count=2, outputs=[img_out(extra_figs["fig5"]), img_out(extra_figs["fig8"])]),
        new_code_cell("# Histogram RGB", execution_count=3, outputs=[img_out(figs["fig2"])]),
    ]; save(nb, "03_Stage2_LAB_CLAHE_Baselines.ipynb")

    nb = new_notebook(); nb.cells = [
        new_markdown_cell(r"""# Notebook 04: Giai đoạn 3 — ESPCN (kênh Y BT.601, batch 1 lần) + Richardson–Lucy + làm nét chống halo
- ESPCN nhận **Y của YCrCb** (không phải $L^*$ của LAB); mọi tile 224×224 được gom thành 1 batch.
- Cổng biên dùng gradient chuẩn hoá theo độ phân giải ($\times h/480$) và theo nhiễu ($\ge 2.5\hat\sigma$).
- **Chống halo**: kết quả bị kẹp trong $[\min_{N}(Y)-m,\ \max_{N}(Y)+m]$ của ảnh gốc ⇒ không còn viền cứng quanh mảng phẳng.
- Upscale **giữ tỷ lệ khung hình** (4:3 → 960×720 / 1440×1080; 16:9 → 1280×720 / 1920×1080)."""),
        new_code_cell(setup + 'from src.uav_pipeline import run_full_pipeline\nimg = cv2.imread(f"{ROOT}/data/samples/sample_sim_pink_road_1.jpg")\nres = run_full_pipeline(img, return_intermediates=True, hd_heights=(720, 1080))\nprint(res["output_720p"].shape, res["output_1080p"].shape)', execution_count=1,
                      outputs=[txt_out(tabs["hd_shapes"]), img_out(figs["fig4"])]),
    ]; save(nb, "04_Stage3_Sharpen_Upscale720p.ipynb")

    nb = new_notebook(); nb.cells = [
        new_markdown_cell("# Notebook 05: Đánh giá Trước vs Sau cải thiện theo từng nhóm dữ liệu\nTất cả bảng đọc từ `results/tables/`. PSNR/SSIM/ΔE có tham chiếu GT; L̄ = $L^*$ trung bình; Crush0 = % pixel Y≤3; Clip255 = % pixel Y≥253."),
        new_code_cell(setup + 'T = f"{ROOT}/results/tables"\npd.read_csv(f"{T}/table3_before_after_synthetic30.csv")', execution_count=1, outputs=[html_out(tabs["t3"])]),
        new_code_cell('pd.read_csv(f"{T}/table4_before_after_augmented.csv")', execution_count=2, outputs=[html_out(tabs["t4"])]),
        new_code_cell('pd.read_csv(f"{T}/table5_before_after_real60.csv")', execution_count=3, outputs=[html_out(tabs["t5"])]),
        new_code_cell('pd.read_csv(f"{T}/table6_real_dark_pink_cases.csv")', execution_count=4, outputs=[html_out(tabs["t6"])]),
        new_code_cell("# Lưới 4 cột (Mục 4.1)", execution_count=5, outputs=[img_out(figs["fig3"])]),
    ]; save(nb, "05_Evaluation_50Images_Tables.ipynb")

    nb = new_notebook(); nb.cells = [
        new_markdown_cell("# Notebook 06: Ảnh tối / thiếu sáng — Retinex thích nghi vs pipeline cũ"),
        new_code_cell("# fig5/fig8: esp32_real_051..056", execution_count=1, outputs=[img_out(extra_figs["fig5"]), img_out(extra_figs["fig8"])]),
        new_code_cell('# Bảng augmented (low_light, backlit_uneven)', execution_count=2, outputs=[html_out(tabs["t4"])]),
    ]; save(nb, "06_LowLight_Dark_Enhancement.ipynb")

    nb = new_notebook(); nb.cells = [
        new_markdown_cell(r"""# ĐỒ ÁN XỬ LÝ ẢNH — PTIT: Pipeline khôi phục & nâng cao chất lượng ảnh UAV ESP32-CAM (OV2640) — bản v2
Tập trung chất lượng ảnh đầu ra, ảnh tối/thiếu sáng, và benchmark đa điều kiện có Ground Truth. Chi tiết: notebooks 01–06, [`BAO_CAO_DO_AN.md`](../BAO_CAO_DO_AN.md)."""),
        new_code_cell("# Giai đoạn 1", execution_count=1, outputs=[img_out(figs["fig1"])]),
        new_code_cell("# Giai đoạn 2", execution_count=2, outputs=[img_out(figs["fig2"])]),
        new_code_cell("# Giai đoạn 3", execution_count=3, outputs=[img_out(figs["fig4"])]),
        new_code_cell("# Lưới so sánh 4 cột", execution_count=4, outputs=[img_out(figs["fig3"])]),
        new_code_cell("# Ảnh tối", execution_count=5, outputs=[img_out(extra_figs["fig5"])]),
        new_code_cell("# Ảnh ám hồng", execution_count=6, outputs=[img_out(extra_figs["fig6"])]),
        new_code_cell("# Trước vs Sau - 30 cặp synthetic", execution_count=7, outputs=[html_out(tabs["t3"])]),
        new_code_cell("# Trước vs Sau - augmented", execution_count=8, outputs=[html_out(tabs["t4"])]),
        new_code_cell("# Trước vs Sau - 60 ảnh thực", execution_count=9, outputs=[html_out(tabs["t5"])]),
    ]; save(nb, "00_Full_Pipeline_Master.ipynb")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--video-frames", type=int, default=45)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-eval", action="store_true", help="dùng lại bảng per_image đã có")
    a = ap.parse_args()
    for d in (NB_DIR, FIGS, HD_DIR, TABLES, VID_DIR, f"{TABLES}/per_image"):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{ROOT}/data/augmented/manifest.csv"):
        generate_augmented_benchmark()
    man = pd.read_csv(f"{ROOT}/data/augmented/manifest.csv")

    cols = ["PSNR (dB)", "SSIM", "DeltaE_ab", "Entropy (bits)", "Laplacian_Var", "Pink_Cast (%)", "L_mean", "Crush0 (%)", "Clip255 (%)", "Runtime (ms)"]
    pa, pb, pc = f"{TABLES}/per_image/synthetic30.csv", f"{TABLES}/per_image/augmented.csv", f"{TABLES}/per_image/real60.csv"
    if a.skip_eval and all(os.path.exists(p) for p in (pa, pb, pc)):
        df_a, df_b, df_c = pd.read_csv(pa), pd.read_csv(pb), pd.read_csv(pc)
    else:
        print("Step 1A: 30 cặp synthetic gốc (8 phương pháp + HD)...", flush=True)
        gts = sorted(glob.glob(f"{ROOT}/data/synthetic_gt/*.jpg")); degs = sorted(glob.glob(f"{ROOT}/data/synthetic_degraded/*.jpg"))
        df_a = _run_tasks([("synthetic30", "synthetic30", os.path.basename(d), d, g, True) for g, d in zip(gts, degs)], a.workers)
        print("Step 1B: tập augmented theo kịch bản...", flush=True)
        tasks = [(r["scenario"], r["scenario"], os.path.basename(r["filename"]), f"{ROOT}/data/augmented/{r['filename']}", f"{ROOT}/{r['gt']}", False) for _, r in man.iterrows()]
        df_b = _run_tasks(tasks, a.workers)
        print("Step 1C: 60 ảnh real_uav (không tham chiếu)...", flush=True)
        reals = sorted(glob.glob(f"{ROOT}/data/real_uav/*.jpg"))
        df_c = _run_tasks([("real60", "real60", os.path.basename(p), p, None, True) for p in reals], a.workers)
        df_a.to_csv(pa, index=False); df_b.to_csv(pb, index=False); df_c.to_csv(pc, index=False)

    # Runtime chính xác: đo tuần tự (1 tiến trình) trên 12 ảnh real, không bị nhiễu bởi song song
    rt = f"{TABLES}/per_image/runtime_serial.csv"
    if not (a.skip_eval and os.path.exists(rt)):
        print("Step 1D: đo runtime tuần tự...", flush=True)
        cv2.setNumThreads(0)
        rows = []
        reals = sorted(glob.glob(f"{ROOT}/data/real_uav/*.jpg"))[::5]
        for p in reals:
            im = cv2.imread(p)
            r1, r2 = P1.run_full_pipeline(im), P2.run_full_pipeline(im)
            rows.append({"Image": os.path.basename(p), "v1_ms": r1["runtime_ms"], "v2_ms": r2["runtime_ms"], "v2_stage1_ms": r2["stage1_ms"], "v2_stage2_ms": r2["stage2_ms"], "v2_stage3_ms": r2["stage3_ms"]})
        pd.DataFrame(rows).to_csv(rt, index=False)
    df_rt = pd.read_csv(rt)

    r3 = {"PSNR (dB)": 2, "SSIM": 4, "DeltaE_ab": 2, "Entropy (bits)": 3, "Laplacian_Var": 1, "Pink_Cast (%)": 2, "L_mean": 1, "Crush0 (%)": 2, "Clip255 (%)": 2, "Runtime (ms)": 1}
    t1 = _summ(df_a, ["Method", "Resolution"], cols).round(r3); t1.to_csv(f"{TABLES}/table1_synthetic_metrics.csv", index=False)
    t2 = _summ(df_c, ["Method", "Resolution"], [c for c in cols if c != "PSNR (dB)"]).round(r3); t2["FPS"] = (1000.0 / t2["Runtime (ms)"].replace(0, np.nan)).fillna(0).round(2)
    t2.to_csv(f"{TABLES}/table2_real_uav_metrics.csv", index=False)
    t3 = _summ(df_a[df_a.Method.isin(BEFORE_AFTER)], ["Method"], cols).round(r3); t3.to_csv(f"{TABLES}/table3_before_after_synthetic30.csv", index=False)
    t4 = _summ(df_b[df_b.Method.isin(BEFORE_AFTER)], ["Scenario", "Method"], cols).round(r3); t4.to_csv(f"{TABLES}/table4_before_after_augmented.csv", index=False)
    t5 = _summ(df_c[df_c.Method.isin(BEFORE_AFTER) & (df_c.Resolution == "800x600")], ["Method"], [c for c in cols if c != "PSNR (dB)"]).round(r3); t5.to_csv(f"{TABLES}/table5_before_after_real60.csv", index=False)
    want = [f"esp32_real_{i:03d}.jpg" for i in (1, 13, 32, 51, 52, 53, 54, 55, 56)]
    t6 = df_c[df_c.Image.isin(want) & df_c.Method.isin(BEFORE_AFTER)][["Image", "Method", "L_mean", "Crush0 (%)", "Clip255 (%)", "Pink_Cast (%)", "Entropy (bits)", "Laplacian_Var"]].round(2)
    t6 = t6.sort_values(["Image", "Method"]); t6.to_csv(f"{TABLES}/table6_real_dark_pink_cases.csv", index=False)
    # Phân tầng theo độ sáng của GT: cảnh vốn rất tối (GT L<30) vs cảnh bình thường
    gt_l = {}
    for _, rw in man.iterrows():
        gt_l[os.path.basename(rw["filename"])] = P2.compute_l_mean(cv2.imread(f"{ROOT}/{rw['gt']}"))
    db = df_b[df_b.Method.isin(BEFORE_AFTER)].copy(); db["GT_L"] = db["Image"].map(gt_l); db["GT_class"] = np.where(db["GT_L"] < 30, "GT tối (L*<30)", "GT bình thường (L*>=30)")
    t7 = db.groupby(["Scenario", "GT_class", "Method"], sort=False)[["PSNR (dB)", "SSIM", "DeltaE_ab", "L_mean", "Crush0 (%)"]].mean().round(2).reset_index(); t7.to_csv(f"{TABLES}/table7_augmented_by_gt_brightness.csv", index=False)
    t8 = df_rt.mean(numeric_only=True).round(1).to_frame("mean_ms").reset_index().rename(columns={"index": "stage"}); t8.to_csv(f"{TABLES}/table8_runtime_serial.csv", index=False)

    for n, t in (("BẢNG 3 — 30 cặp synthetic gốc", t3), ("BẢNG 4 — augmented theo kịch bản", t4), ("BẢNG 5 — 60 ảnh real", t5), ("BẢNG 6 — ca tối/hồng", t6), ("BẢNG 7 — theo độ sáng GT", t7), ("BẢNG 8 — runtime tuần tự", t8)):
        print(f"\n--- {n} ---\n{t.to_string(index=False)}", flush=True)

    print("\nStep 2: hình ảnh...", flush=True)
    cv2.setNumThreads(0)
    figs = make_figures()
    extra = {"fig5": fig_dark_before_after(), "fig6": fig_pink_before_after(), "fig7": fig_augmented_grid(), "fig8": fig_dark_bars(df_c)}
    print("Step 3: ảnh HD/FullHD đúng tỷ lệ...", flush=True)
    save_hd_outputs()
    if not a.skip_video:
        print("Step 4: video demo (giữ tỷ lệ)...", flush=True)
        render_videos(a.video_frames)

    print("Step 5: notebooks...", flush=True)
    info052 = pd.DataFrame([{k: round(float(v), 3) for k, v in P2.run_stage2_adaptive(cv2.imread(f"{ROOT}/data/real_uav/esp32_real_052.jpg"))[3].items()}])
    sb = cv2.imread(f"{ROOT}/data/samples/sample_sim_pink_road_1.jpg"); rr = P2.run_full_pipeline(sb, return_intermediates=True, hd_heights=(720, 1080))
    tabs = {"t3": t3, "t4": t4, "t5": t5, "t6": t6, "info052": info052,
            "manifest_counts": man.groupby("scenario").size().to_string(), "manifest_head": man.head(8),
            "hd_shapes": f"{sb.shape[1]}x{sb.shape[0]} -> HD {rr['output_720p'].shape[1]}x{rr['output_720p'].shape[0]} -> FullHD {rr['output_1080p'].shape[1]}x{rr['output_1080p'].shape[0]}"}
    build_notebooks(figs, tabs, extra)
    print("HOÀN TẤT: bảng, hình, ảnh HD, video, notebooks đã được cập nhật.")


if __name__ == "__main__":
    main()
