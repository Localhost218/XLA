"""C2ST — kiểm định hai mẫu bằng bộ phân loại (Lopez-Paz & Oquab, 1610.06545), phase 06 bước 4.

CNN nhỏ (≤ 100 k tham số) phân biệt patch thật/tổng hợp trên **phần dư cao tần** (ảnh − Gauss σ=2), để nội dung cảnh không
lấn khác biệt suy giảm. Chia train/val/test theo **ảnh nguồn**: các patch cùng một ảnh không rơi vào hai phía. Chọn epoch
theo val, báo độ chính xác test + CI 95 % bootstrap theo cụm ảnh nguồn. ~0,5 = không phân biệt được.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from xla.eval.patches import Patches

MAX_PARAMS = 100_000


def highpass_residual(pixels_bgr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """[N, S, S, 3] uint8 → [N, 3, S, S] float32: ảnh/255 − GaussianBlur(σ), biên phản xạ (giống nhau cho hai tập)."""
    out = np.empty((len(pixels_bgr), 3) + pixels_bgr.shape[1:3], dtype=np.float32)
    for i, p in enumerate(pixels_bgr):
        f = p.astype(np.float32) / 255.0
        out[i] = (f - cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)).transpose(2, 0, 1)
    return out


def build_net():
    """CNN ~60 k tham số: 4 tầng conv 3×3 (3 tầng stride 2) → pooling trung bình toàn cục → 1 logit."""
    from torch import nn

    def block(cin, cout, stride):
        return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

    net = nn.Sequential(
        *block(3, 16, 1),
        *block(16, 32, 2),
        *block(32, 64, 2),
        *block(64, 64, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
    )
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params <= MAX_PARAMS, n_params
    return net


def split_by_source(sources: np.ndarray, rng, fracs=(0.5, 0.2, 0.3)) -> list[np.ndarray]:
    """Chia chỉ số patch thành train/val/test theo ảnh nguồn (không ảnh nào nằm ở hai phần)."""
    uniq = rng.permutation(np.unique(sources))
    if len(uniq) < 3:
        raise ValueError(f"C2ST cần ≥ 3 ảnh nguồn mỗi tập để chia theo ảnh (có {len(uniq)})")
    cut1 = max(1, round(fracs[0] * len(uniq)))
    cut2 = min(len(uniq) - 1, max(cut1 + 1, round((fracs[0] + fracs[1]) * len(uniq))))
    groups = [uniq[:cut1], uniq[cut1:cut2], uniq[cut2:]]
    return [np.flatnonzero(np.isin(sources, g)) for g in groups]


def _cluster_bootstrap(correct: np.ndarray, clusters: np.ndarray, rng, n_boot: int) -> list[float]:
    uniq = np.unique(clusters)
    per = [correct[clusters == c] for c in uniq]
    accs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per), size=len(per))
        sel = np.concatenate([per[k] for k in pick])
        accs.append(sel.mean())
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def c2st(
    real: Patches,
    synth: Patches,
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    n_boot: int = 1000,
    sigma: float = 2.0,
) -> dict:
    """Độ chính xác test của CNN phân biệt `real` (nhãn 0) với `synth` (nhãn 1)."""
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xr, xs = highpass_residual(real.pixels, sigma), highpass_residual(synth.pixels, sigma)
    parts_r, parts_s = split_by_source(real.source, rng), split_by_source(synth.source, rng)

    def take(k):
        x = np.concatenate([xr[parts_r[k]], xs[parts_s[k]]])
        y = np.concatenate([np.zeros(len(parts_r[k])), np.ones(len(parts_s[k]))]).astype(np.float32)
        # cụm = ảnh nguồn, đánh số riêng cho hai tập
        c = np.concatenate([real.source[parts_r[k]], synth.source[parts_s[k]] + len(real.paths)])
        return x, y, c

    (xtr, ytr, _), (xva, yva, _), (xte, yte, cte) = take(0), take(1), take(2)
    scale = float(xtr.std()) or 1.0
    to_t = lambda a: torch.from_numpy(a / scale).float().to(device)
    xtr_t, xva_t, xte_t = to_t(xtr), to_t(xva), to_t(xte)
    ytr_t = torch.from_numpy(ytr).to(device)
    # cân lớp trong loss: hai tập có thể lệch số patch sau khi chia theo ảnh
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)

    net = build_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def predict(x):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(x[s : s + 512]).squeeze(1) for s in range(0, len(x), 512)]).cpu().numpy() > 0

    def bal_acc(pred, y):
        return 0.5 * ((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean())

    best = (-1.0, None, -1)
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(xtr_t))).to(device)
        for s in range(0, len(perm), batch_size):
            idx = perm[s : s + batch_size]
            if len(idx) < 2:
                continue
            loss = F.binary_cross_entropy_with_logits(net(xtr_t[idx]).squeeze(1), ytr_t[idx], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = bal_acc(predict(xva_t), yva)
        if va > best[0]:
            best = (va, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    pred = predict(xte_t)
    correct = (pred == (yte == 1)).astype(np.float64)
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(bal_acc(pred, yte)),
        "ci95": _cluster_bootstrap(correct, cte, rng, n_boot),
        "val_balanced_accuracy": float(best[0]),
        "best_epoch": int(best[2]),
        "epochs": epochs,
        "n_train": len(ytr),
        "n_val": len(yva),
        "n_test": len(yte),
        "n_test_images": len(np.unique(cte)),
        "n_params": int(sum(p.numel() for p in net.parameters())),
        "patch": int(real.size),
        "residual": f"img/255 - GaussianBlur(sigma={sigma})",
        "seconds": round(time.perf_counter() - t0, 2),
    }
