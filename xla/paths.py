"""Đường dẫn dùng chung của XLA, đọc từ ``configs/paths.yaml``.

Thứ tự ưu tiên cho mỗi khoá: biến môi trường ``XLA_<KHOÁ>`` (viết hoa) > ``configs/paths.yaml``. Thiếu khoá ở cả hai
nơi là lỗi (``KeyError``), không có giá trị mặc định ẩn trong code.

``configs/paths.yaml`` là ánh xạ phẳng ``khoá: giá trị`` nên module này chỉ dùng thư viện chuẩn: mọi lane import được
kể cả khi venv chưa có pyyaml. Đường dẫn tương đối tính từ gốc repo.
"""

from __future__ import annotations

import os
from pathlib import Path

_CONFIG_REL = Path("configs") / "paths.yaml"


def repo_root() -> Path:
    """Gốc repo (worktree) chứa package ``xla`` đang chạy. Ghi đè bằng ``XLA_REPO_ROOT``."""
    env = os.environ.get("XLA_REPO_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def _parse_flat_yaml(text: str, source: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if raw[:1].isspace() or ":" not in line:
            raise ValueError(f"{source}:{lineno}: chỉ nhận ánh xạ phẳng 'khoá: giá trị', gặp {raw!r}")
        key, value = line.split(":", 1)
        # "D:/x" — dấu ':' đầu tiên là phân cách khoá; phần sau giữ nguyên.
        value = value.strip().strip('"').strip("'")
        if not key.strip() or not value:
            raise ValueError(f"{source}:{lineno}: khoá hoặc giá trị rỗng: {raw!r}")
        out[key.strip()] = value
    return out


def load_config() -> dict[str, str]:
    """Đọc ``configs/paths.yaml`` của repo đang chạy (không áp biến môi trường)."""
    path = repo_root() / _CONFIG_REL
    return _parse_flat_yaml(path.read_text(encoding="utf-8"), path)


def get_path(key: str) -> Path:
    """Đường dẫn theo khoá; ``XLA_<KEY>`` ghi đè. Tương đối thì tính từ gốc repo."""
    env = os.environ.get(f"XLA_{key.upper()}")
    if env:
        value = env
    else:
        cfg = load_config()
        if key not in cfg:
            raise KeyError(f"thiếu khoá '{key}' trong {repo_root() / _CONFIG_REL} và biến XLA_{key.upper()}")
        value = cfg[key]
    p = Path(value)
    if not p.is_absolute() and not _looks_like_windows_drive(value):
        p = repo_root() / p
    return p


def _looks_like_windows_drive(value: str) -> bool:
    return len(value) >= 2 and value[1] == ":" and value[0].isalpha()


def data_root() -> Path:
    """Tập sinh D0/D1/D2 (``XLA_DATA_ROOT``)."""
    return get_path("data_root")


def src_datasets_dir() -> Path:
    """Dataset nguồn đã giải nén."""
    return get_path("src_datasets_dir")


def downloads_dir() -> Path:
    """Thư mục file tải về."""
    return get_path("downloads_dir")


def cache_dir() -> Path:
    """Cache tính toán."""
    return get_path("cache_dir")


def iotcv_root() -> Path:
    """Gốc repo IOT-CV (chỉ đọc)."""
    return get_path("iotcv_root")


def gpu_lock_file() -> Path:
    """File khoá GPU dùng chung (D9)."""
    return get_path("gpu_lock_file")


ALL_KEYS = ("data_root", "src_datasets_dir", "downloads_dir", "cache_dir", "iotcv_root", "gpu_lock_file")
