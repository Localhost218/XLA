"""Phase 01 Accept: package, đường dẫn, archive, web, .gitignore, khoá GPU."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import xla
from xla import paths

ROOT = Path(__file__).resolve().parents[1]
SUBPACKAGES = ("camera", "degrade", "enhance", "detect", "eval")
ARCHIVED = (
    "data/real_uav",
    "data/synthetic_gt",
    "data/synthetic_degraded",
    "data/raw_videos",
    "data/augmented",
    "results/tables",
    "results/figures",
)


def test_version() -> None:
    assert isinstance(xla.__version__, str) and xla.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports_with_docstring(name: str) -> None:
    mod = importlib.import_module(f"xla.{name}")
    assert mod.__doc__ and mod.__doc__.strip()


def test_repo_root_is_this_checkout() -> None:
    assert paths.repo_root().resolve() == ROOT
    assert (paths.repo_root() / "pyproject.toml").is_file()


@pytest.mark.parametrize("key", paths.ALL_KEYS)
def test_paths_not_on_drive_c(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(f"XLA_{key.upper()}", raising=False)
    value = str(getattr(paths, key)())
    assert not value.lower().startswith(("c:", "c\\", "c/")), f"{key} trỏ ổ C: {value}"


def test_paths_config_has_every_key() -> None:
    assert set(paths.load_config()) >= set(paths.ALL_KEYS)


def test_env_overrides_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XLA_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path


def test_missing_key_is_an_error() -> None:
    with pytest.raises(KeyError):
        paths.get_path("khong_co_khoa_nay")


@pytest.mark.parametrize("rel", ARCHIVED)
def test_old_data_moved_to_archive(rel: str) -> None:
    assert not (ROOT / rel).exists(), f"{rel} vẫn còn ở chỗ cũ"
    assert (ROOT / "archive" / "v2-nhom" / rel).is_dir(), f"thiếu archive/v2-nhom/{rel}"


@pytest.mark.parametrize(
    "phrase",
    [
        "ảnh drone phố có biển chữ Hán cộng hồng nhân tạo",
        "không phải camera này",
        "video đã mã hoá lại cảnh khác",
        "Bảng 1–7 đo trên dữ liệu do chính mô hình suy giảm sinh ra",
        "git checkout 3e9a764",
    ],
)
def test_archive_readme_states_facts(phrase: str) -> None:
    text = (ROOT / "archive" / "README.md").read_text(encoding="utf-8")
    assert phrase in " ".join(text.split()), phrase


def test_web_server_binds_localhost_only() -> None:
    text = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in text
    assert 'host="127.0.0.1"' in text


GITIGNORE_LINES = ("models/release/**/*.onnx", "runs/", "results/**/runs/", "*.pt", ".pytest_cache/")


@pytest.mark.parametrize("line", GITIGNORE_LINES)
def test_gitignore_has_line(line: str) -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert line in [x.strip() for x in lines]


@pytest.mark.skipif(shutil.which("git") is None, reason="không có git")
@pytest.mark.parametrize(
    ("rel", "ignored"),
    [
        ("data/real/frames/a.jpg", False),
        ("results/ab/img/a.jpg", False),
        ("models/release/x-v0.0.0/golden/000.jpg", False),
        ("models/release/x-v0.0.0/detector.onnx", True),
        ("runs/detect/x.txt", True),
    ],
)
def test_git_check_ignore(rel: str, ignored: bool) -> None:
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, capture_output=True, check=False)
    assert r.returncode in (0, 1), r.stderr
    assert (r.returncode == 0) is ignored, f"{rel}: ignored={r.returncode == 0}, mong {ignored}"


@pytest.fixture
def lock_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    p = tmp_path / "locks" / "gpu.lock"
    monkeypatch.setenv("XLA_GPU_LOCK_FILE", str(p))
    return p


def test_gpu_lock_acquires_and_records_holder(lock_file: Path) -> None:
    from xla.gpu_lock import gpu_lock, read_holder

    with gpu_lock("thu-nghiem", timeout_s=1) as f:
        assert f == lock_file and lock_file.exists()
        holder = read_holder()
        assert "label=thu-nghiem" in holder and "pid=" in holder
    assert read_holder().startswith("free")


def test_gpu_lock_second_acquire_waits_then_times_out(lock_file: Path) -> None:
    from xla.gpu_lock import gpu_lock

    with gpu_lock("a", timeout_s=1):
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="label=a"), gpu_lock("b", timeout_s=0.5, poll_s=0.1):
            pytest.fail("lấy được khoá thứ hai khi khoá đầu còn giữ")
        waited = time.monotonic() - t0
        assert 0.45 <= waited < 3.0, waited
    with gpu_lock("c", timeout_s=0.5):
        pass  # đã nhả, lấy lại được


def test_gpu_lock_released_on_exception(lock_file: Path) -> None:
    from xla.gpu_lock import gpu_lock

    with pytest.raises(RuntimeError), gpu_lock("loi", timeout_s=1):
        raise RuntimeError("lỗi giữa chừng")
    with gpu_lock("sau-loi", timeout_s=0.5):
        pass
