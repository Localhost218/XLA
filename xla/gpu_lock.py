"""Khoá GPU dùng chung giữa các tiến trình (D9): một job GPU tại một thời điểm.

Dùng::

    from xla.gpu_lock import gpu_lock
    with gpu_lock("phase10 train yolo11s", timeout_s=3600):
        ...  # việc GPU

Khoá là khoá file của hệ điều hành (``msvcrt.locking`` trên Windows, ``fcntl.flock`` trên POSIX) trên
``xla.paths.gpu_lock_file()``: tiến trình chết thì hệ điều hành tự nhả, không có khoá mồ côi. Khoá theo handle, nên
lần khoá thứ hai trong CÙNG tiến trình cũng phải chờ. Người giữ khoá ghi ``pid``, ``label``, giờ vào đầu file để biết ai
đang giữ (``read_holder()``).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from xla.paths import gpu_lock_file

_INFO_BYTES = 512  # bản ghi người giữ, đệm cố định ở đầu file
_LOCK_OFFSET = 1 << 30  # byte bị khoá (Windows), nằm sau bản ghi để người chờ vẫn đọc được bản ghi

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _write_info(fd: int, text: str) -> None:
    data = text.encode("utf-8")[: _INFO_BYTES - 1]
    data = data + b" " * (_INFO_BYTES - 1 - len(data)) + b"\n"
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, data)


def read_holder(path: Path | None = None) -> str:
    """Bản ghi người đang/đã giữ khoá (chuỗi rỗng nếu chưa có file)."""
    p = path or gpu_lock_file()
    if not p.exists():
        return ""
    with open(p, "rb", buffering=0) as f:
        return f.read(_INFO_BYTES).decode("utf-8", errors="replace").strip()


@contextmanager
def gpu_lock(label: str = "", timeout_s: float | None = None, poll_s: float = 2.0) -> Iterator[Path]:
    """Chờ và giữ khoá GPU trong khối ``with``.

    ``timeout_s=None`` chờ mãi; quá ``timeout_s`` giây chưa lấy được khoá thì ném ``TimeoutError`` (kèm bản ghi người
    đang giữ). Nhả khoá khi ra khỏi khối, kể cả khi có lỗi. Trả về đường dẫn file khoá.
    """
    if poll_s <= 0:
        raise ValueError("poll_s phải > 0")
    path = gpu_lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        start = time.monotonic()
        while not _try_lock(fd):
            waited = time.monotonic() - start
            if timeout_s is not None and waited >= timeout_s:
                raise TimeoutError(
                    f"chờ khoá GPU {path} quá {timeout_s:g} s; đang giữ: {read_holder(path) or 'không rõ'}"
                )
            sleep = poll_s if timeout_s is None else max(0.0, min(poll_s, timeout_s - waited))
            time.sleep(sleep)
        try:
            cmd = " ".join(sys.argv)
            _write_info(
                fd, f"pid={os.getpid()} label={label} since={datetime.now().isoformat(timespec='seconds')} cmd={cmd}"
            )
            yield path
        finally:
            try:
                _write_info(
                    fd,
                    f"free (lần cuối: pid={os.getpid()} label={label} released={datetime.now().isoformat(timespec='seconds')})",
                )
            finally:
                _unlock(fd)
    finally:
        os.close(fd)
