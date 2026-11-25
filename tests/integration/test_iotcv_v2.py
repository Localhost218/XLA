"""Phase 15 (plan 261125-ai-v2) — IOT-CV chạy bundle v2 thật: sha, golden CUDA, mặc định config, ms v1 vs v2.

Đọc repo IOT-CV ở ``xla.paths.iotcv_root()`` (``XLA_IOTCV_ROOT`` ghi đè, vd. worktree trước khi merge) và bench xen kẽ
``results/v2/iotcv/bench.json`` (``results/v2/iotcv/bench_v1_v2.py``, cách chạy ở ``results/v2/iotcv/README.md``).
Bundle ship theo ``results/ab-v2/decision.json`` + Kill (b)–(d) của plan: ``detector-v1.0.1`` (Kill (b): không có
``detector-v2.0.0``), ``verifier-v1.0.0``, ``enhancer-v2.0.0``; ``enhancer-offline`` chỉ khi ``offline_ship``.
Thiếu dữ liệu → FAIL (``require_data``), không skip. Riêng manifest job offline: ``offline_ship`` false thì không có
chuyến offline nào để kiểm → skip kèm lý do (đúng plan: offline không ship thì API trả 503, không có job).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from xla.paths import gpu_lock_file, iotcv_root, repo_root

pytestmark = [pytest.mark.data, pytest.mark.gpu]

SHIP = {"detector": "detector-v1.0.1", "verifier": "verifier-v1.0.0", "enhancer": "enhancer-v2.0.0"}
OFFLINE_KIND = "enhancer-offline"
CEILING_MS = 5.0
TIMEOUT_S = 600

# Nạp config.py của IOT-CV ở thư mục tạm (PROJECT_ROOT = thư mục đó): .env của máy không lọt vào phép thử mặc định.
_CONFIG_PROBE = r"""
import importlib.util, json, os, shutil, sys, tempfile
from pathlib import Path
src = Path(sys.argv[1])
for k in [k for k in os.environ if k.endswith(("_ONNX", "_SHA256")) or k == "ENHANCER_DEFAULT_ON"]:
    del os.environ[k]
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "backend").mkdir()
    shutil.copy(src, Path(d) / "backend" / "config.py")
    spec = importlib.util.spec_from_file_location("_cfg", Path(d) / "backend" / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    root = Path(d).resolve()
    rel = lambda p: Path(p).resolve().relative_to(root).as_posix() if p else ""
    print(json.dumps({
        "detector": [rel(m.DETECTOR_ONNX), m.DETECTOR_SHA256],
        "verifier": [rel(m.VERIFIER_ONNX), m.VERIFIER_SHA256],
        "enhancer": [rel(m.ENHANCER_ONNX), m.ENHANCER_SHA256],
        "enhancer-offline": [rel(m.ENHANCER_OFFLINE_ONNX), m.ENHANCER_OFFLINE_SHA256],
        "enhancer_default_on": m.ENHANCER_DEFAULT_ON,
    }))
"""


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(bundle: str) -> dict:
    return json.loads((repo_root() / "models" / "release" / bundle / "meta.json").read_text(encoding="utf-8"))


@pytest.fixture
def iotcv(require_data) -> Path:
    return require_data(
        Path(iotcv_root()) / "models" / "vision", "chép bundle bằng scripts/vision/lay_model.py"
    ).parent.parent


@pytest.fixture
def decision(require_data) -> dict:
    p = require_data(repo_root() / "results" / "ab-v2" / "decision.json", "phase 14")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def bench(require_data) -> dict:
    p = require_data(repo_root() / "results" / "v2" / "iotcv" / "bench.json", "chạy results/v2/iotcv/bench_v1_v2.py")
    return json.loads(p.read_text(encoding="utf-8"))


def test_sha_bundle_iotcv_bang_meta_release(iotcv):
    for kind, bundle in SHIP.items():
        dst = iotcv / "models" / "vision" / bundle
        meta = _meta(bundle)
        assert _sha(dst / f"{kind}.onnx") == meta["onnx_sha256"], bundle
        assert json.loads((dst / "meta.json").read_text(encoding="utf-8")) == meta, f"{bundle}/meta.json lệch release"


def test_models_vision_khop_decision(iotcv, decision):
    """Có bundle ``enhancer-offline-*`` trong ``models/vision/`` ⇔ ``offline_ship`` (decision.json)."""
    offline = sorted(p.name for p in (iotcv / "models" / "vision").iterdir() if p.name.startswith(f"{OFFLINE_KIND}-"))
    assert bool(offline) == bool(decision["offline_ship"]), (
        f"models/vision có {offline or 'không có'} bundle offline nhưng offline_ship={decision['offline_ship']} "
        f"({decision.get('offline_reason')})"
    )


def test_config_mac_dinh_tro_bundle_ship(iotcv, decision):
    p = subprocess.run(
        [sys.executable, "-c", _CONFIG_PROBE, str(iotcv / "backend" / "config.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert p.returncode == 0, p.stderr[-1500:]
    cfg = json.loads(p.stdout.strip().splitlines()[-1])
    for kind, bundle in SHIP.items():
        assert cfg[kind] == [f"models/vision/{bundle}/{kind}.onnx", _meta(bundle)["onnx_sha256"]], kind
    assert cfg["enhancer_default_on"] is True  # classic mặc định bật (quyết định chủ dự án, Deferred 17)
    assert (cfg[OFFLINE_KIND][0] != "") == bool(decision["offline_ship"]), cfg[OFFLINE_KIND]


def test_golden_cuda_trong_iotcv(iotcv):
    """``scripts/vision/kiem_bundle.py`` của IOT-CV (venv IOT-CV, env con bỏ venv XLA), CUDA đứng đầu provider."""
    from xla.eval.system import iotcv_child_env

    lock = gpu_lock_file()
    if lock.is_file() and not lock.read_text(encoding="utf-8").strip().startswith("free"):
        pytest.fail(
            f"khoá GPU XLA đang bị giữ ({lock.read_text(encoding='utf-8').strip()[:120]}): chạy lại sau", pytrace=False
        )
    env = iotcv_child_env(iotcv=iotcv)
    for bundle in SHIP.values():
        p = subprocess.run(
            ["uv", "run", "--no-sync", "python", "scripts/vision/kiem_bundle.py", f"models/vision/{bundle}", "--json"],
            cwd=iotcv,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
            check=False,
        )
        res = json.loads(p.stdout.strip().splitlines()[-1])
        assert p.returncode == 0 and res["pass"] is True, (bundle, res)
        assert res["providers"][0] == "CUDAExecutionProvider", (bundle, res["providers"])
        assert res["onnx_sha256"] == _meta(bundle)["onnx_sha256"], bundle


def test_bench_dung_bundle_ship(bench):
    for kind, bundle in SHIP.items():
        b = bench["bundles"]["v2"][kind]
        assert (b["bundle"], b["onnx_sha256"]) == (bundle, _meta(bundle)["onnx_sha256"]), kind
        assert b["providers"][0] == "CUDAExecutionProvider", kind
    assert all(bench["golden_cuda"].values()), bench["golden_cuda"]


def test_realtime_khong_loi(bench):
    assert bench["errors"] == {"v1": 0, "v2": 0}, bench["errors"]


def test_ms_v2_khong_qua_v1_cong_5ms(bench):
    """p95 tổng nhịp thật (có người xem "Ảnh rõ") của v2 ≤ v1 + 5 ms, cùng phiên GPU, lượt xen kẽ ABBA."""
    c = bench["ceiling"]["realtime"]
    assert c["v2_p95"] - c["v1_p95"] <= CEILING_MS, (
        f"v2 p95 {c['v2_p95']} ms > v1 {c['v1_p95']} + {CEILING_MS} ms; máy lúc đo {bench['machine_start']}. "
        "Không nới trần, không khoá xung GPU."
    )


def test_manifest_offline_xong(decision, iotcv):
    if not decision["offline_ship"]:
        pytest.skip(
            f"offline_ship = false ({decision.get('offline_reason')}): enhancer-offline không ship, không có "
            "job offline thật để kiểm manifest done == total (API offline trả 503, web tắt nút)"
        )
    manifests = sorted((iotcv / "logs" / ".cache" / "flights-ai").glob(f"*/{OFFLINE_KIND}-v*/manifest.json"))
    assert manifests, "offline_ship nhưng không có manifest job offline nào"
    m = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert m["done"] == m["total"] > 0, m
