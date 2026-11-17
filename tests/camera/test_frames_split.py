"""Phase 03: khung bay thật giữ nguyên byte, ghép độ cao, chia dev/test theo khối thời gian.

Ca không marker chạy trên dữ liệu tổng hợp; ca ``data`` đọc ``data/real/`` + bản ghi IOT-CV local.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from xla.camera import frames as F
from xla.camera import split as S
from xla.paths import repo_root

EVENTS_SAMPLE = """\
id               dài(s) bay  cao   xa    pin V        mAh  link%  cam/s phủ%  cảnh báo  mode
20261101-201225     255 có     15.3  35.3 15.44→15.05  562     -     -    -      1    STABILIZE  [dataflash]
20261102-101046     125 có     18.2  42.0 16.61→16.24  285   3.5   7.0   89      4    LOITER,RTL  [dataflash+tlog+video]
      10:11:14 warning camera Mất hình 2.9 s
20261102-101359      29 không   0.5   1.1 16.41→16.35    2   0.0   9.6  100      0    LOITER  [dataflash+tlog+video]
"""


def test_parse_flight_events_keeps_only_flown_with_video():
    fl = F.parse_flight_events(EVENTS_SAMPLE)
    assert [f.flight_id for f in fl] == ["20261102-101046"]
    # 10:10:46 giờ VN = 03:10:46 UTC; dòng đầu frames.csv của 20261102-100941 (10:09:41) có unix 1793588981.537
    assert fl[0].t0 == 1793588981.0 + 65
    assert fl[0].t1 - fl[0].t0 == 125


def test_spike_mask_flags_jump_not_slow_climb():
    t = np.arange(0, 20, 0.2)
    alt = 5 + 0.5 * t  # leo đều 2,5 m/s
    alt[50] = 57.8
    alt[60] = np.nan
    m = F.spike_mask(t, alt)
    assert m[50] and m.sum() == 1


def test_greedy_spaced_and_alt_bin():
    assert S.greedy_spaced([0.0, 0.4, 1.0, 1.9, 2.0, 3.5], 1.0) == [0, 2, 4, 5]
    assert [S.alt_bin(a) for a in (3.0, 7.99, 8.0, 14.0, 30.0, 2.0)] == ["3-8", "3-8", "8-14", ">14", ">14", "<3"]


def test_frame_name_roundtrip():
    n = F.frame_name("20261102-100941", 3, 1234)
    assert n == "20261102-100941_003_01234.jpg"
    assert F.parse_frame_name(n) == ("20261102-100941", 3, 1234)


# ---------------------------------------------------------------- dữ liệu thật


@pytest.fixture(scope="module")
def manifest():
    return F.read_manifest()


@pytest.fixture(scope="module")
def split_doc():
    return json.loads((repo_root() / S.SPLIT_REL).read_text(encoding="utf-8"))


@pytest.mark.data
def test_every_frame_is_byte_copy_of_avi_chunk(manifest):
    video = F._import_iotcv_video()
    chunks: dict[tuple[str, int], list[tuple[int, int]]] = {}
    frames_dir = repo_root() / F.FRAMES_REL
    for r in manifest:
        rec, seg, vf = r["rec"], int(r["segment"]), int(r["video_frame"])
        avi = F.recordings_dir() / rec / f"video-{seg:03d}.avi"
        idx = chunks.setdefault((rec, seg), video.avi_chunk_index(avi))
        off, size = idx[vf]
        with avi.open("rb") as f:
            f.seek(off)
            src = f.read(size)
        data = (frames_dir / r["frame"]).read_bytes()
        assert hashlib.sha256(src).hexdigest() == r["sha256"] == hashlib.sha256(data).hexdigest(), r["frame"]
    assert {p.name for p in frames_dir.glob("*.jpg")} == {r["frame"] for r in manifest}


@pytest.mark.data
def test_flight_frames_above_3m(manifest, split_doc):
    assert all(float(r["alt_m"]) > F.ALT_MIN_M for r in manifest if r["kind"] == "flight")
    assert all(d["alt_m"] > F.ALT_MIN_M for d in split_doc["frames"] if d["split"] != "neg")
    n = sum(r["kind"] == "flight" for r in manifest)
    assert abs(n - F.FLIGHT_FRAMES_EXPECTED) <= F.FLIGHT_FRAMES_EXPECTED * F.FLIGHT_FRAMES_TOL


@pytest.mark.data
def test_split_disjoint_and_buffered(split_doc):
    fr = split_doc["frames"]
    assert split_doc["seed"] == S.SEED_DEFAULT and split_doc["block_s"] == 30
    dev = {d["frame"] for d in fr if d["split"] == "dev"}
    test = {d["frame"] for d in fr if d["split"] == "test"}
    assert dev and test and not dev & test
    assert len({d["frame"] for d in fr}) == len(fr)
    assert not {d["block"] for d in fr if d["split"] == "dev"} & {d["block"] for d in fr if d["split"] == "test"}
    t_dev = np.sort([d["t_unix"] for d in fr if d["split"] == "dev"])
    t_test = np.sort([d["t_unix"] for d in fr if d["split"] == "test"])
    i = np.clip(np.searchsorted(t_dev, t_test), 1, len(t_dev) - 1)
    gap = np.minimum(np.abs(t_test - t_dev[i - 1]), np.abs(t_test - t_dev[i]))
    assert gap.min() >= S.DEV_BUFFER_S
    assert np.diff(t_test).min() >= S.TEST_SPACING_S


@pytest.mark.data
def test_test_has_person_in_two_alt_bins(split_doc):
    persons = S.read_person_frames(repo_root() / S.PRELIM_REL)
    bins = {S.alt_bin(d["alt_m"]) for d in split_doc["frames"] if d["split"] == "test" and d["frame"] in persons}
    assert len(bins) >= 2


@pytest.mark.data
def test_neg_frames(split_doc):
    neg = [d for d in split_doc["frames"] if d["split"] == "neg"]
    assert 0 < len(neg) <= F.NEG_MAX
    assert all(d["frame"].split("_")[0] in F.NEG_RECS for d in neg)
    assert all(d["alt_m"] is None or math.isfinite(d["alt_m"]) for d in neg)


@pytest.mark.data
def test_split_rebuild_is_byte_identical(tmp_path):
    out, rep = tmp_path / "split.json", tmp_path / "rep.json"
    S.main(["build", "--seed", str(S.SEED_DEFAULT), "--out", str(out), "--report", str(rep)])
    assert out.read_bytes() == (repo_root() / S.SPLIT_REL).read_bytes()
