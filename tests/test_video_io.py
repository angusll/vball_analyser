"""Tests for self-contained PyAV video I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_synthetic_baseline_video
from vball_analyser import video_io
from vball_analyser.video_io import (
    NVDEC_MIN_FRAMES,
    ensure_web_playable_mp4,
    extract_keyframe_jpeg,
    iter_video_frame_batches,
    iter_video_frames,
    parse_byte_range,
    probe_video,
    should_use_nvdec,
    validate_video_magic_bytes,
)


def test_probe_and_iter_synthetic_mp4(synthetic_clip: tuple[Path, dict]) -> None:
    clip_path, gt = synthetic_clip
    meta = probe_video(clip_path)
    assert meta.width == gt["width"]
    assert meta.height == gt["height"]
    assert meta.fps == pytest.approx(gt["fps"], rel=0.05)
    assert meta.frame_count >= gt["frame_count"] - 1
    assert meta.codec == "h264"

    sampled = list(iter_video_frames(clip_path, start_s=0.5, end_s=2.0, sample_fps=10.0))
    assert 12 <= len(sampled) <= 18
    for idx, ts, bgr in sampled:
        assert 0.4 <= ts <= 2.1
        assert bgr.shape == (gt["height"], gt["width"], 3)
        assert idx >= 0


def test_transcode_avi_to_web_playable_h264(tmp_path: Path) -> None:
    avi_path = tmp_path / "legacy_drill.avi"
    make_synthetic_baseline_video(avi_path, duration_s=1.5, fps=24)
    avi_meta = probe_video(avi_path)
    assert avi_meta.codec == "mpeg4"

    proxy_dir = tmp_path / "session_cache"
    web_mp4 = ensure_web_playable_mp4(avi_path, proxy_dir)
    assert web_mp4.name == "web_proxy.mp4"
    assert web_mp4.is_file()
    proxy_meta = probe_video(web_mp4)
    assert proxy_meta.codec == "h264"
    assert proxy_meta.width == 1280
    assert proxy_meta.height == 720


def test_extract_keyframe_jpeg(synthetic_clip: tuple[Path, dict], tmp_path: Path) -> None:
    clip_path, _gt = synthetic_clip
    thumb = extract_keyframe_jpeg(clip_path, frame_idx=30, out_path=tmp_path / "thumb.jpg", max_width=320)
    assert thumb.is_file()
    assert thumb.stat().st_size > 1000


def test_magic_bytes_and_byte_ranges() -> None:
    assert validate_video_magic_bytes(b"\x00\x00\x00\x20ftypisom", ".mp4")
    assert validate_video_magic_bytes(b"\x1a\x45\xdf\xa3\x93\x42", ".mkv")
    assert validate_video_magic_bytes(b"RIFF\x10\x00\x00\x00AVI LIST", ".avi")
    assert not validate_video_magic_bytes(b"<html>not a video</html>", ".mp4")

    assert parse_byte_range("bytes=0-1023", 5000) == (0, 1023)
    assert parse_byte_range("bytes=100-", 5000) == (100, 4999)
    assert parse_byte_range("bytes=-500", 5000) == (4500, 4999)
    with pytest.raises(ValueError):
        parse_byte_range("bytes=6000-7000", 5000)


def test_nvdec_threshold_and_automatic_cpu_fallback(
    synthetic_clip: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_path, _gt = synthetic_clip
    assert NVDEC_MIN_FRAMES == 100_000
    assert should_use_nvdec(None) is False
    assert should_use_nvdec(50_000) is False
    assert should_use_nvdec(100_000) is False
    assert should_use_nvdec(100_001) is True

    cpu_frames = list(iter_video_frames(clip_path, start_s=0.0, end_s=1.0, sample_fps=8.0, use_nvdec=False))
    assert len(cpu_frames) >= 6

    # Simulate NVDEC unavailable -> automatic CPU fallback returns identical frame indices
    monkeypatch.setattr(video_io, "_try_open_nvdec_container", lambda _path: None)
    fallback_frames = list(iter_video_frames(clip_path, start_s=0.0, end_s=1.0, sample_fps=8.0, use_nvdec=True))
    assert [f[0] for f in fallback_frames] == [f[0] for f in cpu_frames]


def test_iter_video_frame_batches_streams_exact_frames_and_stops_cleanly(
    synthetic_clip: tuple[Path, dict],
) -> None:
    clip_path, _gt = synthetic_clip
    direct = list(iter_video_frames(clip_path, start_s=0.0, end_s=1.5, sample_fps=10.0, use_nvdec=False))
    batches = list(
        iter_video_frame_batches(
            clip_path, start_s=0.0, end_s=1.5, sample_fps=10.0, batch_size=4, use_nvdec=False, prefetch_batches=1
        )
    )
    assert len(batches) >= 3
    assert all(1 <= len(b) <= 4 for b in batches)
    flat = [item for b in batches for item in b]
    assert [f[0] for f in flat] == [f[0] for f in direct]

    # Early break should shut down producer thread cleanly without hanging
    import threading

    gen = iter_video_frame_batches(
        clip_path, start_s=0.0, end_s=1.5, sample_fps=10.0, batch_size=2, use_nvdec=False, prefetch_batches=1
    )
    first_batch = next(gen)
    assert len(first_batch) == 2
    gen.close()
    assert not any(t.name == "video-batch-decoder" and t.is_alive() for t in threading.enumerate())


def test_nvdec_pre_first_frame_fallback_vs_mid_stream_error_and_short_slice(
    synthetic_clip: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_path, _gt = synthetic_clip
    cpu_frames = list(iter_video_frames(clip_path, start_s=0.0, end_s=1.0, sample_fps=8.0, use_nvdec=False))

    # 1. Pre-first-frame decode failure on NVDEC container closes container and falls back to CPU
    closed = {"flag": False}

    class _PreFailContainer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed["flag"] = True

    monkeypatch.setattr(video_io, "_try_open_nvdec_container", lambda _p: _PreFailContainer())
    orig_decode = video_io._decode_container_frames

    def _fail_first(container, start_s, end_s, sample_fps):
        if isinstance(container, _PreFailContainer):
            raise RuntimeError("simulated NVDEC pre-frame failure")
        yield from orig_decode(container, start_s, end_s, sample_fps)

    monkeypatch.setattr(video_io, "_decode_container_frames", _fail_first)
    fb = list(iter_video_frames(clip_path, start_s=0.0, end_s=1.0, sample_fps=8.0, use_nvdec=True))
    assert closed["flag"] is True
    assert [f[0] for f in fb] == [f[0] for f in cpu_frames]

    # 2. Mid-stream decode failure after >=1 frame yielded re-raises instead of restarting on CPU
    def _fail_mid(container, start_s, end_s, sample_fps):
        if isinstance(container, _PreFailContainer):
            yield cpu_frames[0]
            raise RuntimeError("simulated NVDEC mid-stream error")
        yield from orig_decode(container, start_s, end_s, sample_fps)

    monkeypatch.setattr(video_io, "_decode_container_frames", _fail_mid)
    with pytest.raises(RuntimeError, match="mid-stream"):
        list(iter_video_frames(clip_path, start_s=0.0, end_s=1.0, sample_fps=8.0, use_nvdec=True))

    # 3. use_nvdec=None does NOT trigger NVDEC for a short [0, 1s] slice even on a >100k-frame video
    nv_called = {"count": 0}

    def _track_nv(_p):
        nv_called["count"] += 1
        return None

    monkeypatch.setattr(video_io, "_decode_container_frames", orig_decode)
    monkeypatch.setattr(video_io, "_estimate_stream_frames", lambda _c, _s: 150_000)
    monkeypatch.setattr(video_io, "_try_open_nvdec_container", _track_nv)
    list(iter_video_frames(clip_path, start_s=0.0, end_s=1.0, sample_fps=8.0, use_nvdec=None))
    assert nv_called["count"] == 0
    # Full video (end_s=None) on >100k frames DOES attempt NVDEC
    list(iter_video_frames(clip_path, start_s=0.0, end_s=None, sample_fps=8.0, use_nvdec=None))
    assert nv_called["count"] == 1


def test_iter_video_frame_batches_propagates_producer_exception(
    synthetic_clip: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_path, _gt = synthetic_clip

    def _boom(*_args, **_kwargs):
        raise ValueError("decoder boom")
        yield  # pragma: no cover

    monkeypatch.setattr(video_io, "iter_video_frames", _boom)
    with pytest.raises(ValueError, match="decoder boom"):
        list(iter_video_frame_batches(clip_path, batch_size=2))
