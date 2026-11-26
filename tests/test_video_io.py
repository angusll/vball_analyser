"""Tests for self-contained PyAV video I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_synthetic_baseline_video
from vball_analyser.video_io import (
    ensure_web_playable_mp4,
    extract_keyframe_jpeg,
    iter_video_frames,
    parse_byte_range,
    probe_video,
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
