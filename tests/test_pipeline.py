"""Tests for two-pass pipeline, cached recalibration/reassignment, and RTX 4090 GPU smoke test."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.conftest import make_synthetic_baseline_video
from vball_analyser.calibration import auto_detect_net_tape
from vball_analyser.detector import (
    SyntheticColorPredictor,
    YoloGpuPredictor,
    analyse_play_window,
    scan_candidate_plays,
)
from vball_analyser.models import AnalysisSession, Point2D
from vball_analyser.storage import SessionStore
from vball_analyser.video_io import iter_video_frames, probe_video


def test_short_clip_direct_pipeline_and_cached_overrides(
    synthetic_clip: tuple[Path, dict], tmp_path: Path
) -> None:
    clip_path, gt = synthetic_clip
    meta = probe_video(clip_path)
    frames = [bgr for _i, _t, bgr in iter_video_frames(clip_path, sample_fps=4.0)]
    cal = auto_detect_net_tape(frames)

    predictor = SyntheticColorPredictor()
    windows = scan_candidate_plays(clip_path, cal, predictor, meta)
    assert len(windows) == 1

    store = SessionStore(tmp_path / "sessions")
    sid = "sess_short_test"
    sdir = store.create_session_dir(sid)
    play, raw_balls = analyse_play_window(
        clip_path, 1, windows[0][0], windows[0][1], cal, predictor, meta, sdir
    )
    assert play.hitter_track_id is not None
    assert play.set_trajectory is not None
    assert play.hitter_metrics is not None
    assert play.hitter_metrics.contact_height_px is not None
    assert play.hitter_metrics.contact_height_px > 20.0
    assert len(play.keyframes) == 3
    assert all(km.thumbnail_rel_path for km in play.keyframes)

    session = AnalysisSession(
        session_id=sid,
        filename=clip_path.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="ready",
        progress_pct=100.0,
        metadata=meta,
        calibration=cal,
        plays=[play],
    )
    store.save_session(session)
    store.save_play_ball_cache(sid, play.play_id, raw_balls)

    # Recalibrate net manually -> cm values populate cleanly
    recal = store.recalibrate_session_net(
        sid,
        left_point=Point2D(x=gt["left_pt"][0], y=gt["left_pt"][1]),
        right_point=Point2D(x=gt["right_pt"][0], y=gt["right_pt"][1]),
    )
    assert recal.calibration.is_manual is True
    assert recal.plays[0].hitter_metrics is not None
    assert recal.plays[0].hitter_metrics.contact_height_cm is not None

    # Reassign hitter to another cached track (setter track 2)
    other_id = next(t.track_id for t in recal.plays[0].tracks if t.track_id != recal.plays[0].hitter_track_id)
    swapped = store.reassign_play_hitter(sid, play.play_id, other_id)
    assert swapped.plays[0].hitter_track_id == other_id


def test_long_uncut_two_pass_segments_two_rallies(tmp_path: Path) -> None:
    long_path = tmp_path / "uncut_drill.mp4"
    # 24s video > 20s short_clip_max_sec with 2 distinct rallies separated by dead time
    make_synthetic_baseline_video(
        long_path,
        duration_s=24.0,
        fps=24,
        tilt_deg=1.5,
        add_stray_ball=True,
        rally_windows=[(2.0, 3.6), (15.0, 16.6)],
    )
    meta = probe_video(long_path)
    frames = [bgr for _i, _t, bgr in iter_video_frames(long_path, sample_fps=1.0)]
    cal = auto_detect_net_tape(frames)

    predictor = SyntheticColorPredictor()
    windows = scan_candidate_plays(long_path, cal, predictor, meta)
    assert len(windows) == 2
    assert windows[0][0] < 3.0 < windows[0][1]
    assert windows[1][0] < 16.0 < windows[1][1]


def test_yolo_gpu_smoke_on_synthetic_clip(synthetic_clip: tuple[Path, dict]) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    clip_path, _gt = synthetic_clip
    sample = list(iter_video_frames(clip_path, start_s=0.4, end_s=1.0, sample_fps=5.0))
    frames = [bgr for _i, _t, bgr in sample]
    cal = auto_detect_net_tape(frames)

    predictor = YoloGpuPredictor(det_weights="yolo11n.pt", pose_weights="yolo11n-pose.pt")
    assert predictor.device == "cuda:0"
    balls, people = predictor.scan_coarse(sample[:2], cal)
    assert isinstance(balls, list)
    assert isinstance(people, list)
    tracks = predictor.track_poses(sample[:2], cal)
    assert isinstance(tracks, list)
