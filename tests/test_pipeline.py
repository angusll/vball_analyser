"""Tests for two-pass pipeline, cached recalibration/reassignment, and RTX 4090 GPU smoke test."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.conftest import make_synthetic_baseline_video
from vball_analyser.calibration import auto_detect_net_tape
from types import SimpleNamespace
from vball_analyser.detector import (
    SyntheticColorPredictor,
    YoloGpuPredictor,
    analyse_play_window,
    detect_inference_device,
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


def test_detect_inference_device_priority_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBALL_DEVICE", raising=False)

    def _fake_torch(*, cuda: bool = False, xpu: bool = False, mps: bool = False) -> object:
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            xpu=SimpleNamespace(is_available=lambda: xpu),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )

    assert detect_inference_device(_fake_torch(cuda=True, xpu=True, mps=True)) == "cuda:0"
    assert detect_inference_device(_fake_torch(cuda=False, xpu=True, mps=True)) == "xpu:0"
    assert detect_inference_device(_fake_torch(cuda=False, xpu=False, mps=True)) == "mps"
    assert detect_inference_device(_fake_torch(cuda=False, xpu=False, mps=False)) == "cpu"

    monkeypatch.setenv("VBALL_DEVICE", "mps")
    assert detect_inference_device(_fake_torch(cuda=True)) == "mps"


def test_predictor_device_batch_defaults_and_weight_resolution(tmp_path: Path) -> None:
    pt = tmp_path / "yolo11m.pt"
    pt.write_bytes(b"")

    # MPS prefers .mlpackage when present and uses batch_size=8
    mlpkg = tmp_path / "yolo11m.mlpackage"
    mlpkg.mkdir()
    mps_pred = YoloGpuPredictor(det_weights=str(pt), device="mps")
    assert mps_pred._batch_size == 8
    assert mps_pred._resolve_weights(str(pt)) == str(mlpkg)
    assert mps_pred._infer_kwargs(str(pt)) == {}

    # CPU uses batch_size=4
    cpu_pred = YoloGpuPredictor(det_weights=str(pt), device="cpu")
    assert cpu_pred._batch_size == 4
    assert cpu_pred._infer_kwargs(str(pt)) == {}

    # XPU uses batch_size=8
    xpu_pred = YoloGpuPredictor(det_weights=str(pt), device="xpu:0")
    assert xpu_pred._batch_size == 8

    # CUDA uses batch_size=16 and FP16 for .pt
    cuda_pred = YoloGpuPredictor(det_weights=str(pt), device="cuda:0")
    assert cuda_pred._batch_size == 16
    assert cuda_pred._infer_kwargs(str(pt)) == {"quantize": "fp16"}
    assert cuda_pred._infer_kwargs(str(tmp_path / "yolo11m.engine")) == {}

    # Bare weight filenames resolve inside models_dir
    bare_pred = YoloGpuPredictor(det_weights="yolo11m.pt", device="cpu", models_dir=tmp_path)
    assert bare_pred._resolve_weights("yolo11m.pt") == str((tmp_path / "yolo11m.pt").resolve())

    # Compiled model load failure falls back cleanly to .pt weights
    calls: list[str] = []

    def _fake_yolo(path: str, *, task: str) -> str:
        calls.append(path)
        if path.endswith(".mlpackage"):
            raise RuntimeError("broken compiled engine")
        return f"loaded:{path}:{task}"

    model, resolved = mps_pred._load_yolo_model(_fake_yolo, str(pt), "detect")
    assert calls == [str(mlpkg), str(pt)]
    assert resolved == str(pt)
    assert model == f"loaded:{pt}:detect"


def test_detect_high_arc_motion_balls_recovers_above_net_dropouts() -> None:
    import cv2
    import numpy as np
    from vball_analyser.detector import _detect_high_arc_motion_balls
    from vball_analyser.models import NetCalibration

    cal = NetCalibration(left_point=Point2D(x=180.0, y=220.0), right_point=Point2D(x=980.0, y=220.0), is_manual=True)
    frames = []
    for f in range(16):
        img = np.full((480, 960, 3), 210, dtype=np.uint8)
        bx = int(600 - f * 20)
        by = int(150 - 8 * f + 0.5 * f * f)
        cv2.circle(img, (bx, by), 9, (40, 210, 235), -1)
        frames.append((f, f / 30.0, img))

    extra = _detect_high_arc_motion_balls(frames, cal, existing_by_frame={})
    assert len(extra) >= 12
    by_f = {d.frame_idx: d for d in extra}
    assert 8 in by_f
    assert abs(by_f[8].x - (600 - 8 * 20)) <= 4.0
    assert abs(by_f[8].y - (150 - 8 * 8 + 0.5 * 64)) <= 4.0
