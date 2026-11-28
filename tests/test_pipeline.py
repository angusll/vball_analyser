"""Tests for two-pass pipeline, cached recalibration/reassignment, and RTX 4090 GPU smoke test."""

from __future__ import annotations

import json
from collections import Counter
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
    analyse_play_windows_shared,
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


class _FrameCountingPredictor(SyntheticColorPredictor):
    """Counts how many times each frame index reaches Pass-2 ball detection."""

    def __init__(self) -> None:
        self.ball_frame_counts: Counter[int] = Counter()

    def detect_balls(self, frames, calibration):  # type: ignore[no-untyped-def]
        self.ball_frame_counts.update(idx for idx, _ts, _bgr in frames)
        return super().detect_balls(frames, calibration)


def test_multi_swing_rally_splits_ours_and_opponent_with_single_pass2_decode(tmp_path: Path) -> None:
    clip = tmp_path / "match_rally.mp4"
    # One rally with two swings 3.8s apart (ours then opponent) plus a later isolated ours swing
    make_synthetic_baseline_video(
        clip,
        duration_s=22.0,
        fps=24,
        tilt_deg=1.5,
        rally_windows=[(3.0, 4.2), (16.0, 17.2)],
        far_side_rallies=[(6.8, 8.0)],
    )
    meta = probe_video(clip)
    cal = auto_detect_net_tape([bgr for _i, _t, bgr in iter_video_frames(clip, sample_fps=1.0)])

    predictor = _FrameCountingPredictor()
    windows = scan_candidate_plays(clip, cal, predictor, meta)
    assert len(windows) == 3
    assert windows[0][1] > windows[1][0]  # the two rally swings overlap
    assert windows[0][0] < 4.0 < windows[0][1]
    assert windows[1][0] < 7.8 < windows[1][1]
    assert windows[2][0] < 17.0 < windows[2][1]

    # An extra dead-time window (no set arc, no jump) is pruned as a false trigger
    ready: list[tuple[int, int]] = []
    results = analyse_play_windows_shared(
        clip,
        windows + [(11.0, 12.4)],
        cal,
        predictor,
        meta,
        tmp_path / "session",
        session_mode="match",
        on_play_ready=lambda idx, total, _play, _balls: ready.append((idx, total)),
    )
    plays = [p for p, _ in results]
    assert [p.index for p in plays] == [1, 2, 3]
    assert ready == [(1, 4), (2, 4), (3, 4)]
    assert [p.team_side for p in plays] == ["ours", "opponent", "ours"]
    assert plays[1].attack_zone.startswith("Opp ")
    assert all(p.set_trajectory is not None for p in plays)
    assert len(list((tmp_path / "session" / "thumbnails").glob("*.jpg"))) == 3 * len(plays)

    # Overlapping rally frames were decoded/detected once, not once per window
    rally_frames = [f for f in predictor.ball_frame_counts if windows[1][0] * meta.fps <= f <= windows[0][1] * meta.fps]
    assert rally_frames
    assert max(predictor.ball_frame_counts.values()) == 1


def test_mid_length_single_swing_clip_returns_full_window(tmp_path: Path) -> None:
    clip = tmp_path / "mid_clip.mp4"
    make_synthetic_baseline_video(clip, duration_s=12.0, fps=24, tilt_deg=1.5, rally_windows=[(5.0, 6.6)])
    meta = probe_video(clip)
    cal = auto_detect_net_tape([bgr for _i, _t, bgr in iter_video_frames(clip, sample_fps=1.0)])
    assert scan_candidate_plays(clip, cal, SyntheticColorPredictor(), meta) == [(0.0, meta.duration_s)]


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


def test_yolo_predictor_skips_vfr_duplicates_and_conditional_tiling() -> None:
    import numpy as np
    from vball_analyser.detector import _find_unique_frame_indices
    from vball_analyser.models import NetCalibration

    cal = NetCalibration(left_point=Point2D(x=100.0, y=200.0), right_point=Point2D(x=540.0, y=200.0), is_manual=True)

    # Build 6 frames: f0 (moving ball at x=200), f1 (dup of f0), f2 (moving ball at x=260),
    # f3 (dup of f2), f4 (no ball -> should trigger tiling), f5 (dup of f4)
    img0 = np.full((360, 640, 3), 120, dtype=np.uint8)
    img0[100:120, 190:210] = 240
    img2 = np.full((360, 640, 3), 120, dtype=np.uint8)
    img2[100:120, 250:270] = 240
    img4 = np.full((360, 640, 3), 130, dtype=np.uint8)

    frames = [
        (0, 0.0 / 30.0, img0),
        (1, 1.0 / 30.0, img0.copy()),
        (2, 2.0 / 30.0, img2),
        (3, 3.0 / 30.0, img2.copy()),
        (4, 4.0 / 30.0, img4),
        (5, 5.0 / 30.0, img4.copy()),
    ]
    uniq, mapping = _find_unique_frame_indices(frames)
    assert uniq == [0, 2, 4]
    assert mapping == [0, 0, 1, 1, 2, 2]

    class _FakeArray:
        def __init__(self, data: list) -> None:
            self._data = data

        def tolist(self) -> list:
            return self._data

        def int(self) -> "_FakeArray":
            return self

    class _FakeBoxes:
        def __init__(self, xyxy: list, conf: list) -> None:
            self.xyxy = _FakeArray(xyxy)
            self.conf = _FakeArray(conf)

        def __len__(self) -> int:
            return len(self.xyxy.tolist())

    class _FakeDetModel:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []  # (imgsz, batch_len)

        def predict(self, imgs: list, *, imgsz: int, **_kw: object) -> list:
            self.calls.append((imgsz, len(imgs)))
            out = []
            for idx, _im in enumerate(imgs):
                if imgsz == 1280:
                    if idx == 0:
                        boxes = _FakeBoxes([[190.0, 100.0, 210.0, 120.0]], [0.8])
                    elif idx == 1:
                        boxes = _FakeBoxes([[250.0, 100.0, 270.0, 120.0]], [0.8])
                    else:
                        boxes = None
                    out.append(SimpleNamespace(boxes=boxes))
                else:
                    # Tile pass: only called for unique frame 2 (2 tiles)
                    out.append(SimpleNamespace(boxes=None))
            return out

    class _FakePoseModel:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def track(self, imgs: list, **_kw: object) -> list:
            self.calls.append(len(imgs))
            out = []
            for _im in imgs:
                boxes = SimpleNamespace(id=_FakeArray([7]), xyxy=_FakeArray([[100.0, 150.0, 160.0, 320.0]]))
                kps = SimpleNamespace(
                    xy=_FakeArray([[[130.0, 170.0 + k * 5.0] for k in range(17)]]),
                    conf=_FakeArray([[0.9] * 17]),
                )
                out.append(SimpleNamespace(boxes=boxes, keypoints=kps))
            return out

    pred = YoloGpuPredictor(device="cpu")
    fake_det = _FakeDetModel()
    fake_pose = _FakePoseModel()
    pred._det_model = fake_det
    pred._pose_model = fake_pose

    balls = pred.detect_balls(frames, cal)
    # Full-frame pass saw only 3 unique frames; tile pass saw only 2 tiles (for the 1 frame without a ball)
    assert fake_det.calls == [(1280, 3), (960, 2)]
    # Detections expanded to duplicate frames f0, f1, f2, f3
    assert [b.frame_idx for b in balls] == [0, 1, 2, 3]

    tracks = pred.track_poses(frames, cal)
    # Pose tracker saw only 3 unique frames, expanded back to all 6 frames
    assert fake_pose.calls == [3]
    assert len(tracks) == 1
    assert [pf.frame_idx for pf in tracks[0].frames] == [0, 1, 2, 3, 4, 5]


def test_yolo_predictor_warmup_and_app_startup_preload(tmp_path: Path) -> None:
    import threading
    from fastapi.testclient import TestClient
    from vball_analyser.app import create_app
    from vball_analyser.config import AppConfig

    # 1. YoloGpuPredictor.warmup() runs a dummy pass on both det and pose models
    calls: list[tuple[str, dict]] = []

    class _FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def predict(self, imgs, **kwargs):
            calls.append((f"{self.name}:predict", kwargs))
            return []

        def track(self, imgs, **kwargs):
            calls.append((f"{self.name}:track", kwargs))
            return []

    pred = YoloGpuPredictor(device="cpu", models_dir=tmp_path)
    pred._det_model = _FakeModel("det")
    pred._pose_model = _FakeModel("pose")
    pred._det_resolved = "yolo11m.pt"
    pred._pose_resolved = "yolo11m-pose.pt"
    pred.warmup()
    assert [name for name, _ in calls] == ["det:predict", "pose:track"]
    assert calls[0][1]["imgsz"] == 1280
    assert calls[1][1]["imgsz"] == 1280

    # 2. FastAPI lifespan triggers predictor.warmup() automatically in background
    warmed = threading.Event()

    class _WarmupPredictor(SyntheticColorPredictor):
        def warmup(self) -> None:
            warmed.set()

    cfg = AppConfig(data_dir=tmp_path / "data")
    app = create_app(config=cfg, predictor=_WarmupPredictor(), warmup_on_startup=True)
    with TestClient(app) as client:
        assert warmed.wait(timeout=3.0)
        resp = client.get("/api/csrf")
        assert resp.status_code == 200


def test_legacy_session_load_mode_toggle_and_side_swap_clears_stale_hitter(
    synthetic_clip: tuple[Path, dict], tmp_path: Path
) -> None:
    clip_path, _gt = synthetic_clip
    meta = probe_video(clip_path)
    cal = auto_detect_net_tape([bgr for _i, _t, bgr in iter_video_frames(clip_path, sample_fps=4.0)])

    store = SessionStore(tmp_path / "sessions")
    sid = "sess_legacy_side"
    sdir = store.create_session_dir(sid)
    (sdir / "video_path.txt").write_text(str(clip_path), encoding="utf-8")
    play, _ = analyse_play_window(clip_path, 1, 0.0, meta.duration_s, cal, SyntheticColorPredictor(), meta, sdir)
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
    # Legacy-shaped session.json: no session_mode / summary keys
    legacy = session.model_dump(mode="json")
    legacy.pop("session_mode")
    legacy.pop("summary")
    (sdir / "session.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = store.load_session(sid)
    assert loaded.session_mode == "drill"
    assert loaded.summary is not None
    assert loaded.summary.ours_count == 1

    matched = store.update_session_mode(sid, "match")
    assert matched.session_mode == "match"
    assert matched.summary is not None and matched.summary.session_mode == "match"
    ours_play = matched.plays[0]
    assert ours_play.team_side == "ours"
    tracks_by_id = {t.track_id: t for t in ours_play.tracks}
    assert tracks_by_id[ours_play.hitter_track_id].court_side == "attacking"

    swapped = store.update_play_team_side(sid, play.play_id, "opponent")
    opp = swapped.plays[0]
    assert opp.team_side == "opponent"
    assert opp.team_side_manual is True
    assert opp.hitter_track_id is not None
    assert tracks_by_id[opp.hitter_track_id].court_side == "blocking"
    assert opp.attack_zone.startswith("Opp ")
    assert swapped.summary is not None
    assert swapped.summary.ours_count == 0
    assert swapped.summary.opponent_count == 1
    assert swapped.summary.ours_all.play_count == 0
    assert all((sdir / "thumbnails" / km.thumbnail_rel_path).is_file() for km in opp.keyframes)

    # Persisted, and "auto" hands the side back to the classifier
    assert store.load_session(sid).plays[0].team_side == "opponent"
    auto = store.update_play_team_side(sid, play.play_id, "auto")
    assert auto.plays[0].team_side_manual is False
    assert auto.plays[0].team_side == "ours"
    assert tracks_by_id[auto.plays[0].hitter_track_id].court_side == "attacking"
    assert auto.summary is not None and auto.summary.ours_count == 1
