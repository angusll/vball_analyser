"""Unit tests for automated serve detection and passer floor projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_synthetic_baseline_video
from vball_analyser.models import (
    BallDetection,
    CourtCalibration,
    NetCalibration,
    PlayerTrack,
    Point2D,
    PoseFrame,
    VideoMetadata,
)
from vball_analyser.serve_detector import detect_serve_events, extract_foot_point
from vball_analyser.serve_geometry import compute_court_homography, project_court_to_image


def _test_calibration() -> CourtCalibration:
    return CourtCalibration(
        front_left=Point2D(x=220.0, y=370.0),
        front_right=Point2D(x=836.0, y=370.0),
        attack_right=Point2D(x=896.0, y=435.0),
        attack_left=Point2D(x=160.0, y=435.0),
    )


def _make_pose_frame(frame_idx: int, ts: float, foot_x: float, foot_y: float, height: float = 150.0) -> PoseFrame:
    w = 54.0
    x1, y1, x2, y2 = foot_x - w / 2, foot_y - height, foot_x + w / 2, foot_y
    kps = [(foot_x, foot_y - height * 0.5, 0.9) for _ in range(17)]
    kps[15] = (foot_x - 8.0, foot_y, 0.9)
    kps[16] = (foot_x + 8.0, foot_y, 0.9)
    return PoseFrame(frame_idx=frame_idx, timestamp_s=ts, bbox=(x1, y1, x2, y2), keypoints=kps)


class FakeServePredictor:
    """Deterministic FramePredictor returning pre-scripted ball & pose tracks."""

    def __init__(self, balls: list[BallDetection], tracks: list[PlayerTrack]) -> None:
        self._balls = balls
        self._tracks = tracks

    def detect_balls(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[BallDetection]:
        del calibration
        valid = {f[0] for f in frames}
        return [b for b in self._balls if b.frame_idx in valid]

    def track_poses(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[PlayerTrack]:
        del calibration
        valid = {f[0] for f in frames}
        out: list[PlayerTrack] = []
        for t in self._tracks:
            sub = [fr for fr in t.frames if fr.frame_idx in valid]
            if sub:
                out.append(PlayerTrack(track_id=t.track_id, court_side=t.court_side, frames=sub))
        return out

    def scan_coarse(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> tuple[list[BallDetection], list[tuple[int, float, tuple[float, float, float, float]]]]:
        return self.detect_balls(frames, calibration), []


def test_extract_foot_point_uses_ankles_or_bbox_fallback() -> None:
    pf = _make_pose_frame(0, 0.0, 400.0, 500.0)
    assert extract_foot_point(pf) == pytest.approx((400.0, 500.0), abs=1e-2)

    low_conf_kps = [(0.0, 0.0, 0.05) for _ in range(17)]
    pf_fallback = PoseFrame(frame_idx=0, timestamp_s=0.0, bbox=(370.0, 350.0, 430.0, 510.0), keypoints=low_conf_kps)
    assert extract_foot_point(pf_fallback) == pytest.approx((400.0, 510.0), abs=1e-2)


def test_detects_played_serve_in_seam_and_ignores_subsequent_set(tmp_path: Path) -> None:
    video_path = tmp_path / "serve_rally.mp4"
    fps = 15.0
    duration_s = 6.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)

    # 3 passers at P1(1.8, 6.2), P2(4.5, 6.5), P3(7.2, 6.2); P2 steps left to (3.15, 6.3) at t=2.2s
    p1_px = project_court_to_image(H_inv, 1.8, 6.2)
    p2_pre_px = project_court_to_image(H_inv, 4.5, 6.5)
    p2_recv_px = project_court_to_image(H_inv, 3.15, 6.3)
    p3_px = project_court_to_image(H_inv, 7.2, 6.2)

    p1_frames: list[PoseFrame] = []
    p2_frames: list[PoseFrame] = []
    p3_frames: list[PoseFrame] = []
    balls: list[BallDetection] = []

    n_frames = int(fps * duration_s)
    recv_frame = int(2.2 * fps)  # frame 33
    for i in range(n_frames):
        ts = round(i / fps, 4)
        p1_frames.append(_make_pose_frame(i, ts, p1_px[0], p1_px[1]))
        # P2 moves from pre-serve spot to seam between frame 26 and 33
        if i <= 26:
            fx, fy = p2_pre_px
        elif i <= recv_frame:
            alpha = (i - 26) / max(1, recv_frame - 26)
            fx = p2_pre_px[0] + alpha * (p2_recv_px[0] - p2_pre_px[0])
            fy = p2_pre_px[1] + alpha * (p2_recv_px[1] - p2_pre_px[1])
        else:
            fx, fy = p2_recv_px
        p2_frames.append(_make_pose_frame(i, ts, fx, fy))
        p3_frames.append(_make_pose_frame(i, ts, p3_px[0], p3_px[1]))

        # Serve crosses net at frame 15 (t=1.0s), descends to P2's chest/head height (y=310 < net_center_y=370) at frame 33 (t=2.2s), rebounds up
        if 15 <= i <= recv_frame:
            u = (i - 15) / (recv_frame - 15)
            bx = 520.0 + u * (p2_recv_px[0] - 520.0)
            by = 150.0 + u * (310.0 - 150.0)
            balls.append(BallDetection(frame_idx=i, timestamp_s=ts, x=bx, y=by, conf=0.9))
        elif recv_frame < i <= 45:
            u = (i - recv_frame) / (45 - recv_frame)
            bx = p2_recv_px[0] + u * 80.0
            by = 310.0 - u * 160.0
            balls.append(BallDetection(frame_idx=i, timestamp_s=ts, x=bx, y=by, conf=0.9))
        # Subsequent set/hit arc at frames 52..68 (t=3.47s..4.53s) — within refractory / same rally
        elif 52 <= i <= 62:
            u = (i - 52) / 10.0
            bx = 600.0 - u * 150.0
            by = 180.0 + u * 110.0
            balls.append(BallDetection(frame_idx=i, timestamp_s=ts, x=bx, y=by, conf=0.9))
        elif 62 < i <= 68:
            u = (i - 62) / 6.0
            bx = 450.0 - u * 50.0
            by = 290.0 - u * 80.0
            balls.append(BallDetection(frame_idx=i, timestamp_s=ts, x=bx, y=by, conf=0.9))

    far_blocker_frames = [_make_pose_frame(i, round(i / fps, 4), p2_recv_px[0], 390.0, height=110.0) for i in range(n_frames)]
    tracks = [
        PlayerTrack(track_id=1, court_side="attacking", frames=p1_frames),
        PlayerTrack(track_id=2, court_side="attacking", frames=p2_frames),
        PlayerTrack(track_id=3, court_side="attacking", frames=p3_frames),
        PlayerTrack(track_id=99, court_side="blocking", frames=far_blocker_frames),
    ]
    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=n_frames, duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, tracks), meta, server_name="#7")

    assert len(events) == 1
    ev = events[0]
    assert ev.source == "auto"
    assert ev.result == "in"
    assert ev.is_seam is True
    assert ev.seam_pair == ("P1", "P2")
    assert ev.timestamp_s is not None and abs(ev.timestamp_s - 2.2) < 0.25
    assert len(ev.passers_snapshot) == 3
    assert ev.server_name == "#7"


def test_detects_unplayed_ace_and_out_long(tmp_path: Path) -> None:
    video_path = tmp_path / "unplayed.mp4"
    fps = 15.0
    duration_s = 7.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)

    ace_px = project_court_to_image(H_inv, 7.5, 6.0)  # Zone 1 in-bounds
    out_px = project_court_to_image(H_inv, 4.5, 9.8)  # Out Long

    balls: list[BallDetection] = []
    # Serve 1: frames 10..24 descending to ace_px, bouncing up frames 25..30
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(
            BallDetection(
                frame_idx=i,
                timestamp_s=round(i / fps, 4),
                x=540.0 + u * (ace_px[0] - 540.0),
                y=160.0 + u * (ace_px[1] - 160.0),
                conf=0.9,
            )
        )
    for i in range(25, 31):
        u = (i - 24) / 6.0
        balls.append(
            BallDetection(
                frame_idx=i,
                timestamp_s=round(i / fps, 4),
                x=ace_px[0],
                y=ace_px[1] - u * 70.0,
                conf=0.9,
            )
        )

    # Serve 2 (> 2.5s later): frames 68..82 descending to out_px, bouncing up 83..88
    for i in range(68, 83):
        u = (i - 68) / 14.0
        balls.append(
            BallDetection(
                frame_idx=i,
                timestamp_s=round(i / fps, 4),
                x=520.0 + u * (out_px[0] - 520.0),
                y=160.0 + u * (out_px[1] - 160.0),
                conf=0.9,
            )
        )
    for i in range(83, 89):
        u = (i - 82) / 6.0
        balls.append(
            BallDetection(
                frame_idx=i,
                timestamp_s=round(i / fps, 4),
                x=out_px[0],
                y=out_px[1] - u * 70.0,
                conf=0.9,
            )
        )

    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=int(fps * duration_s), duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, []), meta)
    assert len(events) == 2
    assert events[0].result == "ace" and events[0].zone == "Zone 1"
    assert events[1].result == "out" and events[1].zone == "Out Long"


def test_empty_video_returns_no_serves(tmp_path: Path) -> None:
    video_path = tmp_path / "empty.mp4"
    make_synthetic_baseline_video(video_path, duration_s=2.0, fps=15)
    cal = _test_calibration()
    meta = VideoMetadata(width=1056, height=576, fps=15.0, frame_count=30, duration_s=2.0, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor([], []), meta)
    assert events == []


def test_ignores_high_net_spike_after_refractory_gap(tmp_path: Path) -> None:
    video_path = tmp_path / "rally_late_spike.mp4"
    fps = 15.0
    duration_s = 7.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)
    land_px = project_court_to_image(H_inv, 4.5, 6.0)

    balls: list[BallDetection] = []
    # Real serve descending into near court at t=1.6s (frame 24)
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=528.0, y=160.0 + u * (land_px[1] - 160.0), conf=0.9))
    for i in range(25, 31):
        u = (i - 24) / 6.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=528.0, y=land_px[1] - u * 80.0, conf=0.9))

    # High set/spike contact 3.5s later (t=5.1s, > 2.5s refractory) that stays high above net centerline (y=260 < 370)
    for i in range(68, 78):
        u = (i - 68) / 9.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=480.0, y=170.0 + u * 90.0, conf=0.9))
    for i in range(78, 84):
        u = (i - 77) / 6.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=480.0, y=260.0 - u * 60.0, conf=0.9))

    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=int(fps * duration_s), duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, []), meta)
    assert len(events) == 1
    assert events[0].timestamp_s is not None and abs(events[0].timestamp_s - 1.6) < 0.25


def test_long_clip_coarse_scan_handles_missed_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "long_clip.mp4"
    fps = 15.0
    duration_s = 22.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)
    land_px = project_court_to_image(H_inv, 7.2, 6.2)

    balls: list[BallDetection] = []
    # Single serve around t=10.0s..11.0s (frames 150..165), dropping middle coarse frames around 156..158
    for i in range(150, 166):
        if i in (156, 157, 158):
            continue
        u = (i - 150) / 15.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=520.0 + u * (land_px[0] - 520.0), y=160.0 + u * (land_px[1] - 160.0), conf=0.9))
    for i in range(166, 173):
        u = (i - 165) / 7.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=land_px[0], y=land_px[1] - u * 70.0, conf=0.9))

    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=int(fps * duration_s), duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, []), meta)
    assert len(events) == 1
    assert events[0].zone == "Zone 1"
