"""Unit tests for automated serve detection and passer floor projection."""

from __future__ import annotations

import math
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
from vball_analyser.serve_detector import PRE_SERVE_OFFSET_S, detect_serve_events, extract_foot_point
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


def test_stops_detection_after_first_serve_and_receive_during_ongoing_rally(tmp_path: Path) -> None:
    video_path = tmp_path / "long_rally.mp4"
    fps = 15.0
    duration_s = 9.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)
    p2_px = project_court_to_image(H_inv, 4.5, 6.2)

    n_frames = int(fps * duration_s)
    p2_frames = [_make_pose_frame(i, round(i / fps, 4), p2_px[0], p2_px[1]) for i in range(n_frames)]
    balls: list[BallDetection] = []

    # First serve & receive at t=1.6s (frame 24)
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=150.0 + u * (310.0 - 150.0), conf=0.9))
    for i in range(25, 33):
        u = (i - 24) / 8.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=310.0 - u * 140.0, conf=0.9))

    # Ongoing rally free-ball / dig 4s later at t=5.6s (frame 84) -> must NOT be logged as a 2nd serve
    for i in range(70, 85):
        u = (i - 70) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=150.0 + u * (320.0 - 150.0), conf=0.9))
    for i in range(85, 92):
        u = (i - 84) / 7.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=320.0 - u * 120.0, conf=0.9))

    tracks = [PlayerTrack(track_id=2, court_side="attacking", frames=p2_frames)]
    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=n_frames, duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, tracks), meta)
    assert len(events) == 1
    assert events[0].timestamp_s is not None and abs(events[0].timestamp_s - 1.6) < 0.25


def test_multi_serve_short_clip_detects_both_serves(tmp_path: Path) -> None:
    video_path = tmp_path / "multi_serve_drill.mp4"
    fps = 15.0
    duration_s = 14.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)
    p1_px = project_court_to_image(H_inv, 2.0, 6.2)
    p3_px = project_court_to_image(H_inv, 7.2, 6.2)

    n_frames = int(fps * duration_s)
    p1_frames = [_make_pose_frame(i, round(i / fps, 4), p1_px[0], p1_px[1]) for i in range(n_frames)]
    p3_frames = [_make_pose_frame(i, round(i / fps, 4), p3_px[0], p3_px[1]) for i in range(n_frames)]
    balls: list[BallDetection] = []

    # Serve #1 received by P1 at t=1.6s (frame 24)
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p1_px[0], y=150.0 + u * (310.0 - 150.0), conf=0.9))
    for i in range(25, 32):
        u = (i - 24) / 7.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p1_px[0], y=310.0 - u * 120.0, conf=0.9))

    # Serve #2 in the same <=20s drill clip received by P3 at t=9.6s (frame 144, >6s after serve #1)
    for i in range(130, 145):
        u = (i - 130) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p3_px[0], y=150.0 + u * (310.0 - 150.0), conf=0.9))
    for i in range(145, 152):
        u = (i - 144) / 7.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p3_px[0], y=310.0 - u * 120.0, conf=0.9))

    tracks = [
        PlayerTrack(track_id=1, court_side="attacking", frames=p1_frames),
        PlayerTrack(track_id=3, court_side="attacking", frames=p3_frames),
    ]
    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=n_frames, duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, tracks), meta)
    assert len(events) == 2
    assert events[0].timestamp_s is not None and abs(events[0].timestamp_s - 1.6) < 0.25
    assert events[1].timestamp_s is not None and abs(events[1].timestamp_s - 9.6) < 0.25


def _build_receive_pass_scenario(
    tmp_path: Path,
    filename: str,
    setter_court_xy: tuple[float, float] | None,
    *,
    overpass_to_blocker: bool = False,
) -> tuple[Path, CourtCalibration, FakeServePredictor, VideoMetadata]:
    """Build a synthetic video + ball/track script with 3 passers and an optional setter or overpass."""
    video_path = tmp_path / filename
    fps = 15.0
    duration_s = 6.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)

    p1_px = project_court_to_image(H_inv, 1.8, 6.2)
    p2_px = project_court_to_image(H_inv, 4.5, 6.5)
    p3_px = project_court_to_image(H_inv, 7.2, 6.2)

    n_frames = int(fps * duration_s)
    p1_frames = [_make_pose_frame(i, round(i / fps, 4), p1_px[0], p1_px[1]) for i in range(n_frames)]
    p2_frames = [_make_pose_frame(i, round(i / fps, 4), p2_px[0], p2_px[1]) for i in range(n_frames)]
    p3_frames = [_make_pose_frame(i, round(i / fps, 4), p3_px[0], p3_px[1]) for i in range(n_frames)]

    tracks = [
        PlayerTrack(track_id=1, court_side="attacking", frames=p1_frames),
        PlayerTrack(track_id=2, court_side="attacking", frames=p2_frames),
        PlayerTrack(track_id=3, court_side="attacking", frames=p3_frames),
    ]

    balls: list[BallDetection] = []
    p2_recv_y = p2_px[1] - 75.0
    # Serve descends onto P2 chest at frame 24 (t=1.6s)
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(
            BallDetection(
                frame_idx=i,
                timestamp_s=round(i / fps, 4),
                x=520.0 + u * (p2_px[0] - 520.0),
                y=150.0 + u * (p2_recv_y - 150.0),
                conf=0.9,
            )
        )

    if setter_court_xy is not None:
        setter_px = project_court_to_image(H_inv, setter_court_xy[0], setter_court_xy[1])
        setter_frames = [
            _make_pose_frame(i, round(i / fps, 4), setter_px[0], setter_px[1])
            for i in range(n_frames)
        ]
        tracks.append(PlayerTrack(track_id=4, court_side="attacking", frames=setter_frames))
        setter_hand_y = setter_px[1] - 75.0
        apex_x = 0.5 * (p2_px[0] + setter_px[0])
        apex_y = 140.0
        # Pass rises frames 25..32 to apex, descends frames 33..40 to setter wrists, rebounds frames 41..45
        for i in range(25, 33):
            u = (i - 24) / 8.0
            balls.append(
                BallDetection(
                    frame_idx=i,
                    timestamp_s=round(i / fps, 4),
                    x=p2_px[0] + u * (apex_x - p2_px[0]),
                    y=p2_recv_y + u * (apex_y - p2_recv_y),
                    conf=0.9,
                )
            )
        for i in range(33, 41):
            u = (i - 32) / 8.0
            balls.append(
                BallDetection(
                    frame_idx=i,
                    timestamp_s=round(i / fps, 4),
                    x=apex_x + u * (setter_px[0] - apex_x),
                    y=apex_y + u * (setter_hand_y - apex_y),
                    conf=0.9,
                )
            )
        for i in range(41, 46):
            u = (i - 40) / 5.0
            balls.append(
                BallDetection(
                    frame_idx=i,
                    timestamp_s=round(i / fps, 4),
                    x=setter_px[0] - u * 40.0,
                    y=setter_hand_y - u * 80.0,
                    conf=0.9,
                )
            )
    elif overpass_to_blocker:
        blocker_x, blocker_y = 540.0, 345.0
        blocker_frames = [
            _make_pose_frame(i, round(i / fps, 4), blocker_x, blocker_y, height=130.0)
            for i in range(n_frames)
        ]
        tracks.append(PlayerTrack(track_id=99, court_side="blocking", frames=blocker_frames))
        # Ball flies high above net tape (y=120 < 203.68 - 10) into far-side blocker's reach
        for i in range(25, 34):
            u = (i - 24) / 9.0
            balls.append(
                BallDetection(
                    frame_idx=i,
                    timestamp_s=round(i / fps, 4),
                    x=p2_px[0] + u * (blocker_x - p2_px[0]),
                    y=p2_recv_y + u * (120.0 - p2_recv_y),
                    conf=0.9,
                )
            )
        for i in range(34, 41):
            u = (i - 33) / 7.0
            balls.append(
                BallDetection(
                    frame_idx=i,
                    timestamp_s=round(i / fps, 4),
                    x=blocker_x,
                    y=120.0 + u * 110.0,
                    conf=0.9,
                )
            )
    else:
        # Shank high and wide off-screen without contacting any setter or other passer
        for i in range(25, 33):
            u = (i - 24) / 8.0
            balls.append(
                BallDetection(
                    frame_idx=i,
                    timestamp_s=round(i / fps, 4),
                    x=p2_px[0] - u * 160.0,
                    y=p2_recv_y - u * 340.0,
                    conf=0.9,
                )
            )

    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=n_frames, duration_s=duration_s, codec="h264")
    return video_path, cal, FakeServePredictor(balls, tracks), meta


def test_receive_perfect_pass_to_setter_scores_3(tmp_path: Path) -> None:
    video_path, cal, predictor, meta = _build_receive_pass_scenario(
        tmp_path, "perfect_pass.mp4", setter_court_xy=(6.2, 1.0)
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.result == "in"
    assert ev.passer_label == "P2"
    assert ev.receive_score == 3
    assert ev.pass_x_m == pytest.approx(6.2, abs=0.15)
    assert ev.pass_y_m == pytest.approx(1.0, abs=0.15)
    assert ev.pass_distance_m is not None and ev.pass_distance_m <= 1.5
    assert ev.pass_timestamp_s is not None and ev.pass_frame_idx is not None


@pytest.mark.parametrize(
    ("setter_xy", "expected_score"),
    [
        ((3.6, 1.0), 2),  # ~2.41m from default target (6.0, 0.8) -> Yellow (2)
        ((2.0, 1.0), 1),  # ~4.00m from default target (6.0, 0.8) -> Red (1)
    ],
)
def test_receive_off_target_passes_score_2_and_1(
    tmp_path: Path, setter_xy: tuple[float, float], expected_score: int
) -> None:
    video_path, cal, predictor, meta = _build_receive_pass_scenario(
        tmp_path, f"pass_{expected_score}.mp4", setter_court_xy=setter_xy
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.passer_label == "P2"
    assert ev.receive_score == expected_score
    assert ev.pass_x_m == pytest.approx(setter_xy[0], abs=0.15)
    assert ev.pass_y_m == pytest.approx(setter_xy[1], abs=0.15)


def test_receive_overpass_to_blocker_scores_0(tmp_path: Path) -> None:
    video_path, cal, predictor, meta = _build_receive_pass_scenario(
        tmp_path, "overpass.mp4", setter_court_xy=None, overpass_to_blocker=True
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.result == "in"
    assert ev.passer_label == "P2"
    assert ev.receive_score == 0
    assert ev.pass_x_m is None
    assert ev.pass_y_m is None


def test_receive_ace_scores_0_and_out_is_none(tmp_path: Path) -> None:
    video_path = tmp_path / "ace_and_out_scores.mp4"
    fps = 15.0
    duration_s = 7.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)

    ace_px = project_court_to_image(H_inv, 7.5, 6.0)
    out_px = project_court_to_image(H_inv, 4.5, 9.8)
    balls: list[BallDetection] = []
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=ace_px[0], y=160.0 + u * (ace_px[1] - 160.0), conf=0.9))
    for i in range(25, 31):
        u = (i - 24) / 6.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=ace_px[0], y=ace_px[1] - u * 70.0, conf=0.9))
    for i in range(68, 83):
        u = (i - 68) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=out_px[0], y=160.0 + u * (out_px[1] - 160.0), conf=0.9))
    for i in range(83, 89):
        u = (i - 82) / 6.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=out_px[0], y=out_px[1] - u * 70.0, conf=0.9))

    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=int(fps * duration_s), duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, []), meta)
    assert len(events) == 2
    assert events[0].result == "ace" and events[0].receive_score == 0
    assert events[1].result == "out" and events[1].receive_score is None


def test_receive_shank_without_setter_contact_scores_0(tmp_path: Path) -> None:
    video_path, cal, predictor, meta = _build_receive_pass_scenario(
        tmp_path, "shank.mp4", setter_court_xy=None, overpass_to_blocker=False
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.result == "in"
    assert ev.passer_label == "P2"
    assert ev.receive_score == 0
    assert ev.pass_x_m is None
    assert ev.pass_y_m is None


def test_filters_stationary_background_fp_and_ignores_bystander_during_rising_pass(tmp_path: Path) -> None:
    """Stationary gym-wall FPs must not create fake serves, and rising passes must not lock onto bystanders."""
    video_path, cal, base_predictor, meta = _build_receive_pass_scenario(
        tmp_path, "static_fp_bystander.mp4", setter_court_xy=(5.5, 1.2)
    )
    _H, H_inv = compute_court_homography(cal)
    p1_px = project_court_to_image(H_inv, 1.8, 6.2)

    # Inject a stationary false-positive wall light on frames 0..24 (higher conf than early noise, spanning >0.8s)
    noisy_balls = list(base_predictor._balls)
    for i in range(0, 25):
        noisy_balls.append(
            BallDetection(frame_idx=i, timestamp_s=round(i / meta.fps, 4), x=1003.5, y=185.5, conf=0.35)
        )

    # Add a front-row bystander standing visually along the rising pass path (before apex)
    bystander_px = project_court_to_image(H_inv, 4.9, 2.4)
    bystander_frames = [
        _make_pose_frame(i, round(i / meta.fps, 4), bystander_px[0], bystander_px[1])
        for i in range(meta.frame_count)
    ]
    tracks = list(base_predictor._tracks) + [
        PlayerTrack(track_id=50, court_side="attacking", frames=bystander_frames)
    ]

    events = detect_serve_events(video_path, cal, FakeServePredictor(noisy_balls, tracks), meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.passer_label == "P2"
    assert ev.receive_score == 3
    assert ev.pass_x_m == pytest.approx(5.5, abs=0.2)
    assert ev.pass_y_m == pytest.approx(1.2, abs=0.2)


def test_continuous_long_rally_beyond_cooldown_does_not_double_record(tmp_path: Path) -> None:
    """An ongoing rally (>6s after serve #1 with continuous ball flight and no dead-ball gap) logs only 1 serve."""
    video_path = tmp_path / "long_rally_single_serve.mp4"
    fps = 15.0
    duration_s = 12.0
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)
    p2_px = project_court_to_image(H_inv, 4.5, 6.2)

    n_frames = int(fps * duration_s)
    p2_frames = [_make_pose_frame(i, round(i / fps, 4), p2_px[0], p2_px[1]) for i in range(n_frames)]
    balls: list[BallDetection] = []

    # Real serve #1 received at t=1.6s (frame 24)
    for i in range(10, 25):
        u = (i - 10) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=150.0 + u * (310.0 - 150.0), conf=0.9))
    for i in range(25, 34):
        u = (i - 24) / 9.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=310.0 - u * 160.0, conf=0.9))

    # Continuous rally activity every ~1.2s (no >=2.2s dead-ball gap)
    for f_idx in (50, 68, 86, 104, 120):
        balls.append(BallDetection(frame_idx=f_idx, timestamp_s=round(f_idx / fps, 4), x=520.0, y=120.0, conf=0.8))

    # Rally return descending at t=9.6s (frame 144, >6s after serve #1) -> must NOT be logged as serve #2
    for i in range(130, 145):
        u = (i - 130) / 14.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=140.0 + u * (310.0 - 140.0), conf=0.9))
    for i in range(145, 152):
        u = (i - 144) / 7.0
        balls.append(BallDetection(frame_idx=i, timestamp_s=round(i / fps, 4), x=p2_px[0], y=310.0 - u * 120.0, conf=0.9))

    tracks = [PlayerTrack(track_id=2, court_side="attacking", frames=p2_frames)]
    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=n_frames, duration_s=duration_s, codec="h264")
    events = detect_serve_events(video_path, cal, FakeServePredictor(balls, tracks), meta)
    assert len(events) == 1
    assert events[0].timestamp_s is not None and abs(events[0].timestamp_s - 1.6) < 0.25


def _build_movement_scenario(
    tmp_path: Path,
    filename: str,
    *,
    p2_first_t: float,
    move_start_t: float,
    ball_dx_px: float,
) -> tuple[Path, CourtCalibration, FakeServePredictor, VideoMetadata]:
    """P2 stands at (4.5, 6.5), moves from move_start_t to receive at (3.1, 5.3) at t_land=2.2s (frame 33)."""
    video_path = tmp_path / filename
    fps = 15.0
    duration_s = 6.5
    make_synthetic_baseline_video(video_path, duration_s=duration_s, fps=int(fps))
    cal = _test_calibration()
    _H, H_inv = compute_court_homography(cal)

    p1_px = project_court_to_image(H_inv, 1.8, 6.2)
    p2_pre_px = project_court_to_image(H_inv, 4.5, 6.5)
    p2_recv_px = project_court_to_image(H_inv, 3.1, 5.3)
    p3_px = project_court_to_image(H_inv, 7.2, 6.2)

    n_frames = int(fps * duration_s)
    recv_frame = 33
    recv_t = recv_frame / fps
    recv_bx = p2_recv_px[0] + ball_dx_px
    recv_by = p2_recv_px[1] - 75.0
    p1_frames: list[PoseFrame] = []
    p2_frames: list[PoseFrame] = []
    p3_frames: list[PoseFrame] = []
    balls: list[BallDetection] = []
    for i in range(n_frames):
        ts = round(i / fps, 4)
        p1_frames.append(_make_pose_frame(i, ts, p1_px[0], p1_px[1]))
        p3_frames.append(_make_pose_frame(i, ts, p3_px[0], p3_px[1]))
        if ts >= p2_first_t - 1e-6:
            alpha = min(1.0, max(0.0, (ts - move_start_t) / (recv_t - move_start_t)))
            fx = p2_pre_px[0] + alpha * (p2_recv_px[0] - p2_pre_px[0])
            fy = p2_pre_px[1] + alpha * (p2_recv_px[1] - p2_pre_px[1])
            pf = _make_pose_frame(i, ts, fx, fy)
            # Shoulders 40px apart -> midline half-width 0.65 * 40 = 26px around torso center
            pf.keypoints[5] = (fx - 20.0, fy - 120.0, 0.9)
            pf.keypoints[6] = (fx + 20.0, fy - 120.0, 0.9)
            p2_frames.append(pf)

        if 15 <= i <= recv_frame:
            u = (i - 15) / (recv_frame - 15)
            balls.append(BallDetection(frame_idx=i, timestamp_s=ts, x=520.0 + u * (recv_bx - 520.0), y=150.0 + u * (recv_by - 150.0), conf=0.9))
        elif recv_frame < i <= 45:
            u = (i - recv_frame) / (45 - recv_frame)
            balls.append(BallDetection(frame_idx=i, timestamp_s=ts, x=recv_bx + u * 80.0, y=recv_by - u * 160.0, conf=0.9))

    tracks = [
        PlayerTrack(track_id=1, court_side="attacking", frames=p1_frames),
        PlayerTrack(track_id=2, court_side="attacking", frames=p2_frames),
        PlayerTrack(track_id=3, court_side="attacking", frames=p3_frames),
    ]
    meta = VideoMetadata(width=1056, height=576, fps=fps, frame_count=n_frames, duration_s=duration_s, codec="h264")
    return video_path, cal, FakeServePredictor(balls, tracks), meta


def test_pre_serve_140ms_lookback_movement_vector_and_body_alignment(tmp_path: Path) -> None:
    assert PRE_SERVE_OFFSET_S == pytest.approx(1.40)
    video_path, cal, predictor, meta = _build_movement_scenario(
        tmp_path, "movement.mp4", p2_first_t=0.0, move_start_t=1.3, ball_dx_px=45.0
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.result == "in"
    # Receive foot (3.1, 5.3) is nearer P1's stance (1.8, 6.2) than P2's, so only direct track attribution yields P2/Mid
    assert ev.passer_label == "P2"
    assert ev.slot_label == "Mid"
    assert ev.pre_passer_x_m == pytest.approx(4.5, abs=0.15)
    assert ev.pre_passer_y_m == pytest.approx(6.5, abs=0.15)
    assert ev.recv_foot_x_m == pytest.approx(3.1, abs=0.15)
    assert ev.recv_foot_y_m == pytest.approx(5.3, abs=0.15)
    assert ev.move_distance_m == pytest.approx(math.hypot(-1.4, -1.2), abs=0.2)
    assert ev.move_dx_m is not None and ev.move_dx_m < -1.0
    assert ev.move_dy_m is not None and ev.move_dy_m < -1.0
    assert ev.contact_alignment == "outside"


def test_late_entering_receiver_track_fallback_and_unmeasured_short_track(tmp_path: Path) -> None:
    # (a) Track enters at t_land - 0.80s, before it starts moving: its earliest frame is the stance
    video_path, cal, predictor, meta = _build_movement_scenario(
        tmp_path, "late_a.mp4", p2_first_t=1.4, move_start_t=1.6, ball_dx_px=0.0
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.pre_passer_x_m == pytest.approx(4.5, abs=0.15)
    assert ev.move_distance_m == pytest.approx(math.hypot(-1.4, -1.2), abs=0.2)
    assert ev.contact_alignment == "midline"

    # (b) Track enters only at t_land - 0.20s (< 0.45s): movement is unmeasured, never 0.0
    video_path, cal, predictor, meta = _build_movement_scenario(
        tmp_path, "late_b.mp4", p2_first_t=2.0, move_start_t=1.6, ball_dx_px=0.0
    )
    events = detect_serve_events(video_path, cal, predictor, meta)
    assert len(events) == 1
    ev = events[0]
    assert ev.result == "in"
    assert ev.recv_foot_x_m is not None
    assert ev.pre_passer_x_m is None
    assert ev.move_dx_m is None and ev.move_dy_m is None
    assert ev.move_distance_m is None
    assert ev.slot_label is not None
