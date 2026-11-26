"""Tests for pose smoothing, court-side classification, hitter metrics, and blocker seams."""

from __future__ import annotations

import math

import pytest

from vball_analyser.biomechanics import analyse_play_mechanics, classify_court_sides, smooth_pose_tracks
from vball_analyser.models import BallDetection, NetCalibration, PlayerTrack, Point2D, PoseFrame, SetTrajectory


def _make_coco_frame(
    frame_idx: int,
    fps: float,
    hip_x: float,
    hip_y: float,
    ankle_y: float,
    elbow_angle_deg: float = 168.0,
    torso_tilt_deg: float = 8.0,
    scale: float = 1.0,
) -> PoseFrame:
    kps = [(0.0, 0.0, 0.0) for _ in range(17)]
    tilt_rad = math.radians(torso_tilt_deg)
    sh_x = hip_x + 55.0 * scale * math.sin(tilt_rad)
    sh_y = hip_y - 55.0 * scale * math.cos(tilt_rad)
    nose_y = sh_y - 20.0 * scale
    kps[0] = (sh_x, nose_y, 0.95)
    kps[5] = (sh_x - 14 * scale, sh_y, 0.95)
    kps[6] = (sh_x + 14 * scale, sh_y, 0.95)
    kps[11] = (hip_x - 10 * scale, hip_y, 0.95)
    kps[12] = (hip_x + 10 * scale, hip_y, 0.95)
    kps[15] = (hip_x - 12 * scale, ankle_y, 0.95)
    kps[16] = (hip_x + 12 * scale, ankle_y, 0.95)

    # Right arm with specified interior shoulder-elbow-wrist angle
    r_sh = kps[6][:2]
    r_el = (r_sh[0], r_sh[1] - 32.0 * scale)
    # Vector elbow->shoulder is (0, +1). Rotate by elbow_angle_deg to elbow->wrist
    theta = math.radians(elbow_angle_deg)
    r_wr = (r_el[0] + 32.0 * scale * math.sin(theta), r_el[1] + 32.0 * scale * math.cos(theta))
    kps[8] = (r_el[0], r_el[1], 0.95)
    kps[10] = (r_wr[0], r_wr[1], 0.95)
    kps[7] = (kps[5][0] - 12 * scale, kps[5][1] + 15 * scale, 0.9)
    kps[9] = (kps[5][0] - 16 * scale, kps[5][1] + 28 * scale, 0.9)

    bbox = (hip_x - 28 * scale, min(r_wr[1], nose_y) - 8, hip_x + 28 * scale, ankle_y + 4)
    return PoseFrame(frame_idx=frame_idx, timestamp_s=frame_idx / fps, bbox=bbox, keypoints=kps)


def test_role_isolation_hitter_swap_and_blocker_seam() -> None:
    fps = 30.0
    cal = NetCalibration(left_point=Point2D(x=180.0, y=320.0), right_point=Point2D(x=1100.0, y=320.0), is_manual=True)

    # Frames 10..50, contact at frame 40
    setter_frames, hitter_frames, decoy_frames, passer_frames = [], [], [], []
    b1_frames, b2_frames = [], []

    for f in range(10, 51):
        # Setter on Zone 2/3 near side
        setter_frames.append(_make_coco_frame(f, fps, 790.0, 455.0, 535.0))
        # Idle back-row passer near baseline
        passer_frames.append(_make_coco_frame(f, fps, 960.0, 595.0, 680.0))

        # Outside hitter jumps peaking at f=40
        dt = abs(f - 40) / fps
        h_jump = 90.0 * max(0.0, 1.0 - (dt / 0.4) ** 2)
        elb_ang = 85.0 if 30 <= f <= 35 else (168.0 if 38 <= f <= 42 else 120.0)
        hitter_frames.append(_make_coco_frame(f, fps, 340.0, 485.0 - h_jump, 565.0 - h_jump, elbow_angle_deg=elb_ang))

        # Middle decoy jumper also jumps simultaneously
        d_jump = 85.0 * max(0.0, 1.0 - (dt / 0.4) ** 2)
        decoy_frames.append(_make_coco_frame(f, fps, 600.0, 480.0 - d_jump, 560.0 - d_jump, elbow_angle_deg=140.0))

        # Two blockers across the net with open seam (~55px gap)
        b_jump = 55.0 * max(0.0, 1.0 - (dt / 0.35) ** 2)
        b1_frames.append(_make_coco_frame(f, fps, 305.0, 405.0 - b_jump, 468.0 - b_jump, scale=0.85))
        b2_frames.append(_make_coco_frame(f, fps, 420.0, 405.0 - b_jump, 468.0 - b_jump, scale=0.85))

    raw_tracks = [
        PlayerTrack(track_id=1, court_side="attacking", frames=setter_frames),
        PlayerTrack(track_id=2, court_side="attacking", frames=hitter_frames),
        PlayerTrack(track_id=3, court_side="attacking", frames=decoy_frames),
        PlayerTrack(track_id=4, court_side="attacking", frames=passer_frames),
        PlayerTrack(track_id=5, court_side="attacking", frames=b1_frames),  # intentionally mislabeled to test classifier
        PlayerTrack(track_id=6, court_side="attacking", frames=b2_frames),
    ]

    classified = classify_court_sides(smooth_pose_tracks(raw_tracks), cal)
    side_by_id = {t.track_id: t.court_side for t in classified}
    assert side_by_id[1] == "attacking"
    assert side_by_id[2] == "attacking"
    assert side_by_id[4] == "attacking"
    assert side_by_id[5] == "blocking"
    assert side_by_id[6] == "blocking"

    # Set trajectory ending near hitter's contact at x=354, y=260 on frame 40
    traj = SetTrajectory(
        release_frame=14,
        release_time_s=14 / fps,
        contact_frame=40,
        contact_time_s=40 / fps,
        peak_frame=28,
        peak_point=Point2D(x=560.0, y=210.0),
        tempo_s=round(26 / fps, 3),
        peak_height_px=110.0,
        peak_height_cm=107.6,
        arc_points=[
            BallDetection(frame_idx=14, timestamp_s=14 / fps, x=790.0, y=350.0, conf=0.9),
            BallDetection(frame_idx=40, timestamp_s=40 / fps, x=354.0, y=250.0, conf=0.9),
        ],
        inlier_count=20,
    )

    setter_id, hitter_id, metrics, block_summary, keyframes = analyse_play_mechanics(
        raw_tracks, traj, cal, fps=fps, frame_shape=(720, 1280)
    )
    assert hitter_id == 2
    assert setter_id == 1
    assert metrics is not None
    assert metrics.arm_extension_deg == pytest.approx(168.0, abs=3.0)
    assert metrics.torso_tilt_deg == pytest.approx(8.0, abs=2.0)
    assert metrics.jump_timing_label == "At peak jump"
    assert metrics.contact_height_px is not None and metrics.contact_height_px > 50.0
    assert len(block_summary.blockers) == 2
    assert block_summary.seam_open is True
    assert block_summary.seam_gap_px is not None and block_summary.seam_gap_px > 40.0
    assert [k.name for k in keyframes] == ["set_release", "max_load", "contact"]
    assert 30 <= keyframes[1].frame_idx <= 36

    # Reassign active hitter to middle decoy (track 3)
    _, swapped_id, swapped_metrics, _, _ = analyse_play_mechanics(
        raw_tracks, traj, cal, fps=fps, frame_shape=(720, 1280), override_hitter_track_id=3
    )
    assert swapped_id == 3
    assert swapped_metrics is not None
    assert swapped_metrics.arm_extension_deg == pytest.approx(140.0, abs=3.0)
