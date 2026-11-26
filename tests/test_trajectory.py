"""Tests for gravity-constrained RANSAC set-arc fitting."""

from __future__ import annotations

import random

from vball_analyser.models import BallDetection, NetCalibration, Point2D
from vball_analyser.trajectory import fit_setting_path


def test_fit_setting_path_rejects_stray_balls_and_truncates_at_spike() -> None:
    fps = 30.0
    cal = NetCalibration(
        left_point=Point2D(x=180.0, y=320.0),
        right_point=Point2D(x=1100.0, y=340.0),
        is_manual=True,
    )
    rng = random.Random(7)

    release_frame = 15
    contact_frame = 45  # 1.0s tempo
    x0, y0 = 780.0, 360.0
    x1, y1 = 330.0, 265.0
    T = (contact_frame - release_frame) / fps
    g = 900.0
    vx = (x1 - x0) / T
    vy = (y1 - y0 - 0.5 * g * T * T) / T

    detections: list[BallDetection] = []
    dropped_frames = {24, 25, 26, 27}  # 4-frame motion blur gap

    for f in range(release_frame, contact_frame + 1):
        if f in dropped_frames:
            continue
        tau = (f - release_frame) / fps
        bx = x0 + vx * tau + rng.uniform(-2.0, 2.0)
        by = y0 + vy * tau + 0.5 * g * tau * tau + rng.uniform(-2.0, 2.0)
        detections.append(BallDetection(frame_idx=f, timestamp_s=f / fps, x=bx, y=by, conf=0.85))

    # Post-contact downward spike deflection (must NOT be part of set arc)
    for f in range(contact_frame + 1, contact_frame + 12):
        tau_hit = (f - contact_frame) / fps
        detections.append(
            BallDetection(
                frame_idx=f,
                timestamp_s=f / fps,
                x=x1 - 180.0 * tau_hit,
                y=y1 + 520.0 * tau_hit,
                conf=0.88,
            )
        )

    # 20 stray background balls + static cart ball
    for f in range(0, 60, 3):
        detections.append(
            BallDetection(
                frame_idx=f,
                timestamp_s=f / fps,
                x=1150.0 + rng.uniform(-1.5, 1.5),
                y=640.0 + rng.uniform(-1.5, 1.5),
                conf=0.9,
            )
        )
        detections.append(
            BallDetection(
                frame_idx=f,
                timestamp_s=f / fps,
                x=rng.uniform(100.0, 1100.0),
                y=rng.uniform(480.0, 650.0),
                conf=0.45,
            )
        )

    traj = fit_setting_path(detections, cal, fps=fps, frame_height=720)
    assert traj is not None
    assert abs(traj.release_frame - release_frame) <= 1
    assert abs(traj.contact_frame - contact_frame) <= 1
    assert 0.90 <= traj.tempo_s <= 1.10
    assert traj.peak_height_px > 40.0
    assert traj.peak_height_cm is not None and traj.peak_height_cm > 35.0
    # Interpolated gap frames must be filled cleanly
    arc_by_f = {p.frame_idx: p for p in traj.arc_points}
    for df in dropped_frames:
        assert df in arc_by_f
        assert arc_by_f[df].is_interpolated


def test_fit_setting_path_sparse_returns_none() -> None:
    cal = NetCalibration(left_point=Point2D(x=100, y=300), right_point=Point2D(x=1100, y=300))
    sparse = [
        BallDetection(frame_idx=5, timestamp_s=5 / 30.0, x=500, y=220, conf=0.8),
        BallDetection(frame_idx=6, timestamp_s=6 / 30.0, x=490, y=215, conf=0.8),
    ]
    assert fit_setting_path(sparse, cal, fps=30.0, frame_height=720) is None


def test_fit_setting_path_picks_set_over_longer_serve_and_pass_arcs() -> None:
    fps = 30.0
    cal = NetCalibration(left_point=Point2D(x=50.0, y=260.0), right_point=Point2D(x=1020.0, y=260.0))
    dets: list[BallDetection] = []

    # Arc 1: Serve (24 frames, f=20..44) ending in backcourt (no jumping attacker)
    for f in range(20, 45):
        tau = (f - 20) / fps
        dets.append(BallDetection(frame_idx=f, timestamp_s=f / fps, x=560 - 140 * tau, y=240 - 260 * tau + 400 * tau * tau, conf=0.8))

    # Arc 2: Pass (20 frames, f=55..75)
    for f in range(55, 76):
        tau = (f - 55) / fps
        dets.append(BallDetection(frame_idx=f, timestamp_s=f / fps, x=380 + 240 * tau, y=290 - 420 * tau + 600 * tau * tau, conf=0.8))

    # Arc 3: Set (only 10 observed frames with ceiling-light dropout f=99..112, ending at hitter f=120)
    # x(t) = 560 - 300*tau, y(t) = 240 - 450*tau + 380*tau^2
    for f in list(range(90, 99)) + [112, 113, 114]:
        tau = (f - 90) / fps
        dets.append(BallDetection(frame_idx=f, timestamp_s=f / fps, x=560 - 300 * tau, y=240 - 450 * tau + 380 * tau * tau, conf=0.7))

    # Stray floor false positive at f=116,117 far from trajectory
    dets.append(BallDetection(frame_idx=116, timestamp_s=116 / fps, x=410.0, y=490.0, conf=0.35))
    dets.append(BallDetection(frame_idx=117, timestamp_s=117 / fps, x=405.0, y=488.0, conf=0.35))

    # Airborne attacker contact at f=120 (tau=1.0s -> x=260, y=170)
    cand_contacts = [(120, Point2D(x=262.0, y=174.0))]

    traj = fit_setting_path(dets, cal, fps=fps, frame_height=576, candidate_contact_frames=cand_contacts)
    assert traj is not None
    assert 89 <= traj.release_frame <= 91
    assert traj.contact_frame == 120


def test_fit_setting_path_rejects_vertical_toss_duplicates_and_low_conf_bridge() -> None:
    fps = 30.0
    cal = NetCalibration(
        left_point=Point2D(x=240.0, y=200.0),
        right_point=Point2D(x=910.0, y=200.0),
        is_manual=True,
    )
    dets: list[BallDetection] = []

    # 1. Nearly-vertical server toss with duplicate VFR frames below net tape (f=43..61)
    for f in range(43, 62):
        step = (f - 43) // 2
        dets.append(
            BallDetection(
                frame_idx=f,
                timestamp_s=f / fps,
                x=380.0 + 0.3 * step,
                y=248.0 - 4.0 * step + 0.4 * step * step,
                conf=0.65,
            )
        )

    # 2. Low-confidence motion-damped static FP bridged to a falling serve (f=81 + f=88..92)
    dets.append(BallDetection(frame_idx=81, timestamp_s=81 / fps, x=327.0, y=308.0, conf=0.04))
    for f, yv in [(88, 163.0), (89, 163.0), (90, 185.0), (91, 186.0), (92, 215.0)]:
        dets.append(BallDetection(frame_idx=f, timestamp_s=f / fps, x=376.0 + (f - 88), y=yv, conf=0.55))

    # 3. True high set arc from setter (625, 220) to left outside hitter (250, 140) across f=142..178
    for f in range(142, 179, 2):
        tau = (f - 142) / fps
        bx = 625.0 - 310.0 * tau
        by = 220.0 - 520.0 * tau + 380.0 * tau * tau
        dets.append(BallDetection(frame_idx=f, timestamp_s=f / fps, x=bx, y=by, conf=0.45))

    cand_contacts = [(178, Point2D(x=252.0, y=145.0))]
    traj = fit_setting_path(dets, cal, fps=fps, frame_height=576, candidate_contact_frames=cand_contacts)
    assert traj is not None
    assert 141 <= traj.release_frame <= 144
    assert 176 <= traj.contact_frame <= 180
    assert traj.peak_point.y < 100.0


