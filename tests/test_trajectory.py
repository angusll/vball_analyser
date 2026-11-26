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
