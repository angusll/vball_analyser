"""Tests for net top-tape calibration and geometry calculations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vball_analyser.calibration import (
    auto_detect_net_tape,
    classify_attack_zone,
    cm_per_px,
    height_above_tape,
    net_ground_y_at,
    tape_y_at,
)
from vball_analyser.models import NetCalibration, Point2D
from vball_analyser.video_io import iter_video_frames


def test_tilted_calibration_interpolation_and_zones() -> None:
    cal = NetCalibration(
        left_point=Point2D(x=100.0, y=300.0),
        right_point=Point2D(x=1000.0, y=345.0),
        net_height_m=2.43,
        is_manual=True,
    )
    assert tape_y_at(cal, 550.0) == pytest.approx(322.5, abs=0.1)

    px_above, cm_above = height_above_tape(cal, 550.0, 272.5)
    assert px_above == pytest.approx(50.0, abs=0.1)
    span = math.hypot(900.0, 45.0)
    expected_cm = 50.0 * (900.0 / span)
    assert cm_above == pytest.approx(expected_cm, abs=0.2)
    assert cm_per_px(cal) == pytest.approx(900.0 / span, rel=1e-4)

    # Below tape yields negative height
    px_below, _ = height_above_tape(cal, 550.0, 350.0)
    assert px_below < 0

    assert net_ground_y_at(cal, 550.0) > 322.5
    assert classify_attack_zone(cal, 250.0) == "Zone 4 (Left)"
    assert classify_attack_zone(cal, 550.0) == "Zone 3 (Middle)"
    assert classify_attack_zone(cal, 880.0) == "Zone 2 (Right)"


def test_auto_detect_tilted_net_tape(synthetic_clip: tuple[Path, dict]) -> None:
    clip_path, gt = synthetic_clip
    frames = [bgr for _idx, _ts, bgr in iter_video_frames(clip_path, sample_fps=4.0)]
    cal = auto_detect_net_tape(frames)

    gt_lx, gt_ly = gt["left_pt"]
    gt_rx, gt_ry = gt["right_pt"]

    def gt_tape_y(x: float) -> float:
        t = (x - gt_lx) / (gt_rx - gt_lx)
        return gt_ly + t * (gt_ry - gt_ly)

    for test_x in (320.0, 640.0, 960.0):
        assert abs(tape_y_at(cal, test_x) - gt_tape_y(test_x)) <= 4.0
    assert not cal.is_manual


def test_auto_detect_fallback_on_blank_frames() -> None:
    blank = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    cal = auto_detect_net_tape(blank)
    assert cal.left_point.x < cal.right_point.x
    assert not cal.is_manual


def test_auto_detect_prefers_top_tape_over_bottom_tape() -> None:
    import cv2

    h, w = 576, 1168
    frame = np.full((h, w, 3), 135, dtype=np.uint8)
    # Top tape at y=200, bottom tape at y=275, vertical net mesh strictly between them
    cv2.line(frame, (180, 200), (980, 200), (235, 235, 235), 4)
    cv2.line(frame, (180, 275), (980, 275), (240, 240, 240), 4)
    for x in range(184, 980, 10):
        cv2.line(frame, (x, 204), (x, 271), (55, 55, 55), 1)
    for y in range(210, 270, 10):
        cv2.line(frame, (180, y), (980, y), (75, 75, 75), 1)

    cal = auto_detect_net_tape([frame, frame.copy(), frame.copy()])
    assert abs(tape_y_at(cal, 580.0) - 200.0) <= 6.0
    assert cal.net_band_px is not None
    assert abs(cal.net_band_px - 75.0) <= 6.0
    px_above, cm_above = height_above_tape(cal, 580.0, 125.0)
    assert px_above == pytest.approx(75.0, abs=6.0)
    assert cm_above is not None
    assert abs(cm_above - 100.0) <= 10.0
