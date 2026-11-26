"""Unit tests for court homography, zone/seam classification, and serve distribution stats."""

from __future__ import annotations

import pytest

from vball_analyser.models import CourtCalibration, PasserPosition, Point2D, ServeEvent
from vball_analyser.serve_geometry import (
    classify_serve,
    compute_court_homography,
    compute_distribution_summary,
    default_passer_formation,
    interpolate_court_from_net,
    project_court_to_image,
    project_image_to_court,
)


def test_homography_roundtrip_and_degenerate_quad() -> None:
    cal = CourtCalibration(
        front_left=Point2D(x=200.0, y=320.0),
        front_right=Point2D(x=850.0, y=320.0),
        attack_right=Point2D(x=940.0, y=420.0),
        attack_left=Point2D(x=110.0, y=420.0),
    )
    H, H_inv = compute_court_homography(cal)
    for cx, cy in [(0.0, 0.0), (9.0, 0.0), (9.0, 3.0), (0.0, 3.0), (4.5, 6.5)]:
        px, py = project_court_to_image(H_inv, cx, cy)
        rx, ry = project_image_to_court(H, px, py)
        assert rx == pytest.approx(cx, abs=1e-2)
        assert ry == pytest.approx(cy, abs=1e-2)

    # Collinear points must raise ValueError (Review Focus #1)
    bad = CourtCalibration(
        front_left=Point2D(x=100.0, y=300.0),
        front_right=Point2D(x=500.0, y=300.0),
        attack_right=Point2D(x=700.0, y=300.0),
        attack_left=Point2D(x=200.0, y=300.0),
    )
    with pytest.raises(ValueError):
        compute_court_homography(bad)

    # Twisted (self-intersecting) quad must also raise ValueError
    twisted = CourtCalibration(
        front_left=Point2D(x=200.0, y=320.0),
        front_right=Point2D(x=850.0, y=320.0),
        attack_right=Point2D(x=110.0, y=420.0),
        attack_left=Point2D(x=940.0, y=420.0),
    )
    with pytest.raises(ValueError):
        compute_court_homography(twisted)

    # Clicking clockwise starting from bottom-left (3m-Left -> Net-Left -> Net-Right -> 3m-Right)
    # must normalize to the exact same homography as top-left first
    rotated = CourtCalibration(
        front_left=Point2D(x=110.0, y=420.0),
        front_right=Point2D(x=200.0, y=320.0),
        attack_right=Point2D(x=850.0, y=320.0),
        attack_left=Point2D(x=940.0, y=420.0),
    )
    H_rot, _ = compute_court_homography(rotated)
    assert project_image_to_court(H_rot, 200.0, 320.0) == pytest.approx((0.0, 0.0), abs=1e-2)
    assert project_image_to_court(H_rot, 940.0, 420.0) == pytest.approx((9.0, 3.0), abs=1e-2)


def test_interpolate_court_from_net_handles_clipped_sidelines_and_click_order() -> None:
    lt = Point2D(x=215.0, y=252.0)
    rt = Point2D(x=978.0, y=260.0)
    # Clicking right-to-left must produce the same normalized calibration as left-to-right
    cal_lr = interpolate_court_from_net(lt, rt, image_width=1056, net_height_m=2.43)
    cal_rl = interpolate_court_from_net(rt, lt, image_width=1056, net_height_m=2.43)
    assert cal_lr.method == "net"
    assert cal_lr.front_left.x == pytest.approx(cal_rl.front_left.x, abs=1e-2)
    assert cal_lr.attack_right.x > 1056.0  # Clipped right 3m corner outside frame still valid!
    H, H_inv = compute_court_homography(cal_lr)
    px, py = project_court_to_image(H_inv, 4.5, 6.0)
    rx, ry = project_image_to_court(H, px, py)
    assert (rx, ry) == pytest.approx((4.5, 6.0), abs=1e-2)

    # Too-short net span rejected
    with pytest.raises(ValueError):
        interpolate_court_from_net(Point2D(x=500.0, y=250.0), Point2D(x=510.0, y=250.0), image_width=1056)


def test_interpolate_court_from_net_off_center_sideline_point() -> None:
    lt = Point2D(x=215.0, y=252.0)
    rt = Point2D(x=978.0, y=260.0)
    # Camera shifted left: left sideline drops almost vertically below front_left (~x=213, y=458)
    side_pt = Point2D(x=205.0, y=530.0)
    cal = interpolate_court_from_net(lt, rt, image_width=1056, net_height_m=2.43, sideline_point=side_pt)
    H, _ = compute_court_homography(cal)
    # The clicked sideline point must project onto the left sideline (x_m == 0.0)
    sx, sy = project_image_to_court(H, side_pt.x, side_pt.y)
    assert sx == pytest.approx(0.0, abs=0.05)
    assert sy > 2.0
    # Right sideline must flare much wider than left sideline
    assert (cal.attack_right.x - cal.front_right.x) > 3.0 * abs(cal.attack_left.x - cal.front_left.x)

    # Sideline point above the net floor must raise ValueError
    with pytest.raises(ValueError):
        interpolate_court_from_net(lt, rt, image_width=1056, sideline_point=Point2D(x=200.0, y=300.0))


def test_classify_serve_zones_seams_and_result_invariants() -> None:
    passers = default_passer_formation(3)  # P1(1.8,6.2), P2(4.5,6.5), P3(7.2,6.2)

    # Seam hit between P1 and P2 in Zone 6 deep
    x, y, res, zone, depth, is_seam, pair = classify_serve(3.15, 6.3, "in", passers)
    assert (res, zone, depth, is_seam, pair) == ("in", "Zone 6", "deep", True, ("P1", "P2"))

    # Directly on P2 body -> NOT a seam
    _, _, _, zone2, _, is_seam2, pair2 = classify_serve(4.55, 6.5, "in", passers)
    assert zone2 == "Zone 6" and is_seam2 is False and pair2 is None

    # Front-row short Zone 4
    _, _, _, zone3, depth3, is_seam3, _ = classify_serve(1.5, 2.0, "ace", passers)
    assert (zone3, depth3, is_seam3) == ("Zone 4", "short", False)

    # Outside court auto-marks out and clears seam
    _, _, res4, zone4, depth4, is_seam4, _ = classify_serve(4.5, 9.6, "in", passers)
    assert (res4, zone4, depth4, is_seam4) == ("out", "Out Long", "out", False)

    # Net result snaps y_m=0.0, forces Net / Short and no seam
    nx, ny, res5, zone5, depth5, is_seam5, _ = classify_serve(3.2, 5.0, "net", passers)
    assert (nx, ny, res5, zone5, depth5, is_seam5) == (3.2, 0.0, "net", "Net / Short", "out", False)

    # Fewer than 2 passers & close passers (Review Focus #2)
    _, _, _, _, _, seam_one, _ = classify_serve(3.0, 6.2, "in", [PasserPosition(label="P1", x_m=1.8, y_m=6.2)])
    assert seam_one is False
    close_passers = [PasserPosition(label="P1", x_m=3.0, y_m=6.0), PasserPosition(label="P2", x_m=4.0, y_m=6.0)]
    _, _, _, _, _, seam_close, pair_close = classify_serve(3.5, 6.0, "in", close_passers)
    assert seam_close is True and pair_close == ("P1", "P2")


def test_distribution_summary_formulas_and_filter() -> None:
    empty = compute_distribution_summary([])
    assert empty.total == 0 and empty.in_pct == 0.0 and empty.seam_pct == 0.0

    serves = [
        ServeEvent(
            serve_id="s1",
            index=1,
            source="manual",
            server_name="#7",
            x_m=3.1,
            y_m=6.3,
            result="in",
            zone="Zone 6",
            depth="deep",
            is_seam=True,
            seam_pair=("P1", "P2"),
        ),
        ServeEvent(
            serve_id="s2",
            index=2,
            source="manual",
            server_name="#7",
            x_m=7.5,
            y_m=2.0,
            result="ace",
            zone="Zone 2",
            depth="short",
            is_seam=False,
        ),
        ServeEvent(
            serve_id="s3",
            index=3,
            source="manual",
            server_name="#7",
            x_m=4.0,
            y_m=9.5,
            result="out",
            zone="Out Long",
            depth="out",
            is_seam=False,
        ),
        ServeEvent(
            serve_id="s4",
            index=4,
            source="manual",
            server_name="#9",
            x_m=3.0,
            y_m=0.0,
            result="net",
            zone="Net / Short",
            depth="out",
            is_seam=False,
        ),
    ]
    all_sum = compute_distribution_summary(serves)
    assert all_sum.total == 4
    assert all_sum.in_pct == 50.0
    assert all_sum.ace_pct == 25.0
    assert all_sum.error_pct == 50.0
    assert all_sum.in_pct + all_sum.error_pct == pytest.approx(100.0)
    assert all_sum.seam_count == 1 and all_sum.seam_pct == 50.0
    assert all_sum.seam_by_pair == {"P1-P2": 1}

    s7 = compute_distribution_summary(serves, server_filter="#7")
    assert s7.total == 3 and s7.net_count == 0
