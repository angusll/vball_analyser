"""Unit tests for court homography, zone/seam classification, and serve distribution stats."""

from __future__ import annotations

import pytest

from vball_analyser.models import CourtCalibration, PasserPosition, Point2D, ServeEvent
from vball_analyser.serve_geometry import (
    DEFAULT_SETTER_TARGET,
    clamp_setter_target,
    classify_serve,
    compute_court_homography,
    compute_distribution_summary,
    compute_receive_summary,
    default_passer_formation,
    infer_slot_label,
    interpolate_court_from_net,
    normalize_receive_ring_radii,
    project_court_to_image,
    project_image_to_court,
    score_receive,
    slot_names_for_count,
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


def test_interpolate_court_from_net_with_off_center_sideline_point() -> None:
    lt = Point2D(x=215.0, y=252.0)
    rt = Point2D(x=978.0, y=260.0)
    # Off-center camera shifted left: left sideline goes almost straight down in image
    centered = interpolate_court_from_net(lt, rt, image_width=1056, net_height_m=2.43)
    skewed = interpolate_court_from_net(
        lt,
        rt,
        image_width=1056,
        net_height_m=2.43,
        sideline_point=Point2D(x=180.0, y=540.0),
    )
    assert skewed.sideline_point is not None
    # Both 3m corners shift strongly right compared to the centered estimate
    assert skewed.attack_left.x > centered.attack_left.x + 80.0
    assert skewed.attack_right.x > centered.attack_right.x + 80.0

    # Round-trip homography still exact
    H, H_inv = compute_court_homography(skewed)
    px, py = project_court_to_image(H_inv, 2.0, 5.0)
    assert project_image_to_court(H, px, py) == pytest.approx((2.0, 5.0), abs=1e-2)

    # Sideline click above the floor centerline must raise ValueError
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


def test_clamp_setter_target_bounds_and_non_finite() -> None:
    assert clamp_setter_target(Point2D(x=6.0, y=0.8)) == Point2D(x=6.0, y=0.8)
    assert clamp_setter_target(Point2D(x=-2.0, y=-1.0)) == Point2D(x=0.5, y=0.2)
    assert clamp_setter_target(Point2D(x=12.0, y=7.5)) == Point2D(x=8.5, y=4.0)

    for bad in (
        Point2D(x=float("nan"), y=1.0),
        Point2D(x=4.5, y=float("inf")),
        Point2D(x=float("-inf"), y=1.0),
    ):
        with pytest.raises(ValueError):
            clamp_setter_target(bad)


def test_score_receive_rings_boundaries_and_special_cases() -> None:
    target = DEFAULT_SETTER_TARGET  # (6.0, 0.8)

    # Ring boundaries: 1.5 -> 3, 3.2 -> 2, 4.8 -> 1, >4.8 -> 0
    assert score_receive(6.0, 0.8, "in", target) == (0.0, 3)
    assert score_receive(6.0, 2.3, "in", target) == (1.5, 3)
    assert score_receive(6.0, 2.31, "in", target) == (1.51, 2)
    assert score_receive(6.0, 4.0, "in", target) == (3.2, 2)
    assert score_receive(6.0, 4.01, "in", target) == (3.21, 1)
    assert score_receive(6.0, 5.6, "in", target) == (4.8, 1)
    assert score_receive(6.0, 5.61, "in", target) == (4.81, 0)

    # Overpass (py < 0) scores 0 even when close to setter target
    dist_op, sc_op = score_receive(6.0, -0.2, "in", target)
    assert dist_op == pytest.approx(1.0, abs=1e-3)
    assert sc_op == 0

    # Ace -> (None, 0); out / net -> (None, None)
    assert score_receive(None, None, "ace", target) == (None, 0)
    assert score_receive(6.0, 0.8, "out", target) == (None, None)
    assert score_receive(6.0, 0.8, "net", target) == (None, None)

    # Unplayable shank vs unrecorded manual serve
    assert score_receive(None, None, "in", target, unplayable=True) == (None, 0)
    assert score_receive(None, None, "in", target, unplayable=False) == (None, None)

    with pytest.raises(ValueError):
        score_receive(float("nan"), 1.0, "in", target)

    # Custom resizable ring radii
    assert normalize_receive_ring_radii((0.1, 0.2, 20.0)) == (0.4, 0.7, 8.5)
    with pytest.raises(ValueError):
        normalize_receive_ring_radii((1.0, float("inf"), 4.0))
    dist_c, sc_c = score_receive(target.x + 2.0, target.y, "in", target, ring_radii_m=(2.2, 4.0, 6.0))
    assert dist_c == pytest.approx(2.0, abs=1e-3) and sc_c == 3
    _, sc_tight = score_receive(target.x + 2.0, target.y, "in", target, ring_radii_m=(0.8, 1.5, 2.5))
    assert sc_tight == 1


def test_compute_receive_summary_metrics_passer_order_and_filter() -> None:
    empty = compute_receive_summary([])
    assert empty.scored_count == 0
    assert empty.avg_rating == 0.0
    assert empty.in_system_pct == 0.0
    assert empty.perfect_pct == 0.0
    assert empty.by_passer == []

    serves = [
        ServeEvent(
            serve_id="r1",
            index=1,
            server_name="#7",
            x_m=1.8,
            y_m=6.2,
            result="in",
            passer_label="P1",
            receive_score=3,
        ),
        ServeEvent(
            serve_id="r2",
            index=2,
            server_name="#7",
            x_m=4.5,
            y_m=6.5,
            result="in",
            passer_label="P2",
            receive_score=2,
        ),
        ServeEvent(
            serve_id="r3",
            index=3,
            server_name="#7",
            x_m=1.8,
            y_m=6.2,
            result="in",
            passer_label="P1",
            receive_score=1,
        ),
        ServeEvent(
            serve_id="r4",
            index=4,
            server_name="#7",
            x_m=7.5,
            y_m=6.0,
            result="ace",
            passer_label=None,
            receive_score=0,
        ),
        ServeEvent(
            serve_id="r5",
            index=5,
            server_name="#9",
            x_m=7.2,
            y_m=6.2,
            result="in",
            passer_label="P10",
            receive_score=3,
        ),
        ServeEvent(
            serve_id="r6",
            index=6,
            server_name="#7",
            x_m=4.5,
            y_m=9.6,
            result="out",
            receive_score=None,
        ),
    ]

    all_sum = compute_receive_summary(serves)
    assert all_sum.scored_count == 5
    assert all_sum.avg_rating == pytest.approx((3 + 2 + 1 + 0 + 3) / 5, abs=1e-2)
    assert all_sum.in_system_count == 3
    assert all_sum.in_system_pct == 60.0
    assert all_sum.perfect_count == 2
    assert all_sum.perfect_pct == 40.0
    assert all_sum.score_counts == {"0": 1, "1": 1, "2": 1, "3": 2}
    assert all_sum.score_pcts == {"0": 20.0, "1": 20.0, "2": 20.0, "3": 40.0}

    # Passer ordering: P1, P2, P10 (numeric), Unassigned last
    assert [p.passer_label for p in all_sum.by_passer] == ["P1", "P2", "P10", "Unassigned"]
    p1 = all_sum.by_passer[0]
    assert p1.count == 2
    assert p1.avg_rating == 2.0
    assert p1.in_system_pct == 50.0
    assert p1.score_counts == {"0": 0, "1": 1, "2": 0, "3": 1}

    # Server filter excludes #9 (P10) and still ignores out serve (r6)
    s7 = compute_receive_summary(serves, server_filter="#7")
    assert s7.scored_count == 4
    assert s7.avg_rating == 1.5
    assert s7.in_system_pct == 50.0
    assert s7.perfect_pct == 25.0
    assert [p.passer_label for p in s7.by_passer] == ["P1", "P2", "Unassigned"]


def test_infer_slot_label_handles_short_manual_serves_aces_and_errors() -> None:
    assert slot_names_for_count(1) == ["Mid"]
    assert slot_names_for_count(2) == ["Left", "Right"]
    assert slot_names_for_count(3) == ["Left", "Mid", "Right"]
    assert slot_names_for_count(4) == ["Left", "Mid-L", "Mid-R", "Right"]
    assert slot_names_for_count(5) == ["Left", "Mid-L", "Mid", "Mid-R", "Right"]

    passers = default_passer_formation(3)  # P1 (1.8, 6.2), P2 (4.5, 6.5), P3 (7.2, 6.2)
    # Short "in" serve 4.0m in front of P1: no 2.5m cap, still the Left slot.
    assert infer_slot_label(1.9, 2.2, "in", passers) == "Left"
    assert infer_slot_label(4.6, 8.1, "ace", passers) == "Mid"
    assert infer_slot_label(4.6, 8.1, "out", passers) is None
    assert infer_slot_label(4.6, 0.0, "net", passers) is None
    # Matching passer_label wins over nearest geometry; snapshot order is by x, not list order.
    assert infer_slot_label(1.0, 6.0, "in", passers, passer_label="P3") == "Right"
    assert infer_slot_label(1.0, 6.0, "in", list(reversed(passers))) == "Left"
    # Unknown passer_label falls back to nearest passer.
    assert infer_slot_label(8.0, 6.0, "in", passers, passer_label="Libero") == "Right"

    # Empty snapshot: fixed thirds.
    assert infer_slot_label(2.9, 6.0, "in", []) == "Left"
    assert infer_slot_label(3.0, 6.0, "in", []) == "Mid"
    assert infer_slot_label(6.0, 6.0, "ace", []) == "Mid"
    assert infer_slot_label(6.1, 6.0, "in", []) == "Right"
    assert infer_slot_label(6.1, 6.0, "out", []) is None


def test_vulnerability_splits_by_slot_and_by_passer_with_legacy_fallback() -> None:
    snap = default_passer_formation(3)

    def ev(sid: str, **kw: object) -> ServeEvent:
        return ServeEvent(serve_id=sid, index=1, passers_snapshot=snap, **kw)  # type: ignore[arg-type]

    serves = [
        ev("a", x_m=1.8, y_m=3.0, depth="short", slot_label="Left", passer_label="P1",
           move_distance_m=0.4, contact_alignment="midline", receive_score=3),
        ev("b", x_m=3.1, y_m=7.0, depth="deep", is_seam=True, slot_label="Left", passer_label="P1",
           move_distance_m=1.6, contact_alignment="outside", receive_score=1),
        # Manual serve: no movement measured.
        ev("c", x_m=7.2, y_m=6.2, depth="deep", slot_label="Right", passer_label="P3", receive_score=2),
        # Legacy serves (slot_label=None): inferred from passer_label / nearest snapshot passer.
        ev("d", x_m=1.0, y_m=6.5, depth="deep", passer_label="P2", receive_score=0),
        ev("e", x_m=7.5, y_m=7.0, depth="deep", result="ace", receive_score=0),
        ev("f", x_m=4.5, y_m=9.6, depth="out", result="out"),
    ]

    drill = compute_receive_summary(serves)
    assert drill.active_group_by == "slot"
    assert compute_receive_summary(serves, session_mode="match").active_group_by == "passer"
    assert compute_receive_summary([], session_mode="match").active_group_by == "passer"
    assert serves[3].slot_label is None  # summary must not mutate stored serves

    assert drill.scored_count == 5
    assert drill.move_count == 2
    assert drill.avg_move_m == 1.0
    assert drill.midline_count == 1
    assert drill.alignment_count == 2
    assert drill.midline_pct == 50.0
    sp = drill.splits
    assert (sp.short.count, sp.short.avg_rating, sp.short.in_system_pct) == (1, 3.0, 100.0)
    assert (sp.deep.count, sp.deep.avg_rating, sp.deep.in_system_pct) == (4, 0.75, 25.0)
    assert (sp.seam.count, sp.seam.avg_rating, sp.seam.in_system_pct) == (1, 1.0, 0.0)
    assert (sp.direct.count, sp.direct.avg_rating, sp.direct.in_system_pct) == (4, 1.25, 50.0)
    assert (sp.stationary.count, sp.stationary.avg_rating) == (1, 3.0)
    assert (sp.moving.count, sp.moving.avg_rating) == (1, 1.0)

    assert [r.passer_label for r in drill.by_slot] == ["Left", "Mid", "Right"]
    left, mid, right = drill.by_slot
    assert (left.count, left.avg_rating, left.in_system_pct) == (2, 2.0, 50.0)
    assert (left.avg_move_m, left.midline_pct) == (1.0, 50.0)
    assert (left.splits.short.avg_rating, left.splits.deep.avg_rating) == (3.0, 1.0)
    assert (left.splits.stationary.count, left.splits.moving.count) == (1, 1)
    assert (mid.count, mid.avg_rating) == (1, 0.0)
    assert (right.count, right.avg_rating, right.in_system_pct) == (2, 1.0, 50.0)
    assert right.move_count == 0
    assert right.avg_move_m is None
    assert right.midline_pct is None
    assert right.splits.stationary.count == 0
    assert right.splits.stationary.avg_rating is None
    assert right.splits.stationary.in_system_pct is None

    assert [r.passer_label for r in drill.by_passer] == ["P1", "P2", "P3", "Unassigned"]
    assert drill.by_passer[0].splits.seam.count == 1

    # Old JSON (no new fields) still loads with defaults.
    from vball_analyser.models import ServeSession

    old = ServeSession.model_validate(
        {"serve_session_id": "s1", "name": "Old", "created_at": "2026-01-01T00:00:00Z",
         "serves": [{"serve_id": "x", "index": 1, "x_m": 4.5, "y_m": 6.0}]}
    )
    assert old.session_mode == "drill"
    assert old.passers_roster == []
    assert old.serves[0].slot_label is None
    assert old.serves[0].slot_label_manual is False
    assert old.serves[0].move_distance_m is None
    assert old.serves[0].contact_alignment is None
