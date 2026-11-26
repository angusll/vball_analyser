"""Top-down court homography, zone/seam classification, and serve distribution summaries."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from vball_analyser.models import (
    CourtCalibration,
    PasserPosition,
    Point2D,
    ServeDistributionSummary,
    ServeEvent,
)

COURT_WIDTH_M = 9.0
COURT_LENGTH_M = 9.0
ATTACK_LINE_M = 3.0
SHORT_DEEP_SPLIT_M = 4.5
SEAM_LONGITUDINAL_BAND_M = 2.2
SEAM_MAX_BODY_RADIUS_M = 0.75
SEAM_GAP_BODY_FRACTION = 0.30

IN_BOUNDS_ZONES: tuple[str, ...] = (
    "Zone 1",
    "Zone 2",
    "Zone 3",
    "Zone 4",
    "Zone 5",
    "Zone 6",
)


def default_passer_formation(count: int = 3) -> list[PasserPosition]:
    """Return standard practice serve-receive passer formations in court meters."""
    if count == 2:
        return [
            PasserPosition(label="P1", x_m=2.5, y_m=6.3),
            PasserPosition(label="P2", x_m=6.5, y_m=6.3),
        ]
    if count == 4:
        return [
            PasserPosition(label="P1", x_m=1.4, y_m=6.0),
            PasserPosition(label="P2", x_m=3.5, y_m=6.5),
            PasserPosition(label="P3", x_m=5.5, y_m=6.5),
            PasserPosition(label="P4", x_m=7.6, y_m=6.0),
        ]
    return [
        PasserPosition(label="P1", x_m=1.8, y_m=6.2),
        PasserPosition(label="P2", x_m=4.5, y_m=6.5),
        PasserPosition(label="P3", x_m=7.2, y_m=6.2),
    ]


def _validate_convex_quad(pts: np.ndarray) -> None:
    """Reject collinear, duplicate, tiny, or self-intersecting 4-corner calibrations."""
    if not np.all(np.isfinite(pts)):
        raise ValueError("Court calibration points must be finite numbers.")

    area = abs(cv2.contourArea(pts.astype(np.float32)))
    if area < 400.0:
        raise ValueError("Court calibration quad is degenerate or too small.")

    signs: list[float] = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) < 1e-3:
            raise ValueError("Court calibration corners must not be collinear.")
        signs.append(cross)

    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        raise ValueError("Court calibration corners must form a non-twisted convex quadrilateral.")


def normalize_court_calibration(cal: CourtCalibration) -> CourtCalibration:
    """Sort 4 clicked front-3m corners into (front_left, front_right, attack_right, attack_left) regardless of click order."""
    pts = [cal.front_left, cal.front_right, cal.attack_right, cal.attack_left]
    by_y = sorted(pts, key=lambda p: p.y)
    top_two = sorted(by_y[:2], key=lambda p: p.x)
    bot_two = sorted(by_y[2:], key=lambda p: p.x)
    return CourtCalibration(
        front_left=top_two[0],
        front_right=top_two[1],
        attack_right=bot_two[1],
        attack_left=bot_two[0],
        is_calibrated=cal.is_calibrated,
        method=cal.method,
        net_left_top=cal.net_left_top,
        net_right_top=cal.net_right_top,
        sideline_point=cal.sideline_point,
        net_height_m=cal.net_height_m,
    )


def interpolate_court_from_net(
    left_top: Point2D,
    right_top: Point2D,
    image_width: int,
    net_height_m: float = 2.43,
    attack_line_y: float | None = None,
    cam_height_m: float = 1.55,
    sideline_point: Point2D | None = None,
) -> CourtCalibration:
    """Interpolate the full 9m x 9m receiving court homography from the 2 net antenna top-tape points
    and an optional 3rd point on either near-court sideline for off-center camera angles.

    Works even when the baseline and 3m sideline corners are cropped outside the camera frame.
    """
    if not all(math.isfinite(v) for v in (left_top.x, left_top.y, right_top.x, right_top.y, net_height_m)):
        raise ValueError("Net calibration points must be finite numbers.")

    lt, rt = (left_top, right_top) if left_top.x <= right_top.x else (right_top, left_top)
    dx = rt.x - lt.x
    dy = rt.y - lt.y
    span = math.hypot(dx, dy)
    if span < 40.0:
        raise ValueError("Net antenna endpoints are too close together.")

    # Unit downward perpendicular to the net top tape
    nx = -dy / span
    ny = dx / span
    if ny < 0.0:
        nx, ny = -nx, -ny

    net_h_px = (net_height_m / COURT_WIDTH_M) * span
    fl = Point2D(x=round(lt.x + nx * net_h_px, 2), y=round(lt.y + ny * net_h_px, 2))
    fr = Point2D(x=round(rt.x + nx * net_h_px, 2), y=round(rt.y + ny * net_h_px, 2))

    floor_mid_x = 0.5 * (fl.x + fr.x)
    floor_mid_y = 0.5 * (fl.y + fr.y)
    vp_x = floor_mid_x
    vp_y = floor_mid_y - (cam_height_m / net_height_m) * net_h_px

    if sideline_point is not None:
        if not (math.isfinite(sideline_point.x) and math.isfinite(sideline_point.y)):
            raise ValueError("Sideline point must be finite numbers.")
        anchor = fl if abs(sideline_point.x - fl.x) <= abs(sideline_point.x - fr.x) else fr
        if sideline_point.y <= anchor.y + 8.0:
            raise ValueError("Sideline point must be on the near court below the net centerline.")
        t_vp = (vp_y - anchor.y) / (sideline_point.y - anchor.y)
        raw_vp_x = anchor.x + t_vp * (sideline_point.x - anchor.x)
        vp_x = max(floor_mid_x - 4.0 * span, min(floor_mid_x + 4.0 * span, raw_vp_x))

    if attack_line_y is not None and math.isfinite(attack_line_y):
        denom = max(10.0, floor_mid_y - vp_y)
        scale_3m = max(1.08, min(2.5, (attack_line_y - vp_y) / denom))
    else:
        f_px = 0.95 * max(320.0, float(image_width))
        d_cam = max(5.0, (f_px * COURT_WIDTH_M) / span)
        scale_3m = d_cam / max(1.5, d_cam - ATTACK_LINE_M)

    al = Point2D(
        x=round(vp_x + (fl.x - vp_x) * scale_3m, 2),
        y=round(vp_y + (fl.y - vp_y) * scale_3m, 2),
    )
    ar = Point2D(
        x=round(vp_x + (fr.x - vp_x) * scale_3m, 2),
        y=round(vp_y + (fr.y - vp_y) * scale_3m, 2),
    )
    cal = CourtCalibration(
        front_left=fl,
        front_right=fr,
        attack_right=ar,
        attack_left=al,
        is_calibrated=True,
        method="net",
        net_left_top=lt,
        net_right_top=rt,
        sideline_point=sideline_point,
        net_height_m=net_height_m,
    )
    compute_court_homography(cal)
    return cal


def compute_court_homography(cal: CourtCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Compute (H_img_to_court_m, H_court_m_to_img) from the front 3m rectangle."""
    raw_src = np.array(
        [
            [cal.front_left.x, cal.front_left.y],
            [cal.front_right.x, cal.front_right.y],
            [cal.attack_right.x, cal.attack_right.y],
            [cal.attack_left.x, cal.attack_left.y],
        ],
        dtype=np.float32,
    )
    _validate_convex_quad(raw_src)

    norm = normalize_court_calibration(cal)
    src = np.array(
        [
            [norm.front_left.x, norm.front_left.y],
            [norm.front_right.x, norm.front_right.y],
            [norm.attack_right.x, norm.attack_right.y],
            [norm.attack_left.x, norm.attack_left.y],
        ],
        dtype=np.float32,
    )
    _validate_convex_quad(src)

    dst = np.array(
        [
            [0.0, 0.0],
            [COURT_WIDTH_M, 0.0],
            [COURT_WIDTH_M, ATTACK_LINE_M],
            [0.0, ATTACK_LINE_M],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError("Unable to compute invertible court homography.")
    H_inv = np.linalg.inv(H)
    return H, H_inv


def project_image_to_court(H: np.ndarray, x_px: float, y_px: float) -> tuple[float, float]:
    """Project an image pixel (x_px, y_px) through homography H into court meters (x_m, y_m)."""
    vec = np.array([float(x_px), float(y_px), 1.0], dtype=np.float64)
    out = H @ vec
    if abs(float(out[2])) < 1e-9:
        raise ValueError("Homography projection denominator near zero.")
    return round(float(out[0] / out[2]), 3), round(float(out[1] / out[2]), 3)


def project_court_to_image(H_inv: np.ndarray, x_m: float, y_m: float) -> tuple[float, float]:
    """Project a court coordinate (x_m, y_m) into image pixels (x_px, y_px)."""
    vec = np.array([float(x_m), float(y_m), 1.0], dtype=np.float64)
    out = H_inv @ vec
    if abs(float(out[2])) < 1e-9:
        raise ValueError("Inverse homography projection denominator near zero.")
    return round(float(out[0] / out[2]), 2), round(float(out[1] / out[2]), 2)


def _out_of_bounds_zone(x_m: float, y_m: float) -> str | None:
    if y_m < 0.0:
        return "Net / Short"
    if y_m > COURT_LENGTH_M:
        return "Out Long"
    if x_m < 0.0:
        return "Out Left"
    if x_m > COURT_WIDTH_M:
        return "Out Right"
    return None


def _in_bounds_zone(x_m: float, y_m: float) -> str:
    col = 0 if x_m < 3.0 else (1 if x_m < 6.0 else 2)
    if y_m < ATTACK_LINE_M:
        return ("Zone 4", "Zone 3", "Zone 2")[col]
    return ("Zone 5", "Zone 6", "Zone 1")[col]


def _detect_seam(
    x_m: float, y_m: float, passers: list[PasserPosition]
) -> tuple[bool, tuple[str, str] | None]:
    if len(passers) < 2:
        return False, None

    ordered = sorted(passers, key=lambda p: p.x_m)
    for i in range(len(ordered) - 1):
        p_left = ordered[i]
        p_right = ordered[i + 1]
        gap = p_right.x_m - p_left.x_m
        if gap <= 1e-6:
            continue
        if not (p_left.x_m < x_m < p_right.x_m):
            continue
        avg_y = 0.5 * (p_left.y_m + p_right.y_m)
        if abs(y_m - avg_y) > SEAM_LONGITUDINAL_BAND_M:
            continue
        r_body = min(SEAM_MAX_BODY_RADIUS_M, SEAM_GAP_BODY_FRACTION * gap)
        d_left = math.hypot(x_m - p_left.x_m, y_m - p_left.y_m)
        d_right = math.hypot(x_m - p_right.x_m, y_m - p_right.y_m)
        if d_left >= r_body and d_right >= r_body:
            return True, (p_left.label, p_right.label)
    return False, None


def classify_serve(
    x_m: float,
    y_m: float,
    result: Literal["in", "ace", "out", "net"],
    passers: list[PasserPosition],
) -> tuple[
    float,
    float,
    Literal["in", "ace", "out", "net"],
    str,
    Literal["short", "deep", "out"],
    bool,
    tuple[str, str] | None,
]:
    """Normalize coordinates/result and compute zone, depth, and seam classification."""
    nx = round(float(x_m), 3)
    ny = round(float(y_m), 3)

    if result == "net":
        return nx, 0.0, "net", "Net / Short", "out", False, None

    oob_zone = _out_of_bounds_zone(nx, ny)
    if oob_zone is not None:
        return nx, ny, "out", oob_zone, "out", False, None

    if result == "out":
        return nx, ny, "out", "Out", "out", False, None

    zone = _in_bounds_zone(nx, ny)
    depth: Literal["short", "deep", "out"] = "short" if ny < SHORT_DEEP_SPLIT_M else "deep"
    is_seam, seam_pair = _detect_seam(nx, ny, passers)
    return nx, ny, result, zone, depth, is_seam, seam_pair


def compute_distribution_summary(
    serves: list[ServeEvent],
    server_filter: str | None = None,
) -> ServeDistributionSummary:
    """Aggregate serve distribution percentages, seam breakdown, and zone counts."""
    filtered = (
        [s for s in serves if s.server_name == server_filter]
        if server_filter and server_filter != "All"
        else list(serves)
    )
    total = len(filtered)
    zone_counts: dict[str, int] = {z: 0 for z in IN_BOUNDS_ZONES}
    zone_pcts: dict[str, float] = {z: 0.0 for z in IN_BOUNDS_ZONES}
    if total == 0:
        return ServeDistributionSummary(zone_counts=zone_counts, zone_pcts=zone_pcts)

    in_only = sum(1 for s in filtered if s.result == "in")
    ace_count = sum(1 for s in filtered if s.result == "ace")
    out_count = sum(1 for s in filtered if s.result == "out")
    net_count = sum(1 for s in filtered if s.result == "net")
    in_bounds = in_only + ace_count

    seam_count = 0
    seam_by_pair: dict[str, int] = {}
    short_count = 0
    deep_count = 0

    for s in filtered:
        if s.result in ("in", "ace"):
            if s.zone in zone_counts:
                zone_counts[s.zone] += 1
            if s.depth == "short":
                short_count += 1
            elif s.depth == "deep":
                deep_count += 1
            if s.is_seam:
                seam_count += 1
                if s.seam_pair is not None:
                    key = f"{s.seam_pair[0]}-{s.seam_pair[1]}"
                    seam_by_pair[key] = seam_by_pair.get(key, 0) + 1

    if in_bounds > 0:
        for z in IN_BOUNDS_ZONES:
            zone_pcts[z] = round(100.0 * zone_counts[z] / in_bounds, 1)

    return ServeDistributionSummary(
        total=total,
        in_count=in_bounds,
        ace_count=ace_count,
        out_count=out_count,
        net_count=net_count,
        in_pct=round(100.0 * in_bounds / total, 1),
        ace_pct=round(100.0 * ace_count / total, 1),
        error_pct=round(100.0 * (out_count + net_count) / total, 1),
        seam_count=seam_count,
        seam_pct=round(100.0 * seam_count / in_bounds, 1) if in_bounds > 0 else 0.0,
        seam_by_pair=seam_by_pair,
        zone_counts=zone_counts,
        zone_pcts=zone_pcts,
        short_count=short_count,
        deep_count=deep_count,
    )
