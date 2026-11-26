"""Net top-tape calibration, court-zone classification, and height geometry."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vball_analyser.models import NetCalibration, Point2D


def tape_y_at(cal: NetCalibration, x: float) -> float:
    """Linearly interpolate the net top-tape pixel Y coordinate at horizontal position x."""
    dx = cal.right_point.x - cal.left_point.x
    if abs(dx) < 1e-6:
        return float(cal.left_point.y)
    t = (x - cal.left_point.x) / dx
    return float(cal.left_point.y + t * (cal.right_point.y - cal.left_point.y))


def net_span_px(cal: NetCalibration) -> float:
    """Pixel distance between left and right antenna top-tape endpoints."""
    return math.hypot(
        cal.right_point.x - cal.left_point.x,
        cal.right_point.y - cal.left_point.y,
    )


def cm_per_px(cal: NetCalibration) -> float | None:
    """Return cm/px scale from 9.0m regulation net width when manually calibrated."""
    if not cal.is_manual:
        return None
    span = net_span_px(cal)
    if span <= 1e-3:
        return None
    return 900.0 / span


def estimated_cm_per_px(cal: NetCalibration) -> float:
    """Return approximate cm/px scale even before manual antenna confirmation."""
    span = net_span_px(cal)
    if span <= 1e-3:
        return 1.0
    return 900.0 / span


def height_above_tape(cal: NetCalibration, x: float, y: float) -> tuple[float, float | None]:
    """Return (px_above_tape, cm_above_tape). Positive values are above the net tape."""
    px_above = tape_y_at(cal, x) - y
    scale = cm_per_px(cal)
    cm_above = round(px_above * scale, 1) if scale is not None else None
    return round(px_above, 1), cm_above


def net_ground_y_at(cal: NetCalibration, x: float) -> float:
    """Estimate the pixel Y coordinate of the court floor directly beneath the net at x."""
    span = net_span_px(cal)
    return float(tape_y_at(cal, x) + (cal.net_height_m / 9.0) * span)


def classify_attack_zone(cal: NetCalibration, x: float) -> str:
    """Classify horizontal attack position along the net into Zone 4 / 3 / 2."""
    dx = cal.right_point.x - cal.left_point.x
    u = (x - cal.left_point.x) / dx if abs(dx) > 1e-6 else 0.5
    if u < 0.33:
        return "Zone 4 (Left)"
    if u <= 0.67:
        return "Zone 3 (Middle)"
    return "Zone 2 (Right)"


def _fallback_calibration(width: int, height: int) -> NetCalibration:
    return NetCalibration(
        left_point=Point2D(x=round(0.15 * width, 1), y=round(0.45 * height, 1)),
        right_point=Point2D(x=round(0.85 * width, 1), y=round(0.45 * height, 1)),
        is_manual=False,
    )


def auto_detect_net_tape(
    frames_bgr: list[np.ndarray],
    person_boxes: list[tuple[float, float, float, float]] | None = None,
) -> NetCalibration:
    """Auto-estimate the net top tape across frames using median background + Hough lines."""
    if not frames_bgr:
        return _fallback_calibration(1280, 720)

    h, w = frames_bgr[0].shape[:2]
    sample = frames_bgr[: min(15, len(frames_bgr))]
    stack = np.stack(sample, axis=0)
    median_bgr = np.median(stack, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)

    y_min = max(10, int(0.20 * h))
    y_max = min(h - 10, int(0.68 * h))
    roi = gray[y_min:y_max, :]
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    mesh_offsets = np.arange(12, 48, 6)

    # Enhance horizontal bright tape contrast against darker net/background
    blur_v = cv2.GaussianBlur(roi, (1, 21), 0)
    ridge = cv2.subtract(roi, blur_v)
    edges = cv2.Canny(ridge, 30, 90)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(40, int(0.035 * w)),
        minLineLength=max(80, int(0.14 * w)),
        maxLineGap=max(24, int(0.08 * w)),
    )
    if lines is None or len(lines) == 0:
        return _fallback_calibration(w, h)

    candidates: list[tuple[float, float, float, float]] = []
    for seg in lines.reshape(-1, 4):
        x1, y1, x2, y2 = float(seg[0]), float(seg[1] + y_min), float(seg[2]), float(seg[3] + y_min)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy = y2 - y1
        if dx < 20.0:
            continue
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        if angle_deg <= 10.0:
            candidates.append((x1, y1, x2, y2))

    if not candidates:
        return _fallback_calibration(w, h)

    best_score = -1.0
    best_line: tuple[float, float] | None = None
    best_span: tuple[float, float] = (0.15 * w, 0.85 * w)

    for cx1, cy1, cx2, cy2 in candidates:
        slope = (cy2 - cy1) / (cx2 - cx1)
        intercept = cy1 - slope * cx1

        inlier_pts_x: list[float] = []
        inlier_pts_y: list[float] = []
        total_len = 0.0
        for x1, y1, x2, y2 in candidates:
            ym1 = slope * x1 + intercept
            ym2 = slope * x2 + intercept
            if abs(y1 - ym1) <= 10.0 and abs(y2 - ym2) <= 10.0:
                inlier_pts_x.extend([x1, x2])
                inlier_pts_y.extend([y1, y2])
                total_len += math.hypot(x2 - x1, y2 - y1)

        if len(inlier_pts_x) < 2:
            continue

        arr_x = np.asarray(inlier_pts_x, dtype=np.float64)
        arr_y = np.asarray(inlier_pts_y, dtype=np.float64)
        m, b = np.polyfit(arr_x, arr_y, 1)

        min_x = float(np.min(arr_x))
        max_x = float(np.max(arr_x))
        span = max_x - min_x
        if span < 0.25 * w:
            continue

        # Measure bright horizontal ridge support and center the line on the bright tape
        xs = np.linspace(min_x, max_x, 80)
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        y_est = m * xs + b
        offsets = np.arange(-6, 7)
        best_shift = 0.0
        best_band_mean = -1.0
        for off in offsets:
            ys_off = np.clip(np.round(y_est + off).astype(int), 4, h - 5)
            band_mean = float(np.mean(gray[ys_off, xi]))
            if band_mean > best_band_mean:
                best_band_mean = band_mean
                best_shift = float(off)
        b_centered = float(b + best_shift)

        ys = np.clip(np.round(m * xs + b_centered).astype(int), 6, h - 7)
        tape_vals = gray[ys, xi].astype(np.float32)
        above_vals = gray[np.clip(ys - 8, 0, h - 1), xi].astype(np.float32)
        below_vals = gray[np.clip(ys + 8, 0, h - 1), xi].astype(np.float32)
        contrast = float(np.mean(tape_vals - 0.5 * (above_vals + below_vals)))
        brightness = float(np.mean(tape_vals))

        # Net top tape sits directly above high-frequency vertical net mesh, whereas
        # the bottom tape has net mesh above it and open court below.
        ys_below = np.clip(ys[:, None] + mesh_offsets[None, :], 0, h - 1)
        ys_above = np.clip(ys[:, None] - mesh_offsets[None, :], 0, h - 1)
        tex_below = float(np.mean(sobel_x[ys_below, xi[:, None]]))
        tex_above = float(np.mean(sobel_x[ys_above, xi[:, None]]))
        mesh_asym = max(-35.0, min(45.0, tex_below - tex_above))

        center_y = float(m * (0.5 * w) + b_centered)
        center_prior = 1.0 - min(1.0, abs(center_y - 0.42 * h) / (0.30 * h))

        person_bonus = 0.0
        if person_boxes:
            for bx1, by1, bx2, by2 in person_boxes:
                bx_mid = 0.5 * (bx1 + bx2)
                ly_at_b = m * bx_mid + b_centered
                if by1 - 20 <= ly_at_b <= by2:
                    person_bonus += 15.0

        score = (
            (span / w) * 100.0
            + max(0.0, contrast) * 1.4
            + brightness * 0.25
            + center_prior * 20.0
            + mesh_asym * 1.5
            + person_bonus
            + min(40.0, total_len / w * 20.0)
        )
        if score > best_score and contrast > 8.0:
            best_score = score
            best_line = (float(m), b_centered)
            best_span = (min_x, max_x)

    if best_line is None:
        return _fallback_calibration(w, h)

    m, b = best_line
    lx, rx = best_span
    return NetCalibration(
        left_point=Point2D(x=round(lx, 1), y=round(m * lx + b, 1)),
        right_point=Point2D(x=round(rx, 1), y=round(m * rx + b, 1)),
        is_manual=False,
    )
