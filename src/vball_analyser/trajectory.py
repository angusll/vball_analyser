"""Gravity-constrained RANSAC and Huber-loss volleyball set-arc trajectory fitter."""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.optimize import least_squares

from vball_analyser.calibration import height_above_tape, tape_y_at
from vball_analyser.models import BallDetection, NetCalibration, Point2D, SetTrajectory

_MIN_GRAVITY_PX_S2 = 120.0
_MAX_GRAVITY_PX_S2 = 5000.0
_INLIER_DIST_PX = 18.0
_DEFLECTION_DIST_PX = 24.0
_MAX_INTERP_GAP_FRAMES = 6


def _dedupe_per_frame(detections: list[BallDetection]) -> list[BallDetection]:
    best_by_frame: dict[int, BallDetection] = {}
    for det in detections:
        prev = best_by_frame.get(det.frame_idx)
        if prev is None or det.conf > prev.conf:
            best_by_frame[det.frame_idx] = det
    return [best_by_frame[k] for k in sorted(best_by_frame)]


def _filter_static_cart_balls(detections: list[BallDetection]) -> list[BallDetection]:
    """Drop detections that remain stationary within an 8px cluster for most frames."""
    if len(detections) < 6:
        return detections
    pts = np.array([[d.x, d.y] for d in detections], dtype=np.float64)
    keep = np.ones(len(detections), dtype=bool)
    min_static_count = max(5, int(0.45 * len({d.frame_idx for d in detections})))

    for i, pt in enumerate(pts):
        dists = np.linalg.norm(pts - pt, axis=1)
        cluster_frames = {detections[j].frame_idx for j in np.where(dists <= 8.0)[0]}
        if len(cluster_frames) >= min_static_count:
            keep[i] = False

    filtered = [d for d, k in zip(detections, keep, strict=True) if k]
    return filtered if len(filtered) >= 4 else detections


def _solve_triplet(d0: BallDetection, d1: BallDetection, d2: BallDetection) -> tuple[float, float, float, float, float, float] | None:
    """Solve linear x(tau) and quadratic y(tau) anchored at t0 = d0.timestamp_s."""
    t0 = d0.timestamp_s
    tau1 = d1.timestamp_s - t0
    tau2 = d2.timestamp_s - t0
    if tau1 <= 0.04 or tau2 - tau1 <= 0.04 or tau2 > 2.6:
        return None

    # Linear x(tau) = ax + bx * tau
    bx = (d2.x - d0.x) / tau2
    ax = d0.x

    # Quadratic y(tau) = ay + by * tau + 0.5 * g * tau^2
    ay = d0.y
    mat = np.array(
        [
            [tau1, 0.5 * tau1 * tau1],
            [tau2, 0.5 * tau2 * tau2],
        ],
        dtype=np.float64,
    )
    rhs = np.array([d1.y - ay, d2.y - ay], dtype=np.float64)
    try:
        by, g = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None

    if not (_MIN_GRAVITY_PX_S2 <= g <= _MAX_GRAVITY_PX_S2):
        return None
    if abs(bx) * tau2 < 25.0:
        return None
    return float(t0), float(ax), float(bx), float(ay), float(by), float(g)


def _eval_parabola(params: tuple[float, float, float, float, float, float], ts: float) -> tuple[float, float]:
    t0, ax, bx, ay, by, g = params
    tau = ts - t0
    return ax + bx * tau, ay + by * tau + 0.5 * g * tau * tau


def _peak_of_parabola(params: tuple[float, float, float, float, float, float], t_start: float, t_end: float) -> tuple[float, float, float]:
    t0, ax, bx, ay, by, g = params
    tau_apex = -by / g
    t_apex = max(t_start, min(t_end, t0 + tau_apex))
    px, py = _eval_parabola(params, t_apex)
    return t_apex, px, py


def fit_setting_path(
    detections: list[BallDetection],
    calibration: NetCalibration,
    fps: float,
    frame_height: int,
    candidate_contact_frames: list[tuple[int, Point2D]] | None = None,
) -> SetTrajectory | None:
    """Fit a physical gravity-constrained setting arc and truncate at hitter contact."""
    if fps <= 0:
        fps = 30.0

    # Keep all detections per frame for multi-ball RANSAC, filter static cart balls
    cleaned = _filter_static_cart_balls(detections)
    if len({d.frame_idx for d in cleaned}) < 4:
        return None

    by_frame: dict[int, list[BallDetection]] = {}
    for det in cleaned:
        by_frame.setdefault(det.frame_idx, []).append(det)
    frame_keys = sorted(by_frame.keys())
    if len(frame_keys) < 4:
        return None

    flat_sorted = sorted(cleaned, key=lambda d: (d.frame_idx, -d.conf))

    # Build deterministic sample triplets across frame span
    rng = random.Random(42)
    triplets: list[tuple[BallDetection, BallDetection, BallDetection]] = []
    n = len(flat_sorted)
    if n <= 35:
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                if flat_sorted[j].frame_idx <= flat_sorted[i].frame_idx:
                    continue
                for k in range(j + 1, n):
                    if flat_sorted[k].frame_idx <= flat_sorted[j].frame_idx:
                        continue
                    if 0.25 <= flat_sorted[k].timestamp_s - flat_sorted[i].timestamp_s <= 2.5:
                        triplets.append((flat_sorted[i], flat_sorted[j], flat_sorted[k]))
    if len(triplets) > 450:
        triplets = rng.sample(triplets, 450)
    elif not triplets:
        for _ in range(min(300, n * 10)):
            sample = sorted(rng.sample(flat_sorted, 3), key=lambda d: d.frame_idx)
            if sample[0].frame_idx < sample[1].frame_idx < sample[2].frame_idx:
                triplets.append((sample[0], sample[1], sample[2]))

    best_score = -1.0
    best_inliers: list[BallDetection] = []
    best_params: tuple[float, float, float, float, float, float] | None = None

    for d0, d1, d2 in triplets:
        params = _solve_triplet(d0, d1, d2)
        if params is None:
            continue

        # Check peak occurs above net region (y smaller than tape + 50px)
        t_apex, peak_x, peak_y = _peak_of_parabola(params, d0.timestamp_s, d2.timestamp_s + 0.6)
        if peak_y >= min(frame_height * 0.78, tape_y_at(calibration, peak_x) + 50.0):
            continue

        # Collect inliers chronologically, stopping when trajectory sharply deflects post-peak
        inliers: list[BallDetection] = []
        diverged_streak = 0
        past_peak = False
        for f_idx in frame_keys:
            ts = f_idx / fps
            if ts < d0.timestamp_s - 0.45:
                continue
            if ts > d0.timestamp_s + 2.5:
                break
            px, py = _eval_parabola(params, ts)
            best_d = min(by_frame[f_idx], key=lambda d: math.hypot(d.x - px, d.y - py))
            err = math.hypot(best_d.x - px, best_d.y - py)
            if ts >= t_apex:
                past_peak = True
            if err <= _INLIER_DIST_PX:
                inliers.append(best_d)
                diverged_streak = 0
            elif past_peak and len(inliers) >= 4 and err > _DEFLECTION_DIST_PX:
                diverged_streak += 1
                if diverged_streak >= 2:
                    break

        if len(inliers) < 4:
            continue

        span_s = inliers[-1].timestamp_s - inliers[0].timestamp_s
        if span_s < 0.25:
            continue
        score = len(inliers) * 10.0 + span_s * 3.0
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_params = params

    if best_params is None or len(best_inliers) < 4:
        return None

    # Huber-loss nonlinear least-squares refinement on RANSAC inliers
    t0 = best_inliers[0].timestamp_s
    taus = np.array([d.timestamp_s - t0 for d in best_inliers], dtype=np.float64)
    xs_obs = np.array([d.x for d in best_inliers], dtype=np.float64)
    ys_obs = np.array([d.y for d in best_inliers], dtype=np.float64)
    _, ax0, bx0, ay0, by0, g0 = best_params
    x_init = np.array([ax0, bx0, ay0 + by0 * (t0 - best_params[0]) + 0.5 * g0 * (t0 - best_params[0]) ** 2, by0 + g0 * (t0 - best_params[0]), g0], dtype=np.float64)
    # Clamp initial guess inside bounds
    x_init[4] = float(np.clip(x_init[4], _MIN_GRAVITY_PX_S2 + 1.0, _MAX_GRAVITY_PX_S2 - 1.0))

    def residuals(p: np.ndarray) -> np.ndarray:
        ax, bx, ay, by, g = p
        rx = (ax + bx * taus) - xs_obs
        ry = (ay + by * taus + 0.5 * g * taus * taus) - ys_obs
        return np.concatenate([rx, ry])

    res = least_squares(
        residuals,
        x_init,
        bounds=(
            [-np.inf, -np.inf, -np.inf, -np.inf, _MIN_GRAVITY_PX_S2],
            [np.inf, np.inf, np.inf, np.inf, _MAX_GRAVITY_PX_S2],
        ),
        loss="huber",
        f_scale=8.0,
    )
    ax, bx, ay, by, g = [float(v) for v in res.x]
    refined = (t0, ax, bx, ay, by, g)

    # Re-collect final inliers around refined parabola up to contact deflection
    final_inliers: list[BallDetection] = []
    final_errs: list[float] = []
    t_apex_ref, _, _ = _peak_of_parabola(refined, t0, best_inliers[-1].timestamp_s)
    diverged_streak = 0
    deflected = False
    for f_idx in frame_keys:
        if f_idx < best_inliers[0].frame_idx - 5:
            continue
        ts = f_idx / fps
        px, py = _eval_parabola(refined, ts)
        best_d = min(by_frame[f_idx], key=lambda d: math.hypot(d.x - px, d.y - py))
        err = math.hypot(best_d.x - px, best_d.y - py)
        if err <= _INLIER_DIST_PX:
            final_inliers.append(best_d)
            final_errs.append(err)
            diverged_streak = 0
        elif ts >= t_apex_ref and len(final_inliers) >= 4 and err > _DEFLECTION_DIST_PX:
            diverged_streak += 1
            if diverged_streak >= 2:
                deflected = True
                break

    if deflected:
        # Trim trailing frames where spike deflection already began accelerating away from the arc
        while (
            len(final_inliers) > 4
            and final_inliers[-1].timestamp_s >= t_apex_ref
            and (final_errs[-1] > 5.5 or (final_errs[-1] > 3.5 and final_errs[-1] > final_errs[-2] * 1.4))
        ):
            final_inliers.pop()
            final_errs.pop()

    if len(final_inliers) < 4:
        final_inliers = best_inliers

    release_frame = final_inliers[0].frame_idx
    contact_frame = final_inliers[-1].frame_idx

    # Optional snap to candidate hitter wrist contact frame within [-3, +1] frames & 55px
    if candidate_contact_frames:
        best_snap: int | None = None
        best_snap_dist = 55.0
        for cand_frame, cand_pt in candidate_contact_frames:
            if -3 <= (cand_frame - contact_frame) <= 1 and cand_frame > release_frame + 3:
                px, py = _eval_parabola(refined, cand_frame / fps)
                dist = math.hypot(cand_pt.x - px, cand_pt.y - py)
                if dist < best_snap_dist:
                    best_snap_dist = dist
                    best_snap = cand_frame
        if best_snap is not None:
            contact_frame = best_snap

    release_time_s = round(release_frame / fps, 4)
    contact_time_s = round(contact_frame / fps, 4)
    t_peak, peak_x, peak_y = _peak_of_parabola(refined, release_time_s, contact_time_s)
    peak_frame = max(release_frame, min(contact_frame, int(round(t_peak * fps))))
    peak_height_px, peak_height_cm = height_above_tape(calibration, peak_x, peak_y)

    # Build smooth frame-by-frame arc points across [release_frame, contact_frame]
    inlier_map = {d.frame_idx: d for d in final_inliers}
    arc_points: list[BallDetection] = []
    for f_idx in range(release_frame, contact_frame + 1):
        if f_idx in inlier_map:
            arc_points.append(inlier_map[f_idx])
        else:
            ts = f_idx / fps
            px, py = _eval_parabola(refined, ts)
            arc_points.append(
                BallDetection(
                    frame_idx=f_idx,
                    timestamp_s=round(ts, 4),
                    x=round(px, 2),
                    y=round(py, 2),
                    conf=0.5,
                    is_interpolated=True,
                )
            )

    return SetTrajectory(
        release_frame=release_frame,
        release_time_s=release_time_s,
        contact_frame=contact_frame,
        contact_time_s=contact_time_s,
        peak_frame=peak_frame,
        peak_point=Point2D(x=round(peak_x, 2), y=round(peak_y, 2)),
        tempo_s=round(max(0.0, contact_time_s - release_time_s), 3),
        peak_height_px=peak_height_px,
        peak_height_cm=peak_height_cm,
        arc_points=arc_points,
        inlier_count=len(final_inliers),
    )
