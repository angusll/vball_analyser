"""Gravity-constrained RANSAC and Huber-loss volleyball set-arc trajectory fitter."""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.optimize import least_squares

from vball_analyser.calibration import height_above_tape, net_span_px, tape_y_at
from vball_analyser.models import BallDetection, NetCalibration, Point2D, SetTrajectory

_MIN_GRAVITY_PX_S2 = 120.0
_MAX_GRAVITY_PX_S2 = 5000.0
_INLIER_DIST_PX = 18.0
_DEFLECTION_DIST_PX = 24.0
_MAX_INTERP_GAP_FRAMES = 6
_MIN_BALL_CONF = 0.12
_MIN_SET_HORIZ_SPAN_PX = 30.0


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


def _solve_triplet(
    d0: BallDetection,
    d1: BallDetection,
    d2: BallDetection,
    min_g: float = _MIN_GRAVITY_PX_S2,
    max_g: float = _MAX_GRAVITY_PX_S2,
) -> tuple[float, float, float, float, float, float] | None:
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

    if not (min_g <= g <= max_g):
        return None
    if abs(bx) * tau2 < 25.0:
        return None
    return float(t0), float(ax), float(bx), float(ay), float(by), float(g)


def _count_distinct_positions(inliers: list[BallDetection], min_move_px: float = 2.0) -> int:
    """Count temporally ordered inliers that represent distinct ball positions (not VFR duplicates)."""
    if not inliers:
        return 0
    count = 1
    last_x, last_y = inliers[0].x, inliers[0].y
    for det in inliers[1:]:
        if math.hypot(det.x - last_x, det.y - last_y) >= min_move_px:
            count += 1
            last_x, last_y = det.x, det.y
    return count


def _trim_isolated_endpoints(inliers: list[BallDetection], max_gap_s: float = 0.18) -> list[BallDetection]:
    """Drop lone single-point islands separated from the main arc by a large time gap."""
    trimmed = list(inliers)
    while len(trimmed) >= 4 and (trimmed[1].timestamp_s - trimmed[0].timestamp_s) > max_gap_s:
        trimmed.pop(0)
    while len(trimmed) >= 4 and (trimmed[-1].timestamp_s - trimmed[-2].timestamp_s) > max_gap_s:
        trimmed.pop()
    return trimmed


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

    # Filter motion-damped static false positives (conf < 0.12) and static cart balls
    conf_filtered = [d for d in detections if d.conf >= _MIN_BALL_CONF]
    active_dets = conf_filtered if len({d.frame_idx for d in conf_filtered}) >= 4 else detections
    cleaned = _filter_static_cart_balls(active_dets)
    if len({d.frame_idx for d in cleaned}) < 4:
        return None

    span = net_span_px(calibration)
    if span >= 150.0:
        g_ref = 9.81 * (span / 9.0)
        min_g = max(_MIN_GRAVITY_PX_S2, 0.30 * g_ref)
        max_g = min(_MAX_GRAVITY_PX_S2, 2.8 * g_ref)
    else:
        min_g, max_g = _MIN_GRAVITY_PX_S2, _MAX_GRAVITY_PX_S2

    by_frame: dict[int, list[BallDetection]] = {}
    for det in cleaned:
        by_frame.setdefault(det.frame_idx, []).append(det)
    frame_keys = sorted(by_frame.keys())
    if len(frame_keys) < 4:
        return None

    flat_sorted = sorted(cleaned, key=lambda d: (d.frame_idx, -d.conf))

    # Build temporally local triplets within each 2.4s window so multi-arc rallies
    # (serve + pass + set) always sample the set arc reliably.
    rng = random.Random(42)
    triplets: list[tuple[BallDetection, BallDetection, BallDetection]] = []
    n = len(flat_sorted)
    for i in range(n - 2):
        d0 = flat_sorted[i]
        window = [
            flat_sorted[m]
            for m in range(i + 1, n)
            if flat_sorted[m].frame_idx > d0.frame_idx
            and 0.05 <= (flat_sorted[m].timestamp_s - d0.timestamp_s) <= 2.4
        ]
        if len(window) < 2:
            continue
        # Sample up to 12 local (d1, d2) pairs per anchor d0
        pairs: list[tuple[BallDetection, BallDetection]] = []
        w_len = len(window)
        for a in range(w_len - 1):
            for b in range(a + 1, w_len):
                if (
                    window[b].frame_idx > window[a].frame_idx
                    and (window[b].timestamp_s - d0.timestamp_s) >= 0.25
                ):
                    pairs.append((window[a], window[b]))
        if len(pairs) > 12:
            pairs = rng.sample(pairs, 12)
        for d1, d2 in pairs:
            triplets.append((d0, d1, d2))

    if len(triplets) > 900:
        triplets = rng.sample(triplets, 900)

    best_score = -1.0
    best_inliers: list[BallDetection] = []
    best_params: tuple[float, float, float, float, float, float] | None = None

    for d0, d1, d2 in triplets:
        params = _solve_triplet(d0, d1, d2, min_g=min_g, max_g=max_g)
        if params is None:
            continue

        # Check peak occurs above net top tape (allowing minor 12px tape-estimate slack)
        t_apex, peak_x, peak_y = _peak_of_parabola(params, d0.timestamp_s, d2.timestamp_s + 0.6)
        if peak_y >= min(frame_height * 0.78, tape_y_at(calibration, peak_x) + 12.0):
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
            elif (
                past_peak
                and len(inliers) >= 4
                and _DEFLECTION_DIST_PX < err < 140.0
                and math.hypot(best_d.x - inliers[-1].x, best_d.y - inliers[-1].y) >= 3.0
            ):
                diverged_streak += 1
                if diverged_streak >= 2:
                    break

        inliers = _trim_isolated_endpoints(inliers)
        if len(inliers) < 4:
            continue

        span_s = inliers[-1].timestamp_s - inliers[0].timestamp_s
        if span_s < 0.25:
            continue
        horiz_span = abs(inliers[-1].x - inliers[0].x)
        if horiz_span < _MIN_SET_HORIZ_SPAN_PX:
            continue
        distinct_count = _count_distinct_positions(inliers)
        if distinct_count < 4:
            continue

        # Strongly prefer arcs whose descending branch ends at a jumping attacker's hand
        contact_bonus = 0.0
        if candidate_contact_frames:
            last_f = inliers[-1].frame_idx
            peak_f_est = int(round(t_apex * fps))
            max_extrap_f = last_f + int(round(0.35 * fps))
            best_cand_score = 0.0
            for cand_f, cand_pt in candidate_contact_frames:
                if max(peak_f_est, last_f - 3) <= cand_f <= max_extrap_f:
                    px, py = _eval_parabola(params, cand_f / fps)
                    dist = math.hypot(cand_pt.x - px, cand_pt.y - py)
                    if dist < 95.0:
                        above_tape = max(0.0, tape_y_at(calibration, cand_pt.x) - cand_pt.y)
                        c_score = 320.0 - 1.5 * dist + 1.5 * above_tape
                        if c_score > best_cand_score:
                            best_cand_score = c_score
            contact_bonus = best_cand_score

        score = distinct_count * 10.0 + span_s * 3.0 + contact_bonus
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
    dt0 = t0 - best_params[0]
    x_init = np.array(
        [
            ax0 + bx0 * dt0,
            bx0,
            ay0 + by0 * dt0 + 0.5 * g0 * dt0 * dt0,
            by0 + g0 * dt0,
            g0,
        ],
        dtype=np.float64,
    )
    # Clamp initial guess inside bounds
    x_init[4] = float(np.clip(x_init[4], min_g + 1.0, max_g - 1.0))

    def residuals(p: np.ndarray) -> np.ndarray:
        ax, bx, ay, by, g = p
        rx = (ax + bx * taus) - xs_obs
        ry = (ay + by * taus + 0.5 * g * taus * taus) - ys_obs
        return np.concatenate([rx, ry])

    res = least_squares(
        residuals,
        x_init,
        bounds=(
            [-np.inf, -np.inf, -np.inf, -np.inf, min_g],
            [np.inf, np.inf, np.inf, np.inf, max_g],
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
        elif (
            ts >= t_apex_ref
            and len(final_inliers) >= 4
            and _DEFLECTION_DIST_PX < err < 140.0
            and math.hypot(best_d.x - final_inliers[-1].x, best_d.y - final_inliers[-1].y) >= 3.0
        ):
            diverged_streak += 1
            if diverged_streak >= 2:
                deflected = True
                break

    if deflected:
        # Trim at most 2 trailing frames if they sharply diverge from the arc
        # (e.g. first post-contact spike frame barely inside _INLIER_DIST_PX),
        # without chewing back into normal VFR jitter or a frame matching a hitter contact.
        trimmed_count = 0
        while (
            trimmed_count < 2
            and len(final_inliers) > 4
            and final_inliers[-1].timestamp_s >= t_apex_ref
            and final_errs[-1] > 12.0
            and final_errs[-1] > final_errs[-2] * 1.8
        ):
            last_det = final_inliers[-1]
            if candidate_contact_frames and any(
                abs(cf - last_det.frame_idx) <= 1 and math.hypot(cp.x - last_det.x, cp.y - last_det.y) < 55.0
                for cf, cp in candidate_contact_frames
            ):
                break
            final_inliers.pop()
            final_errs.pop()
            trimmed_count += 1

    if len(final_inliers) < 4:
        final_inliers = best_inliers

    release_frame = final_inliers[0].frame_idx
    contact_frame = final_inliers[-1].frame_idx

    # Optional snap to candidate hitter wrist contact frame (extrapolating brief pre-contact dropout if needed)
    if candidate_contact_frames:
        best_snap: int | None = None
        best_snap_dist = 85.0
        peak_f_ref = int(round(t_apex_ref * fps))
        max_forward = max(3, int(round(0.12 * fps))) if deflected else max(2, int(round(0.32 * fps)))
        for cand_frame, cand_pt in candidate_contact_frames:
            if max(release_frame + 4, peak_f_ref, contact_frame - 3) <= cand_frame <= contact_frame + max_forward:
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

    # Build smooth frame-by-frame arc points evaluated on the refined physical parabola
    inlier_map = {d.frame_idx: d for d in final_inliers}
    arc_points: list[BallDetection] = []
    for f_idx in range(release_frame, contact_frame + 1):
        ts = f_idx / fps
        px, py = _eval_parabola(refined, ts)
        obs = inlier_map.get(f_idx)
        arc_points.append(
            BallDetection(
                frame_idx=f_idx,
                timestamp_s=round(ts, 4),
                x=round(px, 2),
                y=round(py, 2),
                conf=obs.conf if obs is not None else 0.5,
                is_interpolated=obs is None,
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
