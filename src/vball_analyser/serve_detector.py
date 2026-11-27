"""Automated serve detection and passer floor-projection pipeline using YOLO11 + homography."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Literal

import numpy as np

from vball_analyser.config import DEFAULT_CONFIG
from vball_analyser.detector import FramePredictor
from vball_analyser.models import (
    BallDetection,
    CourtCalibration,
    NetCalibration,
    PasserPosition,
    PlayerTrack,
    Point2D,
    PoseFrame,
    ServeEvent,
    VideoMetadata,
)
from vball_analyser.serve_geometry import (
    classify_serve,
    compute_court_homography,
    normalize_court_calibration,
    project_image_to_court,
    score_receive,
)
from vball_analyser.video_io import iter_video_frames

REFRACTORY_GAP_S = 2.5
POST_RECEIVE_COOLDOWN_S = 6.0
RALLY_BREAK_MIN_GAP_S = 2.2
MIN_DESCENT_PX = 45.0
MIN_DESCENT_DURATION_S = 0.20
PRE_SERVE_OFFSET_S = 0.60
POST_PASS_MAX_DURATION_S = 2.6
MIN_PASS_FLIGHT_S = 0.25
PASSER_MATCH_MAX_DIST_M = 2.5


def extract_foot_point(frame: PoseFrame) -> tuple[float, float]:
    """Return floor foot pixel coordinate from ankle keypoints (COCO 15/16) or bbox bottom-center."""
    kps = frame.keypoints
    if len(kps) > 16:
        la = kps[15]
        ra = kps[16]
        if la[2] >= 0.35 and ra[2] >= 0.35:
            return (0.5 * (la[0] + ra[0]), 0.5 * (la[1] + ra[1]))
        if la[2] >= 0.35:
            return (float(la[0]), float(la[1]))
        if ra[2] >= 0.35:
            return (float(ra[0]), float(ra[1]))
    x1, _y1, x2, y2 = frame.bbox
    return (0.5 * (x1 + x2), float(y2))


def _proxy_net_calibration(cal: CourtCalibration) -> NetCalibration:
    """Return net top-tape line from calibrated antenna points or estimated from centerline corners."""
    if cal.net_left_top is not None and cal.net_right_top is not None:
        return NetCalibration(
            left_point=cal.net_left_top,
            right_point=cal.net_right_top,
            net_height_m=cal.net_height_m,
            is_manual=True,
        )
    span = math.hypot(
        cal.front_right.x - cal.front_left.x,
        cal.front_right.y - cal.front_left.y,
    )
    lift_px = (cal.net_height_m / DEFAULT_CONFIG.net_width_m) * span
    return NetCalibration(
        left_point=Point2D(x=cal.front_left.x, y=max(0.0, cal.front_left.y - lift_px)),
        right_point=Point2D(x=cal.front_right.x, y=max(0.0, cal.front_right.y - lift_px)),
        net_height_m=cal.net_height_m,
        is_manual=False,
    )


def _filter_stationary_background(detections: list[BallDetection]) -> list[BallDetection]:
    """Drop repeated static background false positives (e.g. gym lights/clocks stuck within <=8px)."""
    if len(detections) < 10:
        return detections

    clusters: list[list[BallDetection]] = []
    anchors: list[tuple[float, float]] = []
    for d in detections:
        if d.conf < 0.16:
            continue
        placed = False
        for idx, (ax, ay) in enumerate(anchors):
            if math.hypot(d.x - ax, d.y - ay) <= 8.0:
                clusters[idx].append(d)
                placed = True
                break
        if not placed:
            anchors.append((d.x, d.y))
            clusters.append([d])

    static_anchors: list[tuple[float, float]] = []
    for anchor, members in zip(anchors, clusters):
        if len(members) < 10:
            continue
        times = [m.timestamp_s for m in members]
        if (max(times) - min(times)) >= 0.8:
            static_anchors.append(anchor)

    if not static_anchors:
        return detections

    return [
        d
        for d in detections
        if not any(math.hypot(d.x - ax, d.y - ay) <= 8.0 for ax, ay in static_anchors)
    ]


def _select_best_ball_series(detections: list[BallDetection]) -> list[BallDetection]:
    """Pick highest-confidence ball per frame, dropping static background FPs and isolated teleport outliers."""
    filtered = _filter_stationary_background(detections)
    by_frame: dict[int, BallDetection] = {}
    for d in filtered:
        if d.conf < 0.16:
            continue
        prev = by_frame.get(d.frame_idx)
        if prev is None or d.conf > prev.conf:
            by_frame[d.frame_idx] = d

    ordered = [by_frame[k] for k in sorted(by_frame)]
    if len(ordered) < 3:
        return ordered

    cleaned: list[BallDetection] = []
    for i, cur in enumerate(ordered):
        if not cleaned:
            cleaned.append(cur)
            continue
        prev = cleaned[-1]
        dt = max(1e-3, cur.timestamp_s - prev.timestamp_s)
        dist = math.hypot(cur.x - prev.x, cur.y - prev.y)
        # Allow fast serves (~180 px per 1/30s frame), reject wild single-frame noise jumps
        if dist <= max(180.0, 2200.0 * dt):
            cleaned.append(cur)
        elif i + 1 < len(ordered):
            nxt = ordered[i + 1]
            if math.hypot(cur.x - nxt.x, cur.y - nxt.y) <= 140.0:
                cleaned.append(cur)
    return cleaned


def _find_serve_landing_balls(
    series: list[BallDetection],
    net_center_y: float,
    net_tape_y: float | None = None,
    *,
    max_gap_s: float = 0.35,
    min_points: int = 5,
) -> list[BallDetection]:
    """Find serve reception/landing ball frames from descending high-to-low trajectories."""
    if len(series) < min_points:
        return []

    tape_y = net_tape_y if net_tape_y is not None else max(0.0, net_center_y - 160.0)
    landings: list[BallDetection] = []
    last_accepted_t = -1e9
    n = len(series)
    i = 1

    while i < n - 1:
        # Look for a descending run starting above/near the net centerline
        init_dt = max(1e-3, series[i].timestamp_s - series[i - 1].timestamp_s)
        init_dist = math.hypot(series[i].x - series[i - 1].x, series[i].y - series[i - 1].y)
        if series[i].y <= series[i - 1].y or init_dt > max_gap_s or init_dist > max(140.0, 1800.0 * init_dt):
            i += 1
            continue

        run_start = i - 1
        j = i
        # Follow descent while enforcing spatial continuity and allowing brief 1-frame plateaus/jitter
        while j < n:
            dt = max(1e-3, series[j].timestamp_s - series[j - 1].timestamp_s)
            step_dist = math.hypot(series[j].x - series[j - 1].x, series[j].y - series[j - 1].y)
            if (
                series[j].y >= series[j - 1].y - 3.0
                and dt <= max_gap_s
                and step_dist <= max(140.0, 1800.0 * dt)
            ):
                j += 1
            else:
                break
        run_end = j - 1

        start_b = series[run_start]
        end_b = series[run_end]
        descent_px = end_b.y - start_b.y
        descent_dt = end_b.timestamp_s - start_b.timestamp_s

        # Serve starts high (above/near net) and descends below the net top tape to either
        # a chest/overhead pass rebound or a floor bounce.
        started_high = start_b.y <= net_center_y + 25.0
        has_rebound = False
        if run_end + 1 < n:
            nxt_b = series[run_end + 1]
            reb_dt = max(1e-3, nxt_b.timestamp_s - end_b.timestamp_s)
            reb_dist = math.hypot(nxt_b.x - end_b.x, nxt_b.y - end_b.y)
            has_rebound = (
                reb_dt <= max_gap_s
                and reb_dist <= max(140.0, 1800.0 * reb_dt)
                and nxt_b.y < end_b.y - 2.0
            )
        # Chest/overhead serve receptions in the backcourt can project near or slightly above
        # the net top tape in baseline perspective; receiver bbox matching filters non-receptions.
        below_tape = end_b.y >= tape_y - 25.0
        at_floor = end_b.y >= net_center_y - 15.0

        if (
            started_high
            and descent_px >= MIN_DESCENT_PX
            and descent_dt >= MIN_DESCENT_DURATION_S
            and ((has_rebound and below_tape) or at_floor)
            and (end_b.timestamp_s - last_accepted_t) >= REFRACTORY_GAP_S
        ):
            landings.append(end_b)
            last_accepted_t = end_b.timestamp_s

        i = max(run_end + 1, i + 1)

    return landings


def _nearest_pose_frame(track: PlayerTrack, target_t: float, max_delta_s: float = 0.25) -> PoseFrame | None:
    best: PoseFrame | None = None
    best_dt = max_delta_s + 1e-6
    for fr in track.frames:
        dt = abs(fr.timestamp_s - target_t)
        if dt < best_dt:
            best_dt = dt
            best = fr
    return best


def _snapshot_passers(
    tracks: list[PlayerTrack],
    H: np.ndarray,
    target_t: float,
    net_center_y: float,
) -> list[PasserPosition]:
    """Project near-side receiving passers onto top-down court meters at target_t."""
    candidates: list[tuple[float, float]] = []
    for tr in tracks:
        if tr.court_side != "attacking":
            continue
        fr = _nearest_pose_frame(tr, target_t, max_delta_s=0.30)
        if fr is None:
            continue
        fx, fy = extract_foot_point(fr)
        if fy <= net_center_y + 10.0:
            continue
        try:
            cx, cy = project_image_to_court(H, fx, fy)
        except ValueError:
            continue
        if -0.5 <= cx <= 9.5 and 1.2 <= cy <= 9.8:
            candidates.append((cx, cy))

    if not candidates:
        return []

    # Exclude net-tucked setter/front-row players (y < 2.8m) when 2+ deeper passers exist
    deeper = [c for c in candidates if c[1] >= 2.8]
    chosen = deeper if len(deeper) >= 2 else candidates
    chosen.sort(key=lambda c: c[0])
    return [
        PasserPosition(label=f"P{idx}", x_m=round(cx, 3), y_m=round(cy, 3))
        for idx, (cx, cy) in enumerate(chosen, start=1)
    ]


def _match_receiving_passer(
    tracks: list[PlayerTrack],
    H: np.ndarray,
    ball: BallDetection,
    net_center_y: float,
) -> tuple[int, PoseFrame, float, float] | None:
    """Find near-side player contacting the serve at ball.timestamp_s, returning (track_id, frame, foot_x_m, foot_y_m)."""
    best_match: tuple[float, int, PoseFrame, float, float] | None = None
    for tr in tracks:
        if tr.court_side != "attacking":
            continue
        fr = _nearest_pose_frame(tr, ball.timestamp_s, max_delta_s=0.18)
        if fr is None:
            continue
        fx, fy = extract_foot_point(fr)
        if fy <= net_center_y + 10.0:
            continue
        try:
            cx, cy = project_image_to_court(H, fx, fy)
        except ValueError:
            continue
        if not (-1.0 <= cx <= 10.0 and 0.5 <= cy <= 10.5):
            continue

        x1, y1, x2, y2 = fr.bbox
        pad = 45.0
        dx = max(x1 - pad - ball.x, 0.0, ball.x - (x2 + pad))
        dy = max(y1 - pad - ball.y, 0.0, ball.y - (y2 + pad))
        box_dist = math.hypot(dx, dy)
        center_dist = math.hypot(ball.x - 0.5 * (x1 + x2), ball.y - 0.5 * (y1 + y2))
        if box_dist <= 55.0 or center_dist <= 110.0:
            if best_match is None or center_dist < best_match[0]:
                best_match = (center_dist, tr.track_id, fr, cx, cy)

    if best_match is None:
        return None
    return (best_match[1], best_match[2], best_match[3], best_match[4])


def _match_passer_label(
    fx_m: float, fy_m: float, passers: list[PasserPosition]
) -> str | None:
    """Return the label of the nearest passer within PASSER_MATCH_MAX_DIST_M, else None."""
    best_label: str | None = None
    best_dist = PASSER_MATCH_MAX_DIST_M + 1e-9
    for p in passers:
        d = math.hypot(fx_m - p.x_m, fy_m - p.y_m)
        if d <= PASSER_MATCH_MAX_DIST_M and d < best_dist:
            best_dist = d
            best_label = p.label
    return best_label


def _upper_body_contact_point(frame: PoseFrame) -> tuple[float, float]:
    """Return wrist midpoint (COCO 9/10 when conf >= 0.30) or upper-half bbox center."""
    kps = frame.keypoints
    if len(kps) > 10:
        lw = kps[9]
        rw = kps[10]
        if lw[2] >= 0.30 and rw[2] >= 0.30:
            return (0.5 * (lw[0] + rw[0]), 0.5 * (lw[1] + rw[1]))
        if lw[2] >= 0.30:
            return (float(lw[0]), float(lw[1]))
        if rw[2] >= 0.30:
            return (float(rw[0]), float(rw[1]))
    x1, y1, x2, y2 = frame.bbox
    return (0.5 * (x1 + x2), y1 + 0.25 * (y2 - y1))


def _is_trajectory_inflection(post_balls: list[BallDetection], idx: int) -> bool:
    """Return True if ball trajectory inflects (vertical direction flip or sharp velocity change) at idx."""
    if idx <= 0 or idx >= len(post_balls) - 1:
        return False
    prev_b = post_balls[idx - 1]
    cur_b = post_balls[idx]
    next_b = post_balls[idx + 1]
    dt1 = max(1e-3, cur_b.timestamp_s - prev_b.timestamp_s)
    dt2 = max(1e-3, next_b.timestamp_s - cur_b.timestamp_s)
    vy1 = (cur_b.y - prev_b.y) / dt1
    vy2 = (next_b.y - cur_b.y) / dt2
    vx1 = (cur_b.x - prev_b.x) / dt1
    vx2 = (next_b.x - cur_b.x) / dt2
    if vy1 > 10.0 and vy2 < -10.0:
        return True
    if (vy1 - vy2) > 60.0:
        return True
    if vx1 * vx2 < -400.0:
        return True
    return False


def _trace_pass_to_setter(
    series: list[BallDetection],
    tracks: list[PlayerTrack],
    H: np.ndarray,
    land_ball: BallDetection,
    receiver_track_id: int,
    net_center_y: float,
    net_tape_y: float,
) -> tuple[float, float, float, int] | None:
    """Trace post-reception ball to the contacting near-side setter and project setter feet to court meters.

    Returns (pass_x_m, pass_y_m, pass_timestamp_s, pass_frame_idx) or None if unplayable/overpass.
    """
    post_balls = [
        b
        for b in series
        if land_ball.timestamp_s < b.timestamp_s <= land_ball.timestamp_s + POST_PASS_MAX_DURATION_S
    ]
    if not post_balls:
        return None

    went_above_net = any(b.y < net_tape_y - 10.0 for b in post_balls)
    min_contact_t = land_ball.timestamp_s + MIN_PASS_FLIGHT_S

    # Find the pass apex (highest point = minimum y) in the initial flight so we ignore
    # mid-flight rising frames where the ball visually passes near another passer/bystander.
    apex_search = [b for b in post_balls if b.timestamp_s <= land_ball.timestamp_s + 1.8]
    apex_ball = min(apex_search, key=lambda b: b.y) if apex_search else post_balls[0]
    has_clear_arc = apex_ball.y < land_ball.y - 30.0
    apex_t = apex_ball.timestamp_s

    # Candidate tuple: (ball, track_id, cx, cy, upper_dist, is_inflection)
    other_candidates: list[tuple[BallDetection, int, float, float, float, bool]] = []
    same_candidates: list[tuple[BallDetection, int, float, float, float, bool]] = []
    first_blocker_t: float | None = None

    for idx, b in enumerate(post_balls):
        if b.timestamp_s < min_contact_t:
            continue

        inflection = _is_trajectory_inflection(post_balls, idx)
        vy_in = 0.0
        if idx > 0:
            prev_b = post_balls[idx - 1]
            dt_prev = max(1e-3, b.timestamp_s - prev_b.timestamp_s)
            vy_in = (b.y - prev_b.y) / dt_prev

        # Skip frames while the pass is still rising toward its apex
        if has_clear_arc and b.timestamp_s < apex_t and not inflection:
            continue
        if vy_in < -40.0 and not inflection:
            continue

        for tr in tracks:
            fr = _nearest_pose_frame(tr, b.timestamp_s, max_delta_s=0.18)
            if fr is None:
                continue

            if tr.court_side == "blocking":
                if went_above_net and first_blocker_t is None and (inflection or vy_in >= -10.0):
                    bx1, by1, bx2, by2 = fr.bbox
                    bpad = 22.0
                    bdx = max(bx1 - bpad - b.x, 0.0, b.x - (bx2 + bpad))
                    bdy = max(by1 - bpad - b.y, 0.0, b.y - (by2 + bpad))
                    bbox_dist = math.hypot(bdx, bdy)
                    if bbox_dist <= 28.0:
                        first_blocker_t = b.timestamp_s
                continue

            if tr.court_side != "attacking":
                continue

            fx, fy = extract_foot_point(fr)
            if fy <= net_center_y - 4.0:
                continue
            try:
                cx, cy = project_image_to_court(H, fx, fy)
            except ValueError:
                continue
            if not (-1.5 <= cx <= 10.5 and -0.3 <= cy <= 10.5):
                continue

            x1, y1, x2, y2 = fr.bbox
            pad = 25.0
            dx = max(x1 - pad - b.x, 0.0, b.x - (x2 + pad))
            dy = max(y1 - pad - b.y, 0.0, b.y - (y2 + pad))
            box_dist = math.hypot(dx, dy)
            ux, uy = _upper_body_contact_point(fr)
            upper_dist = math.hypot(b.x - ux, b.y - uy)
            if box_dist <= 35.0 or upper_dist <= 85.0:
                cand = (b, tr.track_id, cx, cy, upper_dist, inflection)
                if tr.track_id != receiver_track_id:
                    other_candidates.append(cand)
                else:
                    same_candidates.append(cand)

    pool: list[tuple[BallDetection, int, float, float, float, bool]]
    if other_candidates:
        pool = other_candidates
    elif same_candidates:
        valid_same: list[tuple[BallDetection, int, float, float, float, bool]] = []
        for cand in same_candidates:
            earlier = [pb for pb in post_balls if pb.timestamp_s < cand[0].timestamp_s]
            if not earlier:
                continue
            min_y = min(pb.y for pb in earlier)
            if min_y < land_ball.y - 25.0 and cand[0].y >= min_y + 15.0:
                valid_same.append(cand)
        if not valid_same:
            return None
        pool = valid_same
    else:
        return None

    first_setter_t = pool[0][0].timestamp_s
    # Overpass: ball rose above the net tape and contacted a far-side blocker before any near-side setter
    if went_above_net and first_blocker_t is not None and first_blocker_t < first_setter_t - 0.10:
        return None

    # Identify the first contacted setter track
    first_frame_cands = [c for c in pool if c[0].frame_idx == pool[0][0].frame_idx]
    setter_tid = min(first_frame_cands, key=lambda c: c[4])[1]

    # Pick the best contact frame within the initial contact window for that setter
    window_cands = [
        c for c in pool if c[1] == setter_tid and c[0].timestamp_s <= first_setter_t + 0.55
    ]
    best = min(window_cands, key=lambda c: c[4] - (35.0 if c[5] else 0.0))
    best_ball, _tid, pass_x_m, pass_y_m, _udist, _infl = best
    return (pass_x_m, pass_y_m, round(best_ball.timestamp_s, 3), best_ball.frame_idx)


def _extract_candidate_windows(
    video_path: Path,
    metadata: VideoMetadata,
    net_cal: NetCalibration,
    predictor: FramePredictor,
    net_center_y: float,
    net_tape_y: float,
) -> list[tuple[float, float]]:
    if metadata.duration_s <= DEFAULT_CONFIG.short_clip_max_sec:
        return [(0.0, metadata.duration_s)]

    coarse_frames = list(iter_video_frames(video_path, start_s=0.0, end_s=metadata.duration_s, sample_fps=6.0))
    coarse_balls, _ = predictor.scan_coarse(coarse_frames, net_cal)
    series = _select_best_ball_series(coarse_balls)
    landings = _find_serve_landing_balls(
        series, net_center_y=net_center_y, net_tape_y=net_tape_y, max_gap_s=0.65, min_points=3
    )
    if not landings:
        return []

    raw_windows = [
        (max(0.0, b.timestamp_s - 2.5), min(metadata.duration_s, b.timestamp_s + 3.0))
        for b in landings
    ]
    merged: list[tuple[float, float]] = []
    for ws, we in sorted(raw_windows):
        if merged and ws <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], we))
        else:
            merged.append((ws, we))
    return merged


def _has_rally_break(
    series: list[BallDetection],
    prev_receive_t: float,
    candidate_t: float,
    *,
    min_gap_s: float = RALLY_BREAK_MIN_GAP_S,
) -> bool:
    """Return True if there was a dead-ball pause (>= min_gap_s) between prev_receive_t and candidate_t."""
    start_t = prev_receive_t + 0.5
    end_t = candidate_t - 0.6
    if end_t - start_t < min_gap_s:
        return False
    mid_times = [b.timestamp_s for b in series if start_t < b.timestamp_s < end_t]
    checkpoints = [start_t, *mid_times, end_t]
    max_gap = max(checkpoints[i] - checkpoints[i - 1] for i in range(1, len(checkpoints)))
    return max_gap >= min_gap_s


def detect_serve_events(
    video_path: Path,
    calibration: CourtCalibration,
    predictor: FramePredictor,
    metadata: VideoMetadata,
    server_name: str = "Unassigned",
    setter_target: Point2D | None = None,
    ring_radii_m: tuple[float, float, float] | None = None,
) -> list[ServeEvent]:
    """Detect serve events in video_path and project landing spots, passers & receive quality onto court meters."""
    H, _H_inv = compute_court_homography(calibration)
    norm_cal = normalize_court_calibration(calibration)
    net_cal = _proxy_net_calibration(norm_cal)
    net_center_y = 0.5 * (norm_cal.front_left.y + norm_cal.front_right.y)
    net_tape_y = 0.5 * (net_cal.left_point.y + net_cal.right_point.y)

    windows = _extract_candidate_windows(video_path, metadata, net_cal, predictor, net_center_y, net_tape_y)
    events: list[ServeEvent] = []
    next_allowed_t = -1e9
    last_received_t: float | None = None

    for w_start, w_end in windows:
        frames = list(iter_video_frames(video_path, start_s=w_start, end_s=w_end))
        if not frames:
            continue
        combined_fn = getattr(predictor, "detect_balls_and_poses", None)
        if callable(combined_fn):
            balls, tracks = combined_fn(frames, net_cal)
        else:
            balls = predictor.detect_balls(frames, net_cal)
            tracks = predictor.track_poses(frames, net_cal)
        series = _select_best_ball_series(balls)
        landings = _find_serve_landing_balls(series, net_center_y=net_center_y, net_tape_y=net_tape_y)

        for land_ball in landings:
            if land_ball.timestamp_s < next_allowed_t:
                continue
            if last_received_t is not None and not _has_rally_break(series, last_received_t, land_ball.timestamp_s):
                continue

            t_pre = max(w_start, land_ball.timestamp_s - PRE_SERVE_OFFSET_S)
            passers = _snapshot_passers(tracks, H, t_pre, net_center_y)

            receiver = _match_receiving_passer(tracks, H, land_ball, net_center_y)
            raw_result: Literal["in", "ace", "out", "net"]
            rec_track_id = -1
            fx_m = 0.0
            fy_m = 0.0
            if receiver is not None:
                rec_track_id, rec_frame, fx_m, fy_m = receiver
                x1, _y1, x2, _y2 = rec_frame.bbox
                bbox_w = max(1.0, x2 - x1)
                bbox_cx = 0.5 * (x1 + x2)
                lateral_reach = max(-0.6, min(0.6, ((land_ball.x - bbox_cx) / bbox_w) * 0.6))
                sx_m = fx_m + lateral_reach
                sy_m = fy_m
                raw_result = "in"
            else:
                # Unplayed bounces must reach floor level (rejects unmatched mid-air sets/spikes)
                if land_ball.y < net_center_y - 15.0:
                    continue
                try:
                    sx_m, sy_m = project_image_to_court(H, land_ball.x, land_ball.y)
                except ValueError:
                    continue
                raw_result = "ace"

            nx, ny, norm_res, zone, depth, is_seam, seam_pair = classify_serve(
                sx_m, sy_m, raw_result, passers
            )

            passer_label: str | None = None
            pass_x_m: float | None = None
            pass_y_m: float | None = None
            pass_timestamp_s: float | None = None
            pass_frame_idx: int | None = None
            pass_distance_m: float | None = None
            receive_score: Literal[0, 1, 2, 3] | None = None

            if receiver is not None and norm_res == "in":
                passer_label = _match_passer_label(fx_m, fy_m, passers)
                setter_contact = _trace_pass_to_setter(
                    series=series,
                    tracks=tracks,
                    H=H,
                    land_ball=land_ball,
                    receiver_track_id=rec_track_id,
                    net_center_y=net_center_y,
                    net_tape_y=net_tape_y,
                )
                if setter_contact is not None:
                    pass_x_m, pass_y_m, pass_timestamp_s, pass_frame_idx = setter_contact
                    pass_distance_m, receive_score = score_receive(
                        pass_x_m, pass_y_m, "in", setter_target, ring_radii_m=ring_radii_m
                    )
                else:
                    pass_distance_m, receive_score = score_receive(
                        None, None, "in", setter_target, ring_radii_m=ring_radii_m, unplayable=True
                    )
            elif norm_res == "ace":
                pass_distance_m, receive_score = score_receive(
                    None, None, "ace", setter_target, ring_radii_m=ring_radii_m
                )
            else:
                pass_distance_m, receive_score = score_receive(
                    None, None, norm_res, setter_target
                )

            events.append(
                ServeEvent(
                    serve_id=f"s_{uuid.uuid4().hex[:10]}",
                    index=len(events) + 1,
                    source="auto",
                    timestamp_s=round(land_ball.timestamp_s, 3),
                    frame_idx=land_ball.frame_idx,
                    server_name=server_name.strip()[:32] or "Unassigned",
                    x_m=nx,
                    y_m=ny,
                    result=norm_res,
                    zone=zone,
                    depth=depth,
                    is_seam=is_seam,
                    seam_pair=seam_pair,
                    passers_snapshot=passers,
                    passer_label=passer_label,
                    pass_x_m=pass_x_m,
                    pass_y_m=pass_y_m,
                    pass_timestamp_s=pass_timestamp_s,
                    pass_frame_idx=pass_frame_idx,
                    pass_distance_m=pass_distance_m,
                    receive_score=receive_score,
                )
            )
            cooldown = POST_RECEIVE_COOLDOWN_S if receiver is not None else REFRACTORY_GAP_S
            next_allowed_t = land_ball.timestamp_s + cooldown
            if receiver is not None:
                last_received_t = land_ball.timestamp_s

    return events
