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
)
from vball_analyser.video_io import iter_video_frames

REFRACTORY_GAP_S = 2.5
POST_RECEIVE_COOLDOWN_S = 6.0
MIN_DESCENT_PX = 45.0
MIN_DESCENT_DURATION_S = 0.20
PRE_SERVE_OFFSET_S = 0.60


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


def _select_best_ball_series(detections: list[BallDetection]) -> list[BallDetection]:
    """Pick highest-confidence ball per frame and drop isolated spatial teleport outliers."""
    by_frame: dict[int, BallDetection] = {}
    for d in detections:
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
        if series[i].y <= series[i - 1].y:
            i += 1
            continue

        run_start = i - 1
        j = i
        # Follow descent while allowing brief 1-frame plateaus/jitter
        while j < n:
            if series[j].y >= series[j - 1].y - 3.0 and (series[j].timestamp_s - series[j - 1].timestamp_s) <= max_gap_s:
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
        has_rebound = (
            run_end + 1 < n
            and (series[run_end + 1].timestamp_s - end_b.timestamp_s) <= max_gap_s
            and series[run_end + 1].y < end_b.y - 2.0
        )
        below_tape = end_b.y >= tape_y + 15.0
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
) -> tuple[PoseFrame, float, float] | None:
    """Find near-side player contacting the serve at ball.timestamp_s, returning (frame, foot_x_m, foot_y_m)."""
    best_match: tuple[float, PoseFrame, float, float] | None = None
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
                best_match = (center_dist, fr, cx, cy)

    if best_match is None:
        return None
    return (best_match[1], best_match[2], best_match[3])


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
        (max(0.0, b.timestamp_s - 2.5), min(metadata.duration_s, b.timestamp_s + 1.5))
        for b in landings
    ]
    merged: list[tuple[float, float]] = []
    for ws, we in sorted(raw_windows):
        if merged and ws <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], we))
        else:
            merged.append((ws, we))
    return merged


def detect_serve_events(
    video_path: Path,
    calibration: CourtCalibration,
    predictor: FramePredictor,
    metadata: VideoMetadata,
    server_name: str = "Unassigned",
) -> list[ServeEvent]:
    """Detect serve events in video_path and project landing spots & passers onto top-down court meters."""
    H, _H_inv = compute_court_homography(calibration)
    norm_cal = normalize_court_calibration(calibration)
    net_cal = _proxy_net_calibration(norm_cal)
    net_center_y = 0.5 * (norm_cal.front_left.y + norm_cal.front_right.y)
    net_tape_y = 0.5 * (net_cal.left_point.y + net_cal.right_point.y)

    windows = _extract_candidate_windows(video_path, metadata, net_cal, predictor, net_center_y, net_tape_y)
    events: list[ServeEvent] = []
    next_allowed_t = -1e9

    for w_start, w_end in windows:
        frames = list(iter_video_frames(video_path, start_s=w_start, end_s=w_end))
        if not frames:
            continue
        balls = predictor.detect_balls(frames, net_cal)
        tracks = predictor.track_poses(frames, net_cal)
        series = _select_best_ball_series(balls)
        landings = _find_serve_landing_balls(series, net_center_y=net_center_y, net_tape_y=net_tape_y)

        for land_ball in landings:
            if land_ball.timestamp_s < next_allowed_t:
                continue

            t_pre = max(w_start, land_ball.timestamp_s - PRE_SERVE_OFFSET_S)
            passers = _snapshot_passers(tracks, H, t_pre, net_center_y)

            receiver = _match_receiving_passer(tracks, H, land_ball, net_center_y)
            raw_result: Literal["in", "ace", "out", "net"]
            if receiver is not None:
                rec_frame, fx_m, fy_m = receiver
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
                )
            )
            cooldown = POST_RECEIVE_COOLDOWN_S if receiver is not None else REFRACTORY_GAP_S
            next_allowed_t = land_ball.timestamp_s + cooldown

    return events
