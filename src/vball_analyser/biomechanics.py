"""Pose smoothing, near/far court role isolation, hitter mechanics, and blocker seam analysis."""

from __future__ import annotations

import math
import statistics
from typing import Literal

from vball_analyser.calibration import (
    cm_per_px,
    compute_pin_distance_m,
    estimated_cm_per_px,
    height_above_tape,
    net_ground_y_at,
    net_span_px,
    normalized_net_u,
    tape_y_at,
)
from vball_analyser.models import (
    BlockerCallout,
    BlockIsolationStats,
    BlockSummary,
    HitterMetrics,
    KeyMoment,
    NetCalibration,
    OffenseBucketSummary,
    PlayAnalysis,
    PlayerTrack,
    Point2D,
    PoseFrame,
    SetTrajectory,
    StudioSessionSummary,
    ZoneSetWindowStats,
)

# COCO 17 keypoint indices
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_ANKLE = 15
R_ANKLE = 16

_CONF_THRESH = 0.25
_SMOOTH_WEIGHTS = (0.1, 0.2, 0.4, 0.2, 0.1)


def _kp_valid(kp: tuple[float, float, float]) -> bool:
    return kp[2] >= _CONF_THRESH and (kp[0] > 0 or kp[1] > 0)


def _angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    """Interior angle ABC in degrees."""
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-4 or n2 < 1e-4:
        return None
    cos_val = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return round(math.degrees(math.acos(cos_val)), 1)


def smooth_pose_tracks(raw_tracks: list[PlayerTrack]) -> list[PlayerTrack]:
    """Smooth visible pose keypoints across frames using a 5-tap symmetric kernel."""
    smoothed: list[PlayerTrack] = []
    half = len(_SMOOTH_WEIGHTS) // 2

    for track in raw_tracks:
        ordered = sorted(track.frames, key=lambda f: f.frame_idx)
        n = len(ordered)
        new_frames: list[PoseFrame] = []

        for i, pf in enumerate(ordered):
            new_kps: list[tuple[float, float, float]] = []
            for k_idx in range(len(pf.keypoints)):
                kp = pf.keypoints[k_idx]
                if not _kp_valid(kp):
                    new_kps.append(kp)
                    continue
                wx, wy, wsum = 0.0, 0.0, 0.0
                for offset, weight in enumerate(_SMOOTH_WEIGHTS, start=-half):
                    j = i + offset
                    if 0 <= j < n and abs(ordered[j].frame_idx - pf.frame_idx) <= 3:
                        nkp = ordered[j].keypoints[k_idx]
                        if _kp_valid(nkp):
                            wx += weight * nkp[0]
                            wy += weight * nkp[1]
                            wsum += weight
                if wsum > 0:
                    new_kps.append((round(wx / wsum, 2), round(wy / wsum, 2), kp[2]))
                else:
                    new_kps.append(kp)

            new_frames.append(
                PoseFrame(
                    frame_idx=pf.frame_idx,
                    timestamp_s=pf.timestamp_s,
                    bbox=pf.bbox,
                    keypoints=new_kps,
                )
            )

        smoothed.append(
            PlayerTrack(track_id=track.track_id, court_side=track.court_side, frames=new_frames)
        )
    return smoothed


def _standing_ankle_y(track: PlayerTrack) -> tuple[float, float]:
    """Return (median_x, 90th_percentile_bottom_y) representing resting floor contact."""
    xs: list[float] = []
    ys: list[float] = []
    for pf in track.frames:
        xs.append(0.5 * (pf.bbox[0] + pf.bbox[2]))
        ank_ys = [
            pf.keypoints[idx][1]
            for idx in (L_ANKLE, R_ANKLE)
            if idx < len(pf.keypoints) and _kp_valid(pf.keypoints[idx])
        ]
        if ank_ys:
            ys.append(max(max(ank_ys), pf.bbox[3] - 4.0))
        else:
            ys.append(pf.bbox[3])
    if not ys:
        return 0.0, 0.0
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    x_med = xs_sorted[len(xs_sorted) // 2]
    y_p90 = ys_sorted[min(len(ys_sorted) - 1, int(0.90 * len(ys_sorted)))]
    return x_med, y_p90


def classify_court_sides(tracks: list[PlayerTrack], calibration: NetCalibration) -> list[PlayerTrack]:
    """Classify each track as near-side attacking or far-side blocking using net geometry."""
    span = net_span_px(calibration)
    classified: list[PlayerTrack] = []
    for track in tracks:
        x_med, y_p90 = _standing_ankle_y(track)
        # Account for elevated baseline tripod pitch: far-side blockers' feet
        # project above ~tape_y + 0.19 * net_span_px, near-side players stand below it.
        net_floor_thresh = tape_y_at(calibration, x_med) + 0.19 * span
        side = "attacking" if y_p90 >= net_floor_thresh else "blocking"
        classified.append(PlayerTrack(track_id=track.track_id, court_side=side, frames=track.frames))
    return classified


def _closest_pose_frame(track: PlayerTrack, frame_idx: int, max_delta: int = 3) -> PoseFrame | None:
    best: PoseFrame | None = None
    best_d = max_delta + 1
    for pf in track.frames:
        d = abs(pf.frame_idx - frame_idx)
        if d < best_d:
            best_d = d
            best = pf
    return best


def _upper_body_y(pf: PoseFrame) -> float:
    pts = [
        pf.keypoints[idx][1]
        for idx in (NOSE, L_SHOULDER, R_SHOULDER)
        if idx < len(pf.keypoints) and _kp_valid(pf.keypoints[idx])
    ]
    return sum(pts) / len(pts) if pts else pf.bbox[1]


def _hip_y(pf: PoseFrame) -> float:
    pts = [
        pf.keypoints[idx][1]
        for idx in (L_HIP, R_HIP)
        if idx < len(pf.keypoints) and _kp_valid(pf.keypoints[idx])
    ]
    return sum(pts) / len(pts) if pts else 0.5 * (pf.bbox[1] + pf.bbox[3])


def _jump_impulse_px(track: PlayerTrack) -> float:
    if len(track.frames) < 3:
        return 0.0
    hips = sorted(_hip_y(pf) for pf in track.frames)
    baseline = hips[min(len(hips) - 1, int(0.80 * len(hips)))]
    peak = hips[0]
    return max(0.0, baseline - peak)


def _wrist_points(pf: PoseFrame) -> list[tuple[str, tuple[float, float]]]:
    out: list[tuple[str, tuple[float, float]]] = []
    if L_WRIST < len(pf.keypoints) and _kp_valid(pf.keypoints[L_WRIST]):
        out.append(("left", (pf.keypoints[L_WRIST][0], pf.keypoints[L_WRIST][1])))
    if R_WRIST < len(pf.keypoints) and _kp_valid(pf.keypoints[R_WRIST]):
        out.append(("right", (pf.keypoints[R_WRIST][0], pf.keypoints[R_WRIST][1])))
    return out


def _apex_frame_of_track(
    track: PlayerTrack, contact_frame: int | None = None, fps: float = 30.0
) -> int | None:
    if not track.frames:
        return None
    candidates = track.frames
    if contact_frame is not None:
        win_lo = contact_frame - max(4, int(round(0.55 * fps)))
        win_hi = contact_frame + max(4, int(round(0.45 * fps)))
        windowed = [pf for pf in track.frames if win_lo <= pf.frame_idx <= win_hi]
        if windowed:
            candidates = windowed
    best_pf = min(candidates, key=_upper_body_y)
    return best_pf.frame_idx


def _is_symmetric_two_hand_block(pf: PoseFrame) -> bool:
    """Detect textbook two-arm vertical block pose (both wrists raised high and level)."""
    if (
        L_WRIST >= len(pf.keypoints)
        or R_WRIST >= len(pf.keypoints)
        or L_SHOULDER >= len(pf.keypoints)
        or R_SHOULDER >= len(pf.keypoints)
    ):
        return False
    lw, rw = pf.keypoints[L_WRIST], pf.keypoints[R_WRIST]
    ls, rs = pf.keypoints[L_SHOULDER], pf.keypoints[R_SHOULDER]
    if not (_kp_valid(lw) and _kp_valid(rw) and _kp_valid(ls) and _kp_valid(rs)):
        return False
    sh_y = 0.5 * (ls[1] + rs[1])
    both_above_shoulders = lw[1] < sh_y - 12.0 and rw[1] < sh_y - 12.0
    level_wrists = abs(lw[1] - rw[1]) <= 18.0
    return both_above_shoulders and level_wrists


def _find_max_load_frame(track: PlayerTrack, contact_frame: int, fps: float, dominant_hand: str) -> int:
    search_start = contact_frame - max(3, int(round(0.65 * fps)))
    candidates = [pf for pf in track.frames if search_start <= pf.frame_idx < contact_frame]
    if not candidates:
        return max(0, contact_frame - 1)

    sh_idx = R_SHOULDER if dominant_hand == "right" else L_SHOULDER
    el_idx = R_ELBOW if dominant_hand == "right" else L_ELBOW
    wr_idx = R_WRIST if dominant_hand == "right" else L_WRIST

    best_frame = candidates[max(0, len(candidates) // 2)].frame_idx
    best_score = float("inf")
    for pf in candidates:
        sh, el, wr = pf.keypoints[sh_idx], pf.keypoints[el_idx], pf.keypoints[wr_idx]
        if _kp_valid(sh) and _kp_valid(el) and _kp_valid(wr):
            ang = _angle_deg((sh[0], sh[1]), (el[0], el[1]), (wr[0], wr[1]))
            if ang is not None:
                # Favor compact elbow cocking + higher elbow elevation
                score = ang + 0.15 * (el[1] - sh[1])
                if score < best_score:
                    best_score = score
                    best_frame = pf.frame_idx
    return best_frame


def build_contact_candidates(
    tracks: list[PlayerTrack], calibration: NetCalibration
) -> list[tuple[int, Point2D]]:
    """Collect airborne attacking-pose wrist points near/above the net tape from both court sides."""
    span = max(1.0, net_span_px(calibration))
    candidates: list[tuple[int, Point2D]] = []
    for t in tracks:
        if len(t.frames) < 4:
            continue
        impulse = _jump_impulse_px(t)
        # Far-court jumps shrink with perspective, so blocking-side tracks use a lower threshold.
        if impulse < (25.0 if t.court_side == "attacking" else 20.0):
            continue
        hips = sorted(_hip_y(pf) for pf in t.frames)
        baseline_hip = hips[min(len(hips) - 1, int(0.80 * len(hips)))]
        wrist_margin = max(18.0, 0.035 * span)
        for pf in t.frames:
            if baseline_hip - _hip_y(pf) < 0.50 * impulse:
                continue
            if _is_symmetric_two_hand_block(pf):
                continue
            added_wrist = False
            for w_idx in (L_WRIST, R_WRIST):
                if w_idx < len(pf.keypoints) and _kp_valid(pf.keypoints[w_idx]):
                    wx, wy = pf.keypoints[w_idx][0], pf.keypoints[w_idx][1]
                    if wy <= tape_y_at(calibration, wx) + wrist_margin:
                        candidates.append((pf.frame_idx, Point2D(x=wx, y=wy)))
                        added_wrist = True
            if not added_wrist:
                bx_mid = 0.5 * (pf.bbox[0] + pf.bbox[2])
                if pf.bbox[1] <= tape_y_at(calibration, bx_mid):
                    candidates.append((pf.frame_idx, Point2D(x=bx_mid, y=pf.bbox[1])))
    return candidates


def classify_play_team_side(
    classified: list[PlayerTrack],
    set_trajectory: SetTrajectory | None,
    calibration: NetCalibration,
    contact_frame: int | None,
    contact_point: Point2D | None,
    *,
    session_mode: Literal["drill", "match"] = "drill",
    override_team_side: str | None = None,
) -> Literal["ours", "opponent"]:
    """Score near-side (ours) vs far-side (opponent) attacker/setter cues around contact and release."""
    if override_team_side == "ours" or override_team_side == "opponent":
        return override_team_side

    span = max(1.0, net_span_px(calibration))
    net_mid_x = 0.5 * (calibration.left_point.x + calibration.right_point.x)
    ref_x = contact_point.x if contact_point is not None else net_mid_x
    ours_score = 0.0
    opp_score = 0.0

    # Contact attacker vs blocker cue
    for t in classified:
        if not t.frames or _jump_impulse_px(t) < 20.0:
            continue
        ref_f = contact_frame if contact_frame is not None else _apex_frame_of_track(t)
        pf = _closest_pose_frame(t, ref_f, max_delta=4) if ref_f is not None else None
        if pf is None or abs(0.5 * (pf.bbox[0] + pf.bbox[2]) - ref_x) > 0.22 * span:
            continue
        near = t.court_side == "attacking"
        if _is_symmetric_two_hand_block(pf):
            if near:
                xm, yp90 = _standing_ankle_y(t)
                if yp90 <= net_ground_y_at(calibration, xm) + 0.14 * span:
                    opp_score += 1.2
            else:
                ours_score += 1.0
            continue
        if contact_point is None:
            hit = True
        else:
            pts = [w for _, w in _wrist_points(pf)]
            if not near:
                pts.append((0.5 * (pf.bbox[0] + pf.bbox[2]), pf.bbox[1]))
            max_d = 95.0 if near else 85.0
            hit = any(math.hypot(px - contact_point.x, py - contact_point.y) <= max_d for px, py in pts)
        if hit:
            bonus = 2.2 if contact_point is not None else 1.4
            if near:
                ours_score += bonus
            else:
                opp_score += bonus

    # Setter release cue (excluding each side's top jumper, i.e. its likely hitter)
    if set_trajectory is not None and set_trajectory.arc_points:
        rel_f = set_trajectory.release_frame
        rel_pt = set_trajectory.arc_points[0]
        side_d: dict[str, float] = {}
        for side in ("attacking", "blocking"):
            side_tracks = [t for t in classified if t.court_side == side and t.frames]
            top = max(side_tracks, key=_jump_impulse_px, default=None)
            best_d = float("inf")
            for t in side_tracks:
                if top is not None and t is top and _jump_impulse_px(t) >= 20.0:
                    continue
                pf = _closest_pose_frame(t, rel_f, max_delta=5)
                if pf is None:
                    continue
                pts = [w for _, w in _wrist_points(pf)]
                if side == "blocking":
                    pts.append((0.5 * (pf.bbox[0] + pf.bbox[2]), _upper_body_y(pf)))
                for px, py in pts:
                    best_d = min(best_d, math.hypot(px - rel_pt.x, py - rel_pt.y))
            side_d[side] = best_d
        near_d, far_d = side_d["attacking"], side_d["blocking"]
        if near_d <= 0.28 * span and near_d <= far_d:
            ours_score += 1.5
        elif far_d <= 0.24 * span and near_d > 0.18 * span:
            opp_score += 1.5

    if session_mode == "match":
        return "opponent" if opp_score > ours_score + 0.3 else "ours"
    # Drill reps are overwhelmingly near-side, so require a clear margin.
    return "opponent" if opp_score >= ours_score + 1.5 else "ours"


def _compute_system_state(
    set_trajectory: SetTrajectory | None,
    setter: PlayerTrack | None,
    calibration: NetCalibration,
) -> Literal["in_system", "out_of_system"] | None:
    """Classify our set as in-system (overhead, near net, front-centre/right corridor) or out-of-system."""
    if set_trajectory is None or not set_trajectory.arc_points:
        return None
    span = max(1.0, net_span_px(calibration))
    rel = set_trajectory.arc_points[0]
    x_rel_m = normalized_net_u(calibration, rel.x) * 9.0
    in_horizontal_window = 3.5 <= x_rel_m <= 7.6

    is_overhead: bool | None = None
    is_near_net = rel.y <= tape_y_at(calibration, rel.x) + 0.16 * span
    if setter is not None:
        pf = _closest_pose_frame(setter, set_trajectory.release_frame, max_delta=5)
        if pf is not None:
            shoulders = [pf.keypoints[i] for i in (L_SHOULDER, R_SHOULDER) if _kp_valid(pf.keypoints[i])]
            wrists = _wrist_points(pf)
            if shoulders and wrists:
                sh_mid_y = sum(k[1] for k in shoulders) / len(shoulders)
                is_overhead = any(w[1] <= sh_mid_y + 12.0 for _, w in wrists)
        _, y_foot = _standing_ankle_y(setter)
        is_near_net = y_foot <= net_ground_y_at(calibration, rel.x) + 0.25 * span
    if is_overhead is None:
        is_overhead = rel.y <= tape_y_at(calibration, rel.x) + 0.20 * span

    return "in_system" if (in_horizontal_window and is_overhead and is_near_net) else "out_of_system"


def _compute_blocker_hand_span(pf: PoseFrame, calibration: NetCalibration, span: float) -> tuple[float, float]:
    """Horizontal extent of a blocker's hands/arms above the tape (55% central bbox fallback)."""
    xs = [
        pf.keypoints[idx][0]
        for idx in (L_WRIST, R_WRIST, L_ELBOW, R_ELBOW)
        if idx < len(pf.keypoints)
        and _kp_valid(pf.keypoints[idx])
        and pf.keypoints[idx][1] <= tape_y_at(calibration, pf.keypoints[idx][0]) + 0.08 * span
    ]
    if xs:
        pad = max(6.0, 0.009 * span)
        return round(min(xs) - pad, 1), round(max(xs) + pad, 1)
    # Flaring legs/hips widen the bbox, so only trust its central 55% when arm keypoints drop out.
    cx = 0.5 * (pf.bbox[0] + pf.bbox[2])
    bw = pf.bbox[2] - pf.bbox[0]
    return round(cx - 0.275 * bw, 1), round(cx + 0.275 * bw, 1)


def analyse_play_mechanics(
    tracks: list[PlayerTrack],
    set_trajectory: SetTrajectory | None,
    calibration: NetCalibration,
    fps: float,
    frame_shape: tuple[int, int],
    override_hitter_track_id: int | None = None,
) -> tuple[int | None, int | None, HitterMetrics | None, BlockSummary, list[KeyMoment]]:
    """Backward-compatible wrapper around `analyse_play_full` returning the legacy 5-tuple."""
    return analyse_play_full(
        tracks, set_trajectory, calibration, fps, frame_shape, override_hitter_track_id=override_hitter_track_id
    )[3:]


def analyse_play_full(
    tracks: list[PlayerTrack],
    set_trajectory: SetTrajectory | None,
    calibration: NetCalibration,
    fps: float,
    frame_shape: tuple[int, int],
    *,
    session_mode: Literal["drill", "match"] = "drill",
    override_hitter_track_id: int | None = None,
    override_team_side: str | None = None,
) -> tuple[
    Literal["ours", "opponent"],
    Literal["in_system", "out_of_system"] | None,
    float | None,
    int | None,
    int | None,
    HitterMetrics | None,
    BlockSummary,
    list[KeyMoment],
]:
    """Classify ours/opponent, isolate setter/hitter/blockers, compute 2D biomechanics, and build 3 key moments.

    Returns (team_side, system_state, pin_distance_m, setter_id, hitter_id, hitter_metrics, block_summary, keyframes).
    """
    del frame_shape
    if fps <= 0:
        fps = 30.0

    smoothed = smooth_pose_tracks(tracks)
    classified = classify_court_sides(smoothed, calibration)

    contact_point: Point2D | None = None
    contact_frame: int | None = None
    if set_trajectory is not None and set_trajectory.arc_points:
        contact_frame = set_trajectory.contact_frame
        last_pt = set_trajectory.arc_points[-1]
        contact_point = Point2D(x=last_pt.x, y=last_pt.y)

    span = max(1.0, net_span_px(calibration))
    team_side = classify_play_team_side(
        classified,
        set_trajectory,
        calibration,
        contact_frame,
        contact_point,
        session_mode=session_mode,
        override_team_side=override_team_side,
    )
    defense_side = "blocking" if team_side == "ours" else "attacking"

    def _is_offense(t: PlayerTrack) -> bool:
        if not t.frames:
            return False
        if team_side == "opponent":
            return t.court_side == "blocking"
        # Our attacking tracks, plus any front-row jumper close to the net floor
        xm, yp90 = _standing_ankle_y(t)
        return t.court_side == "attacking" or yp90 >= tape_y_at(calibration, xm) + 0.165 * span

    # Select active hitter from the offense side
    hitter: PlayerTrack | None = None
    if override_hitter_track_id is not None:
        for t in classified:
            if t.track_id == override_hitter_track_id and _is_offense(t):
                hitter = t
                break

    if hitter is None:
        hitter_pool = [t for t in classified if _is_offense(t)]
        best_hitter_score = -float("inf")
        for t in hitter_pool:
            impulse = _jump_impulse_px(t)
            xm, yp90 = _standing_ankle_y(t)
            near_depth_bonus = 0.35 * (yp90 - tape_y_at(calibration, xm))
            wrist_penalty = 0.0
            block_penalty = 0.0
            if contact_frame is not None and contact_point is not None:
                pf = _closest_pose_frame(t, contact_frame, max_delta=4)
                if pf is not None:
                    wrists = _wrist_points(pf)
                    if wrists:
                        min_d = min(math.hypot(w[0] - contact_point.x, w[1] - contact_point.y) for _, w in wrists)
                        wrist_penalty = 0.85 * min_d
                    else:
                        wrist_penalty = 200.0
                    if _is_symmetric_two_hand_block(pf):
                        block_penalty = 55.0
                else:
                    wrist_penalty = 300.0
            score = impulse + near_depth_bonus - wrist_penalty - block_penalty
            if score > best_hitter_score:
                best_hitter_score = score
                hitter = t

    if contact_frame is None:
        if hitter is not None:
            contact_frame = _apex_frame_of_track(hitter, None, fps)
        elif set_trajectory is not None:
            contact_frame = set_trajectory.contact_frame
        else:
            all_frames = [pf.frame_idx for t in classified for pf in t.frames]
            contact_frame = all_frames[len(all_frames) // 2] if all_frames else 0

    if contact_point is None and hitter is not None and contact_frame is not None:
        pf_h = _closest_pose_frame(hitter, contact_frame)
        if pf_h is not None:
            wrists = _wrist_points(pf_h)
            if wrists:
                _, best_w = min(wrists, key=lambda item: item[1][1])
                contact_point = Point2D(x=round(best_w[0], 1), y=round(best_w[1], 1))

    # Select setter near release_frame (prefer attacking/front-row tracks with raised hands near ball)
    setter_track_id: int | None = None
    if set_trajectory is not None and set_trajectory.arc_points and classified:
        rel_f = set_trajectory.release_frame
        rel_pt = set_trajectory.arc_points[0]
        best_setter_score = 0.30 * span
        for t in classified:
            if not t.frames:
                continue
            if hitter is not None and t.track_id == hitter.track_id and len(classified) > 1:
                continue
            if not _is_offense(t):
                continue
            pf = _closest_pose_frame(t, rel_f, max_delta=5)
            if pf is None:
                continue
            wrists = _wrist_points(pf)
            if wrists:
                d = min(math.hypot(w[0] - rel_pt.x, w[1] - rel_pt.y) for _, w in wrists)
                if d < best_setter_score:
                    best_setter_score = d
                    setter_track_id = t.track_id

    # Compute HitterMetrics
    hitter_metrics: HitterMetrics | None = None
    hitter_track_id = hitter.track_id if hitter is not None else None
    if hitter is not None and contact_frame is not None:
        pf_con = _closest_pose_frame(hitter, contact_frame, max_delta=4)
        dominant_hand: str = "right"
        if pf_con is not None:
            wrists = _wrist_points(pf_con)
            if len(wrists) == 2 and contact_point is not None:
                nearest_hand, nearest_pt = min(
                    wrists, key=lambda item: math.hypot(item[1][0] - contact_point.x, item[1][1] - contact_point.y)
                )
                if math.hypot(nearest_pt[0] - contact_point.x, nearest_pt[1] - contact_point.y) <= 80.0:
                    dominant_hand = nearest_hand
                else:
                    dominant_hand = min(wrists, key=lambda item: item[1][1])[0]
            elif wrists:
                dominant_hand = min(wrists, key=lambda item: item[1][1])[0]

        arm_ext: float | None = None
        torso_tilt: float | None = None
        if pf_con is not None:
            sh_i, el_i, wr_i = (
                (R_SHOULDER, R_ELBOW, R_WRIST) if dominant_hand == "right" else (L_SHOULDER, L_ELBOW, L_WRIST)
            )
            sh, el, wr = pf_con.keypoints[sh_i], pf_con.keypoints[el_i], pf_con.keypoints[wr_i]
            if _kp_valid(sh) and _kp_valid(el) and _kp_valid(wr):
                arm_ext = _angle_deg((sh[0], sh[1]), (el[0], el[1]), (wr[0], wr[1]))

            sh_pts = [pf_con.keypoints[i] for i in (L_SHOULDER, R_SHOULDER) if _kp_valid(pf_con.keypoints[i])]
            hp_pts = [pf_con.keypoints[i] for i in (L_HIP, R_HIP) if _kp_valid(pf_con.keypoints[i])]
            if sh_pts and hp_pts:
                sx = sum(p[0] for p in sh_pts) / len(sh_pts)
                sy = sum(p[1] for p in sh_pts) / len(sh_pts)
                hx = sum(p[0] for p in hp_pts) / len(hp_pts)
                hy = sum(p[1] for p in hp_pts) / len(hp_pts)
                torso_tilt = round(abs(math.degrees(math.atan2(sx - hx, -(sy - hy)))), 1)

        apex_f = _apex_frame_of_track(hitter, contact_frame, fps)
        apex_delta_ms: int | None = None
        timing_label = "Unknown"
        if apex_f is not None:
            apex_delta_ms = int(round(((contact_frame - apex_f) / fps) * 1000.0))
            if abs(apex_delta_ms) <= 50:
                timing_label = "At peak jump"
            elif apex_delta_ms < -50:
                timing_label = f"Early ({abs(apex_delta_ms)}ms before peak)"
            else:
                timing_label = f"Late ({apex_delta_ms}ms after peak)"

        ch_px: float | None = None
        ch_cm: float | None = None
        if contact_point is not None:
            ch_px, ch_cm = height_above_tape(calibration, contact_point.x, contact_point.y)

        hx_med, hy_p75 = _standing_ankle_y(hitter)
        # A single baseline camera cannot separate far-court 3m takeoff depth behind the net.
        is_back_row = team_side == "ours" and hy_p75 > tape_y_at(calibration, hx_med) + 0.36 * span

        hitter_metrics = HitterMetrics(
            contact_point=contact_point,
            contact_height_px=ch_px,
            contact_height_cm=ch_cm,
            is_back_row_approximate=is_back_row,
            dominant_hand="left" if dominant_hand == "left" else "right",
            arm_extension_deg=arm_ext,
            torso_tilt_deg=torso_tilt,
            apex_frame=apex_f,
            apex_delta_ms=apex_delta_ms,
            jump_timing_label=timing_label,
        )

    # Opposing Blockers & Open Seam analysis
    blocker_callouts: list[BlockerCallout] = []
    hitter_ground_y = _standing_ankle_y(hitter)[1] if hitter is not None else None
    if contact_frame is not None:
        for bt in classified:
            if not bt.frames:
                continue
            if bt.track_id in (hitter_track_id, setter_track_id):
                continue
            pf_b = _closest_pose_frame(bt, contact_frame, max_delta=3)
            if pf_b is None:
                continue
            b_cx = 0.5 * (pf_b.bbox[0] + pf_b.bbox[2])
            if contact_point is not None and abs(b_cx - contact_point.x) > 0.25 * span:
                continue

            if team_side == "ours":
                _, b_ground_y = _standing_ankle_y(bt)
                is_defense = (
                    bt.court_side == "blocking"
                    or _is_symmetric_two_hand_block(pf_b)
                    or (hitter_ground_y is not None and b_ground_y < hitter_ground_y - 6.0)
                )
            else:
                # Our near-side block: only players who actually jump at the net
                is_defense = bt.court_side == "attacking" and _jump_impulse_px(bt) >= 20.0
            if not is_defense:
                continue

            reach_candidates: list[tuple[float, float]] = [(b_cx, pf_b.bbox[1])]
            for idx in (L_WRIST, R_WRIST, NOSE):
                if idx < len(pf_b.keypoints) and _kp_valid(pf_b.keypoints[idx]):
                    reach_candidates.append((pf_b.keypoints[idx][0], pf_b.keypoints[idx][1]))
            top_x, top_y = min(reach_candidates, key=lambda pt: pt[1])
            # Active blocker must actually jump/reach at or above the net tape
            if top_y <= tape_y_at(calibration, top_x) + 0.02 * span:
                r_px, r_cm = height_above_tape(calibration, top_x, top_y)
                blocker_callouts.append(
                    BlockerCallout(
                        track_id=bt.track_id,
                        bbox=pf_b.bbox,
                        top_reach_point=Point2D(x=round(top_x, 1), y=round(top_y, 1)),
                        reach_above_tape_px=r_px,
                        reach_above_tape_cm=r_cm,
                        hand_span_x=_compute_blocker_hand_span(pf_b, calibration, span),
                    )
                )

    blocker_callouts.sort(key=lambda b: 0.5 * (b.hand_span_x[0] + b.hand_span_x[1]))
    seam_open = False
    seam_gap_px: float | None = None
    seam_gap_cm: float | None = None
    seam_interval_x: tuple[float, float] | None = None

    if len(blocker_callouts) >= 2:
        est_scale = estimated_cm_per_px(calibration)
        min_seam_px = max(22.0 / max(est_scale, 1e-4), 0.024 * span)
        best_gap = 0.0
        best_pair: tuple[float, float] | None = None
        for b1, b2 in zip(blocker_callouts, blocker_callouts[1:], strict=False):
            gap = b2.hand_span_x[0] - b1.hand_span_x[1]
            if gap > best_gap:
                best_gap = gap
                best_pair = (b1.hand_span_x[1], b2.hand_span_x[0])
        if best_pair is not None and best_gap >= min_seam_px:
            seam_open = True
            seam_gap_px = round(best_gap, 1)
            scale = cm_per_px(calibration)
            seam_gap_cm = round(best_gap * scale, 1) if scale is not None else None
            seam_interval_x = best_pair

    if not blocker_callouts:
        has_defense_players = any(t.court_side == defense_side and len(t.frames) >= 3 for t in classified)
        block_state = "zero_block" if has_defense_players and session_mode == "match" else "unopposed"
    elif len(blocker_callouts) == 1:
        block_state = "single_block"
    else:
        block_state = "open_seam" if seam_open else "closed_block"

    block_summary = BlockSummary(
        blockers=blocker_callouts,
        seam_open=seam_open,
        seam_gap_px=seam_gap_px,
        seam_gap_cm=seam_gap_cm,
        seam_interval_x=seam_interval_x,
        block_state=block_state,
    )

    # Build 3 KeyMoments: 1. Set Release -> 2. Max Load -> 3. Ball Contact
    min_track_frame = min((pf.frame_idx for t in classified for pf in t.frames), default=0)
    con_f = contact_frame if contact_frame is not None else min_track_frame
    rel_f = (
        set_trajectory.release_frame
        if set_trajectory is not None
        else max(min_track_frame, con_f - max(2, int(round(0.9 * fps))))
    )
    if hitter is not None:
        dom = hitter_metrics.dominant_hand if hitter_metrics is not None else "right"
        load_f = _find_max_load_frame(hitter, con_f, fps, dom)
    else:
        load_f = max(rel_f, con_f - max(1, int(round(0.2 * fps))))

    keyframes = [
        KeyMoment(name="set_release", label="1. Set Release", frame_idx=rel_f, timestamp_s=round(rel_f / fps, 3)),
        KeyMoment(name="max_load", label="2. Max Load", frame_idx=load_f, timestamp_s=round(load_f / fps, 3)),
        KeyMoment(name="contact", label="3. Ball Contact", frame_idx=con_f, timestamp_s=round(con_f / fps, 3)),
    ]

    setter = next((t for t in classified if t.track_id == setter_track_id), None)
    system_state = _compute_system_state(set_trajectory, setter, calibration) if team_side == "ours" else None
    pin_distance_m = compute_pin_distance_m(calibration, contact_point.x if contact_point is not None else None)

    return (
        team_side,
        system_state,
        pin_distance_m,
        setter_track_id,
        hitter_track_id,
        hitter_metrics,
        block_summary,
        keyframes,
    )


_OURS_ZONES = ("Zone 4 (Left)", "Zone 3 (Middle)", "Zone 2 (Right)")
_OPP_ZONES = ("Opp Zone 4 (Left)", "Opp Zone 3 (Middle)", "Opp Zone 2 (Right)")


def _mean_std(values: list[float], ndigits: int) -> tuple[float | None, float | None]:
    """Mean and sample std (ddof=1); std is None below 2 reps so 1-rep zones never claim zero spread."""
    if not values:
        return None, None
    std = round(statistics.stdev(values), ndigits) if len(values) >= 2 else None
    return round(statistics.fmean(values), ndigits), std


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total > 0 else 0.0


def _offense_bucket(plays: list[PlayAnalysis], zone_names: tuple[str, ...]) -> OffenseBucketSummary:
    n = len(plays)
    zones: list[ZoneSetWindowStats] = []
    for zone in zone_names:
        zp = [p for p in plays if p.attack_zone == zone]
        trajs = [p.set_trajectory for p in zp if p.set_trajectory is not None]
        tempo_mean, tempo_std = _mean_std([t.tempo_s for t in trajs], 3)
        peak_mean, peak_std = _mean_std([t.peak_height_cm for t in trajs if t.peak_height_cm is not None], 1)
        pin_mean, pin_std = _mean_std([p.pin_distance_m for p in zp if p.pin_distance_m is not None], 2)
        zones.append(
            ZoneSetWindowStats(
                zone=zone,
                count=len(zp),
                pct=_pct(len(zp), n),
                tempo_mean_s=tempo_mean,
                tempo_std_s=tempo_std,
                peak_height_mean_cm=peak_mean,
                peak_height_std_cm=peak_std,
                pin_distance_mean_m=pin_mean,
                pin_distance_std_m=pin_std,
            )
        )

    states = [p.block_summary.block_state for p in plays]
    opposed = sum(1 for s in states if s != "unopposed")
    zero, single, open_seam, closed = (
        states.count(s) for s in ("zero_block", "single_block", "open_seam", "closed_block")
    )
    seam_cms = [
        p.block_summary.seam_gap_cm
        for p in plays
        if p.block_summary.block_state == "open_seam" and p.block_summary.seam_gap_cm is not None
    ]
    block_stats = BlockIsolationStats(
        opposed_count=opposed,
        unopposed_count=n - opposed,
        zero_block_count=zero,
        single_block_count=single,
        open_seam_count=open_seam,
        closed_block_count=closed,
        isolation_pct=_pct(zero + single + open_seam, opposed) if opposed else None,
        closed_block_pct=_pct(closed, opposed) if opposed else None,
        avg_open_seam_cm=_mean_std(seam_cms, 1)[0],
    )
    back_row = sum(1 for p in plays if p.hitter_metrics is not None and p.hitter_metrics.is_back_row_approximate)
    return OffenseBucketSummary(
        play_count=n,
        back_row_count=back_row,
        back_row_pct=_pct(back_row, n),
        zones=zones,
        block_stats=block_stats,
    )


def compute_studio_session_summary(
    plays: list[PlayAnalysis], session_mode: Literal["drill", "match"]
) -> StudioSessionSummary:
    """Aggregate Set Window, zone distribution, and block isolation stats, keeping ours/opponent separate."""
    ours = [p for p in plays if p.team_side == "ours"]
    opponent = [p for p in plays if p.team_side == "opponent"]
    in_sys = [p for p in ours if p.system_state == "in_system"]
    out_sys = [p for p in ours if p.system_state == "out_of_system"]
    return StudioSessionSummary(
        session_mode=session_mode,
        total_plays=len(plays),
        ours_count=len(ours),
        opponent_count=len(opponent),
        in_system_count=len(in_sys),
        in_system_pct=_pct(len(in_sys), len(in_sys) + len(out_sys)),
        ours_all=_offense_bucket(ours, _OURS_ZONES),
        ours_in_system=_offense_bucket(in_sys, _OURS_ZONES),
        ours_out_of_system=_offense_bucket(out_sys, _OURS_ZONES),
        opponent=_offense_bucket(opponent, _OPP_ZONES),
    )
