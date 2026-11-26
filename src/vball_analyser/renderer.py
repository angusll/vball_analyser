"""Anti-aliased OpenCV overlay renderer and PyAV H.264 MP4 play exporter."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from vball_analyser.biomechanics import _kp_valid
from vball_analyser.calibration import tape_y_at
from vball_analyser.models import NetCalibration, PlayAnalysis, PlayerTrack, PoseFrame
from vball_analyser.video_io import iter_video_frames

# Standard COCO-17 skeleton limb pairs
COCO_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def _draw_badge(
    img: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    border_bgr: tuple[int, int, int],
    text_bgr: tuple[int, int, int] = (248, 250, 252),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = anchor
    x = max(8, min(img.shape[1] - tw - 18, x))
    y = max(th + 14, min(img.shape[0] - 8, y))
    cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + 5), (15, 23, 42), -1)
    cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + 5), border_bgr, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, text_bgr, thickness, cv2.LINE_AA)


def _pose_at_frame(track: PlayerTrack, frame_idx: int) -> PoseFrame | None:
    best: PoseFrame | None = None
    best_d = 3
    for pf in track.frames:
        d = abs(pf.frame_idx - frame_idx)
        if d < best_d:
            best_d = d
            best = pf
    return best


def draw_play_overlays(
    bgr: np.ndarray,
    frame_idx: int,
    play: PlayAnalysis,
    cal: NetCalibration,
) -> np.ndarray:
    """Render clean baseline court overlays onto a copy of a single BGR frame."""
    canvas = bgr.copy()

    # 1. Net top tape reference
    lx, ly = int(round(cal.left_point.x)), int(round(cal.left_point.y))
    rx, ry = int(round(cal.right_point.x)), int(round(cal.right_point.y))
    cv2.line(canvas, (lx, ly), (rx, ry), (248, 250, 252), 2, cv2.LINE_AA)
    cv2.circle(canvas, (lx, ly), 5, (56, 189, 248), -1, cv2.LINE_AA)
    cv2.circle(canvas, (rx, ry), 5, (56, 189, 248), -1, cv2.LINE_AA)

    # 2. Setting path progressive arc
    traj = play.set_trajectory
    if traj is not None and traj.arc_points and frame_idx >= traj.release_frame:
        visible = [p for p in traj.arc_points if p.frame_idx <= frame_idx]
        if len(visible) >= 2:
            for a, b in zip(visible, visible[1:], strict=False):
                t = (b.frame_idx - traj.release_frame) / max(1, traj.contact_frame - traj.release_frame)
                col = (
                    int(248 * (1 - t) + 21 * t),
                    int(189 * (1 - t) + 204 * t),
                    int(56 * (1 - t) + 250 * t),
                )
                cv2.line(
                    canvas,
                    (int(round(a.x)), int(round(a.y))),
                    (int(round(b.x)), int(round(b.y))),
                    col,
                    3,
                    cv2.LINE_AA,
                )
        for idx, pt in enumerate(visible):
            if idx % 4 == 0 or idx == len(visible) - 1:
                cv2.circle(canvas, (int(round(pt.x)), int(round(pt.y))), 4, (248, 224, 71), -1, cv2.LINE_AA)
        if frame_idx >= traj.peak_frame:
            _draw_badge(
                canvas,
                f"Set Arc {traj.tempo_s:.2f}s",
                (int(round(traj.peak_point.x)) - 40, int(round(traj.peak_point.y)) - 16),
                (248, 189, 56),
            )

    # 3. Active hitter smoothed pose skeleton
    if play.hitter_track_id is not None:
        hitter = next((t for t in play.tracks if t.track_id == play.hitter_track_id), None)
        if hitter is not None:
            pf = _pose_at_frame(hitter, frame_idx)
            if pf is not None:
                dom = play.hitter_metrics.dominant_hand if play.hitter_metrics else "right"
                hit_arm_edges = {(6, 8), (8, 10)} if dom == "right" else {(5, 7), (7, 9)}
                for a_i, b_i in COCO_EDGES:
                    ka, kb = pf.keypoints[a_i], pf.keypoints[b_i]
                    if _kp_valid(ka) and _kp_valid(kb):
                        col = (21, 204, 250) if (a_i, b_i) in hit_arm_edges else (128, 222, 74)
                        cv2.line(
                            canvas,
                            (int(round(ka[0])), int(round(ka[1]))),
                            (int(round(kb[0])), int(round(kb[1]))),
                            col,
                            3,
                            cv2.LINE_AA,
                        )
                for kp in pf.keypoints:
                    if _kp_valid(kp):
                        cv2.circle(canvas, (int(round(kp[0])), int(round(kp[1]))), 4, (128, 222, 74), -1, cv2.LINE_AA)

    # 4. Contact point + posture + blocker callouts around contact
    con_km = next((k for k in play.keyframes if k.name == "contact"), None)
    con_f = con_km.frame_idx if con_km is not None else None
    if con_f is not None and abs(frame_idx - con_f) <= 3:
        hm = play.hitter_metrics
        if hm is not None and hm.contact_point is not None:
            cx, cy = int(round(hm.contact_point.x)), int(round(hm.contact_point.y))
            ty = int(round(tape_y_at(cal, hm.contact_point.x)))
            cv2.circle(canvas, (cx, cy), 11, (21, 204, 250), 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, ty), (21, 204, 250), 2, cv2.LINE_AA)
            h_txt = (
                f"+{hm.contact_height_cm:.0f}cm ({hm.contact_height_px:+.0f}px)"
                if hm.contact_height_cm is not None
                else f"{hm.contact_height_px:+.0f}px vs tape"
            )
            _draw_badge(canvas, h_txt, (cx + 16, cy), (21, 204, 250))
            if hm.arm_extension_deg is not None:
                _draw_badge(
                    canvas,
                    f"Arm {hm.arm_extension_deg:.0f} deg | {hm.jump_timing_label}",
                    (cx + 16, cy + 28),
                    (128, 222, 74),
                )

        # Opposing blockers & seam
        for blk in play.block_summary.blockers:
            bx1, by1, bx2, by2 = [int(round(v)) for v in blk.bbox]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (133, 113, 251), 2, cv2.LINE_AA)
            rx_p, ry_p = int(round(blk.top_reach_point.x)), int(round(blk.top_reach_point.y))
            cv2.circle(canvas, (rx_p, ry_p), 5, (133, 113, 251), -1, cv2.LINE_AA)
        if play.block_summary.seam_open and play.block_summary.seam_interval_x is not None:
            sx1, sx2 = [int(round(v)) for v in play.block_summary.seam_interval_x]
            sy = int(round(tape_y_at(cal, 0.5 * (sx1 + sx2)) - 14))
            cv2.line(canvas, (sx1, sy), (sx2, sy), (133, 113, 251), 3, cv2.LINE_AA)
            _draw_badge(canvas, "Open Block Seam", (sx1, sy - 14), (133, 113, 251))

    return canvas


def export_annotated_play_mp4(
    video_path: Path,
    play: PlayAnalysis,
    calibration: NetCalibration,
    out_path: Path,
    freeze_at_contact_s: float = 1.5,
) -> Path:
    """Render an annotated H.264 MP4 clip for a play with optional contact freeze-frame."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(iter_video_frames(video_path, start_s=play.start_time_s, end_s=play.end_time_s))
    if not frames:
        raise ValueError("No frames available to export play.")

    fps_int = max(1, int(round(play.fps or 30.0)))
    h, w = frames[0][2].shape[:2]
    out_w = w - (w % 2)
    out_h = h - (h % 2)

    con_km = next((k for k in play.keyframes if k.name == "contact"), None)
    con_frame = con_km.frame_idx if con_km is not None else frames[-1][0]
    freeze_frames = max(0, int(round(freeze_at_contact_s * fps_int)))
    freeze_inserted = False

    with av.open(str(out_path), mode="w", format="mp4") as out_container:
        stream = out_container.add_stream(
            "libx264",
            rate=Fraction(fps_int, 1),
            options={"preset": "veryfast", "crf": "20", "movflags": "+faststart"},
        )
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"

        pts = 0
        for idx, (frame_idx, _ts, bgr) in enumerate(frames):
            annotated = draw_play_overlays(bgr, frame_idx, play, calibration)
            if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                annotated = annotated[:out_h, :out_w]

            is_contact = (not freeze_inserted) and (
                frame_idx >= con_frame or idx == len(frames) - 1
            )
            repeat = 1 + (freeze_frames if (is_contact and freeze_frames > 0) else 0)
            if is_contact:
                freeze_inserted = True

            for _ in range(repeat):
                av_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                av_frame = av_frame.reformat(format="yuv420p")
                av_frame.pts = pts
                pts += 1
                for pkt in stream.encode(av_frame):
                    out_container.mux(pkt)

        for pkt in stream.encode():
            out_container.mux(pkt)

    return out_path
