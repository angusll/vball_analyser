"""Shared pytest fixtures, including synthetic baseline volleyball video generator."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest


def _draw_person(
    img: np.ndarray,
    hip_x: float,
    hip_y: float,
    scale: float,
    color: tuple[int, int, int],
    arm_lift_deg: float = 150.0,
    left_arm_deg: float = 40.0,
    torso_tilt_deg: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Draw a humanoid silhouette & return ground-truth 2D joint positions."""
    rad_tilt = math.radians(torso_tilt_deg)
    torso_len = 52.0 * scale
    sh_x = hip_x + torso_len * math.sin(rad_tilt)
    sh_y = hip_y - torso_len * math.cos(rad_tilt)
    head_x = sh_x + 18.0 * scale * math.sin(rad_tilt)
    head_y = sh_y - 18.0 * scale * math.cos(rad_tilt)

    # Legs down
    l_knee = (hip_x - 10 * scale, hip_y + 34 * scale)
    r_knee = (hip_x + 10 * scale, hip_y + 34 * scale)
    l_ank = (hip_x - 14 * scale, hip_y + 68 * scale)
    r_ank = (hip_x + 14 * scale, hip_y + 68 * scale)

    # Shoulders
    l_sh = (sh_x - 14 * scale, sh_y)
    r_sh = (sh_x + 14 * scale, sh_y)

    # Right arm (upper + forearm along arm_lift_deg from downward vertical)
    r_rad = math.radians(arm_lift_deg)
    upper_len = 28 * scale
    fore_len = 28 * scale
    r_elb = (r_sh[0] + upper_len * math.sin(r_rad) * 0.25, r_sh[1] - upper_len * math.cos(r_rad))
    r_wri = (r_elb[0] + fore_len * math.sin(r_rad) * 0.15, r_elb[1] - fore_len * math.cos(r_rad))

    # Left arm
    l_rad = math.radians(left_arm_deg)
    l_elb = (l_sh[0] - upper_len * math.sin(l_rad), l_sh[1] + upper_len * math.cos(l_rad))
    l_wri = (l_elb[0] - fore_len * 0.5 * math.sin(l_rad), l_elb[1] + fore_len * 0.5 * math.cos(l_rad))

    # Draw thick limbs for realism
    cv2.line(img, (int(hip_x), int(hip_y)), (int(sh_x), int(sh_y)), color, int(14 * scale))
    cv2.circle(img, (int(head_x), int(head_y)), int(11 * scale), color, -1)
    for a, b in [
        ((hip_x, hip_y), l_knee),
        (l_knee, l_ank),
        ((hip_x, hip_y), r_knee),
        (r_knee, r_ank),
        (r_sh, r_elb),
        (r_elb, r_wri),
        (l_sh, l_elb),
        (l_elb, l_wri),
    ]:
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, int(7 * scale))

    return {
        "head": (head_x, head_y),
        "r_sh": r_sh,
        "r_elb": r_elb,
        "r_wri": r_wri,
        "l_sh": l_sh,
        "l_elb": l_elb,
        "l_wri": l_wri,
        "hip": (hip_x, hip_y),
        "l_ank": l_ank,
        "r_ank": r_ank,
    }


def make_synthetic_baseline_video(
    out_path: Path,
    duration_s: float = 4.0,
    fps: int = 30,
    tilt_deg: float = 2.0,
    add_stray_ball: bool = True,
    codec: str = "libx264",
    rally_windows: list[tuple[float, float]] | None = None,
) -> dict:
    """Render a synthetic baseline volleyball clip with PyAV and return ground truth."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    total_frames = int(round(duration_s * fps))
    if rally_windows is None:
        rally_windows = [(0.5, min(duration_s - 0.3, 2.2))]

    left_pt = (192.0, 320.0 - math.tan(math.radians(tilt_deg)) * 448.0)
    right_pt = (1088.0, 320.0 + math.tan(math.radians(tilt_deg)) * 448.0)

    def tape_y(x: float) -> float:
        t = (x - left_pt[0]) / (right_pt[0] - left_pt[0])
        return left_pt[1] + t * (right_pt[1] - left_pt[1])

    pix_fmt = "yuv420p"
    fmt = "avi" if out_path.suffix.lower() == ".avi" else "mp4"
    enc_name = "mpeg4" if fmt == "avi" else codec

    gt_rallies = []
    for rw_idx, (r_start, r_end) in enumerate(rally_windows):
        f_rel = int(round(r_start * fps))
        f_con = int(round((r_start + 1.0) * fps))
        f_end = int(round(r_end * fps))
        gt_rallies.append(
            {
                "index": rw_idx + 1,
                "release_frame": f_rel,
                "contact_frame": f_con,
                "end_frame": f_end,
                "setter_x": 790.0,
                "hitter_x": 340.0,
                "contact_y": tape_y(340.0) - 56.0,
            }
        )

    with av.open(str(out_path), mode="w", format=fmt) as container:
        stream = container.add_stream(enc_name, rate=Fraction(fps, 1))
        stream.width = w
        stream.height = h
        stream.pix_fmt = pix_fmt

        for frame_idx in range(total_frames):
            ts = frame_idx / fps
            img = np.full((h, w, 3), (38, 44, 56), dtype=np.uint8)
            # Court floor perspective
            pts_far = np.array([[280, 415], [1000, 415], [1088, 495], [192, 495]], dtype=np.int32)
            pts_near = np.array([[192, 495], [1088, 495], [1230, 710], [50, 710]], dtype=np.int32)
            cv2.fillPoly(img, [pts_far], (95, 65, 42))
            cv2.fillPoly(img, [pts_near], (62, 112, 175))
            cv2.polylines(img, [pts_near], True, (240, 240, 240), 2)
            # 3m line
            cv2.line(img, (150, 565), (1130, 565), (240, 240, 240), 2)

            # Net posts, mesh, antennas, and bright white top tape
            lx, ly = int(round(left_pt[0])), int(round(left_pt[1]))
            rx, ry = int(round(right_pt[0])), int(round(right_pt[1]))
            cv2.line(img, (lx, ly - 55), (lx, ly + 175), (210, 210, 210), 6)
            cv2.line(img, (rx, ry - 55), (rx, ry + 175), (210, 210, 210), 6)
            cv2.line(img, (lx + 10, ly - 75), (lx + 10, ly + 150), (40, 40, 235), 4)
            cv2.line(img, (rx - 10, ry - 75), (rx - 10, ry + 150), (40, 40, 235), 4)
            cv2.line(img, (lx, ly), (rx, ry), (255, 255, 255), 7)

            # Active rally animation if inside window
            active = None
            for gr in gt_rallies:
                if gr["release_frame"] - 10 <= frame_idx <= gr["end_frame"]:
                    active = gr
                    break

            # Opposing blockers (far side, ankles near net base ~480)
            b1_jump = 0.0
            b2_jump = 0.0
            if active is not None:
                dt_con = abs(frame_idx - active["contact_frame"]) / fps
                if dt_con < 0.35:
                    b1_jump = 58.0 * max(0.0, 1.0 - (dt_con / 0.35) ** 2)
                    b2_jump = 50.0 * max(0.0, 1.0 - (dt_con / 0.35) ** 2)

            _draw_person(img, 315.0, 405.0 - b1_jump, 0.85, (110, 110, 220), arm_lift_deg=175.0)
            _draw_person(img, 425.0, 410.0 - b2_jump, 0.85, (110, 110, 220), arm_lift_deg=172.0)

            # Idle back-row passer (far down near baseline)
            _draw_person(img, 960.0, 610.0, 1.15, (140, 160, 140), arm_lift_deg=25.0)

            # Setter
            _draw_person(img, 790.0, 465.0, 1.0, (220, 180, 90), arm_lift_deg=160.0)

            # Hitter jump + ball parabola
            hitter_jump = 0.0
            arm_angle = 35.0
            if active is not None:
                dt_con = (frame_idx - active["contact_frame"]) / fps
                if -0.45 <= dt_con <= 0.45:
                    hitter_jump = 95.0 * max(0.0, 1.0 - (dt_con / 0.45) ** 2)
                if -0.35 <= dt_con < -0.10:
                    arm_angle = 95.0
                elif -0.10 <= dt_con <= 0.15:
                    arm_angle = 170.0
                elif dt_con > 0.15:
                    arm_angle = 45.0

            _draw_person(img, 340.0, 495.0 - hitter_jump, 1.05, (90, 215, 130), arm_lift_deg=arm_angle, torso_tilt_deg=-8.0)

            # Set ball along gravity parabola
            if active is not None and active["release_frame"] <= frame_idx <= active["end_frame"]:
                if frame_idx <= active["contact_frame"]:
                    tau = (frame_idx - active["release_frame"]) / fps
                    T = (active["contact_frame"] - active["release_frame"]) / fps
                    x0, y0 = active["setter_x"], 360.0
                    x1, y1 = active["hitter_x"], active["contact_y"]
                    g = 900.0
                    vx = (x1 - x0) / max(T, 0.1)
                    vy = (y1 - y0 - 0.5 * g * T * T) / max(T, 0.1)
                    bx = x0 + vx * tau
                    by = y0 + vy * tau + 0.5 * g * tau * tau
                else:
                    tau_hit = (frame_idx - active["contact_frame"]) / fps
                    bx = active["hitter_x"] - 140.0 * tau_hit
                    by = active["contact_y"] + 480.0 * tau_hit
                cv2.circle(img, (int(round(bx)), int(round(by))), 13, (60, 225, 250), -1)

            if add_stray_ball:
                # Stationary stray ball in bottom-right ball cart
                cv2.circle(img, (1165, 645), 12, (60, 225, 250), -1)

            av_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            av_frame.pts = frame_idx
            for pkt in stream.encode(av_frame):
                container.mux(pkt)

        for pkt in stream.encode():
            container.mux(pkt)

    return {
        "width": w,
        "height": h,
        "fps": fps,
        "frame_count": total_frames,
        "duration_s": duration_s,
        "left_pt": left_pt,
        "right_pt": right_pt,
        "rallies": gt_rallies,
    }


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> tuple[Path, dict]:
    out_path = tmp_path / "synthetic_baseline.mp4"
    gt = make_synthetic_baseline_video(out_path, duration_s=3.5, fps=30, tilt_deg=2.5)
    return out_path, gt
