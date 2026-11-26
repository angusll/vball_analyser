"""Two-pass GPU vision pipeline: YOLO11 ball/pose tracking + synthetic test predictor."""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vball_analyser.biomechanics import analyse_play_mechanics, classify_court_sides
from vball_analyser.calibration import classify_attack_zone, tape_y_at
from vball_analyser.config import DEFAULT_CONFIG
from vball_analyser.models import (
    BallDetection,
    NetCalibration,
    PlayAnalysis,
    PlayerTrack,
    Point2D,
    PoseFrame,
    VideoMetadata,
)
from vball_analyser.trajectory import fit_setting_path
from vball_analyser.video_io import extract_keyframe_jpeg, iter_video_frames

logger = logging.getLogger(__name__)


class FramePredictor(Protocol):
    """Interface for ball detection and multi-person pose tracking."""

    def detect_balls(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[BallDetection]: ...

    def track_poses(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[PlayerTrack]: ...

    def scan_coarse(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> tuple[list[BallDetection], list[tuple[int, float, tuple[float, float, float, float]]]]: ...


class YoloGpuPredictor:
    """Lazy-loading YOLO11 + ByteTrack predictor accelerated on CUDA when available."""

    def __init__(self, det_weights: str = "yolo11m.pt", pose_weights: str = "yolo11m-pose.pt") -> None:
        self._det_weights = det_weights
        self._pose_weights = pose_weights
        self._det_model = None
        self._pose_model = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._device

    def _ensure_models(self) -> tuple[object, object]:
        if self._det_model is None or self._pose_model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO11 models on device=%s", self.device)
            self._det_model = YOLO(self._det_weights)
            self._pose_model = YOLO(self._pose_weights)
        return self._det_model, self._pose_model

    def detect_balls(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[BallDetection]:
        del calibration
        if not frames:
            return []
        det_model, _ = self._ensure_models()
        detections: list[BallDetection] = []
        prev_gray: np.ndarray | None = None

        for frame_idx, ts, bgr in frames:
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            motion_map = cv2.absdiff(gray, prev_gray) if prev_gray is not None else None
            prev_gray = gray

            # Full-frame + overlapping upper-court tiles for small high-arc balls
            roi_h = max(64, int(0.76 * h))
            tile_w = int(0.62 * w)
            crops: list[tuple[int, int, np.ndarray]] = [
                (0, 0, bgr),
                (0, 0, bgr[:roi_h, :tile_w]),
                (w - tile_w, 0, bgr[:roi_h, w - tile_w :]),
            ]
            frame_cands: list[tuple[float, float, float]] = []
            for ox, oy, crop in crops:
                results = det_model.predict(
                    crop,
                    device=self.device,
                    classes=[32],  # COCO sports ball
                    imgsz=960 if ox > 0 or crop.shape[0] < h else 1280,
                    conf=0.16,
                    verbose=False,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        cx = ox + 0.5 * (xyxy[0] + xyxy[2])
                        cy = oy + 0.5 * (xyxy[1] + xyxy[3])
                        if motion_map is not None:
                            x1 = max(0, int(ox + xyxy[0]))
                            y1 = max(0, int(oy + xyxy[1]))
                            x2 = min(w, int(ox + xyxy[2]))
                            y2 = min(h, int(oy + xyxy[3]))
                            if x2 > x1 and y2 > y1:
                                m_score = float(np.mean(motion_map[y1:y2, x1:x2]))
                                if m_score < 2.5:
                                    conf *= 0.25
                        frame_cands.append((cx, cy, conf))

            # Simple spatial NMS merge within 18px
            frame_cands.sort(key=lambda c: -c[2])
            kept: list[tuple[float, float, float]] = []
            for cx, cy, conf in frame_cands:
                if all(math.hypot(cx - kx, cy - ky) > 18.0 for kx, ky, _ in kept):
                    kept.append((cx, cy, conf))
                    detections.append(
                        BallDetection(
                            frame_idx=frame_idx,
                            timestamp_s=round(ts, 4),
                            x=round(cx, 2),
                            y=round(cy, 2),
                            conf=round(conf, 3),
                        )
                    )
        return detections

    def track_poses(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[PlayerTrack]:
        if not frames:
            return []
        _, pose_model = self._ensure_models()
        # Reset ByteTrack state between independent play windows
        if getattr(pose_model, "predictor", None) is not None:
            pose_model.predictor = None

        tracks_map: dict[int, list[PoseFrame]] = {}
        for frame_idx, ts, bgr in frames:
            results = pose_model.track(
                bgr,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=1280,
                conf=0.25,
                verbose=False,
            )
            for r in results:
                if r.boxes is None or r.keypoints is None or r.boxes.id is None:
                    continue
                ids = r.boxes.id.int().tolist()
                boxes = r.boxes.xyxy.tolist()
                kps_xy = r.keypoints.xy.tolist()
                kps_conf = r.keypoints.conf.tolist() if r.keypoints.conf is not None else None
                for i, tid in enumerate(ids):
                    bbox = (
                        round(float(boxes[i][0]), 1),
                        round(float(boxes[i][1]), 1),
                        round(float(boxes[i][2]), 1),
                        round(float(boxes[i][3]), 1),
                    )
                    pts = []
                    for k_i, (kx, ky) in enumerate(kps_xy[i]):
                        kc = float(kps_conf[i][k_i]) if kps_conf is not None else 0.8
                        pts.append((round(float(kx), 1), round(float(ky), 1), round(kc, 3)))
                    tracks_map.setdefault(int(tid), []).append(
                        PoseFrame(frame_idx=frame_idx, timestamp_s=round(ts, 4), bbox=bbox, keypoints=pts)
                    )

        raw = [PlayerTrack(track_id=tid, court_side="attacking", frames=pfs) for tid, pfs in sorted(tracks_map.items())]
        return classify_court_sides(raw, calibration)

    def scan_coarse(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> tuple[list[BallDetection], list[tuple[int, float, tuple[float, float, float, float]]]]:
        del calibration
        if not frames:
            return [], []
        det_model, _ = self._ensure_models()
        balls: list[BallDetection] = []
        people: list[tuple[int, float, tuple[float, float, float, float]]] = []

        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            imgs = [item[2] for item in batch]
            results = det_model.predict(
                imgs,
                device=self.device,
                classes=[0, 32],
                imgsz=960,
                conf=0.22,
                verbose=False,
            )
            for (frame_idx, ts, _bgr), r in zip(batch, results, strict=True):
                if r.boxes is None:
                    continue
                cls_ids = r.boxes.cls.int().tolist()
                xyxys = r.boxes.xyxy.tolist()
                confs = r.boxes.conf.tolist()
                for c, box, conf in zip(cls_ids, xyxys, confs, strict=True):
                    if c == 32:
                        balls.append(
                            BallDetection(
                                frame_idx=frame_idx,
                                timestamp_s=round(ts, 4),
                                x=round(0.5 * (box[0] + box[2]), 2),
                                y=round(0.5 * (box[1] + box[3]), 2),
                                conf=round(float(conf), 3),
                            )
                        )
                    elif c == 0:
                        people.append((frame_idx, ts, (box[0], box[1], box[2], box[3])))
        return balls, people


class SyntheticColorPredictor:
    """Deterministic color-blob predictor for fast unit/E2E tests on synthetic clips."""

    def _find_balls(self, bgr: np.ndarray, frame_idx: int, ts: float) -> list[BallDetection]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 130, 170), (38, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[BallDetection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100.0 <= area <= 1800.0:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = float(m["m10"] / m["m00"])
                    cy = float(m["m01"] / m["m00"])
                    out.append(
                        BallDetection(
                            frame_idx=frame_idx,
                            timestamp_s=round(ts, 4),
                            x=round(cx, 2),
                            y=round(cy, 2),
                            conf=0.92,
                        )
                    )
        return out

    def _find_people(self, bgr: np.ndarray) -> list[tuple[int, tuple[float, float, float, float], list[tuple[float, float, float]]]]:
        # Match synthetic color roles in tests/conftest.py
        roles = [
            (1, np.array([90, 215, 130], dtype=np.uint8)),   # Green hitter
            (2, np.array([220, 180, 90], dtype=np.uint8)),   # Setter
            (3, np.array([110, 110, 220], dtype=np.uint8)),  # Blockers (split left/right)
            (4, np.array([140, 160, 140], dtype=np.uint8)),  # Passer
        ]
        found: list[tuple[int, tuple[float, float, float, float], list[tuple[float, float, float]]]] = []
        next_blocker_id = 30
        for base_id, col in roles:
            diff = np.abs(bgr.astype(np.int16) - col.astype(np.int16)).sum(axis=2)
            mask = (diff < 35).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Group nearby contours horizontally within 60px
            boxes = sorted([cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 60], key=lambda b: b[0])
            merged: list[list[int]] = []
            for x, y, w, h in boxes:
                if merged and x - (merged[-1][0] + merged[-1][2]) < 35:
                    mx1 = min(merged[-1][0], x)
                    my1 = min(merged[-1][1], y)
                    mx2 = max(merged[-1][0] + merged[-1][2], x + w)
                    my2 = max(merged[-1][1] + merged[-1][3], y + h)
                    merged[-1] = [mx1, my1, mx2 - mx1, my2 - my1]
                else:
                    merged.append([x, y, w, h])
            for x, y, w, h in merged:
                if h < 45:
                    continue
                tid = base_id if base_id != 3 else next_blocker_id
                if base_id == 3:
                    next_blocker_id += 1
                cx = x + 0.5 * w
                top = float(y)
                bot = float(y + h)
                sh_y = top + 0.24 * h
                hip_y = top + 0.54 * h
                kps = [(0.0, 0.0, 0.0) for _ in range(17)]
                kps[0] = (cx, top + 0.10 * h, 0.9)
                kps[5] = (cx - 0.22 * w, sh_y, 0.9)
                kps[6] = (cx + 0.22 * w, sh_y, 0.9)
                kps[7] = (cx - 0.28 * w, sh_y + 0.12 * h, 0.9)
                kps[8] = (cx + 0.20 * w, top + 0.08 * h, 0.9)
                kps[9] = (cx - 0.30 * w, sh_y + 0.22 * h, 0.9)
                kps[10] = (cx + 0.12 * w, top, 0.9)
                kps[11] = (cx - 0.16 * w, hip_y, 0.9)
                kps[12] = (cx + 0.16 * w, hip_y, 0.9)
                kps[15] = (cx - 0.18 * w, bot, 0.9)
                kps[16] = (cx + 0.18 * w, bot, 0.9)
                found.append((tid, (float(x), top, float(x + w), bot), kps))
        return found

    def detect_balls(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[BallDetection]:
        del calibration
        out: list[BallDetection] = []
        for frame_idx, ts, bgr in frames:
            out.extend(self._find_balls(bgr, frame_idx, ts))
        return out

    def track_poses(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[PlayerTrack]:
        tracks_map: dict[int, list[PoseFrame]] = {}
        for frame_idx, ts, bgr in frames:
            for tid, bbox, kps in self._find_people(bgr):
                tracks_map.setdefault(tid, []).append(
                    PoseFrame(frame_idx=frame_idx, timestamp_s=round(ts, 4), bbox=bbox, keypoints=kps)
                )
        raw = [PlayerTrack(track_id=tid, court_side="attacking", frames=pfs) for tid, pfs in sorted(tracks_map.items())]
        return classify_court_sides(raw, calibration)

    def scan_coarse(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> tuple[list[BallDetection], list[tuple[int, float, tuple[float, float, float, float]]]]:
        del calibration
        balls: list[BallDetection] = []
        people: list[tuple[int, float, tuple[float, float, float, float]]] = []
        for frame_idx, ts, bgr in frames:
            balls.extend(self._find_balls(bgr, frame_idx, ts))
            for _tid, bbox, _kps in self._find_people(bgr):
                people.append((frame_idx, ts, bbox))
        return balls, people


def scan_candidate_plays(
    video_path: Path,
    calibration: NetCalibration,
    predictor: FramePredictor,
    metadata: VideoMetadata,
) -> list[tuple[float, float]]:
    """Return candidate [start_s, end_s] play windows using duration-aware two-pass routing."""
    if metadata.duration_s <= DEFAULT_CONFIG.short_clip_max_sec:
        return [(0.0, metadata.duration_s)]

    coarse_frames = list(iter_video_frames(video_path, start_s=0.0, end_s=metadata.duration_s, sample_fps=10.0))
    if not coarse_frames:
        return [(0.0, min(metadata.duration_s, 5.0))]

    balls, people = predictor.scan_coarse(coarse_frames, calibration)
    # Identify moving balls above net tape (+35px margin)
    by_ts: dict[float, list[BallDetection]] = {}
    for b in balls:
        by_ts.setdefault(b.timestamp_s, []).append(b)

    sorted_ts = sorted(by_ts.keys())
    airborne_triggers: list[float] = []
    for idx, ts in enumerate(sorted_ts):
        for b in by_ts[ts]:
            if b.y <= tape_y_at(calibration, b.x) + 35.0:
                # Verify ball is actually moving compared to neighbors
                moving = False
                for n_idx in range(max(0, idx - 2), min(len(sorted_ts), idx + 3)):
                    if n_idx == idx:
                        continue
                    for nb in by_ts[sorted_ts[n_idx]]:
                        if 8.0 <= math.hypot(nb.x - b.x, nb.y - b.y) <= 260.0:
                            moving = True
                            break
                if moving:
                    airborne_triggers.append(ts)
                    break

    if not airborne_triggers:
        # Fallback: detect net-zone jumping people
        for _f, ts, bbox in people:
            cx = 0.5 * (bbox[0] + bbox[2])
            if bbox[1] <= tape_y_at(calibration, cx) + 20.0:
                airborne_triggers.append(ts)

    if not airborne_triggers:
        return []

    # Cluster triggers separated by <= 4.0s
    clusters: list[list[float]] = [[airborne_triggers[0]]]
    for ts in airborne_triggers[1:]:
        if ts - clusters[-1][-1] <= 4.0:
            clusters[-1].append(ts)
        else:
            clusters.append([ts])

    windows: list[tuple[float, float]] = []
    for cl in clusters:
        if len(cl) < 2 and len(clusters) > 1:
            continue
        t_center = cl[-1]
        w_start = max(0.0, round(t_center - 3.5, 2))
        w_end = min(metadata.duration_s, round(t_center + 1.5, 2))
        if w_end - w_start >= 1.2:
            if windows and w_start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], w_end))
            else:
                windows.append((w_start, w_end))
    return windows


def analyse_play_window(
    video_path: Path,
    play_index: int,
    start_s: float,
    end_s: float,
    calibration: NetCalibration,
    predictor: FramePredictor,
    metadata: VideoMetadata,
    session_dir: Path,
    override_hitter_track_id: int | None = None,
) -> tuple[PlayAnalysis, list[BallDetection]]:
    """Run Pass 2 high-FPS pose + ball analysis for a single play window."""
    play_id = f"play_{play_index:02d}_{uuid.uuid4().hex[:6]}"
    frames = list(iter_video_frames(video_path, start_s=start_s, end_s=end_s, sample_fps=None))
    if not frames:
        raise ValueError(f"No frames decoded in window [{start_s}, {end_s}]")

    ball_dets = predictor.detect_balls(frames, calibration)
    tracks = predictor.track_poses(frames, calibration)

    # Candidate hitter wrist points near net tape to help snap set termination
    cand_contacts: list[tuple[int, Point2D]] = []
    for t in tracks:
        if t.court_side != "attacking":
            continue
        for pf in t.frames:
            for w_idx in (9, 10):
                if w_idx < len(pf.keypoints) and pf.keypoints[w_idx][2] >= 0.25:
                    wx, wy = pf.keypoints[w_idx][0], pf.keypoints[w_idx][1]
                    if wy <= tape_y_at(calibration, wx) + 40.0:
                        cand_contacts.append((pf.frame_idx, Point2D(x=wx, y=wy)))

    set_traj = fit_setting_path(
        ball_dets,
        calibration=calibration,
        fps=metadata.fps,
        frame_height=metadata.height,
        candidate_contact_frames=cand_contacts,
    )

    setter_id, hitter_id, hitter_metrics, block_summary, keyframes = analyse_play_mechanics(
        tracks,
        set_traj,
        calibration=calibration,
        fps=metadata.fps,
        frame_shape=(metadata.height, metadata.width),
        override_hitter_track_id=override_hitter_track_id,
    )

    # Classify attack zone from contact point or hitter position
    if hitter_metrics is not None and hitter_metrics.contact_point is not None:
        zone_x = hitter_metrics.contact_point.x
    elif set_traj is not None and set_traj.arc_points:
        zone_x = set_traj.arc_points[-1].x
    else:
        zone_x = 0.5 * (calibration.left_point.x + calibration.right_point.x)
    attack_zone = classify_attack_zone(calibration, zone_x)

    # Extract keyframe JPEG thumbnails
    thumbs_dir = session_dir / "thumbnails"
    for km in keyframes:
        fname = f"{play_id}_{km.name}.jpg"
        try:
            extract_keyframe_jpeg(video_path, km.frame_idx, thumbs_dir / fname)
            km.thumbnail_rel_path = fname
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Thumbnail extraction failed for %s frame %d: %s", play_id, km.frame_idx, exc)

    start_frame = frames[0][0]
    end_frame = frames[-1][0]
    play = PlayAnalysis(
        play_id=play_id,
        index=play_index,
        start_time_s=round(start_s, 3),
        end_time_s=round(end_s, 3),
        start_frame=start_frame,
        end_frame=end_frame,
        fps=metadata.fps,
        attack_zone=attack_zone,
        setter_track_id=setter_id,
        hitter_track_id=hitter_id,
        set_trajectory=set_traj,
        hitter_metrics=hitter_metrics,
        block_summary=block_summary,
        keyframes=keyframes,
        tracks=tracks,
    )
    return play, ball_dets
