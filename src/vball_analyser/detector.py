"""Two-pass GPU vision pipeline: YOLO11 ball/pose tracking + synthetic test predictor."""

from __future__ import annotations

import bisect
import importlib.util
import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import cv2
import numpy as np

from vball_analyser.biomechanics import (
    _jump_impulse_px,
    analyse_play_full,
    build_contact_candidates,
    classify_court_sides,
)
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
from vball_analyser.video_io import (
    extract_keyframe_jpeg,
    iter_video_frame_batches,
    iter_video_frames,
    should_use_nvdec,
)

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


def detect_inference_device(torch_module: object | None = None) -> str:
    """Auto-detect the best available accelerator: CUDA (NVIDIA) -> XPU (Intel) -> MPS (Apple) -> CPU.

    Honors the ``VBALL_DEVICE`` environment variable when set (e.g. ``cpu``, ``cuda:0``, ``mps``, ``xpu:0``).
    """
    override = os.environ.get("VBALL_DEVICE", "").strip()
    if override:
        return override

    if torch_module is None:
        import torch as torch_module

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda:0"

    xpu = getattr(torch_module, "xpu", None)
    if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
        return "xpu:0"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"

    return "cpu"


def _default_batch_size_for_device(device: str) -> int:
    if device.startswith("cuda"):
        return 16
    if device.startswith(("xpu", "mps")):
        return 8
    return 4


def _find_unique_frame_indices(
    frames: list[tuple[int, float, np.ndarray]],
    diff_thresh: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Return (unique_frame_positions, frame_pos_to_unique_idx) skipping VFR duplicate frames."""
    unique_pos: list[int] = []
    pos_to_unique: list[int] = []
    last_bgr: np.ndarray | None = None
    for i, (_fi, _ts, bgr) in enumerate(frames):
        if last_bgr is None:
            unique_pos.append(i)
            pos_to_unique.append(0)
            last_bgr = bgr
        else:
            diff = cv2.absdiff(bgr[::8, ::8], last_bgr[::8, ::8])
            if float(np.mean(diff)) >= diff_thresh or int(np.count_nonzero(diff >= 25)) >= 12:
                unique_pos.append(i)
                pos_to_unique.append(len(unique_pos) - 1)
                last_bgr = bgr
            else:
                pos_to_unique.append(len(unique_pos) - 1)
    return unique_pos, pos_to_unique


def _patch_motion_score(
    bgr: np.ndarray,
    ref_bgr: np.ndarray | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Return mean grayscale absdiff inside a box, or +inf when no reference frame exists."""
    if ref_bgr is None:
        return float("inf")
    h, w = bgr.shape[:2]
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(w, int(x2))
    iy2 = min(h, int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return float("inf")
    cur_patch = cv2.cvtColor(bgr[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2GRAY)
    prev_patch = cv2.cvtColor(ref_bgr[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(cur_patch, prev_patch)))


def _detect_high_arc_motion_balls(
    frames: list[tuple[int, float, np.ndarray]],
    calibration: NetCalibration,
    existing_by_frame: dict[int, list[tuple[float, float]]],
) -> list[BallDetection]:
    """Recover compact moving ball blobs strictly above the net tape when YOLO drops out."""
    if len(frames) < 12:
        return []
    h, w = frames[0][2].shape[:2]
    upper_y = min(h, int(min(calibration.left_point.y, calibration.right_point.y) - 10.0))
    if upper_y < 40 or w < 64:
        return []

    # Sample up to 25 evenly spaced frames to build a clean static upper-wall background
    step = max(1, len(frames) // 25)
    sample_rois = [frames[i][2][:upper_y] for i in range(0, len(frames), step)][:25]
    if len(sample_rois) < 6:
        return []
    bg = np.median(np.stack(sample_rois, axis=0), axis=0).astype(np.uint8)

    # Skip supplement on handheld/panning cameras where the upper background shifts heavily
    bg_diffs = [float(np.mean(cv2.absdiff(roi[::4, ::4], bg[::4, ::4]))) for roi in sample_rois]
    if float(np.median(bg_diffs)) > 6.0:
        return []

    kernel = np.ones((3, 3), dtype=np.uint8)
    extra: list[BallDetection] = []
    for frame_idx, ts, bgr in frames:
        diff = cv2.absdiff(bgr[:upper_y], bg)
        gray_d = np.max(diff, axis=2)
        _, mask = cv2.threshold(gray_d, 22, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        existing = existing_by_frame.get(frame_idx, [])
        frame_blobs: list[tuple[float, float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (30.0 <= area <= 1000.0):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect > 2.2:
                continue
            fill = area / float(bw * bh)
            if fill < 0.35:
                continue
            cx = bx + 0.5 * bw
            cy = by + 0.5 * bh
            if any(math.hypot(cx - ex, cy - ey) <= 18.0 for ex, ey in existing):
                continue
            # Prefer compact, ball-sized blobs (higher fill & closer to 1:1 aspect)
            quality = fill / aspect
            frame_blobs.append((quality, cx, cy))

        frame_blobs.sort(key=lambda item: -item[0])
        for _q, cx, cy in frame_blobs[:4]:
            extra.append(
                BallDetection(
                    frame_idx=frame_idx,
                    timestamp_s=round(ts, 4),
                    x=round(cx, 2),
                    y=round(cy, 2),
                    conf=0.28,
                )
            )
    return extra


class YoloGpuPredictor:
    """Lazy-loading YOLO11 + ByteTrack predictor with cross-platform GPU auto-detection (CUDA/XPU/MPS/CPU)."""

    def __init__(
        self,
        det_weights: str = "yolo11m.pt",
        pose_weights: str = "yolo11m-pose.pt",
        batch_size: int | None = None,
        prefer_engine: bool = True,
        device: str | None = None,
        models_dir: Path | None = None,
    ) -> None:
        self._det_weights = det_weights
        self._pose_weights = pose_weights
        self._explicit_batch_size = max(1, int(batch_size)) if batch_size is not None else None
        self._prefer_engine = prefer_engine
        self._models_dir = (models_dir or DEFAULT_CONFIG.models_dir).resolve()
        self._det_model = None
        self._pose_model = None
        self._det_resolved: str | None = None
        self._pose_resolved: str | None = None
        self._device: str | None = device

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = detect_inference_device()
        return self._device

    @property
    def batch_size(self) -> int:
        if self._explicit_batch_size is not None:
            return self._explicit_batch_size
        return _default_batch_size_for_device(self.device)

    @property
    def _batch_size(self) -> int:
        return self.batch_size

    @property
    def _is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def _resolve_weights(self, weights: str) -> str:
        raw = Path(weights)
        if raw.parent == Path(".") and not raw.is_absolute():
            self._models_dir.mkdir(parents=True, exist_ok=True)
            base = self._models_dir / raw.name
        else:
            base = raw

        if not (self._prefer_engine and base.suffix == ".pt"):
            return str(base)

        if self._is_cuda and importlib.util.find_spec("tensorrt") is not None:
            engine_path = base.with_suffix(".engine")
            if engine_path.is_file():
                return str(engine_path)
        elif self.device == "mps":
            coreml_path = base.with_suffix(".mlpackage")
            if coreml_path.is_dir():
                return str(coreml_path)
        elif (self.device == "cpu" or self.device.startswith("xpu")) and importlib.util.find_spec("openvino") is not None:
            ov_dir = base.with_name(f"{base.stem}_openvino_model")
            if ov_dir.is_dir():
                return str(ov_dir)
        return str(base)

    def _base_weights_path(self, weights: str) -> str:
        raw = Path(weights)
        if raw.parent == Path(".") and not raw.is_absolute():
            self._models_dir.mkdir(parents=True, exist_ok=True)
            return str(self._models_dir / raw.name)
        return str(raw)

    def _infer_kwargs(self, resolved_weights: str) -> dict[str, object]:
        if self._is_cuda and resolved_weights.endswith(".pt"):
            return {"quantize": "fp16"}
        return {}

    def _load_yolo_model(self, yolo_cls: Any, weights: str, task: str) -> tuple[object, str]:
        resolved = self._resolve_weights(weights)
        fallback = self._base_weights_path(weights)
        try:
            return yolo_cls(resolved, task=task), resolved
        except Exception as exc:
            if resolved != fallback:
                logger.warning(
                    "Failed to load compiled model %s (%s); falling back to %s",
                    resolved,
                    exc,
                    fallback,
                )
                return yolo_cls(fallback, task=task), fallback
            raise

    def _ensure_models(self) -> tuple[object, object]:
        if self._det_model is None or self._pose_model is None:
            import torch
            from ultralytics import YOLO

            if self._is_cuda:
                torch.backends.cudnn.benchmark = True

            self._det_model, self._det_resolved = self._load_yolo_model(YOLO, self._det_weights, "detect")
            self._pose_model, self._pose_resolved = self._load_yolo_model(YOLO, self._pose_weights, "pose")
            logger.info(
                "Loaded YOLO11 models on device=%s (det=%s, pose=%s, batch=%d)",
                self.device,
                self._det_resolved,
                self._pose_resolved,
                self._batch_size,
            )
        return self._det_model, self._pose_model

    def warmup(self) -> None:
        """Preload detector & pose models and run a dummy forward pass to initialize GPU/TensorRT."""
        t0 = time.perf_counter()
        det_model, pose_model = self._ensure_models()
        dummy = np.zeros((576, 1024, 3), dtype=np.uint8)
        det_kw = self._infer_kwargs(self._det_resolved or self._det_weights)
        pose_kw = self._infer_kwargs(self._pose_resolved or self._pose_weights)
        det_model.predict(
            [dummy],
            device=self.device,
            classes=[32],
            imgsz=1280,
            conf=0.16,
            verbose=False,
            **det_kw,
        )
        pose_model.track(
            [dummy],
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=1280,
            conf=0.25,
            verbose=False,
            **pose_kw,
        )
        predictor = getattr(pose_model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for tr in predictor.trackers:
                tr.reset()
        logger.info(
            "Warmed up YOLO11 models on device=%s in %.2fs",
            self.device,
            time.perf_counter() - t0,
        )

    def detect_balls_and_poses(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> tuple[list[BallDetection], list[PlayerTrack]]:
        """Run ball detection and pose tracking concurrently on separate CUDA streams when on GPU."""
        if not frames:
            return [], []
        self._ensure_models()
        if not self._is_cuda:
            return self.detect_balls(frames, calibration), self.track_poses(frames, calibration)

        import torch

        s_det = torch.cuda.Stream()
        s_pose = torch.cuda.Stream()

        def _run_det() -> list[BallDetection]:
            with torch.cuda.stream(s_det):
                res = self.detect_balls(frames, calibration)
            s_det.synchronize()
            return res

        def _run_pose() -> list[PlayerTrack]:
            with torch.cuda.stream(s_pose):
                res = self.track_poses(frames, calibration)
            s_pose.synchronize()
            return res

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_det = pool.submit(_run_det)
            f_pose = pool.submit(_run_pose)
            return f_det.result(), f_pose.result()

    def detect_balls(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[BallDetection]:
        if not frames:
            return []
        det_model, _ = self._ensure_models()
        infer_kw = self._infer_kwargs(self._det_resolved or self._det_weights)

        # Deduplicate near-identical VFR frames so YOLO only runs on visually distinct frames
        unique_pos, pos_to_unique = _find_unique_frame_indices(frames)
        unique_frames = [frames[i] for i in unique_pos]
        n_uniq = len(unique_frames)

        # Per-unique-frame raw candidates: (ox, oy, x1, y1, x2, y2, conf)
        per_unique_raw: list[list[tuple[int, int, float, float, float, float, float]]] = [[] for _ in range(n_uniq)]

        # 1) Batched full-frame pass at imgsz=1280 on unique frames only
        full_imgs = [item[2] for item in unique_frames]
        for start in range(0, n_uniq, self._batch_size):
            batch_imgs = full_imgs[start : start + self._batch_size]
            results = det_model.predict(
                batch_imgs,
                device=self.device,
                classes=[32],  # COCO sports ball
                imgsz=1280,
                conf=0.16,
                verbose=False,
                **infer_kw,
            )
            for offset, r in enumerate(results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxys = r.boxes.xyxy.tolist()
                confs = r.boxes.conf.tolist()
                bucket = per_unique_raw[start + offset]
                for box, conf in zip(xyxys, confs, strict=True):
                    bucket.append((0, 0, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))

        # 2) Conditional upper-court tiles at imgsz=960 ONLY on frames lacking a moving high-conf ball
        tile_meta: list[tuple[int, int, int]] = []
        tile_imgs: list[np.ndarray] = []
        for u_i, (_frame_idx, _ts, bgr) in enumerate(unique_frames):
            ref_bgr = unique_frames[u_i - 1][2] if u_i > 0 else None
            has_moving_ball = any(
                conf >= 0.35 and _patch_motion_score(bgr, ref_bgr, bx1, by1, bx2, by2) >= 2.5
                for _ox, _oy, bx1, by1, bx2, by2, conf in per_unique_raw[u_i]
            )
            if has_moving_ball:
                continue
            h, w = bgr.shape[:2]
            roi_h = max(64, int(0.76 * h))
            tile_w = int(0.62 * w)
            tile_meta.append((u_i, 0, 0))
            tile_imgs.append(bgr[:roi_h, :tile_w])
            tile_meta.append((u_i, w - tile_w, 0))
            tile_imgs.append(bgr[:roi_h, w - tile_w :])

        for start in range(0, len(tile_imgs), self._batch_size):
            batch_imgs = tile_imgs[start : start + self._batch_size]
            batch_meta = tile_meta[start : start + self._batch_size]
            results = det_model.predict(
                batch_imgs,
                device=self.device,
                classes=[32],  # COCO sports ball
                imgsz=960,
                conf=0.16,
                verbose=False,
                **infer_kw,
            )
            for (u_i, ox, oy), r in zip(batch_meta, results, strict=True):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxys = r.boxes.xyxy.tolist()
                confs = r.boxes.conf.tolist()
                bucket = per_unique_raw[u_i]
                for box, conf in zip(xyxys, confs, strict=True):
                    bucket.append((ox, oy, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))

        # 3) Crop-only motion damping against prior unique frame + spatial NMS within 18px
        unique_kept: list[list[tuple[float, float, float]]] = [[] for _ in range(n_uniq)]
        for u_i, (_frame_idx, _ts, bgr) in enumerate(unique_frames):
            raw_cands = per_unique_raw[u_i]
            if not raw_cands:
                continue
            ref_bgr = unique_frames[u_i - 1][2] if u_i > 0 else None
            frame_cands: list[tuple[float, float, float]] = []
            for ox, oy, bx1, by1, bx2, by2, conf in raw_cands:
                cx = ox + 0.5 * (bx1 + bx2)
                cy = oy + 0.5 * (by1 + by2)
                if _patch_motion_score(bgr, ref_bgr, ox + bx1, oy + by1, ox + bx2, oy + by2) < 2.5:
                    conf *= 0.25
                frame_cands.append((cx, cy, conf))

            frame_cands.sort(key=lambda c: -c[2])
            kept: list[tuple[float, float, float]] = []
            for cx, cy, conf in frame_cands:
                if all(math.hypot(cx - kx, cy - ky) > 18.0 for kx, ky, _ in kept):
                    kept.append((cx, cy, conf))
            unique_kept[u_i] = kept

        # Expand unique-frame YOLO detections back to all frame indices (including VFR duplicates)
        detections: list[BallDetection] = []
        existing_by_unique_frame: dict[int, list[tuple[float, float]]] = {}
        for u_i, (u_frame_idx, _u_ts, _u_bgr) in enumerate(unique_frames):
            if unique_kept[u_i]:
                existing_by_unique_frame[u_frame_idx] = [(cx, cy) for cx, cy, _ in unique_kept[u_i]]

        for f_i, (frame_idx, ts, _bgr) in enumerate(frames):
            for cx, cy, conf in unique_kept[pos_to_unique[f_i]]:
                detections.append(
                    BallDetection(
                        frame_idx=frame_idx,
                        timestamp_s=round(ts, 4),
                        x=round(cx, 2),
                        y=round(cy, 2),
                        conf=round(conf, 3),
                    )
                )

        # 4) Supplement high-arc dropouts above the net tape on unique frames and expand to duplicates
        motion_unique = _detect_high_arc_motion_balls(unique_frames, calibration, existing_by_unique_frame)
        motion_by_uframe: dict[int, list[BallDetection]] = {}
        for md in motion_unique:
            motion_by_uframe.setdefault(md.frame_idx, []).append(md)
        for f_i, (frame_idx, ts, _bgr) in enumerate(frames):
            u_frame_idx = unique_frames[pos_to_unique[f_i]][0]
            for md in motion_by_uframe.get(u_frame_idx, []):
                detections.append(
                    BallDetection(
                        frame_idx=frame_idx,
                        timestamp_s=round(ts, 4),
                        x=md.x,
                        y=md.y,
                        conf=md.conf,
                    )
                )

        detections.sort(key=lambda d: (d.frame_idx, -d.conf))
        return detections

    def track_poses(
        self, frames: list[tuple[int, float, np.ndarray]], calibration: NetCalibration
    ) -> list[PlayerTrack]:
        if not frames:
            return []
        _, pose_model = self._ensure_models()
        infer_kw = self._infer_kwargs(self._pose_resolved or self._pose_weights)

        # Reset ByteTrack state between independent play windows without tearing down the predictor
        predictor = getattr(pose_model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for tr in predictor.trackers:
                tr.reset()

        # Deduplicate near-identical VFR frames before running pose inference + ByteTrack
        unique_pos, pos_to_unique = _find_unique_frame_indices(frames)
        unique_frames = [frames[i] for i in unique_pos]
        n_uniq = len(unique_frames)

        unique_poses: list[list[tuple[int, tuple[float, float, float, float], list[tuple[float, float, float]]]]] = [
            [] for _ in range(n_uniq)
        ]
        for start in range(0, n_uniq, self._batch_size):
            batch = unique_frames[start : start + self._batch_size]
            imgs = [item[2] for item in batch]
            results = pose_model.track(
                imgs,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=1280,
                conf=0.25,
                verbose=False,
                **infer_kw,
            )
            for offset, r in enumerate(results):
                if r.boxes is None or r.keypoints is None or r.boxes.id is None:
                    continue
                ids = r.boxes.id.int().tolist()
                boxes = r.boxes.xyxy.tolist()
                kps_xy = r.keypoints.xy.tolist()
                kps_conf = r.keypoints.conf.tolist() if r.keypoints.conf is not None else None
                bucket = unique_poses[start + offset]
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
                    bucket.append((int(tid), bbox, pts))

        tracks_map: dict[int, list[PoseFrame]] = {}
        for f_i, (frame_idx, ts, _bgr) in enumerate(frames):
            for tid, bbox, pts in unique_poses[pos_to_unique[f_i]]:
                tracks_map.setdefault(tid, []).append(
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
        infer_kw = self._infer_kwargs(self._det_resolved or self._det_weights)
        balls: list[BallDetection] = []
        people: list[tuple[int, float, tuple[float, float, float, float]]] = []

        for i in range(0, len(frames), self._batch_size):
            batch = frames[i : i + self._batch_size]
            imgs = [item[2] for item in batch]
            results = det_model.predict(
                imgs,
                device=self.device,
                classes=[0, 32],
                imgsz=960,
                conf=0.22,
                verbose=False,
                **infer_kw,
            )
            for (frame_idx, ts, _bgr), r in zip(batch, results, strict=True):
                if r.boxes is None or len(r.boxes) == 0:
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
            (5, np.array([40, 100, 230], dtype=np.uint8)),   # Far-side opponent attacker
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


MIN_SWING_PEAK_GAP_S = 2.3


def _select_swing_peaks(triggers: list[tuple[float, float]]) -> list[float]:
    """Greedily pick chronological local maxima of +/-0.8s summed trigger scores, >= MIN_SWING_PEAK_GAP_S apart."""
    ts_list = [t for t, _ in triggers]
    prefix = list(accumulate((sc for _, sc in triggers), initial=0.0))
    cands: list[tuple[float, int, float]] = []
    for t in ts_list:
        lo = bisect.bisect_left(ts_list, t - 0.8)
        hi = bisect.bisect_right(ts_list, t + 0.8)
        cands.append((prefix[hi] - prefix[lo], hi - lo, t))
    # Ignore 1-frame noise spikes unless nothing else triggered
    supported = [c for c in cands if c[1] >= 2] or cands
    peaks: list[float] = []
    for _score, _n, t in sorted(supported, key=lambda c: (-c[0], c[2])):
        if all(abs(t - p) >= MIN_SWING_PEAK_GAP_S for p in peaks):
            peaks.append(t)
    return sorted(peaks)


def scan_candidate_plays(
    video_path: Path,
    calibration: NetCalibration,
    predictor: FramePredictor,
    metadata: VideoMetadata,
) -> list[tuple[float, float]]:
    """Return one candidate [start_s, end_s] play window per distinct swing peak (short clips stay whole)."""
    duration_s = metadata.duration_s
    if duration_s <= 8.0:
        return [(0.0, duration_s)]

    batch_size = int(getattr(predictor, "batch_size", None) or getattr(predictor, "_batch_size", 16) or 16)
    use_nvdec = should_use_nvdec(metadata.frame_count)
    balls: list[BallDetection] = []
    people: list[tuple[int, float, tuple[float, float, float, float]]] = []
    seen_any = False
    for batch in iter_video_frame_batches(
        video_path,
        start_s=0.0,
        end_s=duration_s,
        sample_fps=10.0,
        batch_size=batch_size,
        use_nvdec=use_nvdec,
    ):
        seen_any = True
        b_chunk, p_chunk = predictor.scan_coarse(batch, calibration)
        balls.extend(b_chunk)
        people.extend(p_chunk)
    if not seen_any:
        return [(0.0, min(duration_s, 5.0))]
    # Identify moving balls above net tape (+35px margin), scored by peak elevation
    by_ts: dict[float, list[BallDetection]] = {}
    for b in balls:
        by_ts.setdefault(b.timestamp_s, []).append(b)

    sorted_ts = sorted(by_ts.keys())
    triggers: list[tuple[float, float]] = []
    for idx, ts in enumerate(sorted_ts):
        best = 0.0
        for b in by_ts[ts]:
            margin = tape_y_at(calibration, b.x) + 35.0 - b.y
            if margin < 0.0:
                continue
            # Verify ball is actually moving compared to neighbors
            moving = any(
                8.0 <= math.hypot(nb.x - b.x, nb.y - b.y) <= 260.0
                for n_idx in range(max(0, idx - 2), min(len(sorted_ts), idx + 3))
                if n_idx != idx
                for nb in by_ts[sorted_ts[n_idx]]
            )
            if moving:
                best = max(best, max(10.0, margin))
        if best > 0.0:
            triggers.append((ts, best))

    if not triggers:
        # Fallback: detect net-zone jumping people
        jump_ts = {
            ts
            for _f, ts, bbox in people
            if bbox[1] <= tape_y_at(calibration, 0.5 * (bbox[0] + bbox[2])) + 20.0
        }
        triggers = [(ts, 10.0) for ts in sorted(jump_ts)]

    peaks = _select_swing_peaks(triggers) if triggers else []
    if duration_s <= DEFAULT_CONFIG.short_clip_max_sec and len(peaks) <= 1:
        return [(0.0, duration_s)]

    windows: list[tuple[float, float]] = []
    for t_peak in peaks:
        w_start = max(0.0, round(t_peak - 2.6, 2))
        w_end = min(duration_s, round(t_peak + 1.6, 2))
        if w_end - w_start >= 1.2:
            windows.append((w_start, w_end))
    return windows


def _decode_and_detect(
    video_path: Path,
    start_s: float,
    end_s: float,
    calibration: NetCalibration,
    predictor: FramePredictor,
) -> tuple[list[tuple[int, float, np.ndarray]], list[BallDetection], list[PlayerTrack]]:
    """Decode [start_s, end_s] at full FPS and run Pass-2 ball + pose detection once."""
    frames = list(iter_video_frames(video_path, start_s=start_s, end_s=end_s, sample_fps=None))
    if not frames:
        raise ValueError(f"No frames decoded in window [{start_s}, {end_s}]")

    combined_fn = getattr(predictor, "detect_balls_and_poses", None)
    if callable(combined_fn):
        ball_dets, tracks = combined_fn(frames, calibration)
    else:
        ball_dets = predictor.detect_balls(frames, calibration)
        tracks = predictor.track_poses(frames, calibration)
    return frames, ball_dets, tracks


def _build_play_from_window_data(
    video_path: Path,
    play_index: int,
    start_s: float,
    end_s: float,
    frames: list[tuple[int, float, np.ndarray]],
    ball_dets: list[BallDetection],
    tracks: list[PlayerTrack],
    calibration: NetCalibration,
    metadata: VideoMetadata,
    session_dir: Path,
    *,
    session_mode: Literal["drill", "match"] = "drill",
    override_hitter_track_id: int | None = None,
    override_team_side: Literal["ours", "opponent"] | None = None,
) -> PlayAnalysis:
    """Fit the set arc, analyse mechanics/side, and write keyframe thumbnails for already-detected window data."""
    play_id = f"play_{play_index:02d}_{uuid.uuid4().hex[:6]}"

    # Candidate airborne attacker wrist points near net tape to guide set-arc selection
    cand_contacts = build_contact_candidates(tracks, calibration)

    set_traj = fit_setting_path(
        ball_dets,
        calibration=calibration,
        fps=metadata.fps,
        frame_height=metadata.height,
        candidate_contact_frames=cand_contacts,
    )

    team_side, system_state, pin_distance_m, setter_id, hitter_id, hitter_metrics, block_summary, keyframes = (
        analyse_play_full(
            tracks,
            set_traj,
            calibration=calibration,
            fps=metadata.fps,
            frame_shape=(metadata.height, metadata.width),
            session_mode=session_mode,
            override_hitter_track_id=override_hitter_track_id,
            override_team_side=override_team_side,
        )
    )

    # Classify attack zone from contact point or hitter position (mirrored for opponent plays)
    if hitter_metrics is not None and hitter_metrics.contact_point is not None:
        zone_x = hitter_metrics.contact_point.x
    elif set_traj is not None and set_traj.arc_points:
        zone_x = set_traj.arc_points[-1].x
    else:
        zone_x = 0.5 * (calibration.left_point.x + calibration.right_point.x)
    attack_zone = classify_attack_zone(calibration, zone_x, team_side=team_side)

    # Extract keyframe JPEG thumbnails directly from already-decoded frames (avoiding 3 video re-opens)
    thumbs_dir = session_dir / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    frames_by_idx = {idx: bgr for idx, _ts, bgr in frames}
    for km in keyframes:
        fname = f"{play_id}_{km.name}.jpg"
        out_path = thumbs_dir / fname
        try:
            chosen = frames_by_idx.get(km.frame_idx)
            if chosen is not None:
                h, w = chosen.shape[:2]
                if w > 360:
                    new_h = max(2, int(round(h * (360.0 / float(w)))))
                    chosen = cv2.resize(chosen, (360, new_h), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(out_path), chosen, [cv2.IMWRITE_JPEG_QUALITY, 86]):
                    raise IOError(f"Failed to write thumbnail JPEG: {out_path}")
            else:
                extract_keyframe_jpeg(video_path, km.frame_idx, out_path)
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
        team_side=team_side,
        team_side_manual=override_team_side in ("ours", "opponent"),
        system_state=system_state,
        pin_distance_m=pin_distance_m,
    )
    return play


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
    session_mode: Literal["drill", "match"] = "drill",
    override_team_side: Literal["ours", "opponent"] | None = None,
) -> tuple[PlayAnalysis, list[BallDetection]]:
    """Run Pass 2 high-FPS pose + ball analysis for a single play window."""
    frames, ball_dets, tracks = _decode_and_detect(video_path, start_s, end_s, calibration, predictor)
    play = _build_play_from_window_data(
        video_path,
        play_index,
        start_s,
        end_s,
        frames,
        ball_dets,
        tracks,
        calibration,
        metadata,
        session_dir,
        session_mode=session_mode,
        override_hitter_track_id=override_hitter_track_id,
        override_team_side=override_team_side,
    )
    return play, ball_dets


def analyse_play_windows_shared(
    video_path: Path,
    windows: list[tuple[float, float]],
    calibration: NetCalibration,
    predictor: FramePredictor,
    metadata: VideoMetadata,
    session_dir: Path,
    *,
    session_mode: Literal["drill", "match"] = "drill",
    on_play_ready: Callable[[int, int, PlayAnalysis, list[BallDetection]], None] | None = None,
) -> list[tuple[PlayAnalysis, list[BallDetection]]]:
    """Analyse play windows, running Pass-2 detection once per merged span of overlapping/adjacent windows."""
    # Merge windows overlapping or within 0.5s into one decode span (capped at 25s to bound RAM)
    groups: list[list[tuple[float, float]]] = []
    for w in sorted(windows):
        if groups:
            g_end = max(e for _, e in groups[-1])
            if w[0] <= g_end + 0.5 and max(g_end, w[1]) - groups[-1][0][0] <= 25.0:
                groups[-1].append(w)
                continue
        groups.append([w])

    def _drop_thumbnails(pruned: PlayAnalysis) -> None:
        for km in pruned.keyframes:
            if km.thumbnail_rel_path:
                (session_dir / "thumbnails" / km.thumbnail_rel_path).unlink(missing_ok=True)

    total = len(windows)
    half_frame_s = 0.5 / max(metadata.fps, 1.0)
    results: list[tuple[PlayAnalysis, list[BallDetection]]] = []
    fallback: tuple[float, PlayAnalysis, list[BallDetection]] | None = None
    for group in groups:
        frames, ball_dets, tracks = _decode_and_detect(
            video_path, group[0][0], max(e for _, e in group), calibration, predictor
        )
        for w_start, w_end in group:
            w_frames = [f for f in frames if w_start - half_frame_s <= f[1] <= w_end + half_frame_s]
            if not w_frames:
                raise ValueError(f"No frames decoded in window [{w_start}, {w_end}]")
            idx_set = {f[0] for f in w_frames}
            w_balls = [b for b in ball_dets if b.frame_idx in idx_set]
            sliced = [
                PlayerTrack(track_id=t.track_id, court_side=t.court_side, frames=pfs)
                for t in tracks
                if (pfs := [pf for pf in t.frames if pf.frame_idx in idx_set])
            ]
            w_tracks = classify_court_sides(sliced, calibration)
            play = _build_play_from_window_data(
                video_path,
                len(results) + 1,
                w_start,
                w_end,
                w_frames,
                w_balls,
                w_tracks,
                calibration,
                metadata,
                session_dir,
                session_mode=session_mode,
            )
            if total > 1 and play.set_trajectory is None:
                activity = max((_jump_impulse_px(t) for t in w_tracks), default=0.0)
                if activity < 25.0:
                    # Empty false-trigger window; keep only the most active one in case every window is empty
                    if not results and (fallback is None or activity > fallback[0]):
                        if fallback is not None:
                            _drop_thumbnails(fallback[1])
                        fallback = (activity, play, w_balls)
                    else:
                        _drop_thumbnails(play)
                    continue
            results.append((play, w_balls))
            if on_play_ready is not None:
                on_play_ready(play.index, total, play, w_balls)

    if fallback is not None:
        if results:
            _drop_thumbnails(fallback[1])
        else:
            results.append((fallback[1], fallback[2]))
            if on_play_ready is not None:
                on_play_ready(fallback[1].index, total, fallback[1], fallback[2])
    return results
