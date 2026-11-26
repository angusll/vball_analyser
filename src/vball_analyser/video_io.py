"""Self-contained video probing, decoding, proxy transcoding, and range parsing via PyAV."""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

from vball_analyser.models import VideoMetadata

logger = logging.getLogger(__name__)


def validate_video_magic_bytes(header: bytes, extension: str) -> bool:
    """Validate container magic bytes against the declared file extension."""
    ext = extension.lower()
    if ext in {".mp4", ".mov"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if ext == ".mkv":
        return header.startswith(b"\x1a\x45\xdf\xa3")
    if ext == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    return False


def _stream_fps(stream: av.video.stream.VideoStream) -> float:
    rate = stream.average_rate or stream.guessed_rate or stream.base_rate
    if rate is not None and float(rate) > 0:
        return float(rate)
    return 30.0


def probe_video(path: Path) -> VideoMetadata:
    """Inspect video metadata using PyAV without requiring system ffmpeg."""
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream found in file.")
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions.")
        fps = _stream_fps(stream)
        codec = str(stream.codec_context.name or "unknown")

        duration_s = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration / av.time_base)

        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and duration_s > 0:
            frame_count = max(1, int(round(duration_s * fps)))
        if duration_s <= 0 or frame_count <= 0:
            # Fallback for containers lacking duration header
            count = 0
            for _ in container.decode(video=0):
                count += 1
            frame_count = max(1, count)
            duration_s = frame_count / fps

        return VideoMetadata(
            width=width,
            height=height,
            fps=round(fps, 4),
            frame_count=frame_count,
            duration_s=round(duration_s, 4),
            codec=codec,
        )


def ensure_web_playable_mp4(source_path: Path, dest_dir: Path) -> Path:
    """Return source_path if browser-safe H.264/yuv420p MP4, else transcode proxy via PyAV."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = dest_dir / "web_proxy.mp4"
    if proxy_path.is_file() and proxy_path.stat().st_size > 0:
        return proxy_path

    needs_transcode = source_path.suffix.lower() != ".mp4"
    with av.open(str(source_path)) as in_container:
        if not in_container.streams.video:
            raise ValueError("No video stream found in file.")
        in_stream = in_container.streams.video[0]
        codec_name = str(in_stream.codec_context.name or "").lower()
        pix_fmt = str(in_stream.codec_context.pix_fmt or "").lower()
        if codec_name != "h264" or pix_fmt not in {"yuv420p", "yuvj420p"}:
            needs_transcode = True

    if not needs_transcode:
        return source_path

    logger.info("Transcoding %s to browser-compatible H.264 proxy %s", source_path.name, proxy_path)
    with av.open(str(source_path)) as in_container:
        in_stream = in_container.streams.video[0]
        fps_frac = in_stream.average_rate or in_stream.guessed_rate or Fraction(30, 1)
        if float(fps_frac) <= 0:
            fps_frac = Fraction(30, 1)
        width = int(in_stream.codec_context.width)
        height = int(in_stream.codec_context.height)
        # H.264 yuv420p requires even dimensions
        out_w = width - (width % 2)
        out_h = height - (height % 2)

        with av.open(str(proxy_path), mode="w", format="mp4") as out_container:
            out_stream = out_container.add_stream(
                "libx264",
                rate=fps_frac,
                options={"preset": "veryfast", "crf": "20", "movflags": "+faststart"},
            )
            out_stream.width = out_w
            out_stream.height = out_h
            out_stream.pix_fmt = "yuv420p"

            frame_idx = 0
            for frame in in_container.decode(video=0):
                if frame.width != out_w or frame.height != out_h:
                    frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
                else:
                    frame = frame.reformat(format="yuv420p")
                frame.pts = frame_idx
                frame_idx += 1
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

            for packet in out_stream.encode():
                out_container.mux(packet)

    return proxy_path


def iter_video_frames(
    path: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    sample_fps: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Decode frames from video in [start_s, end_s] optionally downsampled to sample_fps."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            return
        stream = container.streams.video[0]
        fps = _stream_fps(stream)
        time_base = float(stream.time_base) if stream.time_base is not None else (1.0 / fps)

        if start_s > 0.25 and stream.time_base is not None:
            seek_ts = max(0, int((start_s - 0.15) / time_base))
            try:
                container.seek(seek_ts, any_frame=False, backward=True, stream=stream)
            except av.AVError:
                pass

        min_step_s = (1.0 / sample_fps) if (sample_fps and sample_fps > 0) else 0.0
        next_emit_s = max(0.0, start_s)
        sequential_idx = 0

        for frame in container.decode(video=0):
            if frame.pts is not None:
                ts = float(frame.pts * time_base)
                frame_idx = max(0, int(round(ts * fps)))
            else:
                frame_idx = sequential_idx
                ts = frame_idx / fps
            sequential_idx += 1

            if ts < start_s - (0.5 / fps):
                continue
            if end_s is not None and ts > end_s + (0.5 / fps):
                break
            if min_step_s > 0 and ts + 1e-4 < next_emit_s:
                continue

            if min_step_s > 0:
                while next_emit_s <= ts + 1e-4:
                    next_emit_s += min_step_s

            bgr = frame.to_ndarray(format="bgr24")
            yield frame_idx, ts, bgr


def extract_keyframe_jpeg(path: Path, frame_idx: int, out_path: Path, max_width: int = 360) -> Path:
    """Decode a single frame near frame_idx and save a resized JPEG thumbnail."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(path)
    target_idx = max(0, min(frame_idx, max(0, meta.frame_count - 1)))
    target_s = target_idx / max(meta.fps, 1.0)
    chosen_bgr: np.ndarray | None = None

    for idx, _ts, bgr in iter_video_frames(path, start_s=max(0.0, target_s - 0.5), end_s=target_s + 0.5):
        chosen_bgr = bgr
        if idx >= target_idx:
            break

    if chosen_bgr is None:
        for _idx, _ts, bgr in iter_video_frames(path, start_s=0.0, end_s=1.0):
            chosen_bgr = bgr
            break

    if chosen_bgr is None:
        raise ValueError(f"Could not decode frame {frame_idx} from {path}")

    h, w = chosen_bgr.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        new_h = max(2, int(round(h * scale)))
        chosen_bgr = cv2.resize(chosen_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

    ok = cv2.imwrite(str(out_path), chosen_bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise IOError(f"Failed to write thumbnail JPEG: {out_path}")
    return out_path


def parse_byte_range(range_header: str | None, file_size: int) -> tuple[int, int]:
    """Parse HTTP Range header ('bytes=start-end') into inclusive (start, end) offsets."""
    if file_size <= 0:
        raise ValueError("Cannot serve byte range on empty file.")
    if not range_header:
        return 0, file_size - 1

    header = range_header.strip()
    if not header.startswith("bytes="):
        raise ValueError("Unsupported Range unit.")

    spec = header[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in spec:
        raise ValueError("Malformed Range header.")

    start_str, end_str = spec.split("-", 1)
    if start_str == "":
        if not end_str.isdigit():
            raise ValueError("Invalid suffix byte range.")
        suffix_len = int(end_str)
        if suffix_len <= 0:
            raise ValueError("Suffix length must be positive.")
        start = max(0, file_size - suffix_len)
        return start, file_size - 1

    if not start_str.isdigit():
        raise ValueError("Invalid range start offset.")
    start = int(start_str)
    if start >= file_size:
        raise ValueError("Range start exceeds file size.")

    if end_str == "":
        return start, file_size - 1
    if not end_str.isdigit():
        raise ValueError("Invalid range end offset.")
    end = min(file_size - 1, int(end_str))
    if end < start:
        raise ValueError("Range end precedes range start.")
    return start, end
