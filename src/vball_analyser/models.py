"""Pydantic schemas for volleyball baseline video analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Point2D(BaseModel):
    x: float
    y: float


class NetCalibration(BaseModel):
    left_point: Point2D
    right_point: Point2D
    net_height_m: float = 2.43
    is_manual: bool = False


class BallDetection(BaseModel):
    frame_idx: int
    timestamp_s: float
    x: float
    y: float
    conf: float
    is_interpolated: bool = False


class SetTrajectory(BaseModel):
    release_frame: int
    release_time_s: float
    contact_frame: int
    contact_time_s: float
    peak_frame: int
    peak_point: Point2D
    tempo_s: float
    peak_height_px: float
    peak_height_cm: float | None = None
    arc_points: list[BallDetection] = Field(default_factory=list)
    inlier_count: int = 0


class PoseFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    bbox: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]


class PlayerTrack(BaseModel):
    track_id: int
    court_side: Literal["attacking", "blocking"]
    frames: list[PoseFrame] = Field(default_factory=list)


class KeyMoment(BaseModel):
    name: Literal["set_release", "max_load", "contact"]
    label: str
    frame_idx: int
    timestamp_s: float
    thumbnail_rel_path: str | None = None


class BlockerCallout(BaseModel):
    track_id: int
    bbox: tuple[float, float, float, float]
    top_reach_point: Point2D
    reach_above_tape_px: float
    reach_above_tape_cm: float | None = None


class BlockSummary(BaseModel):
    blockers: list[BlockerCallout] = Field(default_factory=list)
    seam_open: bool = False
    seam_gap_px: float | None = None
    seam_gap_cm: float | None = None
    seam_interval_x: tuple[float, float] | None = None


class HitterMetrics(BaseModel):
    contact_point: Point2D | None = None
    contact_height_px: float | None = None
    contact_height_cm: float | None = None
    is_back_row_approximate: bool = False
    dominant_hand: Literal["right", "left"] = "right"
    arm_extension_deg: float | None = None
    torso_tilt_deg: float | None = None
    apex_frame: int | None = None
    apex_delta_ms: int | None = None
    jump_timing_label: str = "Unknown"


class PlayAnalysis(BaseModel):
    play_id: str
    index: int
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int
    fps: float
    attack_zone: str
    setter_track_id: int | None = None
    hitter_track_id: int | None = None
    set_trajectory: SetTrajectory | None = None
    hitter_metrics: HitterMetrics | None = None
    block_summary: BlockSummary = Field(default_factory=BlockSummary)
    keyframes: list[KeyMoment] = Field(default_factory=list)
    tracks: list[PlayerTrack] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str


class AnalysisSession(BaseModel):
    session_id: str
    filename: str
    created_at: str
    status: Literal["queued", "scanning", "analysing", "ready", "error"]
    progress_pct: float = 0.0
    error_message: str | None = None
    metadata: VideoMetadata
    calibration: NetCalibration
    plays: list[PlayAnalysis] = Field(default_factory=list)
