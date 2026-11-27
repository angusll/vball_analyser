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
    net_band_px: float | None = None
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


class CourtCalibration(BaseModel):
    """Court floor homography from either 2-click net interpolation or 4-corner front-3m clicks."""

    front_left: Point2D
    front_right: Point2D
    attack_right: Point2D
    attack_left: Point2D
    is_calibrated: bool = True
    method: Literal["net", "four_corner"] = "four_corner"
    net_left_top: Point2D | None = None
    net_right_top: Point2D | None = None
    sideline_point: Point2D | None = None
    net_height_m: float = 2.43


class PasserPosition(BaseModel):
    label: str = Field(min_length=1, max_length=24)
    x_m: float
    y_m: float


class ServeEvent(BaseModel):
    serve_id: str
    index: int
    source: Literal["auto", "manual"] = "manual"
    timestamp_s: float | None = None
    frame_idx: int | None = None
    server_name: str = "Unassigned"
    x_m: float
    y_m: float
    result: Literal["in", "ace", "out", "net"] = "in"
    zone: str = "Zone 6"
    depth: Literal["short", "deep", "out"] = "deep"
    is_seam: bool = False
    seam_pair: tuple[str, str] | None = None
    passers_snapshot: list[PasserPosition] = Field(default_factory=list)
    passer_label: str | None = None
    pass_x_m: float | None = None
    pass_y_m: float | None = None
    pass_timestamp_s: float | None = None
    pass_frame_idx: int | None = None
    pass_distance_m: float | None = None
    receive_score: Literal[0, 1, 2, 3] | None = None
    receive_score_manual: bool = False


class PasserReceiveSummary(BaseModel):
    passer_label: str
    count: int = 0
    avg_rating: float = 0.0
    in_system_pct: float = 0.0
    score_counts: dict[str, int] = Field(default_factory=lambda: {"0": 0, "1": 0, "2": 0, "3": 0})


class ReceiveQualitySummary(BaseModel):
    scored_count: int = 0
    avg_rating: float = 0.0
    in_system_count: int = 0
    in_system_pct: float = 0.0
    perfect_count: int = 0
    perfect_pct: float = 0.0
    score_counts: dict[str, int] = Field(default_factory=lambda: {"0": 0, "1": 0, "2": 0, "3": 0})
    score_pcts: dict[str, float] = Field(default_factory=lambda: {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0})
    by_passer: list[PasserReceiveSummary] = Field(default_factory=list)


class ServeDistributionSummary(BaseModel):
    total: int = 0
    in_count: int = 0
    ace_count: int = 0
    out_count: int = 0
    net_count: int = 0
    in_pct: float = 0.0
    ace_pct: float = 0.0
    error_pct: float = 0.0
    seam_count: int = 0
    seam_pct: float = 0.0
    seam_by_pair: dict[str, int] = Field(default_factory=dict)
    zone_counts: dict[str, int] = Field(default_factory=dict)
    zone_pcts: dict[str, float] = Field(default_factory=dict)
    short_count: int = 0
    deep_count: int = 0


class ServeSession(BaseModel):
    serve_session_id: str
    name: str
    created_at: str
    video_session_id: str | None = None
    status: Literal["idle", "scanning", "ready", "error"] = "idle"
    progress_pct: float = 0.0
    error_message: str | None = None
    court_calibration: CourtCalibration | None = None
    default_passers: list[PasserPosition] = Field(default_factory=list)
    setter_target: Point2D = Field(default_factory=lambda: Point2D(x=6.0, y=0.8))
    receive_ring_radii_m: tuple[float, float, float] = (1.5, 3.2, 4.8)
    servers: list[str] = Field(default_factory=list)
    serves: list[ServeEvent] = Field(default_factory=list)
