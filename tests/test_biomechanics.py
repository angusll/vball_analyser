"""Tests for pose smoothing, court-side classification, hitter metrics, and blocker seams."""

from __future__ import annotations

import math

import pytest

from vball_analyser.biomechanics import (
    analyse_play_full,
    analyse_play_mechanics,
    classify_court_sides,
    compute_studio_session_summary,
    smooth_pose_tracks,
)
from vball_analyser.models import (
    BallDetection,
    BlockSummary,
    HitterMetrics,
    NetCalibration,
    PlayAnalysis,
    PlayerTrack,
    Point2D,
    PoseFrame,
    SetTrajectory,
)


def _make_coco_frame(
    frame_idx: int,
    fps: float,
    hip_x: float,
    hip_y: float,
    ankle_y: float,
    elbow_angle_deg: float = 168.0,
    torso_tilt_deg: float = 8.0,
    scale: float = 1.0,
) -> PoseFrame:
    kps = [(0.0, 0.0, 0.0) for _ in range(17)]
    tilt_rad = math.radians(torso_tilt_deg)
    sh_x = hip_x + 55.0 * scale * math.sin(tilt_rad)
    sh_y = hip_y - 55.0 * scale * math.cos(tilt_rad)
    nose_y = sh_y - 20.0 * scale
    kps[0] = (sh_x, nose_y, 0.95)
    kps[5] = (sh_x - 14 * scale, sh_y, 0.95)
    kps[6] = (sh_x + 14 * scale, sh_y, 0.95)
    kps[11] = (hip_x - 10 * scale, hip_y, 0.95)
    kps[12] = (hip_x + 10 * scale, hip_y, 0.95)
    kps[15] = (hip_x - 12 * scale, ankle_y, 0.95)
    kps[16] = (hip_x + 12 * scale, ankle_y, 0.95)

    # Right arm with specified interior shoulder-elbow-wrist angle
    r_sh = kps[6][:2]
    r_el = (r_sh[0], r_sh[1] - 32.0 * scale)
    # Vector elbow->shoulder is (0, +1). Rotate by elbow_angle_deg to elbow->wrist
    theta = math.radians(elbow_angle_deg)
    r_wr = (r_el[0] + 32.0 * scale * math.sin(theta), r_el[1] + 32.0 * scale * math.cos(theta))
    kps[8] = (r_el[0], r_el[1], 0.95)
    kps[10] = (r_wr[0], r_wr[1], 0.95)
    kps[7] = (kps[5][0] - 12 * scale, kps[5][1] + 15 * scale, 0.9)
    kps[9] = (kps[5][0] - 16 * scale, kps[5][1] + 28 * scale, 0.9)

    bbox = (hip_x - 28 * scale, min(r_wr[1], nose_y) - 8, hip_x + 28 * scale, ankle_y + 4)
    return PoseFrame(frame_idx=frame_idx, timestamp_s=frame_idx / fps, bbox=bbox, keypoints=kps)


def test_role_isolation_hitter_swap_and_blocker_seam() -> None:
    fps = 30.0
    cal = NetCalibration(left_point=Point2D(x=180.0, y=320.0), right_point=Point2D(x=1100.0, y=320.0), is_manual=True)

    # Frames 10..50, contact at frame 40
    setter_frames, hitter_frames, decoy_frames, passer_frames = [], [], [], []
    b1_frames, b2_frames = [], []

    for f in range(10, 51):
        # Setter on Zone 2/3 near side
        setter_frames.append(_make_coco_frame(f, fps, 790.0, 455.0, 535.0))
        # Idle back-row passer near baseline
        passer_frames.append(_make_coco_frame(f, fps, 960.0, 595.0, 680.0))

        # Outside hitter jumps peaking at f=40
        dt = abs(f - 40) / fps
        h_jump = 90.0 * max(0.0, 1.0 - (dt / 0.4) ** 2)
        elb_ang = 85.0 if 30 <= f <= 35 else (168.0 if 38 <= f <= 42 else 120.0)
        hitter_frames.append(_make_coco_frame(f, fps, 340.0, 485.0 - h_jump, 565.0 - h_jump, elbow_angle_deg=elb_ang))

        # Middle decoy jumper also jumps simultaneously
        d_jump = 85.0 * max(0.0, 1.0 - (dt / 0.4) ** 2)
        decoy_frames.append(_make_coco_frame(f, fps, 600.0, 480.0 - d_jump, 560.0 - d_jump, elbow_angle_deg=140.0))

        # Two blockers across the net with open seam (~55px gap)
        b_jump = 55.0 * max(0.0, 1.0 - (dt / 0.35) ** 2)
        b1_frames.append(_make_coco_frame(f, fps, 305.0, 405.0 - b_jump, 468.0 - b_jump, scale=0.85))
        b2_frames.append(_make_coco_frame(f, fps, 420.0, 405.0 - b_jump, 468.0 - b_jump, scale=0.85))

    raw_tracks = [
        PlayerTrack(track_id=1, court_side="attacking", frames=setter_frames),
        PlayerTrack(track_id=2, court_side="attacking", frames=hitter_frames),
        PlayerTrack(track_id=3, court_side="attacking", frames=decoy_frames),
        PlayerTrack(track_id=4, court_side="attacking", frames=passer_frames),
        PlayerTrack(track_id=5, court_side="attacking", frames=b1_frames),  # intentionally mislabeled to test classifier
        PlayerTrack(track_id=6, court_side="attacking", frames=b2_frames),
    ]

    classified = classify_court_sides(smooth_pose_tracks(raw_tracks), cal)
    side_by_id = {t.track_id: t.court_side for t in classified}
    assert side_by_id[1] == "attacking"
    assert side_by_id[2] == "attacking"
    assert side_by_id[4] == "attacking"
    assert side_by_id[5] == "blocking"
    assert side_by_id[6] == "blocking"

    # Set trajectory ending near hitter's contact at x=354, y=260 on frame 40
    traj = SetTrajectory(
        release_frame=14,
        release_time_s=14 / fps,
        contact_frame=40,
        contact_time_s=40 / fps,
        peak_frame=28,
        peak_point=Point2D(x=560.0, y=210.0),
        tempo_s=round(26 / fps, 3),
        peak_height_px=110.0,
        peak_height_cm=107.6,
        arc_points=[
            BallDetection(frame_idx=14, timestamp_s=14 / fps, x=790.0, y=350.0, conf=0.9),
            BallDetection(frame_idx=40, timestamp_s=40 / fps, x=354.0, y=250.0, conf=0.9),
        ],
        inlier_count=20,
    )

    setter_id, hitter_id, metrics, block_summary, keyframes = analyse_play_mechanics(
        raw_tracks, traj, cal, fps=fps, frame_shape=(720, 1280)
    )
    assert hitter_id == 2
    assert setter_id == 1
    assert metrics is not None
    assert metrics.arm_extension_deg == pytest.approx(168.0, abs=3.0)
    assert metrics.torso_tilt_deg == pytest.approx(8.0, abs=2.0)
    assert metrics.jump_timing_label == "At peak jump"
    assert metrics.contact_height_px is not None and metrics.contact_height_px > 50.0
    assert len(block_summary.blockers) == 2
    assert block_summary.seam_open is True
    assert block_summary.seam_gap_px is not None and block_summary.seam_gap_px > 40.0
    assert [k.name for k in keyframes] == ["set_release", "max_load", "contact"]
    assert 30 <= keyframes[1].frame_idx <= 36

    # Reassign active hitter to middle decoy (track 3)
    _, swapped_id, swapped_metrics, _, _ = analyse_play_mechanics(
        raw_tracks, traj, cal, fps=fps, frame_shape=(720, 1280), override_hitter_track_id=3
    )
    assert swapped_id == 3
    assert swapped_metrics is not None
    assert swapped_metrics.arm_extension_deg == pytest.approx(140.0, abs=3.0)


def _make_block_frame(
    frame_idx: int,
    fps: float,
    cx: float,
    hip_y: float,
    ankle_y: float,
    wrist_gap: float = 35.0,
    bbox: tuple[float, float, float, float] | None = None,
    arm_conf: float = 0.95,
) -> PoseFrame:
    """Symmetric two-hand block pose: both wrists raised level above the head."""
    kps = [(0.0, 0.0, 0.0) for _ in range(17)]
    sh_y = hip_y - 55.0
    kps[0] = (cx, sh_y - 20.0, 0.95)
    kps[5] = (cx - 14.0, sh_y, 0.95)
    kps[6] = (cx + 14.0, sh_y, 0.95)
    kps[7] = (cx - 0.5 * wrist_gap, sh_y - 30.0, arm_conf)
    kps[8] = (cx + 0.5 * wrist_gap, sh_y - 30.0, arm_conf)
    kps[9] = (cx - 0.5 * wrist_gap, sh_y - 60.0, arm_conf)
    kps[10] = (cx + 0.5 * wrist_gap, sh_y - 60.0, arm_conf)
    kps[11] = (cx - 10.0, hip_y, 0.95)
    kps[12] = (cx + 10.0, hip_y, 0.95)
    kps[15] = (cx - 12.0, ankle_y, 0.95)
    kps[16] = (cx + 12.0, ankle_y, 0.95)
    if bbox is None:
        bbox = (cx - 30.0, sh_y - 68.0, cx + 30.0, ankle_y + 4.0)
    return PoseFrame(frame_idx=frame_idx, timestamp_s=frame_idx / fps, bbox=bbox, keypoints=kps)


def _jump(f: int, height: float, width_s: float = 0.4, fps: float = 30.0) -> float:
    return height * max(0.0, 1.0 - ((abs(f - 40) / fps) / width_s) ** 2)


def _near_hitter_track(fps: float) -> PlayerTrack:
    frames = []
    for f in range(10, 51):
        h = _jump(f, 90.0)
        elb = 85.0 if 30 <= f <= 35 else (168.0 if 38 <= f <= 42 else 120.0)
        frames.append(_make_coco_frame(f, fps, 340.0, 485.0 - h, 565.0 - h, elbow_angle_deg=elb))
    return PlayerTrack(track_id=2, court_side="attacking", frames=frames)


def _near_traj(fps: float, release_x: float = 790.0, release_y: float = 350.0) -> SetTrajectory:
    return SetTrajectory(
        release_frame=14,
        release_time_s=14 / fps,
        contact_frame=40,
        contact_time_s=40 / fps,
        peak_frame=28,
        peak_point=Point2D(x=560.0, y=210.0),
        tempo_s=round(26 / fps, 3),
        peak_height_px=110.0,
        arc_points=[
            BallDetection(frame_idx=14, timestamp_s=14 / fps, x=release_x, y=release_y, conf=0.9),
            BallDetection(frame_idx=40, timestamp_s=40 / fps, x=354.0, y=250.0, conf=0.9),
        ],
        inlier_count=20,
    )


def _near_setter_track(fps: float) -> PlayerTrack:
    frames = [_make_coco_frame(f, fps, 790.0, 455.0, 535.0) for f in range(10, 51)]
    return PlayerTrack(track_id=1, court_side="attacking", frames=frames)


_CAL = NetCalibration(left_point=Point2D(x=180.0, y=320.0), right_point=Point2D(x=1100.0, y=320.0), is_manual=True)


def test_hand_span_seam_detects_gap_despite_overlapping_leg_bboxes() -> None:
    fps = 30.0
    setter = _near_setter_track(fps)
    # Flared legs/hips: full-body bboxes overlap by 10px, but raised hands leave an open seam.
    b1 = PlayerTrack(
        track_id=5,
        court_side="blocking",
        frames=[_make_block_frame(f, fps, 347.5, 400.0, 466.0, bbox=(300, 260, 395, 470)) for f in range(10, 51)],
    )
    # b2 arm keypoints drop out (conf < 0.25) -> central 55% bbox fallback
    b2 = PlayerTrack(
        track_id=6,
        court_side="blocking",
        frames=[
            _make_block_frame(f, fps, 432.5, 400.0, 466.0, bbox=(385, 260, 480, 470), arm_conf=0.1)
            for f in range(10, 51)
        ],
    )
    tracks = [setter, _near_hitter_track(fps), b1, b2]

    team_side, _, _, _, hitter_id, _, block_summary, _ = analyse_play_full(
        tracks, _near_traj(fps), _CAL, fps, (720, 1280)
    )
    assert team_side == "ours"
    assert hitter_id == 2
    assert sorted(b.track_id for b in block_summary.blockers) == [5, 6]
    b2_span = next(b.hand_span_x for b in block_summary.blockers if b.track_id == 6)
    assert b2_span == pytest.approx((406.4, 458.6), abs=0.2)
    assert block_summary.seam_open is True
    assert block_summary.block_state == "open_seam"
    assert block_summary.seam_gap_cm is not None and block_summary.seam_gap_cm >= 22.0


def test_opponent_play_classification_role_swap_and_in_system_split() -> None:
    fps = 30.0
    # (a) Far-side opponent attacker hits near the right antenna; two near-side blockers jump.
    far_hitter = PlayerTrack(
        track_id=11,
        court_side="attacking",
        frames=[
            _make_coco_frame(f, fps, 1000.0, 400.0 - _jump(f, 45.0), 455.0 - _jump(f, 45.0), scale=0.7)
            for f in range(10, 51)
        ],
    )
    far_setter = PlayerTrack(
        track_id=12,
        court_side="attacking",
        frames=[_make_coco_frame(f, fps, 760.0, 400.0, 455.0, scale=0.7) for f in range(10, 51)],
    )
    near_blockers = [
        PlayerTrack(
            track_id=tid,
            court_side="attacking",
            frames=[
                _make_block_frame(f, fps, cx, 480.0 - _jump(f, 60.0, 0.35), 560.0 - _jump(f, 60.0, 0.35))
                for f in range(10, 51)
            ],
        )
        for tid, cx in ((21, 980.0), (22, 1070.0))
    ]
    opp_traj = SetTrajectory(
        release_frame=14,
        release_time_s=14 / fps,
        contact_frame=40,
        contact_time_s=40 / fps,
        peak_frame=28,
        peak_point=Point2D(x=900.0, y=200.0),
        tempo_s=round(26 / fps, 3),
        peak_height_px=120.0,
        arc_points=[
            BallDetection(frame_idx=14, timestamp_s=14 / fps, x=780.0, y=312.0, conf=0.9),
            BallDetection(frame_idx=40, timestamp_s=40 / fps, x=1020.0, y=268.0, conf=0.9),
        ],
        inlier_count=20,
    )
    team_side, system_state, pin_m, setter_id, hitter_id, metrics, block_summary, _ = analyse_play_full(
        [far_hitter, far_setter, *near_blockers], opp_traj, _CAL, fps, (720, 1280), session_mode="match"
    )
    assert team_side == "opponent"
    assert hitter_id == 11
    assert setter_id == 12
    assert sorted(b.track_id for b in block_summary.blockers) == [21, 22]
    assert system_state is None
    assert metrics is not None and metrics.is_back_row_approximate is False
    assert pin_m == pytest.approx(0.78, abs=0.02)

    # Manual override wins over auto classification
    forced = analyse_play_full(
        [far_hitter, far_setter, *near_blockers], opp_traj, _CAL, fps, (720, 1280), override_team_side="ours"
    )
    assert forced[0] == "ours"

    # (b) Near-side overhead release at x_rel = 6.0m -> in_system
    in_x = 180.0 + 920.0 * 6.0 / 9.0
    setter_in = PlayerTrack(
        track_id=1, court_side="attacking", frames=[_make_coco_frame(f, fps, in_x, 455.0, 535.0) for f in range(10, 51)]
    )
    res_in = analyse_play_full([setter_in, _near_hitter_track(fps)], _near_traj(fps, in_x), _CAL, fps, (720, 1280))
    assert res_in[0] == "ours"
    assert res_in[3] == 1
    assert res_in[1] == "in_system"

    # Low forearm bump released from x_rel = 1.8m -> out_of_system
    bump_x = 180.0 + 920.0 * 1.8 / 9.0
    bump_frames = []
    for f in range(10, 51):
        pf = _make_coco_frame(f, fps, bump_x, 485.0, 565.0)
        kps = list(pf.keypoints)
        kps[9] = (bump_x - 8.0, 490.0, 0.95)
        kps[10] = (bump_x + 8.0, 490.0, 0.95)
        bump_frames.append(PoseFrame(frame_idx=f, timestamp_s=pf.timestamp_s, bbox=pf.bbox, keypoints=kps))
    setter_bump = PlayerTrack(track_id=1, court_side="attacking", frames=bump_frames)
    res_out = analyse_play_full(
        [setter_bump, _near_hitter_track(fps)], _near_traj(fps, bump_x, 480.0), _CAL, fps, (720, 1280)
    )
    assert res_out[0] == "ours"
    assert res_out[1] == "out_of_system"


def _summary_play(
    idx: int,
    team_side: str,
    zone: str,
    tempo: float,
    block_state: str,
    pin: float | None = None,
    system_state: str | None = None,
    seam_cm: float | None = None,
) -> PlayAnalysis:
    traj = SetTrajectory(
        release_frame=0,
        release_time_s=0.0,
        contact_frame=30,
        contact_time_s=1.0,
        peak_frame=15,
        peak_point=Point2D(x=500.0, y=200.0),
        tempo_s=tempo,
        peak_height_px=100.0,
        peak_height_cm=100.0 + idx,
    )
    return PlayAnalysis(
        play_id=f"p{idx}",
        index=idx,
        start_time_s=0.0,
        end_time_s=2.0,
        start_frame=0,
        end_frame=60,
        fps=30.0,
        attack_zone=zone,
        team_side=team_side,
        system_state=system_state,
        pin_distance_m=pin,
        set_trajectory=traj,
        hitter_metrics=HitterMetrics(is_back_row_approximate=idx == 3),
        block_summary=BlockSummary(block_state=block_state, seam_gap_cm=seam_cm),
    )


def test_unopposed_drill_vs_match_and_session_summary_one_rep_std() -> None:
    fps = 30.0
    setter = _near_setter_track(fps)
    # Far-side defender standing with arms down, never reaching the tape
    defender = PlayerTrack(
        track_id=7,
        court_side="blocking",
        frames=[_make_coco_frame(f, fps, 400.0, 410.0, 460.0, elbow_angle_deg=10.0, scale=0.7) for f in range(10, 51)],
    )
    tracks = [setter, _near_hitter_track(fps), defender]

    def _as_play(mode: str) -> PlayAnalysis:
        side, sys_state, pin, setter_id, hitter_id, metrics, block, keys = analyse_play_full(
            tracks, _near_traj(fps), _CAL, fps, (720, 1280), session_mode=mode
        )
        return PlayAnalysis(
            play_id="p0", index=0, start_time_s=0.0, end_time_s=2.0, start_frame=0, end_frame=60, fps=fps,
            attack_zone="Zone 4 (Left)", team_side=side, system_state=sys_state, pin_distance_m=pin,
            setter_track_id=setter_id, hitter_track_id=hitter_id, hitter_metrics=metrics,
            block_summary=block, keyframes=keys,
        )

    drill_play = _as_play("drill")
    assert drill_play.team_side == "ours"
    assert drill_play.block_summary.blockers == []
    assert drill_play.block_summary.block_state == "unopposed"
    assert compute_studio_session_summary([drill_play], "drill").ours_all.block_stats.isolation_pct is None

    match_play = _as_play("match")
    assert match_play.block_summary.block_state == "zero_block"
    match_summary = compute_studio_session_summary([match_play], "match")
    assert match_summary.session_mode == "match"
    assert match_summary.ours_all.block_stats.isolation_pct == pytest.approx(100.0)

    plays = [
        _summary_play(1, "ours", "Zone 4 (Left)", 1.0, "open_seam", pin=0.9, system_state="in_system", seam_cm=30.0),
        _summary_play(2, "ours", "Zone 4 (Left)", 1.2, "closed_block", pin=1.1, system_state="out_of_system"),
        _summary_play(3, "ours", "Zone 3 (Middle)", 0.8, "unopposed"),
        _summary_play(4, "opponent", "Opp Zone 4 (Left)", 1.5, "closed_block", pin=0.5),
    ]
    summary = compute_studio_session_summary(plays, "match")
    assert (summary.total_plays, summary.ours_count, summary.opponent_count) == (4, 3, 1)
    assert summary.in_system_count == 1
    assert summary.in_system_pct == pytest.approx(50.0)

    ours = summary.ours_all
    assert ours.play_count == 3
    assert ours.back_row_count == 1
    assert [z.zone for z in ours.zones] == ["Zone 4 (Left)", "Zone 3 (Middle)", "Zone 2 (Right)"]
    z4, z3, z2 = ours.zones
    assert z4.count == 2 and z4.pct == pytest.approx(66.7)
    assert z4.tempo_mean_s == pytest.approx(1.1)
    assert z4.tempo_std_s == pytest.approx(0.141, abs=1e-3)  # sample std (ddof=1)
    assert z4.pin_distance_mean_m == pytest.approx(1.0)
    assert z3.count == 1 and z3.tempo_mean_s == pytest.approx(0.8)
    assert z3.tempo_std_s is None
    assert z3.peak_height_std_cm is None
    assert z3.pin_distance_mean_m is None
    assert z2.count == 0 and z2.tempo_mean_s is None

    bs = ours.block_stats
    assert (bs.opposed_count, bs.unopposed_count, bs.open_seam_count, bs.closed_block_count) == (2, 1, 1, 1)
    assert bs.isolation_pct == pytest.approx(50.0)
    assert bs.closed_block_pct == pytest.approx(50.0)
    assert bs.avg_open_seam_cm == pytest.approx(30.0)

    assert summary.ours_in_system.play_count == 1
    assert summary.ours_out_of_system.play_count == 1
    opp = summary.opponent
    assert opp.play_count == 1
    assert [z.zone for z in opp.zones] == ["Opp Zone 4 (Left)", "Opp Zone 3 (Middle)", "Opp Zone 2 (Right)"]
    assert opp.zones[0].count == 1
    assert opp.block_stats.closed_block_pct == pytest.approx(100.0)
