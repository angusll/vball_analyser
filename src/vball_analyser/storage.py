"""Disk-backed session persistence and instant cached recalibration/role reassignment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from vball_analyser.biomechanics import analyse_play_full, build_contact_candidates, compute_studio_session_summary
from vball_analyser.calibration import classify_attack_zone
from vball_analyser.models import AnalysisSession, BallDetection, NetCalibration, PlayAnalysis, Point2D
from vball_analyser.trajectory import fit_setting_path
from vball_analyser.video_io import extract_keyframe_jpeg

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


def validate_safe_id(value: str) -> str:
    if not _SAFE_ID_RE.match(value):
        raise ValueError(f"Invalid identifier: {value}")
    return value


class SessionStore:
    """Stores analysis sessions, raw ball caches, and exports inside data/sessions/<id>."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir.resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.sessions_dir, 0o700)
        except OSError:
            pass

    def session_dir(self, session_id: str) -> Path:
        safe_id = validate_safe_id(session_id)
        candidate = (self.sessions_dir / safe_id).resolve()
        if not candidate.is_relative_to(self.sessions_dir):
            raise ValueError("Path escapes sessions directory.")
        return candidate

    def create_session_dir(self, session_id: str) -> Path:
        sdir = self.session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "thumbnails").mkdir(parents=True, exist_ok=True)
        (sdir / "exports").mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(sdir, 0o700)
        except OSError:
            pass
        return sdir

    def save_session(self, session: AnalysisSession) -> None:
        session.summary = compute_studio_session_summary(session.plays, session.session_mode)
        sdir = self.create_session_dir(session.session_id)
        target = sdir / "session.json"
        tmp = sdir / "session.json.tmp"
        tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, target)

    def load_session(self, session_id: str) -> AnalysisSession:
        sdir = self.session_dir(session_id)
        target = sdir / "session.json"
        if not target.is_file():
            raise FileNotFoundError(f"Session not found: {session_id}")
        session = AnalysisSession.model_validate_json(target.read_text(encoding="utf-8"))
        # Legacy session.json files predate `summary`; always serve a fresh one
        session.summary = compute_studio_session_summary(session.plays, session.session_mode)
        return session

    def list_sessions(self) -> list[AnalysisSession]:
        sessions: list[AnalysisSession] = []
        if not self.sessions_dir.is_dir():
            return sessions
        for entry in sorted(self.sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta_file = entry / "session.json"
            if entry.is_dir() and meta_file.is_file():
                try:
                    sessions.append(AnalysisSession.model_validate_json(meta_file.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return sessions

    def save_play_ball_cache(self, session_id: str, play_id: str, detections: list[BallDetection]) -> None:
        sdir = self.create_session_dir(session_id)
        safe_pid = validate_safe_id(play_id)
        cache_file = sdir / f"ball_cache_{safe_pid}.json"
        payload = [d.model_dump() for d in detections]
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

    def load_play_ball_cache(self, session_id: str, play_id: str) -> list[BallDetection]:
        sdir = self.session_dir(session_id)
        safe_pid = validate_safe_id(play_id)
        cache_file = sdir / f"ball_cache_{safe_pid}.json"
        if not cache_file.is_file():
            return []
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return [BallDetection.model_validate(item) for item in raw]

    def resolve_web_video_path(self, session_id: str) -> Path:
        sdir = self.session_dir(session_id)
        proxy = sdir / "web_proxy.mp4"
        if proxy.is_file():
            return proxy
        for ext in (".mp4", ".mov", ".mkv", ".avi"):
            cand = sdir / f"source{ext}"
            if cand.is_file():
                return cand
        ptr = sdir / "video_path.txt"
        if ptr.is_file():
            resolved = Path(ptr.read_text(encoding="utf-8").strip()).resolve()
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(f"Playable video missing for session {session_id}")

    def recalibrate_session_net(
        self,
        session_id: str,
        left_point: Point2D,
        right_point: Point2D,
        net_height_m: float = 2.43,
    ) -> AnalysisSession:
        """Update net calibration and recompute all play metrics purely from cached tracks."""
        session = self.load_session(session_id)
        cal = NetCalibration(
            left_point=left_point,
            right_point=right_point,
            net_height_m=net_height_m,
            is_manual=True,
        )
        session.calibration = cal

        for play in session.plays:
            ball_dets = self.load_play_ball_cache(session_id, play.play_id)
            if ball_dets:
                play.set_trajectory = fit_setting_path(
                    ball_dets,
                    calibration=cal,
                    fps=session.metadata.fps,
                    frame_height=session.metadata.height,
                    candidate_contact_frames=build_contact_candidates(play.tracks, cal),
                )
            elif play.set_trajectory is not None:
                # Recompute peak height using existing arc points
                from vball_analyser.calibration import height_above_tape

                px, cm = height_above_tape(cal, play.set_trajectory.peak_point.x, play.set_trajectory.peak_point.y)
                play.set_trajectory.peak_height_px = px
                play.set_trajectory.peak_height_cm = cm

            self._reanalyse_play(session, play, None, play.hitter_track_id, _manual_side(play))

        self.save_session(session)
        return session

    def _video_or_none(self, session_id: str) -> Path | None:
        try:
            return self.resolve_web_video_path(session_id)
        except FileNotFoundError:
            return None

    def _reanalyse_play(
        self,
        session: AnalysisSession,
        play: PlayAnalysis,
        video_path: Path | None,
        override_hitter_track_id: int | None,
        override_team_side: str | None,
    ) -> None:
        """Re-run cached side/mechanics analysis for one play; extract changed keyframe JPEGs when video exists."""

        def _run(hitter_override: int | None):  # noqa: ANN202
            return analyse_play_full(
                play.tracks,
                play.set_trajectory,
                calibration=session.calibration,
                fps=session.metadata.fps,
                frame_shape=(session.metadata.height, session.metadata.width),
                session_mode=session.session_mode,
                override_hitter_track_id=hitter_override,
                override_team_side=override_team_side,
            )

        result = _run(override_hitter_track_id)
        if result[0] != play.team_side and override_hitter_track_id is not None:
            # Side flipped: the stored hitter belongs to the other court, let the new offense side pick
            result = _run(None)
        team_side, system_state, pin_distance_m, setter_id, hitter_id, hitter_metrics, block_summary, keyframes = result
        play.team_side = team_side
        play.system_state = system_state
        play.pin_distance_m = pin_distance_m
        play.setter_track_id = setter_id
        play.hitter_track_id = hitter_id
        play.hitter_metrics = hitter_metrics
        play.block_summary = block_summary

        sdir = self.session_dir(session.session_id)
        old_thumbs = {k.name: k for k in play.keyframes}
        for km in keyframes:
            prev = old_thumbs.get(km.name)
            if prev is not None and prev.frame_idx == km.frame_idx and prev.thumbnail_rel_path:
                km.thumbnail_rel_path = prev.thumbnail_rel_path
            elif video_path is not None:
                fname = f"{play.play_id}_{km.name}.jpg"
                extract_keyframe_jpeg(video_path, km.frame_idx, sdir / "thumbnails" / fname)
                km.thumbnail_rel_path = fname
        play.keyframes = keyframes

        # Same zone fallback chain as the detector (contact -> set arc end -> net centre), mirrored per side
        cal = session.calibration
        if hitter_metrics is not None and hitter_metrics.contact_point is not None:
            zone_x = hitter_metrics.contact_point.x
        elif play.set_trajectory is not None and play.set_trajectory.arc_points:
            zone_x = play.set_trajectory.arc_points[-1].x
        else:
            zone_x = 0.5 * (cal.left_point.x + cal.right_point.x)
        play.attack_zone = classify_attack_zone(cal, zone_x, team_side=team_side)

    def update_session_mode(self, session_id: str, session_mode: Literal["drill", "match"]) -> AnalysisSession:
        """Switch Drill/Match and re-evaluate auto sides, block states, and summary from cached tracks."""
        session = self.load_session(session_id)
        session.session_mode = session_mode
        video_path = self._video_or_none(session_id)
        for play in session.plays:
            self._reanalyse_play(session, play, video_path, play.hitter_track_id, _manual_side(play))
        self.save_session(session)
        return session

    def update_play_team_side(
        self, session_id: str, play_id: str, team_side: Literal["ours", "opponent", "auto"]
    ) -> AnalysisSession:
        """Pin a play to Ours/Opponent (or hand it back to the classifier with "auto")."""
        session = self.load_session(session_id)
        play = next((p for p in session.plays if p.play_id == play_id), None)
        if play is None:
            raise KeyError(f"Play not found: {play_id}")
        play.team_side_manual = team_side != "auto"
        self._reanalyse_play(
            session,
            play,
            self._video_or_none(session_id),
            play.hitter_track_id,
            team_side if play.team_side_manual else None,
        )
        self.save_session(session)
        return session

    def reassign_play_hitter(self, session_id: str, play_id: str, track_id: int) -> AnalysisSession:
        """Swap active hitter to another cached track ID without re-running GPU detection."""
        session = self.load_session(session_id)
        target_play = next((p for p in session.plays if p.play_id == play_id), None)
        if target_play is None:
            raise KeyError(f"Play not found: {play_id}")
        track = next((t for t in target_play.tracks if t.track_id == track_id), None)
        if track is None:
            raise ValueError(f"Track {track_id} not present in play {play_id}")

        video_path = self._video_or_none(session_id)
        self._reanalyse_play(session, target_play, video_path, track_id, _manual_side(target_play))
        other_court = "blocking" if target_play.team_side == "ours" else "attacking"
        if target_play.hitter_track_id != track_id and track.court_side == other_court:
            # User explicitly picked a hitter on the other court: keep team_side consistent with that track
            target_play.team_side = "opponent" if target_play.team_side == "ours" else "ours"
            target_play.team_side_manual = True
            self._reanalyse_play(session, target_play, video_path, track_id, target_play.team_side)

        self.save_session(session)
        return session

    def delete_play(self, session_id: str, play_id: str) -> AnalysisSession:
        session = self.load_session(session_id)
        before = len(session.plays)
        session.plays = [p for p in session.plays if p.play_id != play_id]
        if len(session.plays) == before:
            raise KeyError(f"Play not found: {play_id}")
        for idx, p in enumerate(session.plays, start=1):
            p.index = idx
        self.save_session(session)
        return session


def _manual_side(play: PlayAnalysis) -> str | None:
    return play.team_side if play.team_side_manual else None
