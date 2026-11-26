"""Disk-backed JSON persistence for Serve Distribution practice sessions."""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from vball_analyser.models import CourtCalibration, PasserPosition, ServeEvent, ServeSession
from vball_analyser.serve_geometry import (
    classify_serve,
    compute_court_homography,
    default_passer_formation,
)
from vball_analyser.storage import validate_safe_id

_UNSET: Any = object()


class ServeSessionStore:
    """Stores serve distribution sessions inside data/serve_sessions/<id>/serve_session.json."""

    def __init__(self, serve_sessions_dir: Path) -> None:
        self.serve_sessions_dir = serve_sessions_dir.resolve()
        self.serve_sessions_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.serve_sessions_dir, 0o700)
        except OSError:
            pass

    def session_dir(self, serve_session_id: str) -> Path:
        safe_id = validate_safe_id(serve_session_id)
        candidate = (self.serve_sessions_dir / safe_id).resolve()
        if not candidate.is_relative_to(self.serve_sessions_dir):
            raise ValueError("Path escapes serve_sessions directory.")
        return candidate

    def create_session(
        self,
        name: str,
        video_session_id: str | None = None,
    ) -> ServeSession:
        clean_name = name.strip() or "Practice Session"
        if video_session_id is not None:
            validate_safe_id(video_session_id)
        sid = f"srv_{uuid.uuid4().hex[:12]}"
        session = ServeSession(
            serve_session_id=sid,
            name=clean_name[:80],
            created_at=datetime.now(timezone.utc).isoformat(),
            video_session_id=video_session_id,
            status="idle",
            progress_pct=0.0,
            default_passers=default_passer_formation(3),
            servers=[],
            serves=[],
        )
        self.save_session(session)
        return session

    def save_session(self, session: ServeSession) -> None:
        sdir = self.session_dir(session.serve_session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(sdir, 0o700)
        except OSError:
            pass
        target = sdir / "serve_session.json"
        tmp = sdir / "serve_session.json.tmp"
        tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, target)

    def load_session(self, serve_session_id: str) -> ServeSession:
        sdir = self.session_dir(serve_session_id)
        target = sdir / "serve_session.json"
        if not target.is_file():
            raise FileNotFoundError(f"Serve session not found: {serve_session_id}")
        return ServeSession.model_validate_json(target.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[ServeSession]:
        sessions: list[ServeSession] = []
        if not self.serve_sessions_dir.is_dir():
            return sessions
        for entry in sorted(
            self.serve_sessions_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            meta = entry / "serve_session.json"
            if entry.is_dir() and meta.is_file():
                try:
                    sessions.append(ServeSession.model_validate_json(meta.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return sessions

    def delete_session(self, serve_session_id: str) -> None:
        sdir = self.session_dir(serve_session_id)
        if not sdir.is_dir():
            raise FileNotFoundError(f"Serve session not found: {serve_session_id}")
        shutil.rmtree(sdir)

    def update_session(
        self,
        serve_session_id: str,
        *,
        name: str | None = None,
        video_session_id: str | None = _UNSET,
        servers: list[str] | None = None,
        default_passers: list[PasserPosition] | None = None,
    ) -> ServeSession:
        session = self.load_session(serve_session_id)
        if name is not None:
            clean = name.strip()
            if clean:
                session.name = clean[:80]
        if video_session_id is not _UNSET:
            if video_session_id is not None:
                validate_safe_id(video_session_id)
            session.video_session_id = video_session_id
        if servers is not None:
            deduped: list[str] = []
            for raw in servers:
                label = raw.strip()[:32]
                if label and label not in deduped:
                    deduped.append(label)
            session.servers = deduped
        if default_passers is not None:
            # Updating default_passers applies to future manual serves only;
            # historical serves keep their own passers_snapshot.
            session.default_passers = [
                PasserPosition(label=p.label.strip()[:24], x_m=round(p.x_m, 3), y_m=round(p.y_m, 3))
                for p in default_passers
            ]
        self.save_session(session)
        return session

    def save_calibration(self, serve_session_id: str, cal: CourtCalibration) -> ServeSession:
        compute_court_homography(cal)
        session = self.load_session(serve_session_id)
        session.court_calibration = cal.model_copy(update={"is_calibrated": True})
        self.save_session(session)
        return session

    def add_serve(
        self,
        serve_session_id: str,
        *,
        x_m: float,
        y_m: float,
        result: Literal["in", "ace", "out", "net"] = "in",
        server_name: str = "Unassigned",
        source: Literal["auto", "manual"] = "manual",
        timestamp_s: float | None = None,
        frame_idx: int | None = None,
        passers_snapshot: list[PasserPosition] | None = None,
    ) -> ServeSession:
        session = self.load_session(serve_session_id)
        snapshot = (
            [PasserPosition(label=p.label, x_m=round(p.x_m, 3), y_m=round(p.y_m, 3)) for p in passers_snapshot]
            if passers_snapshot is not None
            else [p.model_copy() for p in session.default_passers]
        )
        nx, ny, norm_res, zone, depth, is_seam, seam_pair = classify_serve(x_m, y_m, result, snapshot)
        clean_server = server_name.strip()[:32] or "Unassigned"
        if clean_server != "Unassigned" and clean_server not in session.servers:
            session.servers.append(clean_server)

        event = ServeEvent(
            serve_id=f"s_{uuid.uuid4().hex[:10]}",
            index=len(session.serves) + 1,
            source=source,
            timestamp_s=round(timestamp_s, 3) if timestamp_s is not None else None,
            frame_idx=frame_idx,
            server_name=clean_server,
            x_m=nx,
            y_m=ny,
            result=norm_res,
            zone=zone,
            depth=depth,
            is_seam=is_seam,
            seam_pair=seam_pair,
            passers_snapshot=snapshot,
        )
        session.serves.append(event)
        self.save_session(session)
        return session

    def update_serve(
        self,
        serve_session_id: str,
        serve_id: str,
        *,
        x_m: float | None = None,
        y_m: float | None = None,
        result: Literal["in", "ace", "out", "net"] | None = None,
        server_name: str | None = None,
        passers_snapshot: list[PasserPosition] | None = None,
    ) -> ServeSession:
        validate_safe_id(serve_id)
        session = self.load_session(serve_session_id)
        target = next((s for s in session.serves if s.serve_id == serve_id), None)
        if target is None:
            raise KeyError(f"Serve not found: {serve_id}")

        geometry_changed = False
        new_x = target.x_m if x_m is None else float(x_m)
        new_y = target.y_m if y_m is None else float(y_m)
        new_res = target.result if result is None else result
        if x_m is not None or y_m is not None or result is not None:
            geometry_changed = True

        if passers_snapshot is not None:
            target.passers_snapshot = [
                PasserPosition(label=p.label.strip()[:24], x_m=round(p.x_m, 3), y_m=round(p.y_m, 3))
                for p in passers_snapshot
            ]
            geometry_changed = True

        if server_name is not None:
            clean_server = server_name.strip()[:32] or "Unassigned"
            target.server_name = clean_server
            if clean_server != "Unassigned" and clean_server not in session.servers:
                session.servers.append(clean_server)

        if geometry_changed:
            nx, ny, norm_res, zone, depth, is_seam, seam_pair = classify_serve(
                new_x, new_y, new_res, target.passers_snapshot
            )
            target.x_m = nx
            target.y_m = ny
            target.result = norm_res
            target.zone = zone
            target.depth = depth
            target.is_seam = is_seam
            target.seam_pair = seam_pair
            # Flip auto-detected serves to manual once user edits them so re-scan preserves fixes.
            target.source = "manual"

        self.save_session(session)
        return session

    def delete_serve(self, serve_session_id: str, serve_id: str) -> ServeSession:
        validate_safe_id(serve_id)
        session = self.load_session(serve_session_id)
        remaining = [s for s in session.serves if s.serve_id != serve_id]
        if len(remaining) == len(session.serves):
            raise KeyError(f"Serve not found: {serve_id}")
        for idx, s in enumerate(remaining, start=1):
            s.index = idx
        session.serves = remaining
        self.save_session(session)
        return session

    def replace_auto_serves(
        self,
        serve_session_id: str,
        auto_serves: list[ServeEvent],
    ) -> ServeSession:
        session = self.load_session(serve_session_id)
        kept_manual = [s for s in session.serves if s.source == "manual"]
        combined = kept_manual + list(auto_serves)
        combined.sort(
            key=lambda s: (
                0 if s.timestamp_s is not None else 1,
                s.timestamp_s if s.timestamp_s is not None else 0.0,
                s.index,
            )
        )
        for idx, s in enumerate(combined, start=1):
            s.index = idx
        session.serves = combined
        self.save_session(session)
        return session
