"""FastAPI backend for the Volleyball Baseline Training Analyser."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vball_analyser.calibration import auto_detect_net_tape
from vball_analyser.config import DEFAULT_CONFIG, STATIC_DIR, AppConfig
from vball_analyser.detector import (
    FramePredictor,
    YoloGpuPredictor,
    analyse_play_window,
    scan_candidate_plays,
)
from vball_analyser.models import AnalysisSession, Point2D
from vball_analyser.renderer import export_annotated_play_mp4
from vball_analyser.security import (
    ALLOWED_HTTP_METHODS,
    SECURITY_HEADERS,
    SESSION_COOKIE_NAME,
    SessionAuthManager,
    SlidingWindowRateLimiter,
    resolve_secret_key,
    validate_local_video_path,
)
from vball_analyser.storage import SessionStore, validate_safe_id
from vball_analyser.video_io import (
    ensure_web_playable_mp4,
    iter_video_frames,
    parse_byte_range,
    probe_video,
    validate_video_magic_bytes,
)

logger = logging.getLogger(__name__)


class LocalVideoRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    net_height_m: float = Field(default=2.43, ge=1.8, le=2.7)


class CalibrationRequest(BaseModel):
    left_point: Point2D
    right_point: Point2D
    net_height_m: float = Field(default=2.43, ge=1.8, le=2.7)


class AddPlayRequest(BaseModel):
    center_time_s: float | None = Field(default=None, ge=0.0)
    start_time_s: float | None = Field(default=None, ge=0.0)
    end_time_s: float | None = Field(default=None, gt=0.0)


class ReassignHitterRequest(BaseModel):
    track_id: int


class ExportPlayRequest(BaseModel):
    freeze_contact: bool = True
    show_inline_stats: bool = True


def _set_session_cookie(response: Response, cookie_val: str, ttl_seconds: int) -> None:
    # TODO(security): Enforce __Host-/Secure flags, OAuth, and MFA if deployed behind TLS for multi-user remote access.
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=cookie_val,
        max_age=ttl_seconds,
        httponly=True,
        samesite="strict",
        path="/",
    )


def create_app(
    config: AppConfig = DEFAULT_CONFIG,
    predictor: FramePredictor | None = None,
) -> FastAPI:
    """Create configured FastAPI app with security middleware and GPU worker state."""
    secret_key = resolve_secret_key(config.secret_file)
    auth = SessionAuthManager(secret_key=secret_key, ttl_seconds=config.session_ttl_seconds)
    rate_limiter = SlidingWindowRateLimiter(
        max_requests=config.rate_limit_requests,
        window_s=config.rate_limit_window_s,
    )
    store = SessionStore(config.sessions_dir)
    active_predictor: FramePredictor = predictor if predictor is not None else YoloGpuPredictor()
    active_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        for task in list(active_tasks):
            task.cancel()

    app = FastAPI(
        title="VBall Analyser — Baseline Studio",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        if request.method not in ALLOWED_HTTP_METHODS:
            resp = JSONResponse({"detail": "Method not allowed."}, status_code=405)
            for k, v in SECURITY_HEADERS.items():
                resp.headers[k] = v
            return resp

        if request.url.path.startswith("/api/"):
            client_ip = request.client.host if request.client else "local"
            if not rate_limiter.allow(client_ip):
                resp = JSONResponse({"detail": "Rate limit exceeded."}, status_code=429)
                for k, v in SECURITY_HEADERS.items():
                    resp.headers[k] = v
                return resp

        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/api/") and "application/json" in (response.headers.get("content-type") or ""):
            response.headers["Cache-Control"] = "no-store"
        return response

    def _require_csrf(request: Request) -> str:
        sid = auth.verify_cookie(request.cookies.get(SESSION_COOKIE_NAME))
        if sid is None:
            raise HTTPException(status_code=403, detail="Missing or expired session.")
        token = request.headers.get("X-CSRF-Token")
        if not auth.verify_csrf_token(sid, token):
            raise HTTPException(status_code=403, detail="Invalid CSRF token.")
        return sid

    async def _run_session_analysis(session_id: str, source_path: Path) -> None:
        sdir = store.session_dir(session_id)
        try:
            session = store.load_session(session_id)
            session.status = "scanning"
            session.progress_pct = 8.0
            store.save_session(session)

            web_mp4 = await asyncio.to_thread(ensure_web_playable_mp4, source_path, sdir)
            if web_mp4 != source_path:
                session.metadata = await asyncio.to_thread(probe_video, web_mp4)

            if not session.calibration.is_manual:
                sample_frames = await asyncio.to_thread(
                    lambda: [bgr for _i, _t, bgr in iter_video_frames(web_mp4, start_s=0.0, end_s=min(8.0, session.metadata.duration_s), sample_fps=2.0)]
                )
                detected_cal = await asyncio.to_thread(auto_detect_net_tape, sample_frames)
                detected_cal.net_height_m = session.calibration.net_height_m
                session.calibration = detected_cal

            session.progress_pct = 25.0
            store.save_session(session)

            windows = await asyncio.to_thread(
                scan_candidate_plays, web_mp4, session.calibration, active_predictor, session.metadata
            )
            if not windows:
                windows = [(0.0, min(session.metadata.duration_s, 6.0))]

            session.status = "analysing"
            session.progress_pct = 35.0
            store.save_session(session)

            total = max(1, len(windows))
            for idx, (w_start, w_end) in enumerate(windows, start=1):
                # Re-read calibration in case user adjusted 2-click endpoints while scanning
                latest = store.load_session(session_id)
                cal_to_use = latest.calibration
                play, raw_balls = await asyncio.to_thread(
                    analyse_play_window,
                    web_mp4,
                    idx,
                    w_start,
                    w_end,
                    cal_to_use,
                    active_predictor,
                    latest.metadata,
                    sdir,
                )
                store.save_play_ball_cache(session_id, play.play_id, raw_balls)
                latest.plays.append(play)
                latest.progress_pct = round(35.0 + 65.0 * (idx / total), 1)
                store.save_session(latest)

            final_sess = store.load_session(session_id)
            final_sess.status = "ready"
            final_sess.progress_pct = 100.0
            store.save_session(final_sess)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Analysis pipeline failed for session %s", session_id)
            try:
                err_sess = store.load_session(session_id)
                err_sess.status = "error"
                err_sess.error_message = "Video analysis failed. Check server logs for details."
                store.save_session(err_sess)
            except Exception:
                pass

    def _schedule_analysis(session_id: str, source_path: Path) -> None:
        task = asyncio.create_task(_run_session_analysis(session_id, source_path))
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)

    @app.get("/")
    async def index(request: Request) -> FileResponse:
        sid = auth.verify_cookie(request.cookies.get(SESSION_COOKIE_NAME))
        resp = FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")
        if sid is None:
            _sid, cookie_val = auth.create_session()
            _set_session_cookie(resp, cookie_val, config.session_ttl_seconds)
        return resp

    @app.get("/api/csrf")
    async def get_csrf(request: Request) -> JSONResponse:
        sid = auth.verify_cookie(request.cookies.get(SESSION_COOKIE_NAME))
        cookie_val: str | None = None
        if sid is None:
            sid, cookie_val = auth.create_session()
        token = auth.issue_csrf_token(sid)
        resp = JSONResponse({"csrf_token": token})
        if cookie_val is not None:
            _set_session_cookie(resp, cookie_val, config.session_ttl_seconds)
        return resp

    @app.post("/api/session/reset")
    async def reset_auth_session(request: Request) -> JSONResponse:
        old_sid = _require_csrf(request)
        auth.invalidate_session(old_sid)
        new_sid, cookie_val = auth.create_session()
        token = auth.issue_csrf_token(new_sid)
        resp = JSONResponse({"csrf_token": token, "status": "reset"})
        _set_session_cookie(resp, cookie_val, config.session_ttl_seconds)
        return resp

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict]:
        return [s.model_dump() for s in store.list_sessions()]

    @app.post("/api/sessions/upload")
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),
        net_height_m: float = Form(2.43),
    ) -> JSONResponse:
        _require_csrf(request)
        if not (1.8 <= net_height_m <= 2.7):
            raise HTTPException(status_code=400, detail="Invalid net height.")

        raw_name = Path(file.filename or "upload.mp4").name
        ext = Path(raw_name).suffix.lower()
        if ext not in config.allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported video file extension.")

        session_id = uuid.uuid4().hex[:16]
        sdir = store.create_session_dir(session_id)
        dest_path = (sdir / f"source{ext}").resolve()
        if not dest_path.is_relative_to(sdir):
            raise HTTPException(status_code=400, detail="Invalid upload destination.")

        # TODO(security): Integrate sandboxed AV/CDR scanning for untrusted external uploads.
        total_bytes = 0
        first_chunk = b""
        try:
            with dest_path.open("wb") as out_fh:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    if not first_chunk:
                        first_chunk = chunk[:32]
                        if not validate_video_magic_bytes(first_chunk, ext):
                            raise HTTPException(status_code=400, detail="Invalid video container signature.")
                    total_bytes += len(chunk)
                    if total_bytes > config.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Uploaded video exceeds maximum allowed size.")
                    out_fh.write(chunk)
            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Empty video upload.")
            os.chmod(dest_path, 0o600)
            meta = await asyncio.to_thread(probe_video, dest_path)
        except HTTPException:
            shutil.rmtree(sdir, ignore_errors=True)
            raise
        except Exception as exc:
            logger.warning("Uploaded video probe failed: %s", exc)
            shutil.rmtree(sdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Unable to decode uploaded video.") from exc

        initial_cal = auto_detect_net_tape([])
        initial_cal.net_height_m = net_height_m
        session = AnalysisSession(
            session_id=session_id,
            filename=raw_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="queued",
            progress_pct=0.0,
            metadata=meta,
            calibration=initial_cal,
            plays=[],
        )
        store.save_session(session)
        _schedule_analysis(session_id, dest_path)
        return JSONResponse(session.model_dump())

    @app.post("/api/sessions/local")
    async def load_local_video(request: Request, body: LocalVideoRequest) -> JSONResponse:
        _require_csrf(request)
        try:
            resolved = validate_local_video_path(
                body.path,
                allowed_dirs=config.allowed_local_dirs,
                allowed_extensions=config.allowed_extensions,
            )
            meta = await asyncio.to_thread(probe_video, resolved)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("Local video probe failed: %s", exc)
            raise HTTPException(status_code=400, detail="Unable to decode local video file.") from exc

        session_id = uuid.uuid4().hex[:16]
        sdir = store.create_session_dir(session_id)
        (sdir / "video_path.txt").write_text(str(resolved), encoding="utf-8")

        initial_cal = auto_detect_net_tape([])
        initial_cal.net_height_m = body.net_height_m
        session = AnalysisSession(
            session_id=session_id,
            filename=resolved.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="queued",
            progress_pct=0.0,
            metadata=meta,
            calibration=initial_cal,
            plays=[],
        )
        store.save_session(session)
        _schedule_analysis(session_id, resolved)
        return JSONResponse(session.model_dump())

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        try:
            session = store.load_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        return JSONResponse(session.model_dump())

    @app.get("/api/sessions/{session_id}/video")
    async def stream_session_video(session_id: str, request: Request) -> Response:
        try:
            video_path = store.resolve_web_video_path(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Video not ready.") from exc

        file_size = video_path.stat().st_size
        range_header = request.headers.get("range")
        if not range_header:
            return FileResponse(
                video_path,
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes", "X-Content-Type-Options": "nosniff"},
            )

        try:
            start, end = parse_byte_range(range_header, file_size)
        except ValueError as exc:
            raise HTTPException(
                status_code=416,
                detail=str(exc),
                headers={"Content-Range": f"bytes */{file_size}"},
            ) from exc

        length = end - start + 1

        def _iter_chunks() -> object:
            with video_path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _iter_chunks(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/sessions/{session_id}/thumbnails/{filename}")
    async def get_thumbnail(session_id: str, filename: str) -> FileResponse:
        try:
            sdir = store.session_dir(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if Path(filename).name != filename or not filename.lower().endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Invalid thumbnail filename.")
        thumbs_dir = (sdir / "thumbnails").resolve()
        target = (thumbs_dir / filename).resolve()
        if not target.is_relative_to(thumbs_dir) or not target.is_file():
            raise HTTPException(status_code=404, detail="Thumbnail not found.")
        return FileResponse(target, media_type="image/jpeg")

    @app.post("/api/sessions/{session_id}/calibration")
    async def update_calibration(
        session_id: str, request: Request, body: CalibrationRequest
    ) -> JSONResponse:
        _require_csrf(request)
        try:
            updated = store.recalibrate_session_net(
                session_id,
                left_point=body.left_point,
                right_point=body.right_point,
                net_height_m=body.net_height_m,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        return JSONResponse(updated.model_dump())

    @app.post("/api/sessions/{session_id}/plays/add")
    async def add_play(session_id: str, request: Request, body: AddPlayRequest) -> JSONResponse:
        _require_csrf(request)
        try:
            session = store.load_session(session_id)
            video_path = store.resolve_web_video_path(session_id)
            sdir = store.session_dir(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session or video not found.") from exc

        dur = session.metadata.duration_s
        if body.start_time_s is not None and body.end_time_s is not None:
            start_s = max(0.0, min(dur - 0.5, body.start_time_s))
            end_s = min(dur, max(start_s + 0.5, body.end_time_s))
        else:
            center = body.center_time_s if body.center_time_s is not None else 0.5 * dur
            start_s = max(0.0, center - 2.5)
            end_s = min(dur, center + 1.5)

        play, raw_balls = await asyncio.to_thread(
            analyse_play_window,
            video_path,
            len(session.plays) + 1,
            start_s,
            end_s,
            session.calibration,
            active_predictor,
            session.metadata,
            sdir,
        )
        store.save_play_ball_cache(session_id, play.play_id, raw_balls)
        session.plays.append(play)
        session.plays.sort(key=lambda p: p.start_time_s)
        for idx, p in enumerate(session.plays, start=1):
            p.index = idx
        store.save_session(session)
        return JSONResponse(session.model_dump())

    @app.post("/api/sessions/{session_id}/plays/{play_id}/hitter")
    async def reassign_hitter(
        session_id: str, play_id: str, request: Request, body: ReassignHitterRequest
    ) -> JSONResponse:
        _require_csrf(request)
        try:
            validate_safe_id(play_id)
            updated = store.reassign_play_hitter(session_id, play_id, body.track_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        return JSONResponse(updated.model_dump())

    @app.post("/api/sessions/{session_id}/plays/{play_id}/delete")
    async def delete_play(session_id: str, play_id: str, request: Request) -> JSONResponse:
        _require_csrf(request)
        try:
            validate_safe_id(play_id)
            updated = store.delete_play(session_id, play_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session not found.") from exc
        return JSONResponse(updated.model_dump())

    @app.post("/api/sessions/{session_id}/plays/{play_id}/export")
    async def export_play(
        session_id: str, play_id: str, request: Request, body: ExportPlayRequest
    ) -> JSONResponse:
        _require_csrf(request)
        try:
            safe_pid = validate_safe_id(play_id)
            session = store.load_session(session_id)
            video_path = store.resolve_web_video_path(session_id)
            sdir = store.session_dir(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session or video not found.") from exc

        play = next((p for p in session.plays if p.play_id == safe_pid), None)
        if play is None:
            raise HTTPException(status_code=404, detail="Play not found.")

        export_name = f"{safe_pid}_{'freeze' if body.freeze_contact else 'plain'}{'' if body.show_inline_stats else '_clean'}.mp4"
        out_path = (sdir / "exports" / export_name).resolve()
        freeze_s = 1.5 if body.freeze_contact else 0.0
        await asyncio.to_thread(
            export_annotated_play_mp4,
            video_path,
            play,
            session.calibration,
            out_path,
            freeze_s,
            body.show_inline_stats,
        )
        return JSONResponse(
            {
                "download_url": f"/api/sessions/{session_id}/exports/{export_name}",
                "filename": export_name,
            }
        )

    @app.get("/api/sessions/{session_id}/exports/{filename}")
    async def download_export(session_id: str, filename: str) -> FileResponse:
        try:
            sdir = store.session_dir(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if Path(filename).name != filename or not filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Invalid export filename.")
        exports_dir = (sdir / "exports").resolve()
        target = (exports_dir / filename).resolve()
        if not target.is_relative_to(exports_dir) or not target.is_file():
            raise HTTPException(status_code=404, detail="Export not found.")
        return FileResponse(
            target,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app


app = create_app()


def main() -> None:
    """Launch the local Volleyball Baseline Training Analyser studio on 127.0.0.1."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=DEFAULT_CONFIG.bind_host, port=DEFAULT_CONFIG.default_port)
