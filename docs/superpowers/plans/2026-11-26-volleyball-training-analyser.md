# Volleyball Baseline Training Analyser — Implementation Plan

> **For agentic workers:** Implementing task-by-task following `docs/superpowers/specs/2026-11-26-volleyball-training-analyser-design.md`.

**Goal:** Build a self-contained local GPU-accelerated Volleyball Baseline Training Analyser (`FastAPI` + `YOLO11` + `PyAV` + HTML5 `<video>`/`<canvas>` Interactive Review Studio) supporting both short single-play clips and long uncut baseline recordings.

**Architecture:** Two-pass local GPU pipeline (`YOLO11` + `ByteTrack` + gravity-constrained RANSAC set-arc fitting) feeds cached play tracks and metrics to a FastAPI backend (`127.0.0.1` only) and a HiDPI `<canvas>` overlay studio with `PyAV` H.264 MP4 export.

**Tech Stack:** Python 3.12 (`uv`), FastAPI, Uvicorn, Pydantic, PyAV (`av`), OpenCV (`opencv-python-headless`), Ultralytics (`YOLO11`), NumPy, SciPy, Vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-11-26-volleyball-training-analyser-design.md`

---

## Global Constraints

- Python 3.12 strictly managed via `uv` (`.venv` in repo root, `--link-mode=copy` on WSL2 `/mnt/c`).
- No system `ffmpeg`; all video probing, decoding, H.264 `yuv420p` proxy transcoding, thumbnails, and MP4 export use `PyAV` (`av`) + `opencv-python-headless`.
- Pixel coordinates: origin top-left, $y$ downward; `height_above_tape_px = tape_y_at(x) - y`.
- Security: `127.0.0.1` bind only, strict CSP/headers, `HttpOnly`/`SameSite=strict` cookie session, HMAC-bound `X-CSRF-Token` on all `POST` requests, `GET`/`HEAD`/`POST` allow-list (`405` otherwise), rate limiting, magic-byte + extension + size-cap upload validation, `Path.resolve(strict=True)` + `is_relative_to()` local path confinement, zero `innerHTML`/`alert`/`localStorage` in frontend JS.

## Review Focus

1. Stray drill balls & ball carts rejected by motion weighting + RANSAC parabolic fitting.
2. Short/sparse clips (`< 4` ball frames) still yield a playable `PlayAnalysis` without crashing.
3. Tilted tripod net tape (`y_L != y_R`) interpolates `y_tape(x)` at contact/peak $x$.
4. Simultaneous attackers allow instant cached `hitter_track_id` reassignment (`< 50ms`).
5. `GET /api/sessions/{id}/video` supports `206 Partial Content` HTTP `Range` requests + `PyAV` H.264 proxy transcoding.

## Ordered Steps

- Step 0: Scaffold repo, `.venv`, Python 3.12 dependencies, CUDA verification, git init.
- Step 1: Data schemas (`models.py`), `config.py`, `PyAV` video I/O (`video_io.py`), synthetic baseline fixture (`tests/conftest.py`, `tests/test_video_io.py`).
- Step 2: Net top-tape calibration & zone geometry (`calibration.py`, `tests/test_calibration.py`).
- Step 3: Gravity-constrained RANSAC set-arc fitter (`trajectory.py`, `tests/test_trajectory.py`).
- Step 4: Pose smoothing, role isolation, hitter posture & blocker seam analysis (`biomechanics.py`, `tests/test_biomechanics.py`).
- Step 5: Two-pass GPU pipeline (`detector.py`), session storage (`storage.py`), annotated MP4 exporter (`renderer.py`), tests (`tests/test_pipeline.py`, `tests/test_renderer.py`).
- Step 6: Security-hardened FastAPI backend (`security.py`, `app.py`), Interactive Review Studio (`static/`), security & E2E tests (`tests/test_api_security.py`).
- Step 7: Full test suite, live `127.0.0.1` smoke check, security verification, final commit.

## Verification Plan

### Automated Security Check

- **Security Scanner**: Check `~/.securecoder/api.json` / `$SECURECODER_API_PORT`; if active, run `run_security_scanner` on all newly created files to identify vulnerabilities (XSS, path traversal, secrets, headers), remediate any findings, and re-scan. (If extension sidecar is absent, skip per `run_security_scanner` Step 0.)
- **Automated Security Tests (`tests/test_api_security.py`)**: Pin CSP, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Permissions-Policy`, `Cache-Control: no-store`, `405` on disallowed HTTP methods, `403` on missing/invalid CSRF tokens, `400`/`403` on path traversal & prefix-bypass local paths, `400` on invalid magic bytes, `413` on oversized uploads, and static scan of `static/` and `src/` for unsafe DOM sinks or `0.0.0.0` binding.

### Security Audit

- Audit new code for input validation, secrets resolution, session/CSRF checks, and safe file serving; document findings and remediations in `walkthrough.md` via `generate_security_audit_report` if SecureCoder scanner ran.
