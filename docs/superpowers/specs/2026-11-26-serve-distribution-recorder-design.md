# Serve Distribution Recorder — Design Document

**Date:** 2026-11-26  
**Project:** VBall Analyser (`vball_analyser`)  
**Status:** Ready for user approval (builder-checked)

---

## 1. Intent & Why

VBall Analyser currently focuses on single-play **set-and-hit** mechanics from baseline video. During volleyball **practice**, coaches and players also need a fast way to track **serve distribution** — specifically:

1. **Which area of the receiving court** each serve targets (Zones 1–6, short vs. deep, or out of bounds).
2. **Whether serves land in the seams between passers** (e.g., between left and middle passer in a 3-passer receive formation), which is a primary tactical goal in serving drills.
3. **Automatic logging from baseline practice video** (like `data/test_video.mp4`), backed by an interactive top-down court diagram where coaches can verify, adjust, or manually log serves.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- Add a new **Serve Distribution** tab alongside the existing **Set & Hit Studio** tab without disrupting existing set-and-hit workflows.
- Support **automated serve logging** from baseline video (camera behind the receiving team's baseline, matching `data/test_video.mp4`) using the existing `YOLO11` + `ByteTrack` GPU pipeline and a **4-corner front-3m court calibration**.
- Map each serve's **landing/reception location** and the **pre-serve passer positions** onto a 2D top-down 9m × 9m receiving court diagram.
- Automatically classify every serve into its **court zone** (Z1–Z6, short/deep, or out-of-bounds) and **seam status** (whether it landed in between two adjacent passers, and which pair).
- Allow **full manual control**: click the court to log serves without video, drag passers to match any practice formation (2/3/4-passer presets or free placement), drag/edit/delete auto-detected serves, and click any logged serve to jump the video to its reception frame.
- Persist serve practice sessions locally as JSON in `data/` behind the existing FastAPI + CSRF security layer.

### Non-Goals (v1)
- Match-level rotation tracking, scoreboards, or opponent scouting dashboards (practice-first for now).
- Live webcam/RTSP real-time streaming ingestion (recorded practice videos + manual live clicking only).
- Automatic jersey-number OCR to identify which server served each ball (servers are tagged/filtered manually via a quick server selector).
- Automatic pass-quality grading (0–3 scale) or serve speed (`km/h`) radar reconstruction.

### Anti-Goals (Failure Modes to Avoid)
- **Black-box auto-logging that can't be fixed:** If CV misses a serve or misplaces a dot by 1 meter, the user must be able to correct, delete, or add it in one click.
- **Requiring the near baseline to be visible in frame:** In `data/test_video.mp4`, the near baseline sits at/below the bottom edge of the frame. Requiring baseline corners for calibration would fail on real gym footage. Calibration must use the visible **front 3m rectangle**.
- **Mid-air depth distortion:** Projecting a ball caught 1.2m off the ground straight through a floor homography shifts its apparent depth away from the camera. Reception coordinates must anchor to the receiving passer's floor position at contact (falling back to ball floor contact only for unplayed bounces).
- **Breaking existing Set & Hit Studio state:** Switching tabs must not reset the loaded video, playback position, or overlays in Set & Hit Studio.

---

## 3. Context & Constraints

- **Existing Stack:** Python 3.12 (`uv`), FastAPI (`127.0.0.1:8081`), Pydantic v2, OpenCV (`opencv-python-headless`), NumPy/SciPy, PyAV, Ultralytics `YOLO11` + `ByteTrack`, and vanilla HTML/CSS/JS (`src/vball_analyser/static/`). No frontend build step or npm toolchain.
- **Security Layer:** All mutating `/api/*` endpoints must enforce `_require_csrf(request)`, safe ID validation (`validate_safe_id`), path confinement, and existing security headers/rate limiting.
- **Reference Footage (`data/test_video.mp4`):**
  - `1056×576`, 30 fps, 9.7s, H.264.
  - Camera behind the **receiving team's baseline** looking toward the net.
  - 3 near-side passers (`#13`, `#3`, `#11`) clearly visible on the near court; far-side server serves at ~`3.4s`, ball crosses the net against a teal wall, and `#3` steps left into the `#13`–`#3` seam to pass at ~`4.6s`.
  - All 4 corners of the **front 3m zone** (left/right net centerline + left/right 3m attack line) are clearly visible in frame.

---

## 4. Approaches Considered

| Approach | Summary | Trade-offs |
| --- | --- | --- |
| **1. Video Auto-Logger + Interactive Court Diagram *(Selected)*** | 4-click front-3m court homography + YOLO serve & passer auto-detection projected onto a 2D top-down court, with full manual click/drag/edit overrides and video timestamp jump. | Directly delivers auto-logging on `data/test_video.mp4` and longer practice videos, while remaining fast to correct or use manually. Requires a one-time 4-click calibration per camera angle. |
| **2. Manual Court Recorder + Side-by-Side Video** | Video player next to a top-down court diagram; user scrubs and clicks landing spots manually. | Zero CV edge cases and smaller scope, but does not provide auto-logging. |
| **3. Auto-Logger Using Existing 2-Click Net Calibration** | Estimate left/middle/right and rough short/deep from the 2-click net tape line alone. | Saves 2 calibration clicks, but cannot accurately determine short-vs-deep depth or meter-scale spacing between passers on a 2D court. |

**Decision:** The user selected **Approach 1** after inspecting `data/test_video.mp4`.

---

## 5. System Design

### 5.1 Navigation & Tab Architecture
- Add a clean two-tab switcher in the top header (`index.html`):
  - **Set & Hit Studio** (existing view, default on load)
  - **Serve Distribution** (new view)
- Both views remain mounted in the DOM (toggled via a `.hidden` class) so switching tabs preserves video playback state, session selection, and overlays on both sides.
- Header controls specific to Set & Hit Studio (overlay layer pills, Load Video) adapt cleanly when Serve Distribution is active, while sharing the same dark studio visual language (`styles.css`).

### 5.2 4-Corner Front-3m Court Calibration (Homography)
- **Coordinate System (Top-Down Receiving Court):**
  - Origin `(0.0, 0.0)` = Left sideline at the net centerline.
  - `+x` = Left to Right across the net (`0.0m` to `9.0m`).
  - `+y` = Net centerline toward the receiving baseline (`0.0m` to `9.0m`).
  - Out-of-bounds margin extends conceptually to `x ∈ [-1.5, 10.5]` and `y ∈ [-0.5, 11.0]` for visualization and logging long/wide serves.
- **Calibration Workflow (4 Clicks):**
  - Guided 4-click mode on the video canvas:
    1. **Front-Left** `(0.0, 0.0)` — Left sideline × net centerline
    2. **Front-Right** `(9.0, 0.0)` — Right sideline × net centerline
    3. **3m-Right** `(9.0, 3.0)` — Right sideline × 3m attack line
    4. **3m-Left** `(0.0, 3.0)` — Left sideline × 3m attack line
  - Computes a 3×3 homography `H` (`image_px -> court_m`) via `cv2.getPerspectiveTransform` and its inverse `H_inv` (`court_m -> image_px`).
  - **Live Verification Overlay:** Immediately projects the full 9m × 9m court grid (sidelines, 3m line, 6 zone dividers, and baseline) onto the video canvas via `H_inv` so the user can visually confirm alignment and re-click if needed.

### 5.3 Automated Serve Detection & Projection Pipeline (`serve_detector.py`)
Reuses the existing `FramePredictor` protocol (`YoloGpuPredictor` on CUDA, plus synthetic test predictor for unit tests):

1. **Ball & Pose Extraction:**
   - For short clips (`<= 20s`, e.g., `data/test_video.mp4`): run full-FPS `detect_balls` + `track_poses` directly (or reuse cached detections if the linked video session already ran them).
   - For longer practice recordings (`> 20s`): run a coarse scan (e.g., 4–6 fps) to find candidate serve windows where a ball crosses from above/far-side of the net toward the near court, then run full-FPS ball + pose tracking on each candidate window (`~4–5s` window around net crossing).
2. **Serve Event Identification:**
   - Identify downward-approaching ball trajectories that cross from the far side over the net zone (ball appears high near/above the net tape and descends into the near receiving court, with apparent motion toward the foreground).
   - Locate the **reception/landing frame (`t_land`)**:
     - **Played serve (reception):** The frame where the ball's vertical velocity `vy` abruptly flips from downward (`vy > 0` in image pixels) to upward (`vy < 0`) while within proximity of a near-side player's upper body/arms.
     - **Unplayed serve (ace / out):** The local lowest-point / bounce inflection (or last visible near-floor frame) when no near-side player is within reception reach.
   - Ignore post-reception sets and attacks by enforcing a minimum refractory gap between distinct serves (e.g., `>= 2.5s`) and requiring the ball to arrive from the far side of the net.
3. **Pre-Serve Passer Snapshot (`t_pre`):**
   - Sample near-side player tracks at `t_pre = max(window_start, t_land - 0.6s)` (just as the serve is in flight, before passers lunge out of formation).
   - Filter to active receiving passers: near-side players whose homography-projected foot position lies on the receiving court (`x_m ∈ [-0.5, 9.5]`, `y_m ∈ [2.0, 9.5]`). Exclude front-row net/setter players tucked near the net (`y_m < 2.2m`) when 2+ deeper passers are present.
   - Sort detected passers left-to-right by `x_m` and label `P1, P2, P3, ...`.
   - Foot pixel coordinate: midpoint of left/right ankle keypoints (COCO indices 15, 16) when confidence `>= 0.35`, otherwise bottom-center of the person bounding box.
4. **Landing / Reception Court Projection `(x_m, y_m)`:**
   - **If received by a passer:** Use the receiving passer's floor foot position at `t_land`, projected through `H` into court meters `(x_m, y_m)`. (Optionally apply a small lateral offset toward the ball's horizontal wrist/contact offset if the passer reaches sideways, capped to `±0.6m`.)
   - **If unplayed (ace/out):** Project the ball's bounce pixel at `t_land` through `H` into `(x_m, y_m)`.
   - Default auto-logged result: `Out` if `(x_m, y_m)` lies outside `[0, 9] × [0, 9]`, `Ace` if unplayed and in-bounds, otherwise `In`.

### 5.4 Zone & Seam Classification (`serve_geometry.py`)
Pure, deterministic geometry module used identically by both auto-detection and manual court clicks/edits:

- **Area / Zone Classification:**
  - Out of bounds:
    - `y_m < 0.0` → `Net / Short`
    - `y_m > 9.0` → `Out Long`
    - `x_m < 0.0` → `Out Left`
    - `x_m > 9.0` → `Out Right`
  - In bounds (`0 <= x_m <= 9` and `0 <= y_m <= 9`):
    - Horizontal thirds: Left `[0, 3)`, Middle `[3, 6)`, Right `[6, 9]`.
    - Front row (`y_m < 3.0`): `Zone 4` (Left), `Zone 3` (Middle), `Zone 2` (Right).
    - Back row (`y_m >= 3.0`): `Zone 5` (Left), `Zone 6` (Middle), `Zone 1` (Right).
    - Depth tag: `Short` (`y_m < 4.5`) vs. `Deep` (`y_m >= 4.5`).
- **In-Between-Players ("Seam") Classification:**
  - Requires at least 2 passers sorted left-to-right (`P1, P2, ..., Pn`).
  - Only evaluated for in-bounds serves (`result in ("in", "ace")` and `0 <= x_m <= 9`, `0 <= y_m <= 9`).
  - A serve at `(sx, sy)` is classified as **In Seam** (`is_seam = True`, `seam_pair = (Pi.label, Pj.label)`) when for an adjacent pair `(Pi, Pj)`:
    1. `Pi.x_m < sx < Pj.x_m` (horizontally between the two passers),
    2. `|sy - 0.5 * (Pi.y_m + Pj.y_m)| <= 2.2m` (within the longitudinal receive band of that pair), AND
    3. Neither passer is directly on top of the ball: both `hypot(sx - Pi.x_m, sy - Pi.y_m) >= R_body` and `hypot(sx - Pj.x_m, sy - Pj.y_m) >= R_body`, where `R_body = min(0.75, 0.30 * (Pj.x_m - Pi.x_m))` meters (adapts smoothly if passers stand unusually close).
- **Result vs. Coordinate Invariants *(design call)*:**
  - Clicking outside `[0, 9] × [0, 9]` automatically sets `result = "out"`.
  - When `result == "net"`, `serve_geometry.py` snaps `y_m = 0.0` (preserving `x_m` along the net), and forces `zone = "Net / Short"`, `depth = "out"`, `is_seam = False`, `seam_pair = None`.
  - When `result == "out"`, `serve_geometry.py` forces `depth = "out"`, `is_seam = False`, `seam_pair = None` (keeping `Out Left` / `Out Right` / `Out Long` from coordinates, or `"Out"` if manually marked out inside the lines).
  - **Coaching Summary Formulas *(design call)*:** `In % = (in + ace) / total`, `Ace % = ace / total`, `Error % = (out + net) / total` (so `In % + Error % = 100%`), and `Seam % = seam_hits / in_bounds_serves`.
- **Passer Dragging & Re-computation Scope *(design call)*:**
  - Dragging a passer or picking a formation preset with **no serve selected** updates `default_passers` for *future* manual serves only (historical serves keep their own `passers_snapshot`, protecting video-derived positions).
  - Dragging a passer while a **specific serve is selected** updates *that serve's* `passers_snapshot` via `PATCH /api/serve-sessions/{id}/serves/{serve_id}` and recomputes its `is_seam` / `seam_pair` immediately.

### 5.5 Data Models & Persistence (`models.py`, `serve_storage.py`)
Stored under `data/serve_sessions/<serve_session_id>/serve_session.json` (atomic temp-file write + `0o600` permissions, strict `validate_safe_id`):

- `CourtCalibration`: 4 image `Point2D` corners (`front_left`, `front_right`, `attack_right`, `attack_left`), `is_calibrated: bool`.
- `PasserPosition`: `label: str` (`P1`, `P2`, ...), `x_m: float`, `y_m: float`.
- `ServeEvent`:
  - `serve_id: str`, `index: int`
  - `source: Literal["auto", "manual"]`
  - `timestamp_s: float | None`, `frame_idx: int | None` (links back to video when available)
  - `server_name: str` (default `"Unassigned"` or active server)
  - `x_m: float`, `y_m: float`
  - `result: Literal["in", "ace", "out", "net"]`
  - `zone: str` (`Zone 1`–`Zone 6`, `Out Long`, `Out Left`, `Out Right`, `Net / Short`)
  - `depth: Literal["short", "deep", "out"]`
  - `is_seam: bool`, `seam_pair: tuple[str, str] | None`
  - `passers_snapshot: list[PasserPosition]` (the passer positions active when this serve was logged, so hovering/selecting a serve shows where the passers were on *that* serve)
- `ServeSession`:
  - `serve_session_id: str`, `name: str`, `created_at: str`
  - `video_session_id: str | None` (optional link to an existing uploaded/local video session in `SessionStore`)
  - `status: Literal["idle", "scanning", "ready", "error"]`, `progress_pct: float`, `error_message: str | None`
  - `court_calibration: CourtCalibration | None`
  - `default_passers: list[PasserPosition]` (current formation on the diagram; defaults to standard 3-passer receive formation: `P1 (1.8, 6.2)`, `P2 (4.5, 6.5)`, `P3 (7.2, 6.2)`)
  - `servers: list[str]` (list of server names/numbers in this practice)
  - `serves: list[ServeEvent]`

### 5.6 API Endpoints (`app.py`)
All mutating routes require `_require_csrf(request)`:
- `GET /api/serve-sessions` — list saved serve practice sessions
- `POST /api/serve-sessions` — create a new serve session (optionally linked to a `video_session_id`)
- `GET /api/serve-sessions/{id}` — fetch full session + computed distribution summary
- `PATCH /api/serve-sessions/{id}` — update session name, linked `video_session_id`, `servers` list, or `default_passers`
- `DELETE /api/serve-sessions/{id}` — delete a serve session
- `POST /api/serve-sessions/{id}/calibration` — save 4-corner court calibration
- `POST /api/serve-sessions/{id}/auto-detect` — trigger background YOLO serve auto-detection on the linked video (reuses existing video stream endpoint `/api/sessions/{video_session_id}/video`). **Merge rule *(design call)*:** replaces existing `source == "auto"` serves while preserving all `source == "manual"` serves; manually editing an auto-detected serve flips its `source` to `"manual"` so user fixes are never overwritten on re-scan.
- **Video-Only Loading & GPU Locking *(design call)*:** Add an optional `skip_analysis: bool = False` parameter to `/api/sessions/upload` and `/api/sessions/local`. When loading a video from the Serve Distribution tab, pass `skip_analysis=True` so it probes and prepares `web_proxy.mp4` without triggering Set & Hit Studio's GPU analysis. Serialize GPU inference across Set & Hit analysis and Serve auto-detection using a shared `asyncio.Lock`.
- `POST /api/serve-sessions/{id}/serves` — manually add a serve (server computes zone + seam)
- `PATCH /api/serve-sessions/{id}/serves/{serve_id}` — move `(x_m, y_m)`, change `result`, `server_name`, or `passers_snapshot` (server recomputes zone + seam)
- `DELETE /api/serve-sessions/{id}/serves/{serve_id}` — remove a serve (also powers Undo)

### 5.7 Serve Distribution Tab UI
- **Layout:** Responsive two-column studio layout matching the existing app:
  - **Left/Main Stage:**
    - Top bar: Session picker (`+ New`), optional `Link Video` picker (shows sessions already loaded in Set & Hit Studio, plus a `Load Video` button that opens the existing loader), `Calibrate Court (4 Clicks)`, and `Auto-Detect Serves` with progress indicator.
    - When a video is linked: collapsible/side-by-side **Video Preview** canvas (shows 4-corner calibration overlay and jumps directly to `timestamp_s` when clicking any serve in the diagram or list).
    - **Interactive Top-Down Court Canvas (HiDPI `<canvas>`):**
      - 9m × 9m court with out-of-bounds apron, net bar at top, 3m line, subtle Zone 1–6 grid, draggable passer circles (`P1, P2, P3...`), and shaded **seam corridors** between adjacent passers.
      - Formation presets: `2-Passer`, `3-Passer`, `4-Passer`, plus `Reset Passers`.
      - View toggles: `Dots` / `Heatmap` / `Seam Bands`.
      - Color-coded serve markers (e.g., Cyan = In, Gold = Ace, Purple/Ring = Seam hit, Rose = Out/Net). Clicking empty court logs a serve; dragging a passer moves the formation; clicking a serve selects it (highlighting its `passers_snapshot` and jumping the video); `Undo` / `Clear` buttons below.
  - **Right Sidebar:**
    - **Server Selector:** Quick chips (`All Servers`, plus individual server tags and `+ Add Server`).
    - **Quick Result Mode (for manual clicks):** `In` (default), `Ace`, `Net` (clicking outside court lines auto-sets `Out`).
    - **Distribution Summary Cards:**
      - Total serves, `In %`, `Ace %`, `Error %`
      - **Between Players (Seam) Rate:** count & `% of in-bounds serves`, broken down by pair (`P1–P2`, `P2–P3`, etc.)
      - **Area Breakdown:** Zone 1–6 counts & `%` bars + `Short vs. Deep` split
    - **Serve Log List:** Compact scrollable list of recorded serves (`#1 · Z5 · P1–P2 Seam · 4.6s`), clickable to jump video / delete.

---

## 6. User Decisions vs. Proposed Defaults & Builder Calls

- **Decided by User:**
  - Built for **practice** first in a **new tab**.
  - Top-down **court diagram** showing **court area** and **whether the serve landed in between players**.
  - **Movable passer markers** with **auto-detected seams**.
  - **Approach 1:** Video auto-logging (tested against `data/test_video.mp4`) + interactive court diagram with manual overrides.
- **Proposed Defaults (open for user override):**
  - Per-server tagging/filtering (`All Servers` + individual players) and quick result toggles (`In` / `Ace` / `Net` / auto `Out`).
  - Local JSON persistence under `data/serve_sessions/`.
  - Seam geometry threshold (`R_body = min(0.75m, 0.30 * gap)` and `±2.2m` longitudinal band).
- **Left to the Builder:**
  - Exact visual styling tokens (aligned with existing `styles.css` dark theme), heatmap Gaussian kernel radius, and canvas hit-testing tolerances.
  - Background task polling cadence and exact coarse-scan FPS / velocity thresholds (tuned and verified against `data/test_video.mp4`).
  - Test organization (unit tests for homography/zone/seam math, API/CSRF tests, synthetic `FramePredictor` pipeline tests, and GPU smoke verification on `data/test_video.mp4`).
