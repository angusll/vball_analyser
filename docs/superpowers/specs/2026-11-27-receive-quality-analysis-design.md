# Receive Quality Analysis — Design Document

**Date:** 2026-11-27  
**Project:** VBall Analyser (`vball_analyser`)  
**Status:** Ready for user approval (builder-checked)

---

## 1. Intent & Why

The **Serve Distribution** tab currently tracks where serves land on the `9m × 9m` receiving court (Zones 1–6, short/deep, and passer seams). During volleyball practice, coaches and players also need to evaluate **receive (serve-receive) quality** — specifically, how accurately each pass is delivered to the **setter target area** after the serve is received.

By tracing the ball from serve reception to **where the setter contacts it** and scoring the pass on a standard `0–3` scale based on distance from the setter target, coaches get immediate quantitative feedback (team passing average, in-system rate, and per-passer breakdown) from the exact same practice video and court calibration they already set up.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- Add a **Receive Quality** analysis mode directly inside the existing **Serve Distribution** tab, sharing the same practice sessions, linked videos, court calibration, server filter, and GPU auto-detection pass.
- Extend automated serve detection ([serve_detector.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/receive_quality_tab/src/vball_analyser/serve_detector.py)) to trace the ball *past* reception contact to **setter contact**, projecting the setter's floor position onto the `9m × 9m` court to avoid single-camera mid-air depth distortion.
- Score every received serve on a **0–3 scale** using concentric distance rings around a draggable **Setter Target** spot near the net (Green = `3`, Yellow = `2`, Red = `1`, outside red / unplayable / ace = `0`).
- Automatically attribute each received pass to the receiving passer (`P1`, `P2`, `P3`, ...) from the pre-serve passer snapshot and display both **team/session** and **per-passer** summaries.
- Provide full **manual override and manual charting** in Receive Quality mode: drag the setter target, drag/place pass destination dots, override a pass score, or chart passes manually without video.
- Preserve full backward compatibility with existing saved `serve_session.json` files.

### Non-Goals (v1)
- Separating Receive Quality into a third top-level navigation tab or a duplicate session store.
- Tracking setter-to-hitter set quality or attack outcome (handled by **Set & Hit Studio**).
- Jersey-number OCR or persistent named passer rosters (automatic `P1 / P2 / P3` formation attribution is sufficient for v1).
- Scoring serve errors (`out` / `net` serves have no reception and are excluded from receive-quality averages).

### Anti-Goals (Failure Modes to Avoid)
- **Mid-air homography distortion:** Projecting a pass ball caught 2m in the air directly through the floor homography shifts its apparent depth far away from the camera. Pass destination coordinates MUST anchor to the contacting setter's floor foot position.
- **Visual clutter on the court canvas:** Overlaying filled green/yellow/red concentric rings on top of purple seam corridors, passer circles, and serve dots simultaneously makes the court unreadable and dragging ambiguous. Receive Quality must have its own clean sub-mode view.
- **Duplicate GPU scans:** Users should never have to calibrate or run YOLO twice on the same video to get both Serve Distribution and Receive Quality.
- **Overwriting manual fixes:** Re-running Auto-Detect must preserve any serve/pass edited by the user (`source == "manual"`).

---

## 3. Context & Constraints

- **Existing Stack:** Python 3.12 (`uv`), FastAPI (`127.0.0.1:8081`), Pydantic v2, OpenCV, NumPy, PyAV, Ultralytics `YOLO11` + `ByteTrack`, and vanilla HTML/CSS/JS (`src/vball_analyser/static/`). No frontend build step.
- **Security Layer:** All mutating `/api/serve-sessions/*` endpoints enforce `_require_csrf(request)`, `validate_safe_id`, path confinement, and existing security headers/rate limiting.
- **Coordinate System:** Existing top-down receiving court in [serve_geometry.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/receive_quality_tab/src/vball_analyser/serve_geometry.py): `x ∈ [0.0, 9.0]` left-to-right across the net, `y ∈ [0.0, 9.0]` net centerline to receiving baseline, attack line at `y = 3.0`.
- **Camera Geometry:** Single tripod behind the receiving team's baseline looking toward the net.

---

## 4. Approaches Considered

| Approach | Summary | Trade-offs |
| --- | --- | --- |
| **1. Mode Switch Inside Serve Distribution (`Serve Distribution` \| `Receive Quality`) *(Selected)*** | Extend `ServeEvent` / `ServeSession` and `detect_serve_events` so one YOLO scan captures both serve landing and setter contact. A 2-way mode pill inside the Serve Distribution tab switches the court canvas and right sidebar between Serve Distribution and Receive Quality. | Uncluttered court and sidebar, unambiguous drag interactions, single calibration and GPU scan, direct serve-to-pass link, backward-compatible storage. |
| **2. Always-On Overlay Layer on the Same Court** | Same backend, but add a `Receive Rings` toggle pill next to `Dots / Heatmap / Seams` and stack receive KPIs into the existing sidebar. | Shows serve and pass simultaneously without switching modes, but crowds the canvas, creates drag ambiguity, and makes the sidebar too tall. |
| **3. Separate Receive-Only Scanner & Store** | Parallel `receive_sessions` backend and UI sub-tab independent of Serve Distribution. | Clean separation for receive-only drills, but duplicates calibration and GPU scans and loses the link between serve zone/seam and pass score. |

**Decision:** The user selected **Approach 1**.

---

## 5. System Design

### 5.1 Receive Quality Geometry & Scoring (`serve_geometry.py`)
Pure, deterministic scoring and summary functions used by both auto-detection and manual edits:

- **Default Setter Target:** `Point2D(x=6.0, y=0.8)` (classic right-of-center setter target between Zone 3 and Zone 2, `0.8m` off the net, matching the user's reference diagram). Clamped to `x ∈ [0.5, 8.5]`, `y ∈ [0.2, 4.0]` when dragged/updated.
- **Concentric Distance Thresholds (`RECEIVE_RING_RADII_M`):**
  - **Score `3` (Green — Perfect):** `d <= 1.5m`
  - **Score `2` (Yellow — Good / Playable):** `1.5m < d <= 3.2m`
  - **Score `1` (Red — Out-of-System):** `3.2m < d <= 4.8m`
  - **Score `0` (Bad Pass):** `d > 4.8m`, OR overpass (`pass_y_m < 0.0`), OR serve `result == "ace"`, OR unplayable shank (no setter contact).
- **Serve Error Exclusion *(design call)*:** Serves with `result in ("out", "net")` are not received serves; their `receive_score` is `None` and they are excluded from receive-quality totals and averages.
- **Distance & Score Function (`score_receive`):**
  - Given `(pass_x_m, pass_y_m, result, setter_target)`:
    - If `result in ("out", "net")` → `(None, None)`.
    - If `result == "ace"` → `(None, 0)`.
    - If `pass_x_m is None or pass_y_m is None` → `(None, 0)` when explicitly marked as unplayable/shank, or `(None, None)` if no pass data has been recorded yet on a manual serve.
    - If `pass_y_m < 0.0` (overpass over the net) → `(round(d, 3), 0)`.
    - Otherwise compute `d = hypot(pass_x_m - setter_target.x, pass_y_m - setter_target.y)` and map via `(1.5, 3.2, 4.8)` to `3 / 2 / 1 / 0`.
- **Summary Metrics (`compute_receive_summary`):**
  - Respects the existing `server_filter` (`All` or specific server).
  - Evaluates all filtered events where `receive_score is not None`:
    - `scored_count: int`
    - `avg_rating: float` (0.00–3.00, rounded to 2 decimals)
    - `in_system_count: int` (`receive_score >= 2`), `in_system_pct: float`
    - `perfect_count: int` (`receive_score == 3`), `perfect_pct: float`
    - `score_counts: dict[str, int]` (`"0"`, `"1"`, `"2"`, `"3"`) and `score_pcts: dict[str, float]`
    - `by_passer: list[PasserReceiveSummary]` sorted by label (`P1`, `P2`, `P3`, ..., `Unassigned` last), each with `passer_label`, `count`, `avg_rating`, `in_system_pct`, and `score_counts`.

### 5.2 Automated Post-Reception Ball & Setter Contact Pipeline (`serve_detector.py`)
Extends `detect_serve_events(...)` without adding a second YOLO pass:

1. **Candidate Window Extension:**
   - Widen coarse-scan windows from `(t - 2.5s, t + 1.5s)` to `(t - 2.5s, t + 3.0s)` so the full pass trajectory from reception to setter contact is captured.
   - Pass `setter_target: Point2D` into `detect_serve_events` (defaulting to `(6.0, 0.8)`).
2. **Passer Attribution (`passer_label`):**
   - When `_match_receiving_passer` matches a near-side player `(rec_track_id, rec_frame, fx_m, fy_m)` at `land_ball`, match `(fx_m, fy_m)` to the nearest `PasserPosition` in `passers_snapshot` within `2.5m` to assign `passer_label` (e.g., `"P1"`, `"P2"`, `"P3"`).
3. **Post-Reception Ball Tracing & Setter Contact (`_trace_pass_to_setter`):**
   - Only runs for received serves (`receiver is not None` and `norm_res == "in"`). Aces immediately get `receive_score = 0`; `out` / `net` get `receive_score = None`.
   - Slice the cleaned ball series to `t_land < b.timestamp_s <= t_land + 2.6s` (`POST_PASS_MAX_DURATION_S = 2.6`).
   - **Overpass detection:** If the ball travels upward past the net tape (`b.y < net_tape_y - 10.0`) and either enters proximity of a far-side (`court_side == "blocking"`) player or disappears above the net without any subsequent near-side player contact, classify as an overpass (`receive_score = 0`, `pass_x_m = None`, `pass_y_m = None`).
   - **Setter contact search:** Starting after `t_land + 0.25s` (`MIN_PASS_FLIGHT_S = 0.25`), scan each ball detection `b` for proximity to any near-side (`court_side == "attacking"`) player track:
     - Prefer a player with `track_id != receiver_track_id`. (Only allow the same player if no other near-side player contacts the ball and the ball clearly rose and descended back.)
     - Match criterion: ball within padded bounding box (`pad = 50px`, `box_dist <= 55px` or `center_dist <= 115px`), preferring frames where the ball's trajectory inflects (vertical direction flip or velocity vector change) or reaches local minimum distance to the player's upper body/wrists (COCO keypoints 9/10 when conf `>= 0.30`, else upper half of bbox).
     - Project the contacting setter's **floor foot coordinate** (`extract_foot_point(setter_frame)`) through `H` into `(pass_x_m, pass_y_m)`.
     - Compute `pass_distance_m` and `receive_score` via `score_receive(pass_x_m, pass_y_m, "in", setter_target)`, and record `pass_timestamp_s` and `pass_frame_idx`.
   - **Unplayable / shank fallback:** If no near-side setter contact is found within `2.6s`, score `receive_score = 0` (`pass_x_m = None`, `pass_y_m = None`).

### 5.3 Data Models & Persistence (`models.py`, `serve_storage.py`)
All new fields are optional or defaulted so existing JSON sessions deserialize seamlessly:

- `ServeEvent` additions:
  - `passer_label: str | None = None`
  - `pass_x_m: float | None = None`
  - `pass_y_m: float | None = None`
  - `pass_timestamp_s: float | None = None`
  - `pass_frame_idx: int | None = None`
  - `pass_distance_m: float | None = None`
  - `receive_score: Literal[0, 1, 2, 3] | None = None`
  - `receive_score_manual: bool = False` (true only if the user explicitly picked a manual 0–3 score override rather than letting coordinates compute it)
- `ServeSession` additions:
  - `setter_target: Point2D = Field(default_factory=lambda: Point2D(x=6.0, y=0.8))`
- New summary schemas:
  - `PasserReceiveSummary`: `passer_label`, `count`, `avg_rating`, `in_system_pct`, `score_counts`
  - `ReceiveQualitySummary`: `scored_count`, `avg_rating`, `in_system_count`, `in_system_pct`, `perfect_count`, `perfect_pct`, `score_counts`, `score_pcts`, `by_passer`
- **Recomputation Rules (`serve_storage.py`):**
  - Updating `setter_target` on a session clamps it to `[0.5, 8.5] × [0.2, 4.0]` and recomputes `pass_distance_m` and `receive_score` for every serve that has `(pass_x_m, pass_y_m)` and `not receive_score_manual`.
  - Updating a serve's `(pass_x_m, pass_y_m)` recomputes `pass_distance_m` and `receive_score` (clearing `receive_score_manual = False`) and flips `source = "manual"`.
  - Explicitly setting `receive_score` without moving `(pass_x_m, pass_y_m)` sets `receive_score_manual = True` and flips `source = "manual"`.
  - Changing a serve's `result` to `"out"` or `"net"` clears `receive_score = None` and pass coordinates; changing to `"ace"` sets `receive_score = 0` and clears pass coordinates.

### 5.4 API Endpoints (`app.py`)
Reuses existing `/api/serve-sessions/*` endpoints (with `_require_csrf`):
- `_serve_payload(session, server_filter)` returns:
  - `"session"`, `"summary"` (existing), and `"receive_summary"` (new `ReceiveQualitySummary`).
- `POST /api/serve-sessions/{id}/update`: accepts optional `setter_target: Point2D | None`.
- `POST /api/serve-sessions/{id}/serves`: accepts optional `pass_x_m`, `pass_y_m`, `passer_label`, `receive_score` (supports manual Receive Quality charting when clicking the court in Receive Quality mode).
- `POST /api/serve-sessions/{id}/serves/{serve_id}/update`: accepts optional `pass_x_m`, `pass_y_m`, `passer_label`, `receive_score`, `clear_pass: bool = False`.
- `POST /api/serve-sessions/{id}/auto-detect`: passes `serve_sess.setter_target` into `detect_serve_events`.

### 5.5 Frontend UI (`index.html`, `serve_tab.js`, `styles.css`)
- **Mode Switcher:** A compact segmented toggle in the Serve Distribution toolbar:
  - **`Serve Distribution`** (default, unchanged behavior)
  - **`Receive Quality`**
- **Court Canvas in Receive Quality Mode:**
  - Draws the `9m × 9m` court with filled concentric scoring rings clipped to the court boundaries around `setter_target`:
    - Outer red (`r = 4.8m`, score `1`), middle yellow (`r = 3.2m`, score `2`), inner green (`r = 1.5m`, score `3`), with subtle ring score labels.
  - Draws the draggable **Setter Target** handle (white circle with dark border labeled `Setter`, matching the user's sketch).
  - Draws color-coded **pass destination dots** (`3` = green, `2` = yellow, `1` = red, `0` = muted rose/gray).
  - When a serve is selected: draws a dashed arrow from its serve reception spot `(x_m, y_m)` to its pass destination `(pass_x_m, pass_y_m)` (if present) and jumps the linked video to `pass_timestamp_s ?? timestamp_s`.
  - **Interactions in Receive Quality Mode:**
    - Drag the **Setter Target** circle → live-preview rings + persist via `/update` on drop (recomputes all scores).
    - Drag any **pass destination dot** → move `(pass_x_m, pass_y_m)` and recompute score.
    - Click empty court with a **selected serve** → set/move that serve's pass destination to the clicked spot.
    - Click empty court with **no serve selected** → log a new manual receive event at that pass destination (with `result = "in"`, default reception spot `(4.5, 6.5)` or nearest passer, and score from the clicked spot).
    - Footer controls when a receive is selected: quick **Score Override** selector (`Auto` / `3` / `2` / `1` / `0`) and **Passer** selector (`Unassigned`, `P1`, `P2`, `P3`, ...).
- **Right Sidebar in Receive Quality Mode:**
  - Keeps the shared **Server** filter chips at the top.
  - Replaces the Serve Distribution summary + Zone Breakdown sections with:
    - **Receive Quality Summary:** KPI cards for **Avg Rating** (`X.XX / 3.0`), **In-System %** (`2+3`), **Perfect %** (`3`), and horizontal distribution bars for `3 (Perfect)`, `2 (Good)`, `1 (Out-of-Sys)`, `0 (Bad)`.
    - **By Passer (`P1 / P2 / P3`):** Compact table/rows showing passer label, pass count, avg rating, and in-system `%`.
  - Keeps the shared **Serve / Receive Log** at the bottom, adding a color-coded score pill (`3` / `2` / `1` / `0`) to every scored row.

---

## 6. User Decisions vs. Proposed Defaults & Builder Calls

- **Decided by User:**
  - Lives **inside the existing Serve Distribution tab** using **Approach 1** (mode switcher between `Serve Distribution` and `Receive Quality`).
  - Pass landing measured at **where the setter contacts** the ball (anchored to setter floor position).
  - Concentric scoring rings around the setter target: **Green = 3**, **Yellow = 2**, **Red = 1**, **rest of court = 0 (bad passing)**.
  - **Team/session summary + automatic `P1/P2/P3` passer attribution** (no mandatory manual player tagging).
- **Proposed Defaults (open for user adjustment):**
  - Default setter target at `(6.0m, 0.8m)` (draggable).
  - Ring radii: `1.5m` (Green / 3), `3.2m` (Yellow / 2), `4.8m` (Red / 1).
  - Serve errors (`out` / `net`) excluded from receive averages; `ace` and overpasses scored `0`.
  - In-system definition = score `2` or `3`.
- **Left to the Builder:**
  - Exact visual styling tokens (aligned with `styles.css` dark theme and ring opacity so dots remain legible over green/yellow/red fills).
  - Exact velocity/inflection heuristics for setter contact detection inside `_trace_pass_to_setter`, tuned against `data/test_video.mp4` if present and covered by synthetic `FramePredictor` unit tests.
  - Test organization across `test_serve_geometry.py`, `test_serve_detector.py`, `test_serve_storage.py`, and `test_serve_api.py`.
