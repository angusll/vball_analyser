# Set & Hit Match Isolation, Opponent Scouting & Offensive Analytics Implementation Plan (Build 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Drill | Match` session analytics, multi-swing rally splitting, `Ours | Opponent` side classification with mirrored zones/roles, above-tape blocker hand-seam measurement, and a `Play | Session` Studio summary view without extra GPU passes.

**Architecture:** `scan_candidate_plays` splits coarse above-tape activity into distinct local swing peaks (`>= 2.3s` apart) while `detector.py` reuses Pass-2 YOLO detections across overlapping rally windows. `biomechanics.py` unlocks far-side contact candidates, classifies each play as `ours` vs. `opponent` (mirroring `Zone 4 <-> Zone 2` and evaluating near-side net jumpers as `Our Block` on opponent sets), fixes blocker seams to use above-tape hand/arm spans, and computes `StudioSessionSummary` live from cached JSON whenever `Drill | Match` or `Ours <-> Opponent` is toggled.

**Tech Stack:** Python 3.12 (`uv`), FastAPI, Pydantic v2, NumPy, OpenCV, PyAV, Vanilla ES6 JS (`static/app.js`), HTML5 `<canvas>`, CSS, pytest, Node.js `node:test`.

**Spec:** `docs/superpowers/specs/2026-11-27-set-hit-match-analytics-design.md`

## Global Constraints

- Always execute Python/pytest commands via `/home/angus/.local/bin/uv run` using the Python 3.12 `.venv` in the project root.
- Camera is behind the user's team baseline across sets (`court_side == "attacking"` = near court = `ours`; `court_side == "blocking"` = far court = `opponent`).
- Strict frontend & backend hygiene (`tests/test_api_security.py::test_static_frontend_and_backend_security_hygiene`):
  - Zero occurrences (even in comments) of `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `alert(`, `confirm(`, `prompt(`, `localStorage`, or `sessionStorage` in any `src/vball_analyser/static/*.js` file.
  - Zero occurrences of `<script>` or `style=` in `src/vball_analyser/static/index.html`.
  - Zero occurrences of `0.0.0.0` in `src/vball_analyser/*.py`; test servers bind strictly to `127.0.0.1` or ASGI `TestClient`.
- Every mutating `POST /api/sessions/*` route enforces `_require_csrf(request)` and `validate_safe_id(...)`.
- All new Pydantic fields must carry defaults so legacy `session.json` files deserialize without migration errors.

## Review Focus

- **Reversed net calibration click order (`left_point.x > right_point.x`):** Normalization across left/right antenna points must ensure `classify_attack_zone`, `pin_distance_m`, and `x_rel_m` never invert Zone 4 / Zone 2 or divide by zero (pinned in Task 1 `test_calibration.py`).
- **Wide lower-body blocker bbox with missing wrist keypoints:** When two blockers' full-body `bbox` intervals overlap horizontally due to flared legs/hips and wrist keypoints drop out (`conf < 0.25`), the `55%` central bbox fallback must still detect a `>= 22 cm` open seam (pinned in Task 1 `test_biomechanics.py`).
- **Multi-swing rally frame reuse & short-clip compatibility:** Clips `<= 8.0s` return `[(0.0, duration_s)]`, while a `> 20s` match clip with two swings `3.2s` apart inside one rally emits 2 `PlayAnalysis` entries and runs `detect_balls` / `track_poses` only once across their overlapping frames (pinned in Task 2 `test_pipeline.py`).
- **Side swap (`Ours <-> Opponent`) clearing stale hitter override:** Flipping a play's `team_side` when `hitter_track_id` pointed to a near-side track must clear that override, pick the far-side hitter, mirror `attack_zone` (`Zone 2 -> Opp Zone 4`), and exclude the play from `summary.ours_all` (pinned in Task 3 `test_pipeline.py`).
- **Legacy `session.json` load & 1-rep sample std (`ddof=1`):** Loading an older `session.json` without `session_mode` or `summary` populates `summary` automatically, and zones with `count == 1` emit `tempo_std_s = None` instead of `0.0` or raising `ValueError` (pinned in Task 1 `test_biomechanics.py` & Task 3 `test_pipeline.py`).

---

### Task 1: Models, Mirrored Zone/Pin Geometry, Hand-Span Seam, `Ours | Opponent` Mechanics & Session Summary

**Files:**
- Modify: `src/vball_analyser/models.py:67-134`
- Modify: `src/vball_analyser/calibration.py:62-71`
- Modify: `src/vball_analyser/biomechanics.py:250-566`
- Test: `tests/test_calibration.py`
- Test: `tests/test_biomechanics.py`

**Interfaces:**
- Produces in `src/vball_analyser/models.py`:
  - `BlockerCallout.hand_span_x: tuple[float, float] | None = None`
  - `BlockSummary.block_state: Literal["unopposed", "zero_block", "single_block", "open_seam", "closed_block"] = "unopposed"`
  - `PlayAnalysis`: `team_side: Literal["ours", "opponent"] = "ours"`, `team_side_manual: bool = False`, `system_state: Literal["in_system", "out_of_system"] | None = None`, `pin_distance_m: float | None = None`
  - `ZoneSetWindowStats`, `BlockIsolationStats`, `OffenseBucketSummary`, `StudioSessionSummary` (exact fields per Spec §5.4)
  - `AnalysisSession`: `session_mode: Literal["drill", "match"] = "drill"`, `summary: StudioSessionSummary | None = None`
- Produces in `src/vball_analyser/calibration.py`:
  - `normalized_net_u(cal: NetCalibration, x: float) -> float` (orders `lx, rx = sorted((cal.left_point.x, cal.right_point.x))`, returns `(x - lx) / dx` or `0.5` when `dx < 1e-6`)
  - `classify_attack_zone(cal: NetCalibration, x: float, team_side: Literal["ours", "opponent"] = "ours") -> str`
  - `compute_pin_distance_m(cal: NetCalibration, x: float | None) -> float | None` (`None` when `0.33 <= u <= 0.67` or `x is None`; else `round(u * 9.0, 2)` for `u < 0.33` and `round((1.0 - u) * 9.0, 2)` for `u > 0.67`)
- Produces in `src/vball_analyser/biomechanics.py`:
  - `PlayMechanicsResult` dataclass / NamedTuple (or extended return) — keep backward-compatible `analyse_play_mechanics(...)` returning `(setter_track_id, hitter_track_id, hitter_metrics, block_summary, keyframes)` AND add `analyse_play_full(tracks, set_trajectory, calibration, fps, frame_shape, *, session_mode="drill", override_hitter_track_id=None, override_team_side=None) -> tuple[Literal["ours", "opponent"], Literal["in_system", "out_of_system"] | None, float | None, int | None, int | None, HitterMetrics | None, BlockSummary, list[KeyMoment]]` (with `analyse_play_mechanics` delegating to `analyse_play_full` so existing callers/tests stay unbroken)
  - `compute_studio_session_summary(plays: list[PlayAnalysis], session_mode: Literal["drill", "match"]) -> StudioSessionSummary`

- [ ] **Step 1: Write failing unit tests in `tests/test_calibration.py` and `tests/test_biomechanics.py`**
  - In `tests/test_calibration.py`:
    - `test_mirrored_attack_zones_and_pin_distance_reversed_clicks()`: verifies `u = 0.10` -> `"Zone 4 (Left)"` (`ours`) vs `"Opp Zone 2 (Right)"` (`opponent`) with `pin_distance_m == 0.9`, `u = 0.50` -> `"Zone 3 (Middle)"` / `"Opp Zone 3 (Middle)"` with `pin_distance_m is None`, `u = 0.92` -> `"Zone 2 (Right)"` (`ours`) vs `"Opp Zone 4 (Left)"` (`opponent`) with `pin_distance_m == 0.72`, and identical results when `cal.left_point` and `cal.right_point` are swapped.
  - In `tests/test_biomechanics.py`:
    - `test_hand_span_seam_detects_gap_despite_overlapping_leg_bboxes()`: two blockers whose full-body `bbox` edges overlap (`b1.bbox=(300, 260, 395, 470)`, `b2.bbox=(385, 260, 480, 470)`) — both with narrow raised wrists (`35px` apart) AND with `b2` wrist keypoints set to `conf=0.1` (`55%` bbox fallback) — assert `block_summary.seam_open is True` and `block_summary.block_state == "open_seam"`.
    - `test_opponent_play_classification_role_swap_and_in_system_split()`: constructs (a) a far-side attacker (`y_p90` above net floor threshold) hitting near right antenna (`x=1020`) against two near-side symmetric two-hand blockers -> asserts `team_side == "opponent"`, far-side `hitter_id`, near-side blockers in `block_summary` (`Our Block`), `system_state is None`, `is_back_row_approximate is False`; and (b) near-side overhead release at `x_rel_m = 6.0m` (`"in_system"`) vs low forearm bump release from `x_rel_m = 1.8m` (`"out_of_system"`).
    - `test_unopposed_drill_vs_match_and_session_summary_one_rep_std()`: verifies `0` blockers in `"drill"` -> `block_state == "unopposed"` and `isolation_pct is None`, `0` jumping blockers in `"match"` with far-side defender present -> `block_state == "zero_block"` and `isolation_pct == 100.0`, `opponent` plays excluded from `ours_all`, and 1-rep zone returns `tempo_std_s is None` while 2-rep zone returns sample std (`ddof=1`).

- [ ] **Step 2: Run tests to verify they fail**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_calibration.py tests/test_biomechanics.py -v`
  - Expected: FAIL (`ImportError` on `compute_pin_distance_m`, `analyse_play_full`, `compute_studio_session_summary`).

- [ ] **Step 3: Implement schemas, calibration helpers, `analyse_play_full`, and `compute_studio_session_summary`**
  - Add Pydantic models in `src/vball_analyser/models.py` matching Spec §5.4–5.5.
  - Implement `normalized_net_u`, `classify_attack_zone(..., team_side="ours")`, and `compute_pin_distance_m` in `src/vball_analyser/calibration.py`.
  - In `src/vball_analyser/biomechanics.py`:
    - Update `build_contact_candidates` to include both `"attacking"` (`impulse >= 25.0`) and `"blocking"` (`impulse >= 20.0`) non-symmetric-block jumpers (Spec §5.2.1).
    - Implement `classify_play_team_side`, `_compute_system_state` (`x_rel_m` in `[3.5, 7.6]`, overhead `wrist_y <= shoulder_mid_y + 12.0`, near-net `y_foot <= net_ground_y_at + 0.25 * span`), `_compute_blocker_hand_span` (`pad = max(6.0, 0.009 * span)`, `55%` bbox fallback, `min_seam_px = max(22.0 / max(est_scale, 1e-4), 0.024 * span)`), `analyse_play_full`, and `compute_studio_session_summary` (Spec §5.2–5.4).

- [ ] **Step 4: Run tests to verify Task 1 passes**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_calibration.py tests/test_biomechanics.py -v`
  - Expected: PASS.

- [ ] **Step 5: Commit**
  - `git add src/vball_analyser/models.py src/vball_analyser/calibration.py src/vball_analyser/biomechanics.py tests/test_calibration.py tests/test_biomechanics.py && git commit -m "feat(studio): add match side isolation, hand-span seam, and session summary models"`

---

### Task 2: Peak-Split Multi-Swing Rally Scanner & Shared-Frame Pass-2 Pipeline

**Files:**
- Modify: `src/vball_analyser/detector.py:759-945`
- Modify: `tests/conftest.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `analyse_play_full`, `build_contact_candidates`, `classify_attack_zone`, `compute_pin_distance_m` from Task 1.
- Produces in `src/vball_analyser/detector.py`:
  - `MIN_SWING_PEAK_GAP_S = 2.3`
  - `scan_candidate_plays(video_path, calibration, predictor, metadata) -> list[tuple[float, float]]` (short clips `<= 8.0s` return `[(0.0, duration_s)]`; `8.0s < duration_s <= 20.0s` returns `[(0.0, duration_s)]` when `<= 1` peak found; longer/multi-peak clips emit `[max(0, t_peak - 2.6), min(dur, t_peak + 1.6)]` per local peak separated by `>= 2.3s`)
  - `analyse_play_window(..., session_mode: Literal["drill", "match"] = "drill", override_hitter_track_id: int | None = None, override_team_side: Literal["ours", "opponent"] | None = None) -> tuple[PlayAnalysis, list[BallDetection]]`
  - `analyse_play_windows_shared(video_path, windows, calibration, predictor, metadata, session_dir, *, session_mode="drill", on_play_ready=None) -> list[tuple[PlayAnalysis, list[BallDetection]]]` (merges overlapping/adjacent windows within `<= 0.5s` up to `25.0s` max span, runs `detect_balls_and_poses` / `detect_balls` + `track_poses` **once** per merged span, slices frames/detections/tracks per `[w_start, w_end]`, prunes empty non-jump non-arc trigger windows when `len(windows) > 1`, re-indexes surviving plays `1..N`, and invokes `on_play_ready(idx, total, play, raw_balls)` if provided).

- [ ] **Step 1: Write failing tests in `tests/test_pipeline.py` (and opponent rally helper in `tests/conftest.py`)**
  - Extend `make_synthetic_baseline_video` in `tests/conftest.py` (or add a counting wrapper around `SyntheticColorPredictor` in `tests/test_pipeline.py`) to support optional `far_side_rallies: list[tuple[float, float]] = []` where a far-side red/orange jumper attacks an arc while near-side players block.
  - Add `test_multi_swing_rally_splits_ours_and_opponent_with_single_pass2_decode(tmp_path)` in `tests/test_pipeline.py`:
    - `22.0s` synthetic video with a 2-swing rally (`ours` at `3.0..4.2s`, `opponent` at `6.8..8.0s` — `3.8s` apart, so their `[t-2.6, t+1.6]` windows overlap!) plus a later `ours` swing at `16.0..17.2s`.
    - Asserts `len(windows) == 3`, `analyse_play_windows_shared(..., session_mode="match")` produces 3 plays (`plays[0].team_side == "ours"`, `plays[1].team_side == "opponent"`, `plays[2].team_side == "ours"`), and a frame-counting `FramePredictor` wrapper verifies no frame index in the overlapping `[0.4s, 9.6s]` rally span is passed to `detect_balls` more than once.

- [ ] **Step 2: Run tests to verify they fail**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_pipeline.py -k "multi_swing_rally or two_rallies or short_clip" -v`
  - Expected: FAIL (`analyse_play_windows_shared` missing, `scan_candidate_plays` merges `3.0..4.2` and `6.8..8.0` into `cl[-1]`).

- [ ] **Step 3: Implement peak-split `scan_candidate_plays` and `analyse_play_windows_shared` in `src/vball_analyser/detector.py`**
  - Implement local-maxima peak selection (`MIN_SWING_PEAK_GAP_S = 2.3`, window `[t_peak - 2.6, t_peak + 1.6]`, `<= 8.0s` fast return, `<= 20.0s` single-peak full-clip return) per Spec §5.1.1.
  - Refactor `analyse_play_window` around a pure frame/detection helper `_build_play_from_window_data(...)` calling `analyse_play_full`, `classify_attack_zone(..., team_side=team_side)`, and `compute_pin_distance_m`.
  - Implement `analyse_play_windows_shared` grouping overlapping windows (`gap <= 0.5s`, merged span `<= 25.0s`), running detection/pose tracking once per merged group, slicing `frames`, `ball_dets`, and `tracks` per window, and pruning false-trigger windows (`set_trajectory is None` and max track impulse `< 25.0`) when `len(windows) > 1`.

- [ ] **Step 4: Run pipeline tests to verify Task 2 passes**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_pipeline.py -v`
  - Expected: PASS (including existing short-clip and 2-rally tests).

- [ ] **Step 5: Commit**
  - `git add src/vball_analyser/detector.py tests/conftest.py tests/test_pipeline.py && git commit -m "feat(detector): split multi-swing rallies and share overlapping Pass-2 frames"`

---

### Task 3: Cached `SessionStore` Mode/Side Switching, FastAPI Routes & Renderer Overlays

**Files:**
- Modify: `src/vball_analyser/storage.py:54-232`
- Modify: `src/vball_analyser/app.py:59-148`, `269-305`, `357-475`, `548-666`
- Modify: `src/vball_analyser/renderer.py:39-103`, `231-246`
- Test: `tests/test_renderer.py`
- Test: `tests/test_api_security.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `analyse_play_full`, `compute_studio_session_summary` (Task 1); `analyse_play_windows_shared`, `analyse_play_window` (Task 2).
- Produces in `src/vball_analyser/storage.py`:
  - `SessionStore.save_session` & `SessionStore.load_session`: always populate/refresh `session.summary = compute_studio_session_summary(session.plays, session.session_mode)` so legacy JSONs gain `summary` automatically.
  - `SessionStore.update_session_mode(session_id: str, session_mode: Literal["drill", "match"]) -> AnalysisSession`
  - `SessionStore.update_play_team_side(session_id: str, play_id: str, team_side: Literal["ours", "opponent", "auto"]) -> AnalysisSession` (when effective `team_side` flips, clears stale `hitter_track_id`, re-runs `analyse_play_full`, refreshes changed keyframe JPEGs if video exists, updates `attack_zone`, `pin_distance_m`, and `session.summary`).
  - Updates `recalibrate_session_net`, `reassign_play_hitter`, and `delete_play` to preserve `session_mode`/`team_side` (updating `team_side` if user explicitly reassigns hitter to a track on the other court side) and refresh `session.summary`.
- Produces in `src/vball_analyser/app.py`:
  - `LocalVideoRequest.session_mode: Literal["drill", "match"] = "drill"` and `upload_video(..., session_mode: Literal["drill", "match"] = Form("drill"))`
  - `POST /api/sessions/{session_id}/mode` (`UpdateSessionModeRequest(session_mode: Literal["drill", "match"])`)
  - `POST /api/sessions/{session_id}/plays/{play_id}/side` (`UpdatePlaySideRequest(team_side: Literal["ours", "opponent", "auto"])`)
  - `_run_session_analysis` uses `analyse_play_windows_shared(..., session_mode=session.session_mode, on_play_ready=...)`.

- [ ] **Step 1: Write failing tests in `tests/test_pipeline.py`, `tests/test_api_security.py`, and `tests/test_renderer.py`**
  - In `tests/test_pipeline.py`:
    - `test_legacy_session_load_mode_toggle_and_side_swap_clears_stale_hitter(tmp_path)`: writes a legacy-shaped `session.json` (no `session_mode`/`summary`), loads via `SessionStore.load_session` -> asserts `summary is not None`; calls `update_session_mode(..., "match")`; assigns near-side `hitter_track_id`, then calls `update_play_team_side(..., play_id, "opponent")` -> asserts `team_side == "opponent"`, `team_side_manual is True`, `hitter_track_id` switched to far-side track, `attack_zone` mirrored (`"Opp Zone ..."`), and `summary.ours_count == 0`, `summary.opponent_count == 1`.
  - In `tests/test_api_security.py`:
    - `test_studio_mode_and_side_endpoints_csrf_and_validation(tmp_path)`: verifies `POST /api/sessions/{sid}/mode` and `POST /api/sessions/{sid}/plays/{pid}/side` return `403` without CSRF, `400` on `../evil` IDs, `422` on invalid `session_mode="invalid"`, and `200` with updated `summary` when valid.
  - In `tests/test_renderer.py`:
    - `test_renderer_draws_opponent_header_and_unopposed_label()`: verifies `draw_play_overlays` succeeds on both `team_side="opponent"` and `block_state="unopposed"` plays without OpenCV errors.

- [ ] **Step 2: Run tests to verify they fail**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_pipeline.py tests/test_api_security.py tests/test_renderer.py -v`
  - Expected: FAIL (`update_session_mode` / `/mode` / `/side` not defined).

- [ ] **Step 3: Implement `SessionStore` methods, FastAPI endpoints, and `renderer.py` HUD labels**
  - Update `src/vball_analyser/storage.py` with `update_session_mode`, `update_play_team_side`, and automatic `summary` refresh in `load_session` / `save_session` / `recalibrate_session_net` / `reassign_play_hitter` / `delete_play`.
  - Wire `session_mode`, `/mode`, `/plays/{play_id}/side`, and `analyse_play_windows_shared` in `src/vball_analyser/app.py` (with `_require_csrf` and `validate_safe_id`).
  - Update `_draw_top_left_hud` in `src/vball_analyser/renderer.py` to show `OPPONENT` in the header when `play.team_side == "opponent"`, label row `"Our Block"` on opponent plays, and format `block_state == "unopposed"` as `"Unopposed"`.

- [ ] **Step 4: Run tests to verify Task 3 passes**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_pipeline.py tests/test_api_security.py tests/test_renderer.py -v`
  - Expected: PASS.

- [ ] **Step 5: Commit**
  - `git add src/vball_analyser/storage.py src/vball_analyser/app.py src/vball_analyser/renderer.py tests/test_pipeline.py tests/test_api_security.py tests/test_renderer.py && git commit -m "feat(api): add cached Drill/Match mode and Ours/Opponent side endpoints"`

---

### Task 4: Set & Hit Studio Frontend (`Drill | Match`, `Ours | Opp` Filters/Swap & `Play | Session` Sidebar)

**Files:**
- Modify: `src/vball_analyser/static/index.html:58-92`, `234-258`, `475-500`
- Modify: `src/vball_analyser/static/styles.css`
- Modify: `src/vball_analyser/static/app.js`
- Create: `tests/test_studio_analytics_ui.mjs`
- Modify: `tests/test_media_player_frontend.py`

**Interfaces:**
- Consumes: `session.session_mode`, `session.summary`, `play.team_side`, `play.system_state`, `play.pin_distance_m`, `play.block_summary.block_state`, `POST /api/sessions/{id}/mode`, and `POST /api/sessions/{id}/plays/{play_id}/side` from Task 3.
- DOM IDs added in `src/vball_analyser/static/index.html`:
  - `#studioModeGroup` (`button[data-session-mode="drill"]`, `button[data-session-mode="match"]`) in `.cal-controls`
  - `#loadSessionModeSelect` in `#loadDialog` (initial mode choice for Upload & Local File, defaulting to `"drill"`)
  - `#swapPlaySideBtn` in `.cal-controls` next to `#reassignHitterBtn`
  - `#playTeamFilterGroup` (`button[data-team-filter="all"]`, `"ours"`, `"opponent"`) inside `.plays-section`
  - `#studioSidebarViewGroup` (`button[data-sidebar-view="play"]`, `button[data-sidebar-view="session"]`) in `.metrics-panel .sidebar-header`, plus `#sessionSummaryContainer` (`.hidden` when `sidebarView === "play"`) alongside `#metricsList`.

- [ ] **Step 1: Write failing frontend tests (`tests/test_studio_analytics_ui.mjs` and `tests/test_media_player_frontend.py`)**
  - In `tests/test_media_player_frontend.py`: assert `#studioModeGroup`, `#swapPlaySideBtn`, `#playTeamFilterGroup`, `#studioSidebarViewGroup`, `#sessionSummaryContainer`, and `#loadSessionModeSelect` exist in `GET /`, and add `node --test tests/test_studio_analytics_ui.mjs` to the Node test runner check.
  - In `tests/test_studio_analytics_ui.mjs`: load `app.js` in a `vm` sandbox with lightweight DOM + `fetch` stubs and verify:
    1. Clicking `Match` in `#studioModeGroup` POSTs `{"session_mode": "match"}` to `/api/sessions/{id}/mode` and shows `#playTeamFilterGroup`.
    2. Filtering by `Opponent` in `#playTeamFilterGroup` hides `ours` play cards and updates active play selection cleanly.
    3. Clicking `#swapPlaySideBtn` POSTs `{"team_side": "opponent"}` (or `"ours"`) to `/api/sessions/{id}/plays/{play_id}/side`.
    4. Switching `#studioSidebarViewGroup` to `session` hides `#metricsList`, renders Set Window rows (`Z4 / Z3 / Z2` tempo `mean ± std`, peak cm, pin m) and `Unopposed` vs `Setter Isolation %` in `#sessionSummaryContainer`, and in `Match` mode toggles cleanly between `Our Offense` (`All | In-System | Out-of-Sys`) and `Opponent / Our Block`.
    5. `findNearestAttackingTrackId` targets `"blocking"` tracks when `play.team_side === "opponent"` and `"attacking"` tracks when `play.team_side === "ours"`.

- [ ] **Step 2: Run tests to verify they fail**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_media_player_frontend.py tests/test_api_security.py::test_static_frontend_and_backend_security_hygiene -v`
  - Expected: FAIL (new Studio DOM IDs and `test_studio_analytics_ui.mjs` missing).

- [ ] **Step 3: Implement Studio HTML, CSS, and `app.js` (`Drill | Match`, `Ours/Opp`, `Play | Session`)**
  - Add the new controls/containers in `src/vball_analyser/static/index.html` (no `style=` attributes) and style segmented pills, `Ours`/`Opp`/`In-Sys`/`OOS` badges, and compact Set Window summary cards/tables in `src/vball_analyser/static/styles.css` (using CSS width classes or canvas/CSS custom properties—note: setting `el.style.width` in JS is allowed, just never `style=` in `index.html`, and never forbidden JS tokens even in comments).
  - Update `src/vball_analyser/static/app.js` using strictly `document.createElement`, `textContent`, and `replaceChildren()` to wire mode switching, side swap, play list team filtering, amber opponent timeline bars, side-aware `findNearestAttackingTrackId`, per-play `Pin Distance` + `Our Block` / `Unopposed` rows (plus `(Far-court est.)` muted note on `Arm Extension` / `Torso Tilt` for `opponent` plays), and `#sessionSummaryContainer` rendering per Spec §5.6.

- [ ] **Step 4: Run full frontend and backend test suite**
  - Run: `/home/angus/.local/bin/uv run pytest -v`
  - Expected: ALL tests PASS (including `test_static_frontend_and_backend_security_hygiene` and Node `vm` tests).

- [ ] **Step 5: Commit**
  - `git add src/vball_analyser/static/index.html src/vball_analyser/static/styles.css src/vball_analyser/static/app.js tests/test_studio_analytics_ui.mjs tests/test_media_player_frontend.py && git commit -m "feat(studio-ui): add Drill/Match toggle, Ours/Opponent controls, and Session Summary view"`
