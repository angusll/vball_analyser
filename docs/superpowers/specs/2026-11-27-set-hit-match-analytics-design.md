# Set & Hit Match Isolation, Opponent Scouting & Offensive Analytics — Design Document (Build 1)

**Date:** 2026-11-27  
**Project:** VBall Analyser (`vball_analyser`)  
**Status:** Ready for user approval (builder-checked)

---

## 1. Intent & Why

The **Set & Hit Studio** currently excels on short single-play clips and unopposed practice hitting lines, but falls short on **6v6 scrimmage and match recordings** (filmed from behind the user's team baseline, moving ends with the team between sets) and lacks **session-level setting & blocking analytics**:

1. **Multi-swing rally truncation:** `scan_candidate_plays` in [detector.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/explore_new_features/src/vball_analyser/detector.py) merges all above-tape balls separated by `<= 4.0s` into one cluster and only keeps the final window around `cl[-1]`, dropping earlier sets inside multi-exchange rallies.
2. **Opposite-side set confusion:** When the opponent sets on the far court, the ball arc still rises above the net tape in 2D image coordinates while [biomechanics.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/explore_new_features/src/vball_analyser/biomechanics.py) ignores far-side attackers—either failing to lock onto the opponent's set or misattributing a near-side blocker as the "hitter". Coaches actually want **both** sides detected and cleanly separated into **`Our Set`** vs. **`Opponent Set`**.
3. **Blocker seam & drill distortion:** `seam_gap_px` currently measures full-body `bbox` edges (`b2.bbox[0] - b1.bbox[2]`), causing flaring legs/hips to report a split block as "Closed", while unopposed practice reps report `"No Block"` in the same bucket as real match open-net plays.
4. **No session-level coaching view:** Coaches need a **`Drill | Match`** session toggle and a **Session Summary** view for **Set Window consistency** (tempo, peak height, antenna pin spacing), **In-System vs. Out-of-System** distribution splits, **Setter Isolation %**, and **Opponent Scouting / Our Block Seam %**—all recomputed instantly from cached JSON without re-running GPU inference.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- Add a **`Drill | Match`** mode toggle (`session_mode: Literal["drill", "match"]`) to `AnalysisSession` that switches coaching logic and recalculates play/session summaries in milliseconds from cached tracks and ball detections (zero GPU re-run).
- Replace single-tail rally clustering in [detector.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/explore_new_features/src/vball_analyser/detector.py) with **peak-split play windows** so multi-swing rallies emit one `PlayAnalysis` per distinct set-and-hit exchange while reusing Pass-2 YOLO detections across overlapping rally frames.
- Unlock far-side airborne contact candidates in [biomechanics.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/explore_new_features/src/vball_analyser/biomechanics.py) and classify every play as **`ours` (Near Court)** or **`opponent` (Far Court)**, with a **1-click `Ours ↔ Opponent` manual override** per play.
- Automatically **mirror attack zones (`Zone 4 ↔ Zone 2`)** and **swap attacker/blocker court sides** on `opponent` plays so opponent scouting reflects their true court zones and near-side jumpers are evaluated as **Your Block**.
- Fix blocker seam calculation in [biomechanics.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/explore_new_features/src/vball_analyser/biomechanics.py) to use **above-tape hand/arm horizontal span** rather than full-body bounding boxes, and separate `unopposed` drill reps from live `zero_block` plays.
- Compute per-play **pin distance from nearest antenna (`pin_distance_m`)** and **`in_system | out_of_system`** classification, plus a deterministic **Studio Session Summary** (`Set Window` consistency, `Zone Distribution %`, `Setter Isolation %`, and separate `Opponent Scouting / Our Block` stats).
- Add **`All | Ours | Opponent`** play list filters and a **`Play | Session`** switcher inside the Set & Hit Studio right sidebar while keeping full backward compatibility with existing saved `session.json` files.

### Non-Goals
- Supporting a stationary single-end tripod when the user's team switches to the far court (user confirmed camera stays behind their team across sets).
- Emphasizing or aggregating noisy far-court 2D hitter joint angles (`Arm Extension`, `Torso Tilt`) in Opponent session summaries.
- Post-contact attack landing / kill-efficiency tracking or linking to `ServeSession` (linking is handled separately in Build 3).
- Serve net-clearance height or float-vs-topspin classification (intentionally dropped due to single-camera perspective distortion).

### Anti-Goals (Failure Modes to Avoid)
- **Cross-contaminating team stats:** `Opponent` plays must never be averaged into `Our Offense` Set Window, Zone Distribution, or Setter Isolation metrics.
- **Duplicate GPU inference on overlapping rally windows:** Back-to-back swings in the same rally (`2.3–4.0s` apart) have overlapping `[t - 2.6s, t + 1.6s]` windows; Pass 2 must decode and run YOLO once per unique frame across merged rally spans rather than running duplicate inference.
- **Counting hitting lines as Setter Deception:** Plays with zero opposing-side player tracks (or zero jumping blockers in `Drill` mode) must be labeled `unopposed` and excluded from the Setter Isolation % denominator.
- **Unsafe DOM rendering:** Frontend updates in `static/app.js` must strictly use `document.createElement()`, `textContent`, and `replaceChildren()`—never `innerHTML`, `outerHTML`, or `insertAdjacentHTML`.

---

## 3. Context & Constraints

- **Stack:** Python 3.12 (`uv`), FastAPI (`127.0.0.1:8081`), Pydantic v2, OpenCV, NumPy, SciPy, PyAV, Ultralytics `YOLO11` + `ByteTrack`, vanilla HTML/CSS/JS (`src/vball_analyser/static/`). No frontend build step.
- **Camera Geometry:** Single tripod behind the **user's team baseline** looking toward the net (`0.0m` left antenna to `9.0m` right antenna). Near-side players belong to `"ours"`; far-side players belong to `"opponent"`.
- **Security Constraints (`mandatory-secure-web-skills`):**
  - Every mutating `/api/sessions/*` `POST` endpoint enforces `_require_csrf(request)`, `validate_safe_id(...)`, path confinement inside `sessions_dir`, HTTP method allow-listing (`GET`, `POST`, `HEAD`), rate limiting, and `Cache-Control: no-store` on JSON responses.
  - No `innerHTML` or native `alert()`/`confirm()`/`prompt()` calls in frontend JS; no sensitive tokens in `localStorage`/`sessionStorage` or console logs; test servers bind strictly to `127.0.0.1`.

---

## 4. Approaches Considered

| Approach | Summary | Trade-offs |
| --- | --- | --- |
| **1A. Peak-Split Play Windows + Sidebar `Play \| Session` Switcher *(Selected)*** | Split coarse above-tape ball activity into distinct local swing peaks (`>= 2.3s` apart), reuse Pass-2 frame detections across overlapping windows in the same rally, emit 1 `PlayAnalysis` per swing, and add a `Play \| Session` toggle inside the right sidebar's `Play Mechanics` panel. | Lean `session.json` per play, 100% compatible with existing 3-keyframe strip, `ball_cache`, `Reassign Hitter`, and annotated MP4 export; preserves full video canvas height; requires toggling `Play \| Session` in the sidebar. |
| **1B. Whole-Rally Windows with Iterative Multi-Arc Peeling + Bottom Drawer** | Keep merged rally windows up to `25s`, iteratively peel RANSAC set-arc inliers in `trajectory.py`, and render session summaries in a collapsible drawer beneath the video stage. | Slightly longer ByteTrack spans, but requires complex sub-window track/export slicing to avoid 3–5× JSON bloat, risks noisy first-arc fits masking adjacent sets, and shrinks vertical video height on 1080p screens. |

**Decision:** The user selected **Approach 1A**.

---

## 5. System Design

### 5.1 Multi-Swing Rally Peak Splitting & Frame Reuse (`detector.py`, `app.py`)

1. **Candidate Peak Splitting (`scan_candidate_plays`):**
   - Short single-play clips (`metadata.duration_s <= 8.0s`): return `[(0.0, metadata.duration_s)]` directly (preserving fast single-clip behavior and all existing short-clip tests).
   - Longer clips (`> 8.0s`):
     - If `8.0s < duration_s <= DEFAULT_CONFIG.short_clip_max_sec` (`20.0s`), run coarse scan at `sample_fps = 10.0` to check for multiple swings; if `<= 1` swing peak is found, return `[(0.0, duration_s)]`.
     - Score each coarse timestamp `ts` with moving above-tape balls (`b.y <= tape_y_at(calibration, b.x) + 35.0`) by peak elevation `score(ts) = max(10.0, tape_y_at(calibration, b.x) + 35.0 - b.y)` (falling back to net-zone jumping person triggers with score `10.0` when no ball triggers exist).
     - Smooth/bucket trigger timestamps and greedily select local maxima separated by at least `MIN_SWING_PEAK_GAP_S = 2.3s` (requiring `>= 2` trigger frames within `±0.8s` of a peak when multiple peaks exist, filtering 1-frame noise spikes).
     - Sort selected peak timestamps `t_peak` chronologically and emit one play window per peak:
       `[max(0.0, round(t_peak - 2.6, 2)), min(duration_s, round(t_peak + 1.6, 2))]` (minimum duration `1.2s`).
2. **Shared Pass-2 Detection Across Overlapping Rally Windows (`analyse_play_windows_batch` / `_run_session_analysis`) *(builder call)*:**
   - Group chronologically ordered play windows whose `[w_start, w_end]` intervals overlap (or sit within `<= 0.5s` of each other, capped at `25.0s` per merged decode span so RAM stays bounded).
   - For each group containing `>= 2` overlapping windows: decode frames `[group_start, group_end]` once and run `detect_balls_and_poses` (or `detect_balls` + `track_poses`) once over the contiguous span, then slice `ball_dets` and `tracks` (`frame_idx` within `[w_start, w_end]`) into each individual window's `analyse_play_from_detections(...)` pass.
   - Single isolated windows continue calling `analyse_play_window(...)` directly, streaming each completed `PlayAnalysis` to `session.json` as it finishes.
   - **False-Trigger Pruning on Multi-Window Scans *(builder call)*:** When `len(windows) > 1`, if a candidate window yields `set_trajectory is None` AND no player track has `_jump_impulse_px(t) >= 25.0`, drop that empty trigger window (e.g., high serve-receive pass that was immediately caught or whistled dead) unless all windows would be dropped, in which case keep the highest-activity fallback window.

### 5.2 Unlocking Far-Side Candidates & `Ours | Opponent` Classification (`biomechanics.py`, `calibration.py`)

1. **Both-Sides Contact Candidates (`build_contact_candidates`):**
   - Collect airborne non-symmetric-block wrist/top-bbox candidates from **both** `court_side == "attacking"` (`impulse >= 25.0px`) and `court_side == "blocking"` (`impulse >= 20.0px`, accounting for far-court perspective shrink) when `wy <= tape_y_at(cal, wx) + wrist_margin`.
   - This allows `fit_setting_path` in [trajectory.py](file:///home/angus/.gemini/antigravity/worktrees/vball_analyser/explore_new_features/src/vball_analyser/trajectory.py) to award its attacker-hand termination bonus to legitimate sets on **either** side of the net.
2. **Team-Side Classification (`classify_play_team_side`):**
   - Returns `team_side: Literal["ours", "opponent"]`:
   - If `override_team_side in ("ours", "opponent")`: return `override_team_side` immediately (`team_side_manual = True`).
   - Otherwise score `ours_score` vs. `opp_score` around `contact_frame` (`±4` frames) and `release_frame` (`±5` frames):
     - **Contact Attacker vs. Blocker Cue:** Among tracks horizontally within `0.22 * span` of `contact_point.x` (or net midpoint if `contact_point is None`) that jump (`impulse >= 20.0px`):
       - Near-side (`court_side == "attacking"`) track with asymmetric swing (`not _is_symmetric_two_hand_block`) and wrist within `95px` of `contact_point`: `ours_score += 2.2` (or `+1.4` if `contact_point is None`).
       - Far-side (`court_side == "blocking"`) track with asymmetric swing and wrist/top within `85px` of `contact_point`: `opp_score += 2.2` (or `+1.4` if `contact_point is None`).
       - Near-side jumper in `_is_symmetric_two_hand_block` standing near the net floor line (`y_p90 <= net_ground_y_at + 0.14 * span`): `opp_score += 1.2` (near side is blocking an opponent attack).
       - Far-side jumper in `_is_symmetric_two_hand_block`: `ours_score += 1.0`.
     - **Setter Release Cue:** When `set_trajectory` has `arc_points`: find closest near-side wrist (`<= 0.28 * span`) vs. closest far-side wrist/upper-body (`<= 0.24 * span`) to `arc_points[0]` at `release_frame ± 5` (excluding the top hitter candidate on that side):
       - Near-side release match: `ours_score += 1.5`.
       - Far-side release match (with no near-side wrist within `0.18 * span`): `opp_score += 1.5`.
     - **Decision Rule:** In `session_mode == "drill"`, classify as `"opponent"` only if `opp_score >= ours_score + 1.5` (biasing drills toward near-side reps); in `session_mode == "match"`, classify as `"opponent"` if `opp_score > ours_score + 0.3`, else `"ours"`.
3. **Role Assignment & Mirrored Zones (`analyse_play_mechanics`, `classify_attack_zone`):**
   - Let `offense_side = "attacking" if team_side == "ours" else "blocking"` and `defense_side = "blocking" if team_side == "ours" else "attacking"`.
   - Select `hitter` and `setter` from `offense_side` tracks (preserving manual `override_hitter_track_id` when valid).
   - Select opposing `blocker_callouts` (`BlockSummary`) from `defense_side` tracks jumping near `contact_point.x` whose reach crosses `tape_y_at(cal, top_x) + 0.02 * span`. (On `opponent` plays, `BlockSummary` therefore measures **Your Team's Block**!)
   - **Back-Row Flag (`is_back_row_approximate`):**
     - For `team_side == "ours"`: `hy_p90 > tape_y_at(cal, hx_med) + 0.36 * span` (existing rule).
     - For `team_side == "opponent"`: `False` *(builder call: single baseline camera cannot reliably separate far-court 3m takeoff depth behind the net)*.
   - **Mirrored Attack Zones (`classify_attack_zone(cal, x, team_side="ours")`):**
     - `u = (x - cal.left_point.x) / dx` (`0.0` = left antenna, `1.0` = right antenna).
     - `team_side == "ours"`:
       - `u < 0.33` → `"Zone 4 (Left)"`
       - `0.33 <= u <= 0.67` → `"Zone 3 (Middle)"`
       - `u > 0.67` → `"Zone 2 (Right)"`
     - `team_side == "opponent"` (mirrored because opponent faces the camera):
       - `u < 0.33` → `"Opp Zone 2 (Right)"`
       - `0.33 <= u <= 0.67` → `"Opp Zone 3 (Middle)"`
       - `u > 0.67` → `"Opp Zone 4 (Left)"`

### 5.3 Pin Distance, System State & Above-Tape Blocker Hand Seam (`biomechanics.py`, `calibration.py`)

1. **Pin-to-Antenna Distance (`pin_distance_m: float | None`):**
   - Let `u = (contact_x - cal.left_point.x) / dx` and `x_net_m = u * 9.0`.
   - When the play is in a pin zone (`u < 0.33` or `u > 0.67`):
     - Left screen pin (`u < 0.33`): `pin_distance_m = round(x_net_m, 2)` (`0.0m` = right on the left antenna; positive = meters inside the court; negative = outside antenna).
     - Right screen pin (`u > 0.67`): `pin_distance_m = round(9.0 - x_net_m, 2)` (`0.0m` = right on the right antenna; positive = meters inside the court).
   - When in Zone 3 Middle (`0.33 <= u <= 0.67`) or `contact_point is None`: `pin_distance_m = None`.
2. **In-System vs. Out-of-System (`system_state: Literal["in_system", "out_of_system"] | None`):**
   - For `team_side == "opponent"` or `set_trajectory is None`: `system_state = None` *(far-side setter foot depth and hand vs. forearm contact are occluded behind the net)*.
   - For `team_side == "ours"` with `set_trajectory`:
     - Compute normalized setter release position `x_rel_m = ((arc_points[0].x - cal.left_point.x) / dx) * 9.0`.
     - `in_horizontal_window`: `3.5 <= x_rel_m <= 7.6` (front-center to right-front setter target corridor).
     - `is_overhead_release`: if `setter_track_id` has a pose frame within `±5` frames of `release_frame` with valid shoulder & wrist keypoints, at least one wrist satisfies `wrist_y <= shoulder_mid_y + 12.0`; fallback when setter pose is missing: `arc_points[0].y <= tape_y_at(cal, arc_points[0].x) + 0.20 * span`.
     - `is_near_net`: if `setter_track_id` has resting/frame ankle `y_foot`, `y_foot <= net_ground_y_at(cal, arc_points[0].x) + 0.25 * span` (~within 3m of net); fallback `True` when `arc_points[0].y <= tape_y_at(cal, arc_points[0].x) + 0.16 * span`.
     - `system_state = "in_system"` if (`in_horizontal_window` and `is_overhead_release` and `is_near_net`) else `"out_of_system"`.
3. **Above-Tape Blocker Hand-Span Seam Fix (`BlockerCallout`, `BlockSummary`):**
   - On each `BlockerCallout`, compute `hand_span_x: tuple[float, float]`:
     - Gather valid upper-limb/head keypoints (`L_WRIST`, `R_WRIST`, `L_ELBOW`, `R_ELBOW`, `L_SHOULDER`, `R_SHOULDER`, `NOSE`) whose `y <= tape_y_at(cal, x) + 0.08 * span`.
     - Prefer wrist/elbow keypoints (`L_WRIST`, `R_WRIST`, `L_ELBOW`, `R_ELBOW`) when at least one is valid near/above the tape: `pad = max(6.0, 0.009 * span)` (~8 cm palm margin), `hand_span_x = (round(min_x - pad, 1), round(max_x + pad, 1))`.
     - Fallback when upper keypoints drop out: central `55%` of `bbox` width `(round(cx - 0.275 * bw, 1), round(cx + 0.275 * bw, 1))` so flaring legs/hips never falsely close a block seam.
   - Sort `blocker_callouts` by `0.5 * (b.hand_span_x[0] + b.hand_span_x[1])` and compute adjacent gap:
     `gap = b2.hand_span_x[0] - b1.hand_span_x[1]`.
   - `seam_open = True` when `best_gap >= min_seam_px` (`max(22.0 / max(est_scale, 1e-4), 0.024 * span)` ≈ `22 cm`), setting `seam_interval_x = (b1.hand_span_x[1], b2.hand_span_x[0])`.
   - **Block State (`block_state: Literal["unopposed", "zero_block", "single_block", "open_seam", "closed_block"]`):**
     - Let `has_defense_players = any(t.court_side == defense_side and len(t.frames) >= 3 for t in classified)`.
     - If `len(blocker_callouts) == 0`:
       - If `not has_defense_players` or `session_mode == "drill"`: `block_state = "unopposed"`.
       - Else (`session_mode == "match"` and defense players are on the court): `block_state = "zero_block"`.
     - Else if `len(blocker_callouts) == 1`: `block_state = "single_block"`.
     - Else if `seam_open`: `block_state = "open_seam"`.
     - Else: `block_state = "closed_block"`.

### 5.4 Studio Session Summary (`biomechanics.py`, `models.py`)

Pure deterministic aggregation `compute_studio_session_summary(plays, session_mode)` stored/returned on `AnalysisSession.summary`:

- **`ZoneSetWindowStats`** (per zone, e.g. `Zone 4 (Left)`, `Zone 3 (Middle)`, `Zone 2 (Right)`):
  - `zone: str`, `count: int`, `pct: float`
  - `tempo_mean_s: float | None`, `tempo_std_s: float | None` (`std` computed when `count >= 2`, population or sample documented—use sample std `ddof=1` when `count >= 2`, else `0.0` / `None` -> let's use `None` when `count < 2` so 1-rep zones don't claim `±0.00s` perfection!)
  - `peak_height_mean_cm: float | None`, `peak_height_std_cm: float | None` (`None` when `count < 2` or uncalibrated)
  - `pin_distance_mean_m: float | None`, `pin_distance_std_m: float | None` (`None` for Zone 3 Middle)
- **`BlockIsolationStats`**:
  - `opposed_count: int` (plays with `block_state != "unopposed"`)
  - `unopposed_count: int`
  - `zero_block_count: int`, `single_block_count: int`, `open_seam_count: int`, `closed_block_count: int`
  - `isolation_pct: float | None` (`100 * (zero_block + single_block + open_seam) / opposed_count` when `opposed_count > 0`, else `None`—so `Drill` hitting lines show `"Unopposed"` instead of `100%`)
  - `closed_block_pct: float | None` (`100 * closed_block / opposed_count` when `opposed_count > 0`—used as **Our Block Closure %** on `Opponent` sets!)
  - `avg_open_seam_cm: float | None`
- **`OffenseBucketSummary`** (used for `ours_all`, `ours_in_system`, `ours_out_of_system`, and `opponent_all`):
  - `play_count: int`
  - `back_row_count: int`, `back_row_pct: float`
  - `zones: list[ZoneSetWindowStats]` (ordered `Zone 4`, `Zone 3`, `Zone 2` for `ours`; `Opp Zone 4`, `Opp Zone 3`, `Opp Zone 2` for `opponent`)
  - `block_stats: BlockIsolationStats`
- **`StudioSessionSummary`**:
  - `session_mode: Literal["drill", "match"]`
  - `total_plays: int`, `ours_count: int`, `opponent_count: int`
  - `in_system_count: int`, `in_system_pct: float` (over `ours` plays with a `system_state`)
  - `ours_all: OffenseBucketSummary`
  - `ours_in_system: OffenseBucketSummary`
  - `ours_out_of_system: OffenseBucketSummary`
  - `opponent: OffenseBucketSummary`

### 5.5 Data Models, Storage & API (`models.py`, `storage.py`, `app.py`)

All new Pydantic fields carry defaults so existing `session.json` files load without migration errors:

- **`BlockerCallout` additions:** `hand_span_x: tuple[float, float] | None = None`
- **`BlockSummary` additions:** `block_state: Literal["unopposed", "zero_block", "single_block", "open_seam", "closed_block"] = "unopposed"`
- **`PlayAnalysis` additions:**
  - `team_side: Literal["ours", "opponent"] = "ours"`
  - `team_side_manual: bool = False`
  - `system_state: Literal["in_system", "out_of_system"] | None = None`
  - `pin_distance_m: float | None = None`
- **`AnalysisSession` additions:**
  - `session_mode: Literal["drill", "match"] = "drill"`
  - `summary: StudioSessionSummary | None = None` (always populated/refreshed in `SessionStore.load_session` and `save_session` so legacy sessions get summaries automatically upon load!)
- **New / Updated `SessionStore` & FastAPI Endpoints (`app.py`):**
  - `POST /api/sessions/upload` & `POST /api/sessions/local`: accept optional `session_mode: Literal["drill", "match"] = "drill"` (Form field on `/upload`, JSON field on `/local`).
  - `POST /api/sessions/{session_id}/mode` (`UpdateSessionModeRequest`: `session_mode: Literal["drill", "match"]`, CSRF-protected):
    - Updates `session.session_mode`, re-evaluates non-manual `team_side`, `block_state`, and `session.summary` across all cached `session.plays` in milliseconds (no GPU pass), saves and returns `session`.
  - `POST /api/sessions/{session_id}/plays/{play_id}/side` (`UpdatePlaySideRequest`: `team_side: Literal["ours", "opponent", "auto"]`, CSRF-protected):
    - Sets `team_side_manual = (team_side != "auto")`, clears `hitter_track_id` override if the effective `team_side` flipped so `analyse_play_mechanics` picks the appropriate near/far hitter, refreshes keyframe thumbnails if keyframe indices changed, recomputes `session.summary`, saves and returns `session`.
  - `SessionStore.reassign_play_hitter`: allows selecting any track matching the play's active `offense_side` (or if the user explicitly clicks a track on the other side during Reassign Hitter, keeps `team_side` consistent with that track's `court_side`).

### 5.6 Studio UI & Renderer (`index.html`, `app.js`, `styles.css`, `renderer.py`)

1. **Calibration / Top Control Bar (`index.html`, `app.js`):**
   - Add a compact **`Drill | Match`** segmented pill (`#studioModeGroup`) in `.cal-controls` (and in the Load Video modal as an initial choice defaulting to `Drill`). Clicking flips `/api/sessions/{id}/mode` immediately.
   - Add a **`Swap Side (Ours/Opp)`** button next to `Reassign Hitter` (or inside the active play header) for 1-click `Ours ↔ Opponent` override.
2. **Detected Plays List (`#playsList`):**
   - Add compact filter chips (`All`, `Ours`, `Opponent`) in `.plays-section` when `opponent_count > 0` or `session_mode === "match"`.
   - Each play card displays a subtle pill badge: `Ours` (cyan) or `Opp` (amber), plus `In-Sys` / `OOS` badge in `Match` mode. Timeline bars in `renderStudioTimeline` tint `Opponent` plays amber so rallies are visually scannable at a glance.
3. **Play Mechanics vs. Session Summary Sidebar (`Play | Session` Switcher):**
   - Replace the static `<h3>Play Mechanics</h3>` header in `.metrics-panel` with a 2-way segmented switcher: **`Play` | `Session`**.
   - **`Play` view:** Existing per-play rows plus **Pin Distance** (`e.g. 0.55 m from antenna` on pin sets) and **System / Side** badge (`Ours · In-System`, `Ours · Out-of-System`, or `Opponent`). On `Opponent` plays, `Block Seam` label dynamically reads **`Our Block`** (instead of `Block Seam`), and noisy far-court `Arm Extension` / `Torso Tilt` rows display a subtle `(Far-court est.)` note or muted styling.
   - **`Session` view:**
     - **In `Drill` mode:** Shows `Our Reps` count, **Setter Isolation** (`Unopposed` or `%` if live blockers jumped), and **Set Window Table** per zone (`Z4 / Z3 / Z2`: `Count`, `Tempo mean ± SD`, `Peak cm`, `Pin m`).
     - **In `Match` mode:** Adds sub-filter pills **`Our Offense` | `Opponent / Our Block`**, and under `Our Offense` a split selector **`All` | `In-System` | `Out-of-Sys`** showing:
       - KPI cards: `In-System %`, `Setter Isolation %` (`0/1/Open Seam`), `Avg Open Seam cm`, `Back-Row %`.
       - **Zone Distribution & Set Window** bars/rows (`Z4 / Z3 / Z2`: `%`, `Tempo mean ± SD`, `Pin m`).
     - Under **`Opponent / Our Block`**: shows Opponent `Opp Z4 / Opp Z3 / Opp Z2` distribution %, avg opponent set tempo, and **Our Block Closure %** (`closed_block_pct`) + `Our Open Seam Avg cm`.
4. **Canvas & MP4 Overlay Updates (`app.js`, `renderer.py`):**
   - `BlockerCallout` seam line uses `seam_interval_x` (now anchored to `hand_span_x`).
   - Top-left HUD header displays `PLAY #N · OPPONENT` on opponent plays and `Unopposed` vs `No Block` accurately from `block_state`.
   - `findNearestAttackingTrackId` in `app.js` filters for `track.court_side === (play.team_side === "opponent" ? "blocking" : "attacking")` so `Reassign Hitter` works seamlessly on both sides.

---

## 6. Security & Verification Plan

Per `mandatory-secure-web-skills`:
- **CSRF & Method Enforcement:** Unit/API tests in `tests/test_api_security.py` / `tests/test_pipeline.py` must verify that `POST /api/sessions/{id}/mode` and `POST /api/sessions/{id}/plays/{play_id}/side` reject missing/invalid CSRF tokens with `403`, reject invalid session/play IDs (`../` traversal) with `400`, and reject non-allowlisted HTTP methods (`PUT`, `DELETE`, `PATCH`) with `405`.
- **Input Validation:** Strict Pydantic `Literal["drill", "match"]` and `Literal["ours", "opponent", "auto"]` validation on all request bodies and form fields.
- **DOM XSS Prevention:** All new UI elements in `static/app.js` must be built with `document.createElement`, `textContent`, and `replaceChildren()` (verified by automated grep/frontend tests ensuring zero `innerHTML`/`insertAdjacentHTML` usage and zero `alert`/`confirm`/`prompt` dialogues).
- **Localhost Test Binding:** All test clients and test servers bind exclusively to `127.0.0.1` / in-memory ASGI `TestClient`.

---

## 7. User Decisions vs. Proposed Defaults & Builder Calls

- **Decided by User:**
  - Camera always stays behind the user's team baseline across sets (near side = `ours`, far side = `opponent`).
  - Session-level **`Drill | Match`** toggle activates drill vs. match coaching logic from cached data.
  - Detect **both** `Our Set` and `Opponent Set`, labeling each play and keeping their stats separate.
  - **Approach 1A** selected: peak-split play windows + sidebar `Play | Session` switcher.
- **Proposed Defaults (open for user adjustment):**
  - Minimum swing peak separation `2.3s` with `[t_peak - 2.6s, t_peak + 1.6s]` play windows.
  - `In-System` threshold for `Ours`: overhead release (`wrist_y <= shoulder_y + 12px`), within `~3m` of net, and horizontal release `x ∈ [3.5m, 7.6m]` along the `9.0m` net.
  - Above-tape blocker seam threshold: `>= 22 cm` between adjacent blockers' `hand_span_x` intervals.
  - Opponent plays automatically mirror `Zone 4 ↔ Zone 2` and treat near-side net jumpers as `Our Block`.
- **Left to the Builder:**
  - Exact peak-smoothing weights in `scan_candidate_plays` and overlapping-window batch slicing mechanics in `detector.py`.
  - Exact CSS tokens/spacing for the `Drill | Match` pill, `Play | Session` switcher, and compact Set Window table inside `styles.css`.
  - Synthetic multi-rally (ours + opponent) test clip helpers in `tests/conftest.py` and test suite organization across `test_biomechanics.py`, `test_pipeline.py`, `test_renderer.py`, and `test_api_security.py`.
