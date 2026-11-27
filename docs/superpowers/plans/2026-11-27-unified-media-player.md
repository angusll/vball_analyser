# Unified Media Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Set & Hit Studio media player into a single shared `MediaPlayer` component (`static/media_player.js`) and use only that single player instance across both Set & Hit Studio and Serve Distribution.

**Architecture:** One `MediaPlayer` class owns a single `#mediaPlayerRoot` DOM subtree (`<video>`, overlay `<canvas>`, and `.transport-bar` with frame stepping, canvas timeline, speed buttons, and readout). On tab switch, `mediaPlayer.mountTo(slotEl, adapter)` pauses playback, reparents `#mediaPlayerRoot` into the active tab's mount slot (`#studioPlayerSlot` or `#servePlayerSlot`), and swaps the active tab adapter (overlay renderer, timeline marker renderer, pointer hooks, and extra hotkeys) while keeping the loaded video, `currentTime`, FPS, and `playbackRate` intact.

**Tech Stack:** Vanilla ES6 browser JS, HTML5 `<video>` & `<canvas>`, CSS, FastAPI static serving, pytest, Node.js `node:test` (v20+).

**Spec:** `docs/superpowers/specs/2026-11-27-unified-media-player-design.md`

## Global Constraints

- Plain ES6 browser JavaScript served from `src/vball_analyser/static/` (no bundler, no new Python/JS dependencies).
- Strict frontend security hygiene: never use `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `alert(`, `confirm(`, `prompt(`, `localStorage`, `sessionStorage`, inline `<script>`, or inline `style=` in `index.html`.
- Exactly **one** `<video>` element in the entire app DOM.
- No changes to backend API routes, Pydantic models, or Python analysis pipelines.
- Responsive layout preserved at `<= 960px` and `<= 640px`, including inside Serve Distribution's side-by-side `.serve-video-panel`.
- Always run Python commands via `/home/angus/.local/bin/uv run` (Python 3.12 `.venv`).

## Review Focus

- **Speed button selector collision:** Clicking `0.25x`/`0.5x`/`1x` must only toggle `.speed-btn` inside `#mediaPlayerRoot`, never `.serve-formation-btn` in Serve Distribution (pinned in Task 1 `node:test`).
- **Tab switch mid-playback:** Switching tabs while playing must pause playback, move `#mediaPlayerRoot`, and preserve `currentTime` and `playbackRate` without reloading when `session_id` is unchanged (pinned in Task 1 `node:test`).
- **Keyboard shortcuts in form fields:** `Space`, `ArrowLeft`, `ArrowRight` must work in both tabs when a video is loaded, and must be ignored when focus is in `INPUT`, `SELECT`, or `TEXTAREA` (pinned in Task 1 `node:test`).
- **Unlinked Serve session with loaded Studio video:** Switching from Studio (with video loaded) to a Serve session with `video_session_id === null` must keep `#serveVideoCard` visible with the shared video loaded, and auto-link that video when starting calibration or auto-detect (pinned in Task 3).
- **Manual-only Serve unlink & return to Studio:** Explicitly choosing "No Linked Video (Manual)" in Serve Distribution must clear/hide the Serve video card cleanly without breaking Studio when returning to the Studio tab (pinned in Task 2 & Task 3).

---

### Task 1: Build `MediaPlayer` module, HTML mount slots, and responsive CSS

**Files:**
- Create: `src/vball_analyser/static/media_player.js`
- Modify: `src/vball_analyser/static/index.html:129-224`, `301-323`, `454-457`
- Modify: `src/vball_analyser/static/styles.css:581-621`, `726-845`, `1734-1763`
- Test: `tests/test_media_player_frontend.py`
- Test: `tests/test_media_player.mjs`

**Interfaces:**
- Produces (`window.VBallMediaPlayer`):
  - `class MediaPlayer(rootEl)` with:
    - `mountTo(containerEl, adapter)` -> `void`
    - `loadSession(videoSession, opts = { resetTime: false })` -> `boolean` (returns `true` if `src` changed)
    - `clear()` -> `void`
    - `play()` / `pause()` / `togglePlay()` -> `void`
    - `seekToTime(seconds)` -> `void`
    - `jumpToFrame(frameIdx)` -> `void`
    - `stepFrames(delta)` -> `void`
    - `setPlaybackRate(rate)` -> `void`
    - `setCanvasInteractive(enabled)` -> `void`
    - `requestRender()` -> `void`
    - `getCurrentTime()` -> `number`
    - `getCurrentFrame()` -> `number`
    - `getSessionId()` -> `string | null`
    - `getSession()` -> `object | null`
    - `getFps()` -> `number`
    - `getDuration()` -> `number`
    - `hasVideo()` -> `boolean`
    - `getVideoDisplayRect()` -> `{ ox, oy, dw, dh, scale }`
    - `toCanvasPt(pt)` -> `{ x, y }`
    - `toVideoPt(clientX, clientY)` -> `{ x, y }`
    - `video` (HTMLVideoElement), `canvas` (HTMLCanvasElement), `root` (HTMLElement)
  - `adapter` contract: `{ onRenderOverlay?(ctx, player, frameIdx, rect), onRenderTimeline?(ctx, player, layout), onTimeUpdate?(timeS, frameIdx), onCanvasMouseMove?(videoPt, ev), onCanvasMouseLeave?(ev), onCanvasClick?(videoPt, ev), onKeyDown?(ev, player), showEmptyState?: boolean, allowFileDrop?: boolean }`
  - `layout` passed to `onRenderTimeline`: `{ cw, ch, pad, usableW, trackY, trackH, duration }`

- [ ] **Step 1: Write failing tests (`tests/test_media_player_frontend.py` and `tests/test_media_player.mjs`)**
  - `tests/test_media_player_frontend.py`:
    - `test_single_video_element_and_media_player_slots(tmp_path)`: GET `/` asserts `html.count("<video") == 1`, `id="studioPlayerSlot"` and `id="servePlayerSlot"` and `id="mediaPlayerRoot"` present, `/static/media_player.js` included before `app.js` and `serve_tab.js`, and legacy IDs (`id="serveVideo"`, `id="serveVideoCanvas"`, `id="servePlayPauseBtn"`, `id="serveSeekSlider"`, `id="serveVideoTime"`) absent. GET `/static/media_player.js` returns 200.
    - `test_js_syntax_and_media_player_node_unit_tests()`: runs `node --check` on all `static/*.js` files and `node --test tests/test_media_player.mjs`.
  - `tests/test_media_player.mjs`:
    - Loads `src/vball_analyser/static/media_player.js` in a `vm` sandbox with lightweight DOM fakes.
    - Tests: (1) `loadSession` sets `src`, `fps`, `duration`, and preserves `currentTime` + `playbackRate` when called again with the same `session_id`; (2) `mountTo` pauses video, reparents `root` into the target slot, swaps adapter, and invokes `onRenderOverlay` + `onRenderTimeline`; (3) `setPlaybackRate(0.5)` only toggles `.speed-btn` inside `root`, leaving an external `.serve-formation-btn.speed-btn` untouched; (4) `stepFrames` / `jumpToFrame` clamp to `[0, duration]`; (5) `toCanvasPt` / `toVideoPt` round-trip accurately; (6) keyboard handler handles `Space`/`ArrowLeft`/`ArrowRight`, delegates to `adapter.onKeyDown`, and ignores events targeting `INPUT`/`SELECT`/`TEXTAREA`.

- [ ] **Step 2: Run tests to verify they fail**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_media_player_frontend.py -v`
  - Expected: FAIL (`media_player.js` missing, `html.count("<video") == 2`, etc.).

- [ ] **Step 3: Implement `src/vball_analyser/static/media_player.js`**
  - Implement `MediaPlayer` class matching the `Interfaces` block above, exporting via `window.VBallMediaPlayer = { MediaPlayer }` (and `globalThis.VBallMediaPlayer` for Node `vm` testing).
  - Scope all DOM queries (`querySelectorAll(".speed-btn")`, transport buttons, `#timelineCanvas`, `#seekSlider`, `#timeReadout`, `#playerVideo`, `#overlayCanvas`) strictly inside `this.root`.
  - Drive rendering via `requestVideoFrameCallback` (with `requestAnimationFrame` fallback) during playback, plus `timeupdate`, `seeked`, `loadedmetadata`, `ResizeObserver` (when available), and `window.resize`.
  - Render base timeline rail + played fill in `renderTimeline()`, then invoke `adapter.onRenderTimeline(ctx, this, layout)`.
  - Handle `Space`, `ArrowLeft`, `ArrowRight` globally on `document` `keydown` (skipping `INPUT`/`SELECT`/`TEXTAREA` and when `!this.hasVideo()`), and forward all keydowns to `this._adapter?.onKeyDown?.(ev, this)`.

- [ ] **Step 4: Update `src/vball_analyser/static/index.html` and `src/vball_analyser/static/styles.css`**
  - In `index.html`:
    - Wrap `#videoStage` and `.transport-bar` in `<div id="studioPlayerSlot" class="player-slot"><div id="mediaPlayerRoot" class="media-player">...</div></div>`.
    - In `#serveVideoCard`: remove `#servePlayPauseBtn`, `#serveVideoTime`, `#serveVideoStage` (`#serveVideo`, `#serveVideoCanvas`), and `.serve-video-scrub` (`#serveSeekSlider`); replace with `<div id="servePlayerSlot" class="player-slot serve-player-slot"></div>`.
    - Add `<script src="/static/media_player.js" defer></script>` before `app.js` and `serve_tab.js`.
  - In `styles.css`:
    - Style `.player-slot` and `.media-player` (`display: flex; flex-direction: column; flex: 1; min-height: 0; gap: 0.625rem;`).
    - Style `.serve-player-slot` so `.media-player` fits inside `.serve-video-panel` with `gap: 0`, square inner borders on `.video-stage` / `.transport-bar`, and responsive wrapping of `.transport-bar` when inside `.serve-video-panel` (`flex-wrap: wrap` with full-width `.timeline-wrap` on top so frame buttons + speeds + readout never overflow the side-by-side card).
    - Remove dead `.serve-video-stage` and `.serve-video-scrub` rules.

- [ ] **Step 5: Run tests to verify Task 1 passes**
  - Run: `/home/angus/.local/bin/uv run pytest tests/test_media_player_frontend.py tests/test_api_security.py::test_static_frontend_and_backend_security_hygiene -v`
  - Expected: PASS.

- [ ] **Step 6: Commit**
  - Commit `media_player.js`, `index.html`, `styles.css`, and the new tests.

---

### Task 2: Refactor Set & Hit Studio (`src/vball_analyser/static/app.js`) to use `MediaPlayer`

**Files:**
- Modify: `src/vball_analyser/static/app.js`

**Interfaces:**
- Consumes: `window.VBallMediaPlayer.MediaPlayer` from Task 1.
- Produces (`window.VBallShared`):
  - `apiPostJson`, `fetchCsrf`, `openLoadModalForServeTab`, `refreshSessionsDropdown`
  - `mediaPlayer` (the single shared `MediaPlayer` instance)
  - `getStudioSession()` -> `object | null`
  - `syncStudioToVideoSession(sessionId)` -> `Promise<void>` (loads the analysis session in Studio if `sessionId` differs from `state.session?.session_id`, without resetting video `currentTime` or `playbackRate`)

- [ ] **Step 1: Instantiate singleton `mediaPlayer` and create the Studio adapter in `app.js`**
  - Create `const mediaPlayer = new window.VBallMediaPlayer.MediaPlayer(document.getElementById("mediaPlayerRoot"));`.
  - Expose `mediaPlayer`, `getStudioSession`, and `syncStudioToVideoSession` on `window.VBallShared`.
  - Implement `studioAdapter`:
    - `showEmptyState: true`, `allowFileDrop: true`
    - `onRenderOverlay(ctx, player, frameIdx, rect)`: existing Studio overlay drawing (net tape, calibration preview, set arc, hitter pose/candidates, contact callout, blockers/seam, mechanics HUD) using `player.toCanvasPt(...)` and `player.getVideoDisplayRect()`.
    - `onRenderTimeline(ctx, player, layout)`: draws Studio play segments and active play keyframe ticks (`set_release`, `max_load`, `contact`).
    - `onTimeUpdate(timeS, frameIdx)`: calls `syncActiveKeyframeHighlight(frameIdx)`.
    - `onCanvasMouseMove(pt)`, `onCanvasMouseLeave()`, `onCanvasClick(pt)`: existing net calibration & hitter reassignment logic.
    - `onKeyDown(ev, player)`: `Escape` cancels interactive modes; `Digit1`/`Digit2`/`Digit3` jump to active play keyframes via `player.jumpToFrame(...)`.

- [ ] **Step 2: Remove duplicated player logic from `app.js` and wire tab switching**
  - Remove `getVideoDisplayRect`, `toCanvasPt`, `toVideoPt`, `currentFrameIdx`, `renderTimeline`, `updatePlayPauseLabel`, `stepFrames`, `jumpToFrame`, direct `els.video`/`els.seekSlider`/`els.playPauseBtn`/`.speed-btn` listeners, and `els.canvas` mouse/click listeners from `app.js` (keep drag-and-drop on `els.videoStage` gated by `state.activeTab === "studio"`).
  - Update `applySession(sess, resetVideoSrc)` to call `mediaPlayer.loadSession(sess, { resetTime: Boolean(resetVideoSrc && prevId !== sess.session_id) })` and `mediaPlayer.requestRender()`.
  - Update `setInteractiveMode(calNet, reassign)` to call `mediaPlayer.setCanvasInteractive(calNet || reassign)`.
  - Update `switchTab(tabName)`:
    - Switching to `"studio"`: mount `mediaPlayer.mountTo(document.getElementById("studioPlayerSlot"), studioAdapter)`. If `mediaPlayer.hasVideo()` and `mediaPlayer.getSessionId() !== (state.session && state.session.session_id)`, call `syncStudioToVideoSession(mediaPlayer.getSessionId())` (which fetches `/api/sessions/<id>` and calls `applySession(fresh, false)` preserving current time & speed).
    - Switching to `"serve"`: pause player cleanly (and exit any active Studio calibration/reassign mode) before invoking `window.VBallServeTab.onTabActivated()`.

- [ ] **Step 3: Verify syntax and tests**
  - Run: `node --check src/vball_analyser/static/app.js && /home/angus/.local/bin/uv run pytest tests/test_media_player_frontend.py tests/test_api_security.py -v`
  - Expected: PASS.

- [ ] **Step 4: Commit**
  - Commit `src/vball_analyser/static/app.js`.

---

### Task 3: Refactor Serve Distribution (`src/vball_analyser/static/serve_tab.js`) to use the shared `MediaPlayer`

**Files:**
- Modify: `src/vball_analyser/static/serve_tab.js`
- Modify: `tests/test_media_player_frontend.py` (add assertion that `serve_tab.js` contains no legacy player IDs or `getServeVideoRect`)

**Interfaces:**
- Consumes: `window.VBallShared.mediaPlayer` from Task 2.

- [ ] **Step 1: Add regression assertion in `tests/test_media_player_frontend.py`**
  - Assert `serve_tab.js` does not reference `serveVideo`, `serveVideoCanvas`, `servePlayPauseBtn`, `serveSeekSlider`, `serveVideoTime`, or `getServeVideoRect`, and does reference `mediaPlayer`.
  - Run `/home/angus/.local/bin/uv run pytest tests/test_media_player_frontend.py -v` and confirm the new assertion fails before editing `serve_tab.js`.

- [ ] **Step 2: Implement Serve Distribution adapter & shared video sync in `serve_tab.js`**
  - Remove `sEls.playPauseBtn`, `sEls.videoTime`, `sEls.video`, `sEls.videoCanvas`, `sEls.seekSlider`, `getServeVideoRect`, and their listeners.
  - Add `sEls.playerSlot = document.getElementById("servePlayerSlot")`.
  - Create `serveAdapter`:
    - `showEmptyState: false`, `allowFileDrop: false`
    - `onRenderOverlay(ctx, player, frameIdx, rect)`: draws saved court calibration net line + homography grid and live 3-click/4-click calibration points using `player.toCanvasPt`.
    - `onRenderTimeline(ctx, player, layout)`: draws vertical tick markers for each visible serve with a non-null `timestamp_s` (colored `#38bdf8` in, `#facc15` ace, `#fb7185` out/net; taller white-bordered highlight for `sState.selectedServeId`).
    - `onCanvasMouseMove(videoPt)` & `onCanvasClick(videoPt)`: update `sState.calHover` / append to `sState.calPoints` and submit calibration (auto-linking `mediaPlayer.getSessionId()` to the Serve session first if `sState.session.video_session_id` is null).
    - `onKeyDown(ev)`: `Escape` cancels calibration mode.
  - Helper `ensureLinkedVideoForServe()`: if `sState.session && !sState.session.video_session_id && mediaPlayer.hasVideo()`, links `mediaPlayer.getSessionId()` via `/api/serve-sessions/<id>/update` before calibration or auto-detect.
  - Update `applyServePayload(payload)`:
    - Determine `effectiveVideoId = sess.video_session_id || (mediaPlayer.hasVideo() ? mediaPlayer.getSessionId() : null)` (unless the user explicitly selected "No Linked Video (Manual)", in which case `clear()` the player and hide `#serveVideoCard`).
    - Show/hide `#serveVideoCard`, `#toggleCalOverlayBtn`, and enable/disable calibration buttons based on `Boolean(effectiveVideoId)`.
    - If `sess.video_session_id` is set and differs from `mediaPlayer.getSessionId()`, look up its metadata in `sState.videoSessions` (or fetch `/api/sessions/<id>`) and call `mediaPlayer.loadSession(vs, { resetTime: true })`.
    - Call `mediaPlayer.requestRender()`.
  - Update `setCalibrationMode(active, mode)` to toggle `mediaPlayer.setCanvasInteractive(active)` and `mediaPlayer.requestRender()`.
  - Update `selectServe(sv)` to call `mediaPlayer.pause()` + `mediaPlayer.seekToTime(sv.timestamp_s)` when `sv?.timestamp_s != null && mediaPlayer.hasVideo()`.
  - Update manual court click logging to read `mediaPlayer.hasVideo() ? mediaPlayer.getCurrentTime() : null`.
  - Update `onTabActivated()` to mount `mediaPlayer.mountTo(sEls.playerSlot, serveAdapter)` and run `ensureInit()`.

- [ ] **Step 3: Run full verification suite**
  - Run: `node --check src/vball_analyser/static/media_player.js && node --check src/vball_analyser/static/app.js && node --check src/vball_analyser/static/serve_tab.js && node --test tests/test_media_player.mjs && /home/angus/.local/bin/uv run pytest -v`
  - Expected: All JS syntax checks, Node unit tests, and all 45+ pytest tests (including security hygiene) PASS.

- [ ] **Step 4: Commit**
  - Commit `src/vball_analyser/static/serve_tab.js` and `tests/test_media_player_frontend.py`.

## Security Verification Plan

- Confirm `tests/test_api_security.py::test_static_frontend_and_backend_security_hygiene` passes across all `src/vball_analyser/static/*.js` (including `media_player.js`) and `index.html`: no `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `alert(`, `confirm(`, `prompt(`, `localStorage`, `sessionStorage`, inline `<script>`, or inline `style=`.
- Confirm all video URLs constructed in `media_player.js` use `encodeURIComponent(sessionId)` and relative same-origin paths (`/api/sessions/.../video`).
- Confirm no new dependencies, no backend auth/CSRF changes, and no `0.0.0.0` bindings.
