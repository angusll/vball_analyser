# Unified Media Player Design

**Date:** 2026-11-27
**Status:** Approved

## Intent & Why

Today the app maintains two separate `<video>` players:
- **Set & Hit Studio** has the full review player (`#playerVideo`, `#overlayCanvas`, `-5f`/`-1f`/Play/`+1f`/`+5f` frame stepping, `0.25x`/`0.5x`/`1x` playback speeds, a canvas-backed timeline rail with play/keyframe markers, `time · frame` readout, `requestVideoFrameCallback` sync, and keyboard shortcuts).
- **Serve Distribution** has a stripped-down duplicate (`#serveVideo`, `#serveVideoCanvas`, a plain Play/Pause button, time readout, and bare range slider).

The goal is to refactor the Set & Hit Studio player into a single unified `MediaPlayer` component and use **only** that media player across the entire app so that:
1. Serve Distribution gets the full frame-stepping, playback-speed, canvas-timeline, and keyboard-shortcut experience.
2. Both tabs share the same loaded video state (video source, current timestamp, FPS, and playback speed) without duplicating `<video>` elements or reloading on tab switch.

## Goals, Non-Goals & Anti-Goals

### Goals
- Extract a standalone `MediaPlayer` module (`src/vball_analyser/static/media_player.js`) owning **one** `<video>`, **one** overlay `<canvas>`, and **one** full `.transport-bar`.
- Remove the separate `#serveVideo` / `#serveVideoCanvas` / `#servePlayPauseBtn` / `#serveVideoTime` / `#serveSeekSlider` elements and all duplicated player code from `serve_tab.js`.
- Reparent the single `MediaPlayer` DOM into whichever tab is active (Studio's stage slot or Serve Distribution's video panel slot) on tab switch, preserving video state.
- Provide tab-specific hooks for overlay rendering, timeline marker rendering, canvas pointer interactions, and drag-and-drop / empty-state overlays.
- Fix the selector collision where `.speed-btn` in `app.js` also matched `.serve-formation-btn` by scoping all player selectors to the `MediaPlayer` root element.

### Non-Goals
- Changing any backend API endpoints, Pydantic models, or Python analysis pipelines.
- Changing the overall two-tab layout (Studio remains video + sidebar; Serve Distribution remains video panel + 9m × 9m court + sidebar).
- Adding new video codecs, audio tracks, or unrelated UI features.

### Anti-Goals
- Leaving a second `<video>` element anywhere in the DOM.
- Reloading or losing playback position/speed when switching between Studio and Serve Distribution for the same video.
- Breaking existing Studio keyboard shortcuts (`Space`, `←`, `→`, `1`/`2`/`3`, `Esc`) or Serve court/calibration interactions.

## Constraints
- Plain ES6 browser JavaScript (no bundler or npm build step), served directly from `src/vball_analyser/static/` via FastAPI `StaticFiles`.
- Must respect existing CSP / security headers (no inline scripts or `innerHTML` with untrusted data; use DOM APIs / `textContent`).
- Must remain responsive at narrow viewports (`<= 960px`, `<= 640px`), including when the player sits inside Serve Distribution's narrower side-by-side panel.

## Approaches Considered

1. **Approach A — Single `MediaPlayer` instance reparented between tabs *(Selected)***
   - One `MediaPlayer` DOM subtree (`<video>` + `<canvas>` + `.transport-bar`) moved between `#studioPlayerSlot` and `#servePlayerSlot` on tab switch.
   - *Trade-off:* True singleton `<video>` in the DOM and zero reload/seek lag when sharing video state across tabs; requires clean adapter swapping and explicit video-sync rules on tab switch.
2. **Approach B — Reusable `MediaPlayer` class instantiated per tab, synced via a shared store**
   - Two `<video>` elements in the DOM synchronized on tab switch.
   - *Trade-off:* Simpler CSS isolation, but duplicates `<video>` elements/decoder memory and introduces seek/buffer flicker on tab switch.

Approach A was selected because it directly satisfies using only one media player app-wide while sharing live video state seamlessly.

## Architecture & Behavior

### 1. `MediaPlayer` Component (`static/media_player.js`)
Exposed as `window.VBallMediaPlayer` (a single class `MediaPlayer`, with one shared singleton instance created at boot).

**Owned DOM structure** (built or adopted inside a `.media-player` wrapper):
- `.video-stage` containing `<video>`, `<canvas class="player-overlay-canvas">`, and an optional stage-slot overlay area (for Studio's empty state and drag-and-drop overlay).
- `.transport-bar` containing:
  - Frame step buttons (`-5f`, `-1f`, Play/Pause with SVG icon swap, `+1f`, `+5f`)
  - `.timeline-wrap` with `<canvas class="timeline-canvas">` + `<input type="range" class="seek-slider">`
  - `.transport-meta` with `.speed-group` (`0.25x`, `0.5x`, `1x`, scoped inside the player root) and `.time-readout` (`0.00s · f0`)

**Core public API:**
- `mountTo(containerEl, adapter)`: pauses playback, moves the player root into `containerEl`, binds the active tab's `adapter`, and triggers an immediate redraw of overlay + timeline.
- `loadSession(videoSession, { resetTime = false } = {})`: loads `/api/sessions/<id>/video` only if the `session_id` changed (preserving `currentTime` and `playbackRate` when the same video is already loaded); updates `fps`, `duration_s`, `width`, `height`, and slider `max`.
- `clear()`: pauses and unloads video source, resets readout and slider, disables controls.
- `play()`, `pause()`, `togglePlay()`, `seekToTime(seconds)`, `jumpToFrame(frameIdx)`, `stepFrames(delta)`, `setPlaybackRate(rate)`.
- `getCurrentTime()`, `getCurrentFrame()`, `getSessionId()`, `hasVideo()`.
- `getVideoDisplayRect()`, `toCanvasPt(pt)`, `toVideoPt(clientX, clientY)` — single source of truth for letterboxed video-to-canvas coordinate mapping.
- `setCanvasInteractive(enabled)` — toggles pointer events / crosshair cursor on the overlay canvas.
- `requestRender()` — redraws both the overlay canvas and timeline canvas (also driven automatically by `requestVideoFrameCallback` / `requestAnimationFrame` during playback, `timeupdate`, `seeked`, `loadedmetadata`, and `ResizeObserver` / `window.resize`).

**Tab Adapter interface** (passed to `mountTo`):
- `onRenderOverlay(ctx, player, frameIdx, rect)` — draws tab-specific canvas graphics (Studio: net tape, set arc, pose, contact, blockers, HUD; Serve: saved court calibration grid + net line, live calibration clicks).
- `onRenderTimeline(ctx, player, layout)` — draws tab-specific timeline markers on top of the base rail and played fill (Studio: play segments + keyframe ticks; Serve: serve timestamp ticks, highlighting the selected serve).
- `onTimeUpdate(timeS, frameIdx)` — optional callback (Studio uses it to sync active keyframe card highlight).
- `onCanvasMouseMove(videoPt, ev)`, `onCanvasMouseLeave(ev)`, `onCanvasClick(videoPt, ev)` — interactive canvas hooks.
- `onKeyDown(ev, player)` — optional extra tab-specific hotkeys (e.g., Studio `1`/`2`/`3` keyframes and `Esc`; Serve `Esc` to cancel calibration) while `Space` and `←`/`→` are handled once inside `MediaPlayer`.
- `showEmptyState` / `allowFileDrop` — Studio mounts its empty-state card and drag-and-drop overlay inside the stage when active; Serve hides them when active.

### 2. Shared Video State & Cross-Tab Sync
- **Single source of truth:** `MediaPlayer` tracks the currently loaded `videoSession` (`session_id`, `metadata: { fps, duration_s, width, height }`), `currentTime`, and `playbackRate`.
- **Switching Studio → Serve Distribution:**
  - Playback pauses cleanly; `MediaPlayer` mounts into `#servePlayerSlot` inside `#serveVideoCard`.
  - If the active Serve session already has a `video_session_id` matching the loaded video, nothing reloads — timestamp and speed carry straight over.
  - If the active Serve session has a *different* `video_session_id`, the player loads that Serve session's video (and updates the shared video state).
  - If the active Serve session has *no* linked video (`video_session_id === null`), and Studio already has a video loaded, Serve Distribution auto-links / displays that loaded video in the Serve video card (or keeps it loaded in the player and syncs `#serveVideoSelect`) so the user doesn't lose their loaded video when switching tabs. *(Call: auto-showing the currently loaded video in the Serve session's card and persisting the link only when the user explicitly selects/links or starts calibration/auto-detect, OR auto-linking the session — let's keep it clean: if the Serve session has no linked video and a video is already loaded in Studio, show `#serveVideoCard` with the shared video loaded and allow calibrating/auto-detecting by auto-linking it on first action or via `#serveVideoSelect`.)*
  - If neither tab has a video loaded, `#serveVideoCard` stays hidden (manual-only court mode).
- **Switching Serve Distribution → Studio:**
  - Playback pauses cleanly; `MediaPlayer` mounts back into `#studioPlayerSlot`.
  - If Serve Distribution loaded or switched to a different video session than Studio's `state.session`, Studio fetches and applies that session (preserving `currentTime` and `playbackRate` because `loadSession` sees the same `session_id` already in the `<video>`).
  - If Studio already had that same video loaded, timestamp and speed carry straight back with zero reload.

### 3. Styling (`styles.css`)
- Keep `.transport-bar` compact enough to fit cleanly inside Serve Distribution's side-by-side `.serve-video-panel` (wrapping gracefully into two rows when the panel is narrower than ~560px via container/media rules, without overflowing).
- Remove dead `#serveVideo`, `#serveVideoCanvas`, `.serve-video-stage`, and `.serve-video-scrub` styles once replaced by `.media-player`.

## What's Left to the Builder
- Exact DOM helper structure inside `media_player.js` (e.g., `ResizeObserver` vs window resize handling).
- Ensuring Serve Distribution's video panel header stays clean (calibration badge, hint, cancel button — no duplicate Play/time controls).
- Writing automated tests verifying the single-player DOM invariant, static asset serving, and existing backend tests passing.
