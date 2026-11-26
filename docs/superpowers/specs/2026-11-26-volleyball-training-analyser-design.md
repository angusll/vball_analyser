# Volleyball Baseline Training Analyser — Design Spec

**Date:** 2026-11-26  
**Status:** Ready for User Approval

---

## 1. Intent and Why

Players and coaches filming training or scrimmages from a single tripod behind the baseline want an immediate **visual breakdown of the set and hit**—without setting up multi-camera rigs or wading through full-match statistics software.

Watching raw clips at normal speed makes it hard to judge:
- The exact trajectory arc and tempo of the set out of the setter's hands.
- Where and how high above the net tape the hitter actually contacts the ball, and whether contact happened at peak jump or on the way down.
- The hitter's body posture through load and contact (hitting-arm extension and torso arch/tilt).
- How the opposing block lined up across the net and whether an open seam was left between blockers.

This tool takes baseline video—either short single-play clips or long uncut training/scrimmage recordings—detects each set-and-hit sequence, and presents an interactive visual review player with clean court overlays, frame-accurate key-moment jumping, and annotated MP4 clip export.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Dual video input support:** Handle both pre-trimmed single-play clips (a few seconds long) and uncut multi-minute practice/game recordings (`MP4`, `MOV`, `MKV`, `AVI`).
- **Automatic set-and-hit play detection:** Scan uncut recordings to extract candidate attack plays around the net, while letting the user manually add, delete, or nudge play start/end boundaries.
- **Visual breakdown of four play elements on baseline footage:**
  1. **Setting path:** Smooth traced arc of the volleyball from setter release to hit contact, displaying peak arc position and set tempo (duration in seconds).
  2. **Hitter posture:** 2D pose skeleton on the active hitter from takeoff through contact, highlighting max load ("bow-and-arrow" / trophy pose), 2D hitting-arm extension angle at contact, and torso tilt/arch in the camera plane.
  3. **Hitting point:** Pinpointed contact frame and ball position, showing vertical height above the net top tape at that lateral net position and jump timing (contact at peak jump vs. early/late relative to jump apex).
  4. **Blockers:** Highlighted bounding boxes and upper-body/arm reach above the net tape for active opposing blockers at the moment of contact, calling out open horizontal seams between multiple blockers.
- **Interactive Review Studio (Web UI):**
  - Live HTML5 `<video>` + `<canvas>` overlay player with instant independent toggles (`Set Arc`, `Hitter Pose`, `Contact Point`, `Blockers`, `Net Tape`).
  - **Detected Plays** sidebar to jump between plays in uncut sessions (with play timestamp and approximate net attack zone: Left / Middle / Right).
  - **3-Keyframe Strip** (`1. Set Release` → `2. Max Load` → `3. Ball Contact`) plus frame-by-frame stepping and slow-motion playback (`0.25x`, `0.5x`, `1.0x`).
  - Simple **Net Tape Calibration** overlay (auto-detected net top tape with 2-click manual override on left/right antenna–top-tape intersections).
  - **Export MP4** button to render a standalone annotated video clip of the selected play (with optional brief contact freeze-frame).

### Non-Goals (v1)
- **Multi-camera 3D reconstruction or optical-axis depth metrics:** We will not estimate how many centimeters off the net a set is or 3D depth of the ball in front of the hitting shoulder along the camera's line of sight.
- **Match stat sheets & team management:** No kill/error percentages, pass ratings, rotation tracking, jersey OCR, player databases, or cloud accounts.
- **Live camera feed analysis:** Operates on recorded video files only.
- **Defending-side baseline camera angle:** v1 assumes the camera sits behind the **attacking team's baseline** looking toward the net.

### Anti-Goals
- **Visual clutter across all 12 players:** Drawing skeletons and boxes on every bystander or back-row player ruins readability. Overlays must isolate only the active play participants (setter release marker, active hitter skeleton, active opposing blockers, ball arc, and net reference).
- **False 3D precision:** Never report foreshortened 2D camera-plane angles as true 3D joint angles, and never invent centimeter heights unless anchored to calibrated net/antenna geometry.
- **Laggy review workflow:** Toggling an overlay or stepping one frame back/forward must never wait on backend video re-encoding.
- **Unrecoverable CV mistakes:** Gym lighting, background courts, and stray balls during drills cause occasional false detections; the UI must allow quick overrides (2-click net tape fix, clicking another player to reassign the active hitter, nudging keyframe/play bounds) so a play is never unusable.

---

## 3. Constraints

- **Platform & Runtime:** WSL2 Linux (`/mnt/c/Users/Angus/Desktop/Git/vball_analyser`), **Python 3.12** managed exclusively via `uv` (`.venv` in repo root, executed via `uv run`).
- **Hardware:** Local **NVIDIA GeForce RTX 4090 (24 GB VRAM)** and 16 CPU threads—enables fast local GPU inference without external paid APIs.
- **Self-Contained Video I/O:** System `ffmpeg` is not installed on `PATH`. Video decoding, frame seeking, and H.264 `.mp4` export must rely on Python wheels bundled inside `.venv` (`PyAV` / `av` + `opencv-python-headless`).
- **Single-Camera Baseline Geometry:**
  - Lateral net movement (Zone 4 ↔ Zone 3 ↔ Zone 2) and vertical elevation in the net plane are well-conditioned once referenced to the tilted/horizontal net top-tape line between the two antennas (`9.0 m` regulation width).
  - Depth perpendicular to the net is foreshortened along the camera ray; height-above-tape readouts are accurate for front-row sets/hits/blocks near the net plane and should be badged as approximate (`back-row perspective`) if a hitter takes off behind the 3m attack line.

---

## 4. Approaches Considered

### Approach 1: Two-Pass Local GPU Vision Backend (FastAPI) + Web Canvas Studio *(Selected)*
- **Summary:** A Python `FastAPI` server runs a fast coarse GPU scan on long videos to isolate candidate set-and-hit windows near the net (skipped on short clips), then runs high-resolution pose tracking (`YOLO11-pose` + multi-object tracking) and ball detection + outlier-resistant parabolic spline fitting per play. The browser plays the original video with a synchronized `<canvas>` overlay layer and exports burned-in MP4 clips on demand via `PyAV`.
- **Trade-offs:** Slightly more frontend canvas code than a pure-Python dashboard, in exchange for 60fps frame-stepping, zero-latency overlay toggles, easy click-to-calibrate interactions on the video frame, and fast scanning of long practice recordings on the RTX 4090.

### Approach 2: Single-Pass Batch Renderer + Streamlit/Gradio App *(Rejected)*
- **Summary:** Runs full inference across every frame upfront, burns overlays into MP4 clips, and serves them in Streamlit or Gradio.
- **Why rejected:** Toggling individual overlays requires re-rendering video, frame-by-frame scrubbing in Streamlit/Gradio is sluggish, interactive canvas calibration is clunky, and running heavy pose/ball inference on 40 minutes of dead-ball drill downtime wastes time.

### Approach 3: Custom Fine-Tuned Sports Pipeline (TrackNet + Action Classifier) + Web Studio *(Rejected for v1)*
- **Summary:** Pairs the Web Studio with a specialized heatmap ball tracker (TrackNet) and a custom-trained temporal action segmentation model (`pass / set / spike / block`).
- **Why rejected for v1:** Requires sourcing or fine-tuning custom volleyball datasets/weights before anything works. Approach 1 runs immediately with pretrained `YOLO11` weights + physics trajectory fitting, while keeping the ball-detector and play-detector interfaces modular so fine-tuned volleyball weights can be dropped in later.

---

## 5. Architecture & Detailed Design

### 5.1 Decisions Made by User
- **Product focus:** Visual analysis of the set and hit (setting path, hitter posture, hitting point, blockers).
- **Input scope:** Support both short single-play clips and long uncut recordings.
- **UI layout:** **Interactive Review Player** (main annotated court view, overlay toggles, 3-keyframe jump strip `Set Release → Max Load → Contact`, detected plays list on the right, and MP4 export).
- **Technical approach:** **Two-Pass GPU Pipeline (FastAPI) + Web Canvas Studio**.

### 5.2 Proposed Defaults & System Behavior *(Author's Calls)*

#### Video Ingestion & Local Caching *(Author's Call)*
- Support **both** browser file drag-and-drop upload and selecting/pasting a **local filesystem path** (so multi-gigabyte uncut GoPro/phone recordings already on disk in `/mnt/c/...` do not need to be copied over HTTP upload).
- If an input video isn't browser-native H.264 `.mp4` (e.g., HEVC `.mov` from iPhone or `.mkv`), transcode a web-playable H.264 proxy clip (or per-play H.264 segment) via `PyAV` while running analysis on full-quality frames.
- Persist analysis artifacts (`metadata.json`, per-play `tracks.json`, keyframe thumbnails) in `data/sessions/<session_id>/` keyed by file hash + size so reopening a previously analysed video is instant.

#### Net Top-Tape Calibration *(Author's Call)*
- On session creation, auto-estimate the net top-tape segment $(x_L, y_L) \to (x_R, y_R)$ using horizontal line/edge detection constrained by player standing/jumping priors in the middle vertical band of the frame.
- Provide a **2-click "Set Net Endpoints"** tool in the player header: the user clicks the **left antenna top-tape intersection** and **right antenna top-tape intersection** once per tripod setup.
  - Interpolating $y_{\text{tape}}(x)$ along this line accounts for slight tripod tilt.
  - Because the regulation distance between volleyball antennas is **$9.0\text{ m}$**, those two clicks automatically provide a net-plane pixel-to-centimeter scale ($\text{cm/px} = 900 / \|p_R - p_L\|$) without needing full 8-point court homography! Before manual antenna confirmation, display heights in `+px above tape` (with an estimated `~cm` badge once confirmed).

#### Two-Pass Vision Pipeline on RTX 4090 *(Author's Call)*
1. **Routing by duration:**
   - **Short clips (`<= 20s`):** Treat as a single play window (or split only if multiple distinct ball-set arcs separated by `> 4s` appear) and run Pass 2 directly.
   - **Uncut recordings (`> 20s`):** Run **Pass 1 (Coarse Play Scanner)** first, streaming detected plays to the UI as they finish Pass 2 so the user can review Play #1 while the rest of the recording finishes processing in the background.
2. **Pass 1 — Coarse Play Scanner (Uncut Videos):**
   - Sample video at `~10 fps` and run batched GPU detection (`YOLO11` person + sports ball on the upper/net ROI).
   - Flag candidate attack timestamps $t_c$ where a ball rises above $y_{\text{tape}}$ for `>= 0.4s` (set arc) followed by at least one foreground/net-zone player vertical jump apex near the descending ball.
   - Merge nearby triggers within `4.0s` into candidate play windows `[t_c - 3.5s, t_c + 1.5s]`.
3. **Pass 2 — High-Resolution Per-Play Analysis (Full Native FPS):**
   - **Ball Detection & Setting Path Fitting:**
     - Run high-res + tiled upper-court ROI ball detection across every frame of the play window to preserve small/distant volleyball resolution.
     - Associate detections and fit a gravity-constrained 2D trajectory curve (RANSAC / Huber quadratic spline $x(t)$ linear, $y(t)$ parabolic under gravity) between **Set Release** ($t_{\text{set}}$, upward velocity launch out of setter's hands) and **Ball Contact** ($t_{\text{hit}}$, sharp velocity vector deflection / wrist coincidence).
     - RANSAC naturally filters out stray balls rolling or bouncing in the background during practice drills and smoothly bridges 1–6 frame motion-blur gaps.
   - **Multi-Person Pose Tracking & Role Assignment:**
     - Run `YOLO11m-pose` / `YOLO11l-pose` (17 COCO keypoints) with persistent track IDs (`ByteTrack` / `BoT-SORT`) across the play window.
     - **Near-court vs. far-court separation:** Classify tracks as attacking-side vs. blocking-side using ankle ground-contact Y-coordinates prior to jump (`y_ankle > y_net_base` = near/attacking side; `y_ankle <= y_net_base` or occluded behind net mesh with torso emerging behind tape = opposing/blocking side).
     - **Active Hitter:** Attacking-side track with clear vertical jump impulse (hip/shoulder elevation spike) and minimum wrist-to-ball distance at $t_{\text{hit}}$. User can click any other tracked player skeleton in the UI ("Set as Hitter") to swap roles instantly from cached multi-person pose tracks without re-running GPU inference!
     - **Setter Release Anchor:** Attacking-side track whose wrists are nearest the ball at $t_{\text{set}}$ (if visible).
     - **Active Blockers:** Opposing-side tracks within a horizontal margin (`±1.8m` net-plane equivalent) of the hitting point at $t_{\text{hit}}$ that elevate wrists/shoulders near or above $y_{\text{tape}}$.
   - **Keyframes & Visual Callout Metrics:**
     - **`1. Set Release` ($t_{\text{set}}$):** Start of fitted set arc; computes **Set Tempo** $= t_{\text{hit}} - t_{\text{set}}$ (s) and **Set Peak Height** above tape.
     - **`2. Max Load` ($t_{\text{load}}$):** Frame in $[t_{\text{takeoff}}, t_{\text{hit}}]$ where the hitter's hitting wrist/elbow reaches maximum draw-back / minimum elbow flexion prior to forward swing.
     - **`3. Ball Contact` ($t_{\text{hit}}$):**
       - **Contact Height:** Vertical offset from ball center $(x_{\text{hit}}, y_{\text{hit}})$ to $y_{\text{tape}}(x_{\text{hit}})$ (`+px` and `+cm` when antenna endpoints are set).
       - **2D Arm Extension:** Shoulder–elbow–wrist interior angle (`deg`) at $t_{\text{hit}}$ (clearly labeled as 2D camera-plane angle).
       - **2D Torso Tilt / Arch:** Hip-midpoint to shoulder-midpoint angle relative to vertical (`deg`).
       - **Jump Timing:** Compares $t_{\text{hit}}$ against active hitter's head/shoulder apex frame $t_{\text{apex}}$ (`At peak` within `±50ms`, otherwise `Early -Xms` or `Late +Xms` on descent).
       - **Blockers & Seam:** Draws each blocker's upper reach relative to tape; if $\ge 2$ blockers jump and horizontal separation between inner wrist/box edges exceeds `25 cm` (`~1 ball width + margin`), highlights the **Open Block Seam** interval.

#### Interactive Review Studio & Export *(Author's Call)*
- **Layout:** Matches approved Option A mockup:
  - Top bar: Video/session selector, processing progress bar (for background Pass 2 on uncut videos), `Set Net Endpoints (2 clicks)` button, `Reassign Hitter` button, and overlay pill toggles (`Set Arc`, `Pose`, `Contact`, `Block`).
  - Center stage: `<video>` + crisp HiDPI `<canvas>` overlay, frame step controls (`-1f`, `Play/Pause`, `+1f`, playback speed `0.25x / 0.5x / 1x`), and **3-Keyframe Cards** (`1. Set Release`, `2. Max Load`, `3. Ball Contact`) with thumbnail previews and timestamps.
  - Right sidebar: **Detected Plays** list (Play #, timestamp, auto-classified net zone `Zone 4 / Left`, `Zone 3 / Mid`, `Zone 2 / Right` from lateral position along `[x_L, x_R]`, `+ Add Play at Current Time` / delete buttons) and **Export Annotated MP4** button (with checkbox for `1.5s freeze-frame at contact`).
- **MP4 Export:** Renders the exact same overlay primitives onto decoded play frames in Python (OpenCV/Pillow Cairo-style anti-aliased drawing) and encodes H.264 `.mp4` via `PyAV` for download.

### 5.3 Left to the Builder
- Internal Python package structure under `src/vball_analyser/`, exact FastAPI endpoint URLs, and background worker implementation (`asyncio` task queue / thread pool around GPU inference).
- Frontend bundling details (lightweight React/Vite + TypeScript or clean vanilla TypeScript + Canvas served statically by FastAPI—no external cloud dependencies).
- Specific `YOLO11` checkpoint variant (`yolo11m` vs `yolo11l`), tile crop overlap parameters for small volleyball detection, and exact smoothing window sizes (e.g., Savitzky–Golay / One-Euro filter on pose keypoints so skeletons don't jitter frame-to-frame).
- Visual polish, color palette tokens (defaulting to high-contrast dark slate studio theme so bright court overlays pop), and keyboard shortcuts (`Space` play/pause, `,`/`.` or `←`/`→` frame step, `1`/`2`/`3` jump to keyframes).
