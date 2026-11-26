# VBall Analyser — Baseline Set & Hit Studio

A local, GPU-accelerated volleyball training analyser that breaks down set-and-hit mechanics from single-camera **attacking-side baseline** video—supporting both short single-play clips and long uncut practice or scrimmage recordings.

## Features

- **Two-Pass GPU Vision Pipeline (`YOLO11` + `ByteTrack`):**
  - **Short clips (`<= 20s`):** Runs full-FPS pose tracking and tiled upper-court volleyball detection directly.
  - **Uncut recordings (`> 20s`):** Runs a fast coarse GPU scan to locate set-and-hit windows, then streams each analysed play to the UI as it finishes.
  - **Multi-Arc Set Isolation:** Uses temporally local RANSAC + Huber gravity-parabola fitting guided by airborne hitting poses to lock onto the actual set arc (ignoring serves, serve-receive passes, stray drill balls, and brief ceiling-light dropouts).
- **Interactive Canvas Review Studio:**
  - Instant layer toggles for **Set Arc**, **Hitter Pose**, **Contact Point**, **Blockers**, **Net Tape**, **Mechanics HUD**, and **Inline Stats** (no video re-encoding required).
  - **Always-On Top-Left Mechanics HUD** + sidebar **Play Mechanics** breakdown with plain-English metric descriptions.
  - **3-Keyframe Jump Strip:** Jump directly to `1. Set Release`, `2. Max Load`, and `3. Ball Contact`, plus frame-by-frame stepping (`-5f`, `-1f`, `+1f`, `+5f`) and slow-motion playback (`0.25x`, `0.5x`, `1x`).
  - **2-Click Net Tape Calibration:** Auto-detects tilted net top tapes, with a 2-click override on the left/right antenna–tape corners to convert heights above the tape into estimated centimeters using the regulation `9.0 m` net width.
  - **1-Click Hitter Reassignment & Manual Play Windows:** Swap the active hitter from cached pose tracks in milliseconds (no GPU re-run), or add/remove plays at any timestamp.
- **Self-Contained Video I/O & MP4 Export (`PyAV`):**
  - Supports `.mp4`, `.mov`, `.mkv`, and `.avi` via browser upload or direct local disk path—no system `ffmpeg` installation required.
  - Exports annotated H.264 `.mp4` play clips with optional `1.5s` contact freeze-frame and matching **Inline Stats** visibility.

## Play Mechanics Reference

| Metric | Description |
| --- | --- |
| **Attack Zone** | Net section where the hit occurs (`Zone 4 Left`, `Zone 3 Middle`, `Zone 2 Right`). |
| **Set Tempo** | Flight time (seconds) from setter release to hitter ball contact. |
| **Set Peak** | Highest point of the fitted set arc measured vertically above the net top tape. |
| **Contact Height** | Ball height above the net top tape at contact (`+px`, plus `+cm` after 2-click antenna calibration). |
| **Arm Extension (2D)** | Shoulder–elbow–wrist interior angle at contact in the camera plane (closer to `180°` = fuller reach). |
| **Torso Tilt (2D)** | Upper-body lean/arch from vertical at contact in the camera plane. |
| **Jump Timing** | Whether ball contact occurs at peak jump height (`±50 ms` of apex) or early/late relative to apex. |
| **Block Seam** | Horizontal spacing between opposing blockers' hands above the net at contact. |

> **Camera Geometry Note:** Designed for a tripod behind the **attacking team's baseline** looking toward the net. Vertical height above the tape and lateral movement across the net are well-conditioned near the net plane; depth along the camera's line of sight (e.g., distance off the net) is not reconstructed from a single view, and `Arm Extension` / `Torso Tilt` are 2D camera-plane angles.

## Requirements

- **Python 3.12** managed with [`uv`](https://docs.astral.sh/uv/)
- **NVIDIA GPU with CUDA** recommended (runs on CPU as fallback, but significantly slower)
- No system `ffmpeg` required (`PyAV` bundles FFmpeg libraries inside `.venv`)

## Quick Start

```bash
# 1. Create Python 3.12 virtual environment and install dependencies
uv venv --python 3.12
uv sync

# 2. Launch the local Review Studio
uv run vball-analyser
```

Open `http://127.0.0.1:8081` in your browser.

*(On WSL2 `/mnt/c/...` filesystems, pass `--link-mode=copy` to `uv venv` / `uv sync` to avoid NTFS hardlink warnings. On first real-video run, Ultralytics downloads `yolo11m.pt` and `yolo11m-pose.pt` automatically.)*

## Studio Controls & Shortcuts

- **Load Video:** Upload a clip (`.mp4`, `.mov`, `.mkv`, `.avi`) or enter a local filesystem path for large uncut recordings.
- **Calibrate Net (2 Clicks):** Click the **left** antenna top-tape corner, then the **right** antenna top-tape corner to lock tilted tape geometry and `cm` height scaling.
- **Reassign Hitter:** Click **Reassign Hitter**, then click any attacking player skeleton on the canvas to recompute mechanics for that player instantly.
- **Keyboard Shortcuts:**
  - `Space` — Play / Pause
  - `←` / `→` — Step back / forward 1 frame
  - `1` / `2` / `3` — Jump to `Set Release` / `Max Load` / `Ball Contact`

## Configuration (Environment Variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `VBALL_DATA_DIR` | `./data` | Directory for cached sessions, thumbnails, proxies, and MP4 exports |
| `VBALL_ALLOWED_LOCAL_DIRS` | Project root & `$HOME` | Path-separator-delimited directories permitted for local disk video loading |
| `VBALL_MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | Maximum allowed HTTP upload size |
| `VBALL_SECRET_KEY` | Auto-generated in `.vball_secret` | Secret key for signing local session cookies and CSRF tokens |

## Project Structure

```text
src/vball_analyser/
├── app.py           # FastAPI server (127.0.0.1), background worker, REST & Range streaming endpoints
├── biomechanics.py  # Pose smoothing, court-side classification, hitter/setter/blocker analysis
├── calibration.py   # Tilted net top-tape auto-detection & 9.0m antenna height/zone geometry
├── config.py        # Application defaults and environment configuration
├── detector.py      # Two-pass YOLO11 GPU pipeline (Pass 1 coarse scanner + Pass 2 play analyser)
├── models.py        # Pydantic data models for sessions, plays, tracks, trajectories, and metrics
├── renderer.py      # OpenCV overlay renderer & PyAV H.264 annotated MP4 clip exporter
├── security.py      # Session cookies, CSRF verification, CSP headers, rate limiting, path confinement
├── storage.py       # Local JSON session cache, instant net recalibration & hitter reassignment
├── trajectory.py    # Local-window RANSAC + Huber gravity-parabola set-arc fitter
├── video_io.py      # PyAV probing, H.264 web-proxy transcoding, thumbnails, HTTP byte-range parser
└── static/          # Interactive HTML5 <video> + HiDPI <canvas> Review Studio (index.html, styles.css, app.js)
tests/               # Unit, pipeline, renderer, GPU smoke, and API security test suites
```

## Running Tests

```bash
uv run pytest -v
```

## License

See [LICENSE](LICENSE).
