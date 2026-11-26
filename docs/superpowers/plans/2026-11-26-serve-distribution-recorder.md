# Serve Distribution Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **Serve Distribution** tab to VBall Analyser that auto-logs serves and passer positions from baseline practice video (`data/test_video.mp4` and longer recordings) onto an interactive 2D top-down 9m × 9m receiving court diagram, with full manual logging, movable passers, and automatic zone + between-players (seam) analytics.

**Architecture:** The backend adds pure court-homography/zone/seam geometry (`serve_geometry.py`), atomic JSON persistence under `data/serve_sessions/` (`serve_storage.py`), and a YOLO11 serve/passer auto-detection pipeline (`serve_detector.py`) reusing the existing `FramePredictor` protocol and serialized behind a shared `asyncio.Lock`. The frontend adds a two-tab header switcher (`Set & Hit Studio` | `Serve Distribution`) and a dedicated vanilla-JS module (`serve_tab.js`) powering 4-corner front-3m video calibration, a HiDPI top-down court canvas with draggable passers and seam shading, video timestamp jump, and live distribution summaries.

**Tech Stack:** Python 3.12 (`uv`), FastAPI, Pydantic v2, OpenCV (`opencv-python-headless`), NumPy, PyAV, Ultralytics `YOLO11` + `ByteTrack`, vanilla HTML5/CSS/JS (no build step), `pytest` + `httpx`/`TestClient`.

**Spec:** `docs/superpowers/specs/2026-11-26-serve-distribution-recorder-design.md`

---

## Global Constraints

- Python 3.12 managed exclusively via `uv` (`/home/angus/.local/bin/uv run pytest -v`).
- Bind local-only (`127.0.0.1`), enforce existing `SECURITY_HEADERS` CSP (`script-src 'self'`, `style-src 'self'` — zero inline scripts or inline `<style>` blocks).
- Keep `ALLOWED_HTTP_METHODS = frozenset({"GET", "HEAD", "POST"})` in `security.py` intact: all mutating serve-session API routes use `POST` with `_require_csrf(request)` and `validate_safe_id`.
- 4-corner court calibration MUST use the visible **front 3m rectangle** (`(0,0), (9,0), (9,3), (0,3)` in meters) so footage where the near baseline is at/below the frame bottom (like `data/test_video.mp4`) calibrates cleanly.
- Reception coordinates MUST anchor to the receiving passer's floor position at contact (not mid-air ball pixels) to avoid perspective depth distortion, falling back to ball floor contact only for unplayed bounces.
- Switching tabs MUST preserve Set & Hit Studio playback state, overlays, and loaded session.
- Loading a video from the Serve Distribution tab MUST NOT trigger Set & Hit Studio GPU analysis (`skip_analysis=True`).

## Review Focus

1. **Degenerate or twisted 4-corner calibration clicks** (collinear/duplicate/self-intersecting points): `compute_court_homography` must reject invalid quads with `ValueError` (HTTP 400) rather than returning `NaN`/`Inf` projections or crashing OpenCV.
2. **Fewer than 2 passers or very close passers on the court:** `classify_serve` must handle `0` or `1` passer gracefully (`is_seam=False`, `seam_pair=None`) and adapt `R_body = min(0.75, 0.30 * gap)` when passers stand `< 2.5m` apart so seam detection never produces negative intervals.
3. **Editing an auto-detected serve then re-running Auto-Detect:** editing an auto-detected serve's position, result, or passer snapshot must flip `source` to `"manual"` so a subsequent `POST /api/serve-sessions/{id}/auto-detect` preserves the user's fix while replacing unedited `"auto"` serves.
4. **Dragging default passers with no serve selected vs. with a serve selected:** updating `default_passers` via `/update` must NEVER overwrite historical serves' `passers_snapshot` or `is_seam`; only updating a specific serve via `/serves/{serve_id}/update` recomputes that serve's seam.
5. **Videos with zero serves or post-reception set/attack arcs:** `detect_serve_events` must return `[]` cleanly on empty/no-serve clips (never crash or fabricate a serve) and enforce a `>= 2.5s` refractory gap so the subsequent set/hit in a rally (like `t=5.4s–7.3s` in `data/test_video.mp4`) is not logged as a second serve.

---

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `src/vball_analyser/models.py` | Modify | Add `CourtCalibration`, `PasserPosition`, `ServeEvent`, `ServeDistributionSummary`, `ServeSession` |
| `src/vball_analyser/serve_geometry.py` | Create | Front-3m homography, court/image projection, zone & seam classification, coaching summary stats |
| `src/vball_analyser/config.py` | Modify | Add `AppConfig.serve_sessions_dir` property |
| `src/vball_analyser/serve_storage.py` | Create | Atomic JSON `ServeSessionStore` under `data/serve_sessions/<id>/serve_session.json` |
| `src/vball_analyser/serve_detector.py` | Create | YOLO11 + ByteTrack serve & pre-serve passer auto-detection pipeline |
| `src/vball_analyser/app.py` | Modify | Shared GPU `asyncio.Lock`, `skip_analysis` video-only load, `/api/serve-sessions/*` endpoints |
| `src/vball_analyser/static/index.html` | Modify | Topbar tab switcher + `#serveTab` markup |
| `src/vball_analyser/static/styles.css` | Modify | Tab switcher + Serve Distribution dark-studio styles |
| `src/vball_analyser/static/app.js` | Modify | Tab switching + `window.VBallShared` bridge for CSRF & video loader |
| `src/vball_analyser/static/serve_tab.js` | Create | Interactive top-down court canvas, 4-corner calibration, video sync, auto-detect polling, stats |
| `tests/test_serve_geometry.py` | Create | Unit tests for homography, zone/seam math, result invariants, summary formulas |
| `tests/test_serve_storage.py` | Create | Unit tests for `ServeSessionStore` CRUD, path safety, merge rules, snapshot isolation |
| `tests/test_serve_detector.py` | Create | Synthetic `FramePredictor` pipeline tests + optional GPU smoke check on `data/test_video.mp4` |
| `tests/test_serve_api.py` | Create | FastAPI CSRF, `skip_analysis`, calibration, auto-detect, and serve CRUD integration tests |

---

### Task 1: Serve Models & Court Geometry / Seam Classification

**Files:**
- Modify: `src/vball_analyser/models.py`
- Create: `src/vball_analyser/serve_geometry.py`
- Test: `tests/test_serve_geometry.py`

**Interfaces:**
- Consumes: `Point2D` from `vball_analyser.models`.
- Produces:
  - In `vball_analyser.models`:
    - `CourtCalibration(front_left: Point2D, front_right: Point2D, attack_right: Point2D, attack_left: Point2D, is_calibrated: bool = True)`
    - `PasserPosition(label: str, x_m: float, y_m: float)`
    - `ServeEvent(serve_id: str, index: int, source: Literal["auto", "manual"], timestamp_s: float | None = None, frame_idx: int | None = None, server_name: str = "Unassigned", x_m: float, y_m: float, result: Literal["in", "ace", "out", "net"] = "in", zone: str = "Zone 6", depth: Literal["short", "deep", "out"] = "deep", is_seam: bool = False, seam_pair: tuple[str, str] | None = None, passers_snapshot: list[PasserPosition] = [])`
    - `ServeDistributionSummary(total: int, in_count: int, ace_count: int, out_count: int, net_count: int, in_pct: float, ace_pct: float, error_pct: float, seam_count: int, seam_pct: float, seam_by_pair: dict[str, int], zone_counts: dict[str, int], zone_pcts: dict[str, float], short_count: int, deep_count: int)`
    - `ServeSession(serve_session_id: str, name: str, created_at: str, video_session_id: str | None = None, status: Literal["idle", "scanning", "ready", "error"] = "idle", progress_pct: float = 0.0, error_message: str | None = None, court_calibration: CourtCalibration | None = None, default_passers: list[PasserPosition] = ..., servers: list[str] = [], serves: list[ServeEvent] = [])`
  - In `vball_analyser.serve_geometry`:
    - `default_passer_formation(count: int = 3) -> list[PasserPosition]`
    - `compute_court_homography(cal: CourtCalibration) -> tuple[np.ndarray, np.ndarray]`
    - `project_image_to_court(H: np.ndarray, x_px: float, y_px: float) -> tuple[float, float]`
    - `project_court_to_image(H_inv: np.ndarray, x_m: float, y_m: float) -> tuple[float, float]`
    - `classify_serve(x_m: float, y_m: float, result: Literal["in", "ace", "out", "net"], passers: list[PasserPosition]) -> tuple[float, float, Literal["in", "ace", "out", "net"], str, Literal["short", "deep", "out"], bool, tuple[str, str] | None]`
    - `compute_distribution_summary(serves: list[ServeEvent], server_filter: str | None = None) -> ServeDistributionSummary`

- [ ] **Step 1: Write failing unit tests in `tests/test_serve_geometry.py`**

```python
import pytest
from vball_analyser.models import CourtCalibration, PasserPosition, Point2D, ServeEvent
from vball_analyser.serve_geometry import (
    classify_serve,
    compute_court_homography,
    compute_distribution_summary,
    default_passer_formation,
    project_court_to_image,
    project_image_to_court,
)


def test_homography_roundtrip_and_degenerate_quad() -> None:
    cal = CourtCalibration(
        front_left=Point2D(x=200.0, y=320.0),
        front_right=Point2D(x=850.0, y=320.0),
        attack_right=Point2D(x=940.0, y=420.0),
        attack_left=Point2D(x=110.0, y=420.0),
    )
    H, H_inv = compute_court_homography(cal)
    for cx, cy in [(0.0, 0.0), (9.0, 0.0), (9.0, 3.0), (0.0, 3.0), (4.5, 6.5)]:
        px, py = project_court_to_image(H_inv, cx, cy)
        rx, ry = project_image_to_court(H, px, py)
        assert rx == pytest.approx(cx, abs=1e-2)
        assert ry == pytest.approx(cy, abs=1e-2)

    # Collinear points must raise ValueError (Review Focus #1)
    bad = CourtCalibration(
        front_left=Point2D(x=100.0, y=300.0),
        front_right=Point2D(x=500.0, y=300.0),
        attack_right=Point2D(x=700.0, y=300.0),
        attack_left=Point2D(x=200.0, y=300.0),
    )
    with pytest.raises(ValueError):
        compute_court_homography(bad)


def test_classify_serve_zones_seams_and_result_invariants() -> None:
    passers = default_passer_formation(3)  # P1(1.8,6.2), P2(4.5,6.5), P3(7.2,6.2)

    # Seam hit between P1 and P2 in Zone 6 deep
    x, y, res, zone, depth, is_seam, pair = classify_serve(3.15, 6.3, "in", passers)
    assert (res, zone, depth, is_seam, pair) == ("in", "Zone 6", "deep", True, ("P1", "P2"))

    # Directly on P2 body -> NOT a seam
    _, _, _, zone2, _, is_seam2, pair2 = classify_serve(4.55, 6.5, "in", passers)
    assert zone2 == "Zone 6" and is_seam2 is False and pair2 is None

    # Front-row short Zone 4
    _, _, _, zone3, depth3, is_seam3, _ = classify_serve(1.5, 2.0, "ace", passers)
    assert (zone3, depth3, is_seam3) == ("Zone 4", "short", False)

    # Outside court auto-marks out and clears seam
    _, _, res4, zone4, depth4, is_seam4, _ = classify_serve(4.5, 9.6, "in", passers)
    assert (res4, zone4, depth4, is_seam4) == ("out", "Out Long", "out", False)

    # Net result snaps y_m=0.0, forces Net / Short and no seam
    nx, ny, res5, zone5, depth5, is_seam5, _ = classify_serve(3.2, 5.0, "net", passers)
    assert (nx, ny, res5, zone5, depth5, is_seam5) == (3.2, 0.0, "net", "Net / Short", "out", False)

    # Fewer than 2 passers & close passers (Review Focus #2)
    _, _, _, _, _, seam_one, _ = classify_serve(3.0, 6.2, "in", [PasserPosition(label="P1", x_m=1.8, y_m=6.2)])
    assert seam_one is False
    close_passers = [PasserPosition(label="P1", x_m=3.0, y_m=6.0), PasserPosition(label="P2", x_m=4.0, y_m=6.0)]
    _, _, _, _, _, seam_close, pair_close = classify_serve(3.5, 6.0, "in", close_passers)
    assert seam_close is True and pair_close == ("P1", "P2")


def test_distribution_summary_formulas_and_filter() -> None:
    empty = compute_distribution_summary([])
    assert empty.total == 0 and empty.in_pct == 0.0 and empty.seam_pct == 0.0

    serves = [
        ServeEvent(serve_id="s1", index=1, source="manual", server_name="#7", x_m=3.1, y_m=6.3, result="in", zone="Zone 6", depth="deep", is_seam=True, seam_pair=("P1", "P2")),
        ServeEvent(serve_id="s2", index=2, source="manual", server_name="#7", x_m=7.5, y_m=2.0, result="ace", zone="Zone 2", depth="short", is_seam=False),
        ServeEvent(serve_id="s3", index=3, source="manual", server_name="#7", x_m=4.0, y_m=9.5, result="out", zone="Out Long", depth="out", is_seam=False),
        ServeEvent(serve_id="s4", index=4, source="manual", server_name="#9", x_m=3.0, y_m=0.0, result="net", zone="Net / Short", depth="out", is_seam=False),
    ]
    all_sum = compute_distribution_summary(serves)
    assert all_sum.total == 4
    assert all_sum.in_pct == 50.0  # (in + ace) / total
    assert all_sum.ace_pct == 25.0
    assert all_sum.error_pct == 50.0
    assert all_sum.in_pct + all_sum.error_pct == pytest.approx(100.0)
    assert all_sum.seam_count == 1 and all_sum.seam_pct == 50.0  # 1 of 2 in-bounds serves
    assert all_sum.seam_by_pair == {"P1-P2": 1}

    s7 = compute_distribution_summary(serves, server_filter="#7")
    assert s7.total == 3 and s7.net_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_geometry.py -v`  
Expected: FAIL (`ImportError` for `CourtCalibration` / `vball_analyser.serve_geometry`).

- [ ] **Step 3: Add serve models to `src/vball_analyser/models.py` and implement `src/vball_analyser/serve_geometry.py`**

In `models.py`, append `CourtCalibration`, `PasserPosition`, `ServeEvent`, `ServeDistributionSummary`, and `ServeSession` matching the Interfaces block above.

In `serve_geometry.py`:
- `default_passer_formation(count: int = 3)`:
  - `2`: `P1(2.5, 6.3)`, `P2(6.5, 6.3)`
  - `3`: `P1(1.8, 6.2)`, `P2(4.5, 6.5)`, `P3(7.2, 6.2)`
  - `4`: `P1(1.4, 6.0)`, `P2(3.5, 6.5)`, `P3(5.5, 6.5)`, `P4(7.6, 6.0)`
- `compute_court_homography`: verify the 4 image corners form a non-degenerate convex quad (area `>= 400 px^2` and consistent cross-product signs), compute `H = cv2.getPerspectiveTransform(src, dst)` where `dst = [(0,0), (9,0), (9,3), (0,3)]`, check `abs(np.linalg.det(H)) > 1e-12`, and return `(H, np.linalg.inv(H))`.
- `classify_serve`: implement the exact zone, depth, net-snap, out-of-bounds, and seam rules from Spec Section 5.4 (`R_body = min(0.75, 0.30 * gap)`, longitudinal band `<= 2.2m`, sorted left-to-right passers).
- `compute_distribution_summary`: compute rounded 1-decimal percentages, `zone_counts`/`zone_pcts` over `Zone 1`–`Zone 6`, `seam_by_pair` keyed as `f"{a}-{b}"`, and `short_count`/`deep_count` over in-bounds serves.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_geometry.py -v`  
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
/home/angus/.local/bin/uv run bash -c "git add src/vball_analyser/models.py src/vball_analyser/serve_geometry.py tests/test_serve_geometry.py && git commit -m 'feat: add serve distribution models and court geometry classifier'"
```

---

### Task 2: Serve Session Store (`config.py`, `serve_storage.py`)

**Files:**
- Modify: `src/vball_analyser/config.py:38-42`
- Create: `src/vball_analyser/serve_storage.py`
- Test: `tests/test_serve_storage.py`

**Interfaces:**
- Consumes: `ServeSession`, `ServeEvent`, `CourtCalibration`, `PasserPosition` (`models.py`), `classify_serve`, `default_passer_formation` (`serve_geometry.py`), `validate_safe_id` (`storage.py`).
- Produces:
  - `AppConfig.serve_sessions_dir -> Path`
  - `ServeSessionStore(serve_sessions_dir: Path)`:
    - `create_session(name: str, video_session_id: str | None = None) -> ServeSession`
    - `load_session(serve_session_id: str) -> ServeSession`
    - `save_session(session: ServeSession) -> None`
    - `list_sessions() -> list[ServeSession]`
    - `delete_session(serve_session_id: str) -> None`
    - `update_session(serve_session_id: str, *, name: str | None = None, video_session_id: str | None = ..., servers: list[str] | None = None, default_passers: list[PasserPosition] | None = None) -> ServeSession`
    - `save_calibration(serve_session_id: str, cal: CourtCalibration) -> ServeSession`
    - `add_serve(serve_session_id: str, *, x_m: float, y_m: float, result: Literal["in", "ace", "out", "net"] = "in", server_name: str = "Unassigned", source: Literal["auto", "manual"] = "manual", timestamp_s: float | None = None, frame_idx: int | None = None, passers_snapshot: list[PasserPosition] | None = None) -> ServeSession`
    - `update_serve(serve_session_id: str, serve_id: str, *, x_m: float | None = None, y_m: float | None = None, result: Literal["in", "ace", "out", "net"] | None = None, server_name: str | None = None, passers_snapshot: list[PasserPosition] | None = None) -> ServeSession`
    - `delete_serve(serve_session_id: str, serve_id: str) -> ServeSession`
    - `replace_auto_serves(serve_session_id: str, auto_serves: list[ServeEvent]) -> ServeSession`

- [ ] **Step 1: Write failing tests in `tests/test_serve_storage.py`**

```python
from pathlib import Path
import pytest
from vball_analyser.models import CourtCalibration, PasserPosition, Point2D, ServeEvent
from vball_analyser.serve_storage import ServeSessionStore


def test_serve_store_crud_and_snapshot_isolation(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    sess = store.create_session("Tuesday Practice")
    assert len(sess.default_passers) == 3

    # Add manual serve in seam between P1 and P2
    sess = store.add_serve(sess.serve_session_id, x_m=3.15, y_m=6.3, result="in", server_name="#7")
    assert len(sess.serves) == 1
    assert sess.serves[0].is_seam is True
    orig_snapshot = list(sess.serves[0].passers_snapshot)

    # Updating default_passers must NOT overwrite past serve snapshots (Review Focus #4)
    new_formation = [PasserPosition(label="P1", x_m=2.5, y_m=6.3), PasserPosition(label="P2", x_m=6.5, y_m=6.3)]
    sess = store.update_session(sess.serve_session_id, default_passers=new_formation)
    assert len(sess.default_passers) == 2
    assert sess.serves[0].passers_snapshot == orig_snapshot
    assert sess.serves[0].is_seam is True

    # Editing a specific serve's passers_snapshot recomputes its seam
    shifted = [PasserPosition(label="P1", x_m=3.15, y_m=6.3), PasserPosition(label="P2", x_m=7.0, y_m=6.3)]
    sess = store.update_serve(sess.serve_session_id, sess.serves[0].serve_id, passers_snapshot=shifted)
    assert sess.serves[0].is_seam is False

    # Invalid IDs rejected
    with pytest.raises(ValueError):
        store.load_session("../escape")


def test_replace_auto_serves_preserves_manual_and_edited_auto(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    sess = store.create_session("Auto Merge Test")
    sess = store.add_serve(sess.serve_session_id, x_m=2.0, y_m=6.0, source="manual", server_name="#1")
    sess = store.add_serve(sess.serve_session_id, x_m=3.15, y_m=6.3, source="auto", timestamp_s=4.5)
    sess = store.add_serve(sess.serve_session_id, x_m=6.0, y_m=6.3, source="auto", timestamp_s=8.5)

    # User manually adjusts the first auto serve -> flips to manual (Review Focus #3)
    edited_id = sess.serves[1].serve_id
    sess = store.update_serve(sess.serve_session_id, edited_id, x_m=3.25)
    assert sess.serves[1].source == "manual"

    # Re-running auto-detect replaces only unedited auto serves
    fresh_auto = [
        ServeEvent(serve_id="auto_new_1", index=1, source="auto", timestamp_s=12.0, x_m=7.5, y_m=7.0, result="in", zone="Zone 1", depth="deep", is_seam=False),
    ]
    sess = store.replace_auto_serves(sess.serve_session_id, fresh_auto)
    ids = [s.serve_id for s in sess.serves]
    assert edited_id in ids
    assert "auto_new_1" in ids
    assert len(sess.serves) == 3  # original manual + edited auto + 1 fresh auto
    assert [s.index for s in sess.serves] == [1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_storage.py -v`  
Expected: FAIL (`ModuleNotFoundError: No module named 'vball_analyser.serve_storage'`).

- [ ] **Step 3: Add `serve_sessions_dir` to `config.py` and implement `ServeSessionStore` in `src/vball_analyser/serve_storage.py`**

- In `config.py`, add `@property def serve_sessions_dir(self) -> Path: return self.data_dir / "serve_sessions"`.
- In `serve_storage.py`, follow `SessionStore`'s atomic write pattern (`serve_session.json.tmp` -> `chmod 0o600` -> `os.replace`) and `validate_safe_id` confinement.
- In `save_calibration`, call `compute_court_homography(cal)` first so degenerate calibrations fail fast with `ValueError`.
- In `update_serve`, flip `serve.source = "manual"` whenever `x_m`, `y_m`, `result`, or `passers_snapshot` is modified, then recompute via `classify_serve`.
- In `replace_auto_serves`, keep `[s for s in session.serves if s.source == "manual"]`, append `auto_serves`, sort by `(timestamp_s if timestamp_s is not None else float("inf"), index)`, and renumber `index = 1..N`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_storage.py -v`  
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
/home/angus/.local/bin/uv run bash -c "git add src/vball_analyser/config.py src/vball_analyser/serve_storage.py tests/test_serve_storage.py && git commit -m 'feat: add disk-backed ServeSessionStore with snapshot isolation'"
```

---

### Task 3: Automated Serve Detection & Passer Projection Pipeline (`serve_detector.py`)

**Files:**
- Create: `src/vball_analyser/serve_detector.py`
- Test: `tests/test_serve_detector.py`

**Interfaces:**
- Consumes: `FramePredictor`, `YoloGpuPredictor` (`detector.py`), `iter_video_frames` (`video_io.py`), `compute_court_homography`, `project_image_to_court`, `classify_serve` (`serve_geometry.py`), `CourtCalibration`, `NetCalibration`, `PasserPosition`, `PoseFrame`, `ServeEvent`, `VideoMetadata` (`models.py`).
- Produces:
  - `extract_foot_point(frame: PoseFrame) -> tuple[float, float]`
  - `detect_serve_events(video_path: Path, calibration: CourtCalibration, predictor: FramePredictor, metadata: VideoMetadata, server_name: str = "Unassigned") -> list[ServeEvent]`

- [ ] **Step 1: Write failing tests in `tests/test_serve_detector.py`**

Create a deterministic `FakeServePredictor` implementing `FramePredictor` that returns synthetic ball trajectories and 3 near-side passer tracks (`P1`, `P2`, `P3`) matching a known front-3m `CourtCalibration`, and test:
1. `test_detects_played_serve_in_seam_and_ignores_subsequent_set`: synthetic 8s clip where a serve crosses from the net (`t=1.0s`), descends to a reception inflection at `t=2.2s` where `P2` steps left into the `P1–P2` seam to pass it upward, followed by a set/hit arc at `t=3.5s–4.5s`. Asserts `len(events) == 1`, `events[0].is_seam is True`, `events[0].seam_pair == ("P1", "P2")`, `events[0].result == "in"`, `abs(events[0].timestamp_s - 2.2) < 0.25`, and `len(events[0].passers_snapshot) == 3` (Review Focus #5).
2. `test_detects_unplayed_ace_and_out_long`: synthetic clip with an unplayed bounce inside Zone 1 (`result == "ace"`) and a second serve 4s later bouncing at `y_m = 9.8` (`result == "out"`, `zone == "Out Long"`).
3. `test_empty_video_returns_no_serves`: synthetic clip with no ball detections returns `[]` cleanly (Review Focus #5).

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_detector.py -v`  
Expected: FAIL (`ModuleNotFoundError: No module named 'vball_analyser.serve_detector'`).

- [ ] **Step 3: Implement `src/vball_analyser/serve_detector.py`**

Algorithm outline:
1. Compute `(H, H_inv) = compute_court_homography(calibration)` and construct a synthetic `NetCalibration` from `front_left` / `front_right` (shifted upward by estimated net height `0.27 * net_span_px`) so `predictor.detect_balls` / `predictor.track_poses` / `predictor.scan_coarse` can run directly.
2. Short clips (`metadata.duration_s <= DEFAULT_CONFIG.short_clip_max_sec`): decode full-FPS frames once and run `detect_balls` + `track_poses`. Longer clips (`> 20s`): run `scan_coarse` at `5.0 fps` to locate candidate downward-crossing windows (`4.5s` each, merged when overlapping), then run full-FPS `detect_balls` + `track_poses` per window.
3. Build a smoothed per-frame best ball series `(frame_idx, timestamp_s, x, y)` (filtering low-conf noise `< 0.18` and big jumps `> 160 px/frame`).
4. Locate serve arrival inflections:
   - Scan for segments where the ball starts near/above the net (`y_px <= net_center_y + 20`) and descends steadily (`dy > 0` for at least `0.25s`, total descent `>= 45 px`).
   - Find `t_land`: the local maximum `y_px` where `vy` reverses from downward to upward (`dy_before > 2.0`, `dy_after < -1.5`), OR the last descending ball frame before the track ends/bounces near the floor.
   - Enforce a `>= 2.5s` refractory gap after each accepted serve so post-reception passes, sets, and hits are ignored.
5. For each accepted `t_land`:
   - **Pre-serve passer snapshot (`t_pre = max(window_start, t_land - 0.6s)`):** collect all tracks visible within `±0.25s` of `t_pre`, project `extract_foot_point(frame)` through `H` into `(x_m, y_m)`, filter to near-court receiving players (`-0.5 <= x_m <= 9.5` and `1.5 <= y_m <= 9.8`), exclude net-tucked players (`y_m < 2.2`) when `>= 2` deeper passers exist, sort left-to-right by `x_m`, and label `P1, P2, ...`.
   - **Landing / reception projection:** check near-side tracks at `t_land` (`±0.15s`) for the closest player whose bounding box (expanded by `40 px`) contains or is within `95 px` of the ball at `t_land`.
     - If matched (played serve): project that passer's foot at `t_land` through `H` to `(px_m, py_m)`, add a capped lateral reach offset `clamp((ball_x - bbox_cx) / max(1, bbox_w) * 0.6, -0.6, 0.6)` to `px_m`, and set default `raw_result = "in"`.
     - If unmatched (unplayed): project `(ball_x, ball_y)` at `t_land` through `H` directly, and set `raw_result = "ace"` (which `classify_serve` automatically turns into `"out"` if outside `[0, 9] × [0, 9]`).
   - Run `classify_serve` and build `ServeEvent(source="auto", ...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_detector.py -v`  
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
/home/angus/.local/bin/uv run bash -c "git add src/vball_analyser/serve_detector.py tests/test_serve_detector.py && git commit -m 'feat: add YOLO serve detection and passer floor projection pipeline'"
```

---

### Task 4: FastAPI Serve Endpoints, Video-Only Load & GPU Lock (`app.py`)

**Files:**
- Modify: `src/vball_analyser/app.py`
- Test: `tests/test_serve_api.py`

**Interfaces:**
- Consumes: `ServeSessionStore` (`serve_storage.py`), `detect_serve_events` (`serve_detector.py`), `compute_distribution_summary` (`serve_geometry.py`), `CourtCalibration`, `PasserPosition` (`models.py`).
- Produces (HTTP JSON API):
  - `POST /api/sessions/upload` & `POST /api/sessions/local` accept optional `skip_analysis: bool = False`.
  - `GET /api/serve-sessions` -> `list[ServeSession]`
  - `POST /api/serve-sessions` -> `{"session": ServeSession, "summary": ServeDistributionSummary}`
  - `GET /api/serve-sessions/{serve_session_id}?server=<optional>` -> `{"session": ServeSession, "summary": ServeDistributionSummary}`
  - `POST /api/serve-sessions/{serve_session_id}/update` -> `{"session": ..., "summary": ...}`
  - `POST /api/serve-sessions/{serve_session_id}/delete` -> `{"status": "deleted"}`
  - `POST /api/serve-sessions/{serve_session_id}/calibration` -> `{"session": ..., "summary": ...}`
  - `POST /api/serve-sessions/{serve_session_id}/auto-detect` -> `{"session": ..., "summary": ...}` (launches background task guarded by `gpu_lock`)
  - `POST /api/serve-sessions/{serve_session_id}/serves` -> `{"session": ..., "summary": ...}`
  - `POST /api/serve-sessions/{serve_session_id}/serves/{serve_id}/update` -> `{"session": ..., "summary": ...}`
  - `POST /api/serve-sessions/{serve_session_id}/serves/{serve_id}/delete` -> `{"session": ..., "summary": ...}`

- [ ] **Step 1: Write failing integration tests in `tests/test_serve_api.py`**

Test using `TestClient` with `SyntheticColorPredictor` (following `tests/test_api_security.py`):
1. `test_serve_endpoints_enforce_csrf_and_method_allowlist`: POST without CSRF -> 403; existing `DELETE /api/sessions` still returns 405.
2. `test_skip_analysis_local_video_load`: `POST /api/sessions/local` with `{"path": ..., "skip_analysis": true}` returns `status == "ready"` and `plays == []` without running Set & Hit analysis.
3. `test_serve_session_full_workflow`: create serve session linked to video -> save 4-corner calibration (and verify degenerate calibration returns 400) -> add manual serve in seam -> verify summary (`in_pct == 100.0`, `seam_pct == 100.0`) -> update serve to `net` -> verify summary (`error_pct == 100.0`, `seam_count == 0`) -> delete serve -> trigger `auto-detect` and poll until `status == "ready"` -> delete serve session.

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_api.py -v`  
Expected: FAIL (404 on `/api/serve-sessions`).

- [ ] **Step 3: Implement endpoints, `skip_analysis`, and `gpu_lock` in `src/vball_analyser/app.py`**

- Inside `create_app`: create `serve_store = ServeSessionStore(config.serve_sessions_dir)` and `gpu_lock = asyncio.Lock()`.
- Wrap GPU inference in `_run_session_analysis` and the new `_run_serve_auto_detect` with `async with gpu_lock:`.
- On `upload_video` (`skip_analysis: bool = Form(False)`) and `load_local_video` (`LocalVideoRequest.skip_analysis: bool = False`): when `True`, run `ensure_web_playable_mp4` (and update metadata if transcoded), save `status = "ready"`, `progress_pct = 100.0`, and return without calling `_schedule_analysis`.
- Add the `/api/serve-sessions/*` routes with Pydantic request models, `_require_csrf(request)` on every `POST`, `validate_safe_id`, and clean `400`/`404` mapping. Return `{"session": session.model_dump(), "summary": compute_distribution_summary(session.serves, server_filter=server).model_dump()}` from session-returning routes.

- [ ] **Step 4: Run tests (new + existing security suite)**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_api.py tests/test_api_security.py -v`  
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
/home/angus/.local/bin/uv run bash -c "git add src/vball_analyser/app.py tests/test_serve_api.py && git commit -m 'feat: add Serve Distribution REST API, video-only loading, and GPU lock'"
```

---

### Task 5: Tab Navigation & Serve Distribution Interactive UI (`index.html`, `styles.css`, `app.js`, `serve_tab.js`)

**Files:**
- Modify: `src/vball_analyser/static/index.html`
- Modify: `src/vball_analyser/static/styles.css`
- Modify: `src/vball_analyser/static/app.js`
- Create: `src/vball_analyser/static/serve_tab.js`
- Test: `tests/test_serve_api.py` (add static asset & HTML tab markup smoke assertion)

**Interfaces:**
- Consumes: `/api/serve-sessions/*`, `/api/sessions`, `/api/sessions/{id}/video`, and `window.VBallShared` (`apiPostJson`, `fetchCsrf`, `openLoadModalForServeTab`).
- Produces: Complete browser UI with two top-level tabs (`Set & Hit Studio` and `Serve Distribution`), 4-corner video calibration with live projected 9m × 9m court grid, interactive HiDPI top-down court canvas (draggable passers, formation presets, shaded seam corridors, dots/heatmap toggle, click-to-log, click-to-jump video, drag-to-adjust, undo/clear), server chips, result toggles, and live coaching summary cards.

- [ ] **Step 1: Add UI asset & tab markup smoke test to `tests/test_serve_api.py`**

```python
def test_index_contains_tabs_and_serves_serve_tab_js(tmp_path: Path) -> None:
    client, _ = _make_test_client(tmp_path)
    with client:
        html = client.get("/").text
        assert 'data-tab="studio"' in html
        assert 'data-tab="serve"' in html
        assert 'id="serveTab"' in html
        assert 'id="serveCourtCanvas"' in html
        assert '/static/serve_tab.js' in html
        js = client.get("/static/serve_tab.js")
        assert js.status_code == 200
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run: `/home/angus/.local/bin/uv run pytest tests/test_serve_api.py::test_index_contains_tabs_and_serves_serve_tab_js -v`  
Expected: FAIL (`AssertionError: assert 'data-tab="studio"' in html`).

- [ ] **Step 3: Implement tab bar & Serve Distribution UI**

1. **`index.html`:**
   - Add `<nav class="tab-switcher" role="tablist" aria-label="Workspace tabs">` in `.topbar` with buttons `data-tab="studio"` (`Set & Hit Studio`, active) and `data-tab="serve"` (`Serve Distribution`).
   - Add `data-studio-only` attribute to the studio-specific topbar controls (`.toolbar-actions`, `.overlay-toggles`) so switching to `serve` hides them cleanly without unmounting.
   - Keep `<main id="stageMain" class="studio-grid">` as the Set & Hit Studio view, and add `<main id="serveTab" class="serve-grid hidden" aria-label="Serve distribution recorder">` with:
     - Top bar: serve session `<select>`, `+ New` button, linked video `<select>`, `Load Video` button, `Calibrate Court (4 Clicks)` button, `Auto-Detect Serves` button, status pill + `<progress>`.
     - Collapsible Video Preview card (`#serveVideoCard`, `#serveVideo`, `#serveVideoCanvas`, 4-click hint + Cancel button).
     - Top-Down Court card: formation preset buttons (`2-Passer`, `3-Passer`, `4-Passer`, `Reset Passers`), view toggles (`Dots`, `Heatmap`, `Seam Bands`), `<canvas id="serveCourtCanvas">`, and `Undo` / `Clear` / selected-serve bar.
     - Right sidebar: Server chips (`All Servers`, `+ Add Server` inline input), Result mode pills (`In`, `Ace`, `Net`), Distribution Summary (Total, In %, Ace %, Error %, Seam % + pair breakdown, Zone 1–6 bars + Short/Deep split), and scrollable Serve Log list.
   - Load `<script src="/static/serve_tab.js" defer></script>` after `app.js` (no inline JS).
2. **`styles.css`:**
   - Add clean dark-studio styling for `.tab-switcher`, `.tab-btn`, `.serve-grid`, `.serve-video-stage`, `.serve-court-wrap`, `.serve-court-canvas`, `.zone-bar-row`, `.server-chips`, and `.serve-log-list`.
3. **`app.js`:**
   - Add `switchTab(tabName)` toggling `.hidden` on `#stageMain` / `#serveTab` and `[data-studio-only]` controls; pause studio video if switching away only? No — keep playback state intact (or pause only if playing to avoid background audio) and resize canvases on switch.
   - Support `state.loadModalMode` (`"studio"` vs. `"serve"`): when `"serve"`, `uploadForm` / `localPathForm` send `skip_analysis: true` and invoke `window.VBallServeTab.onVideoLoaded(session)` instead of replacing the studio's active session.
   - Export `window.VBallShared = { apiPostJson, fetchCsrf, openLoadModalForServeTab, refreshSessionSelect }`.
4. **`serve_tab.js`:**
   - Manage serve session state, API sync, and 4-corner calibration on `#serveVideoCanvas` (clicking corners 1–4 in order, drawing numbered crosshairs + live projected 9m × 9m court grid via client-side 3×3 homography solver so the user sees immediate visual alignment feedback before/after saving).
   - Render HiDPI `#serveCourtCanvas`: 9m × 9m court + out-of-bounds apron, net bar, 3m line, Zone 1–6 labels, shaded seam corridors between adjacent passers, optional Gaussian heatmap layer, color-coded serve dots (`In` cyan, `Ace` gold, `Seam` violet ring, `Out`/`Net` rose), and draggable passer circles (`P1, P2, ...`).
   - Pointer interaction: drag passer (updates `default_passers` if no serve selected, or updates selected serve's `passers_snapshot` if a serve is selected), click/drag serve dot (selects serve + jumps `#serveVideo` to `timestamp_s`, or repositions it), click empty court (adds manual serve via `POST /api/serve-sessions/{id}/serves`).
   - Poll `GET /api/serve-sessions/{id}` while `status === "scanning"` until `"ready"` or `"error"`, then render updated serves and summary.

- [ ] **Step 4: Run full test suite & real-video smoke verification**

Run: `/home/angus/.local/bin/uv run pytest -v`  
Expected: PASS (all existing + new tests pass).

Also run a quick verification on `data/test_video.mp4` via `/home/angus/.local/bin/uv run python -c ...` (invoking `detect_serve_events` on `data/test_video.mp4` with its front-3m corners) to verify real YOLO detection logs the serve at ~`4.5s` in the seam!

- [ ] **Step 5: Commit**

```bash
/home/angus/.local/bin/uv run bash -c "git add src/vball_analyser/static/index.html src/vball_analyser/static/styles.css src/vball_analyser/static/app.js src/vball_analyser/static/serve_tab.js tests/test_serve_api.py && git commit -m 'feat: add Serve Distribution tab UI with interactive court and 4-corner calibration'"
```

---

## Self-Review Checklist (Completed)

- **Spec coverage:** Every section of `docs/superpowers/specs/2026-11-26-serve-distribution-recorder-design.md` (5.1 Tabs, 5.2 4-corner front-3m homography, 5.3 YOLO serve/passer auto-detection, 5.4 Zone & seam math + result/summary invariants, 5.5 JSON persistence, 5.6 REST API + video-only load + GPU lock, 5.7 Interactive UI) maps directly to Tasks 1–5.
- **Type & route consistency:** All mutating endpoints consistently use `POST` (`ALLOWED_HTTP_METHODS` untouched); model names (`CourtCalibration`, `PasserPosition`, `ServeEvent`, `ServeDistributionSummary`, `ServeSession`) and function signatures match across Tasks 1–5.
- **Review Focus:** All 5 failure modes are explicitly pinned by tests in Tasks 1–3.
