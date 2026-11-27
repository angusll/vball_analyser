"""Frontend structure, single-video invariant, and JS unit tests for the unified MediaPlayer."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from vball_analyser.app import create_app
from vball_analyser.config import AppConfig, STATIC_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_client(tmp_path: Path) -> TestClient:
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        allowed_local_dirs=(tmp_path,),
    )
    return TestClient(create_app(cfg))


def test_single_video_element_and_media_player_slots(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    with client:
        html = client.get("/").text
        assert html.count("<video") == 1, "App must contain exactly one <video> element"
        assert 'id="studioPlayerSlot"' in html
        assert 'id="servePlayerSlot"' in html
        assert 'id="mediaPlayerRoot"' in html
        assert "/static/media_player.js" in html

        # media_player.js must load before app.js and serve_tab.js
        mp_idx = html.index("/static/media_player.js")
        app_idx = html.index("/static/app.js")
        serve_idx = html.index("/static/serve_tab.js")
        assert mp_idx < app_idx < serve_idx

        # Legacy Serve Distribution duplicate video elements must be removed
        for legacy_id in (
            'id="serveVideo"',
            'id="serveVideoCanvas"',
            'id="servePlayPauseBtn"',
            'id="serveSeekSlider"',
            'id="serveVideoTime"',
        ):
            assert legacy_id not in html, f"Legacy Serve video element still present: {legacy_id}"

        res = client.get("/static/media_player.js")
        assert res.status_code == 200
        assert "VBallMediaPlayer" in res.text


def test_js_syntax_and_media_player_node_unit_tests() -> None:
    for js_file in sorted(STATIC_DIR.glob("*.js")):
        check = subprocess.run(
            ["node", "--check", str(js_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert check.returncode == 0, f"node --check failed for {js_file.name}: {check.stderr}"

    unit = subprocess.run(
        ["node", "--test", str(PROJECT_ROOT / "tests" / "test_media_player.mjs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert unit.returncode == 0, f"node --test failed:\n{unit.stdout}\n{unit.stderr}"


def test_serve_tab_uses_shared_media_player_only() -> None:
    serve_js = (STATIC_DIR / "serve_tab.js").read_text(encoding="utf-8")
    for legacy_token in (
        '"serveVideo"',
        '"serveVideoCanvas"',
        '"servePlayPauseBtn"',
        '"serveSeekSlider"',
        '"serveVideoTime"',
        "getServeVideoRect",
    ):
        assert legacy_token not in serve_js, f"Legacy Serve player reference still present: {legacy_token}"
    assert "mediaPlayer" in serve_js
    assert "servePlayerSlot" in serve_js
