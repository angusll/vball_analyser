"""API security, HTTP range streaming, and end-to-end session workflow tests."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_synthetic_baseline_video
from vball_analyser.config import PROJECT_ROOT, STATIC_DIR, AppConfig
from vball_analyser.detector import SyntheticColorPredictor
from vball_analyser.app import create_app


def _make_test_client(tmp_path: Path, max_upload_bytes: int = 50 * 1024 * 1024) -> tuple[TestClient, AppConfig]:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        secret_file=tmp_path / ".vball_secret",
        allowed_local_dirs=(allowed_dir,),
        max_upload_bytes=max_upload_bytes,
    )
    app = create_app(config=cfg, predictor=SyntheticColorPredictor())
    return TestClient(app), cfg


def test_security_headers_methods_and_csrf(tmp_path: Path) -> None:
    client, _cfg = _make_test_client(tmp_path)
    with client:
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "default-src 'self'" in res_root.headers["Content-Security-Policy"]
        assert "script-src 'self'" in res_root.headers["Content-Security-Policy"]
        assert res_root.headers["X-Frame-Options"] == "DENY"
        assert res_root.headers["X-Content-Type-Options"] == "nosniff"
        assert "camera=()" in res_root.headers["Permissions-Policy"]

        # Disallowed HTTP methods -> 405
        assert client.put("/api/sessions").status_code == 405
        assert client.delete("/api/sessions").status_code == 405

        # POST without CSRF -> 403
        res_no_csrf = client.post("/api/sessions/local", json={"path": "/etc/passwd"})
        assert res_no_csrf.status_code == 403

        csrf_res = client.get("/api/csrf")
        assert csrf_res.status_code == 200
        assert csrf_res.headers["Cache-Control"] == "no-store"
        csrf_token = csrf_res.json()["csrf_token"]

        # Reset session invalidates old session token
        reset_res = client.post("/api/session/reset", headers={"X-CSRF-Token": csrf_token})
        assert reset_res.status_code == 200
        new_token = reset_res.json()["csrf_token"]
        assert new_token != csrf_token


def test_path_traversal_prefix_bypass_magic_bytes_and_size_limit(tmp_path: Path) -> None:
    client, cfg = _make_test_client(tmp_path, max_upload_bytes=1024)
    allowed = cfg.allowed_local_dirs[0]
    evil_sibling = tmp_path / "allowed-evil"
    evil_sibling.mkdir(parents=True, exist_ok=True)
    evil_video = evil_sibling / "evil.mp4"
    make_synthetic_baseline_video(evil_video, duration_s=1.0, fps=15)

    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]

        # Prefix-matching bypass attempt -> 403
        r_prefix = client.post(
            "/api/sessions/local",
            json={"path": str(evil_video)},
            headers={"X-CSRF-Token": csrf},
        )
        assert r_prefix.status_code == 403

        # Traversal attempt -> 400 or 403
        r_trav = client.post(
            "/api/sessions/local",
            json={"path": str(allowed / ".." / "allowed-evil" / "evil.mp4")},
            headers={"X-CSRF-Token": csrf},
        )
        assert r_trav.status_code in (400, 403)

        # Disguised HTML inside .mp4 upload -> 400
        r_fake = client.post(
            "/api/sessions/upload",
            files={"file": ("fake.mp4", b"<html><script>alert(1)</script></html>", "video/mp4")},
            headers={"X-CSRF-Token": csrf},
        )
        assert r_fake.status_code == 400

        # Oversized valid-header upload -> 413
        big_mp4 = b"\x00\x00\x00\x20ftypisom" + (b"\x00" * 4096)
        r_big = client.post(
            "/api/sessions/upload",
            files={"file": ("big.mp4", big_mp4, "video/mp4")},
            headers={"X-CSRF-Token": csrf},
        )
        assert r_big.status_code == 413


def test_end_to_end_local_workflow_range_calibration_hitter_export(tmp_path: Path) -> None:
    client, cfg = _make_test_client(tmp_path)
    video_path = cfg.allowed_local_dirs[0] / "baseline_clip.mp4"
    gt = make_synthetic_baseline_video(video_path, duration_s=3.0, fps=24)

    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        create_res = client.post(
            "/api/sessions/local",
            json={"path": str(video_path), "net_height_m": 2.43},
            headers={"X-CSRF-Token": csrf},
        )
        assert create_res.status_code == 200
        sid = create_res.json()["session_id"]

        # Poll until background pipeline reaches ready
        sess = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            r = client.get(f"/api/sessions/{sid}")
            assert r.status_code == 200
            sess = r.json()
            if sess["status"] in ("ready", "error"):
                break
            time.sleep(0.25)
        assert sess is not None
        assert sess["status"] == "ready"
        assert len(sess["plays"]) >= 1
        play = sess["plays"][0]

        # HTTP 206 Partial Content range streaming
        r_range = client.get(f"/api/sessions/{sid}/video", headers={"Range": "bytes=0-1023"})
        assert r_range.status_code == 206
        assert r_range.headers["Accept-Ranges"] == "bytes"
        assert r_range.headers["Content-Range"].startswith("bytes 0-1023/")
        assert len(r_range.content) == 1024

        # 2-click calibration update
        r_cal = client.post(
            f"/api/sessions/{sid}/calibration",
            json={
                "left_point": {"x": gt["left_pt"][0], "y": gt["left_pt"][1]},
                "right_point": {"x": gt["right_pt"][0], "y": gt["right_pt"][1]},
                "net_height_m": 2.43,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert r_cal.status_code == 200
        assert r_cal.json()["calibration"]["is_manual"] is True

        # Reassign hitter to another cached track
        alt_track = next(t["track_id"] for t in play["tracks"] if t["track_id"] != play["hitter_track_id"])
        r_hit = client.post(
            f"/api/sessions/{sid}/plays/{play['play_id']}/hitter",
            json={"track_id": alt_track},
            headers={"X-CSRF-Token": csrf},
        )
        assert r_hit.status_code == 200
        assert r_hit.json()["plays"][0]["hitter_track_id"] == alt_track

        # Export annotated MP4 with attachment header
        r_exp = client.post(
            f"/api/sessions/{sid}/plays/{play['play_id']}/export",
            json={"freeze_contact": True},
            headers={"X-CSRF-Token": csrf},
        )
        assert r_exp.status_code == 200
        dl_url = r_exp.json()["download_url"]
        r_dl = client.get(dl_url)
        assert r_dl.status_code == 200
        assert "attachment;" in r_dl.headers["Content-Disposition"]
        assert r_dl.headers["X-Content-Type-Options"] == "nosniff"
        assert len(r_dl.content) > 5000

        # Re-run detection requires CSRF and preserves manual calibration
        r_no_csrf = client.post(f"/api/sessions/{sid}/rerun", json={})
        assert r_no_csrf.status_code == 403

        r_rerun = client.post(f"/api/sessions/{sid}/rerun", headers={"X-CSRF-Token": csrf}, json={})
        assert r_rerun.status_code == 200
        assert r_rerun.json()["status"] == "queued"
        assert r_rerun.json()["calibration"]["is_manual"] is True

        rerun_ready = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            r = client.get(f"/api/sessions/{sid}")
            assert r.status_code == 200
            rerun_ready = r.json()
            if rerun_ready["status"] in ("ready", "error"):
                break
            time.sleep(0.25)
        assert rerun_ready is not None
        assert rerun_ready["status"] == "ready"
        assert rerun_ready["calibration"]["is_manual"] is True
        assert len(rerun_ready["plays"]) >= 1


def test_static_frontend_and_backend_security_hygiene() -> None:
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    forbidden_js = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "alert(", "confirm(", "prompt(", "localStorage", "sessionStorage")
    js_files = sorted(STATIC_DIR.glob("*.js"))
    assert js_files, "Expected at least one static JS file"
    for js_file in js_files:
        js_text = js_file.read_text(encoding="utf-8")
        for token in forbidden_js:
            assert token not in js_text, f"Forbidden JS pattern {token} found in {js_file.name}"
    assert "<script>" not in index_html
    assert "style=" not in index_html

    src_dir = PROJECT_ROOT / "src" / "vball_analyser"
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "0.0.0.0" not in text, f"Server must never bind 0.0.0.0 ({py_file.name})"


def test_app_config_host_port_env_override(monkeypatch) -> None:
    import pytest

    monkeypatch.delenv("VBALL_HOST", raising=False)
    monkeypatch.delenv("VBALL_PORT", raising=False)
    default_cfg = AppConfig()
    assert default_cfg.bind_host == "127.0.0.1"
    assert default_cfg.default_port == 8081

    # Blank / whitespace values fall back cleanly to defaults
    monkeypatch.setenv("VBALL_HOST", "   ")
    monkeypatch.setenv("VBALL_PORT", "  ")
    blank_cfg = AppConfig()
    assert blank_cfg.bind_host == "127.0.0.1"
    assert blank_cfg.default_port == 8081

    monkeypatch.setenv("VBALL_HOST", "0.0.0.0")
    monkeypatch.setenv("VBALL_PORT", "9090")
    override_cfg = AppConfig()
    assert override_cfg.bind_host == "0.0.0.0"
    assert override_cfg.default_port == 9090

    for bad_port in ("not-a-port", "0", "70000"):
        monkeypatch.setenv("VBALL_PORT", bad_port)
        with pytest.raises(ValueError, match="VBALL_PORT"):
            AppConfig()
