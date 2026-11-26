"""Integration tests for Serve Distribution REST API, CSRF protection, and video-only loading."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_synthetic_baseline_video
from vball_analyser.app import create_app
from vball_analyser.config import AppConfig
from vball_analyser.detector import SyntheticColorPredictor


def _make_client(tmp_path: Path) -> tuple[TestClient, AppConfig]:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        secret_file=tmp_path / ".vball_secret",
        allowed_local_dirs=(allowed_dir,),
    )
    app = create_app(config=cfg, predictor=SyntheticColorPredictor())
    return TestClient(app), cfg


def test_serve_endpoints_enforce_csrf_and_method_allowlist(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        # Existing method allowlist remains intact
        assert client.delete("/api/sessions").status_code == 405
        assert client.patch("/api/serve-sessions").status_code == 405

        # Mutating POST without CSRF -> 403
        res = client.post("/api/serve-sessions", json={"name": "Blocked"})
        assert res.status_code == 403


def test_skip_analysis_local_video_load(tmp_path: Path) -> None:
    client, cfg = _make_client(tmp_path)
    vid = cfg.allowed_local_dirs[0] / "practice.mp4"
    make_synthetic_baseline_video(vid, duration_s=1.5, fps=15)

    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        res = client.post(
            "/api/sessions/local",
            json={"path": str(vid), "skip_analysis": True},
            headers={"X-CSRF-Token": csrf},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ready"
        assert body["progress_pct"] == 100.0
        assert body["plays"] == []


def test_serve_session_full_workflow(tmp_path: Path) -> None:
    client, cfg = _make_client(tmp_path)
    vid = cfg.allowed_local_dirs[0] / "serve_clip.mp4"
    make_synthetic_baseline_video(vid, duration_s=2.0, fps=15)

    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        hdr = {"X-CSRF-Token": csrf}

        # Load video without Set & Hit GPU analysis
        v_res = client.post("/api/sessions/local", json={"path": str(vid), "skip_analysis": True}, headers=hdr)
        assert v_res.status_code == 200
        video_session_id = v_res.json()["session_id"]

        # Create serve session linked to video
        c_res = client.post(
            "/api/serve-sessions",
            json={"name": "Thursday Practice", "video_session_id": video_session_id},
            headers=hdr,
        )
        assert c_res.status_code == 200
        sid = c_res.json()["session"]["serve_session_id"]
        assert c_res.json()["summary"]["total"] == 0

        # Degenerate calibration returns 400
        bad_cal = client.post(
            f"/api/serve-sessions/{sid}/calibration",
            json={
                "front_left": {"x": 100, "y": 300},
                "front_right": {"x": 500, "y": 300},
                "attack_right": {"x": 700, "y": 300},
                "attack_left": {"x": 200, "y": 300},
            },
            headers=hdr,
        )
        assert bad_cal.status_code == 400

        # Valid 4-corner front-3m calibration succeeds
        ok_cal = client.post(
            f"/api/serve-sessions/{sid}/calibration",
            json={
                "front_left": {"x": 220, "y": 370},
                "front_right": {"x": 836, "y": 370},
                "attack_right": {"x": 896, "y": 435},
                "attack_left": {"x": 160, "y": 435},
            },
            headers=hdr,
        )
        assert ok_cal.status_code == 200
        assert ok_cal.json()["session"]["court_calibration"]["is_calibrated"] is True

        # Add manual serve in P1-P2 seam
        add_res = client.post(
            f"/api/serve-sessions/{sid}/serves",
            json={"x_m": 3.15, "y_m": 6.3, "result": "in", "server_name": "#7"},
            headers=hdr,
        )
        assert add_res.status_code == 200
        assert add_res.json()["summary"]["in_pct"] == 100.0
        assert add_res.json()["summary"]["seam_pct"] == 100.0
        serve_id = add_res.json()["session"]["serves"][0]["serve_id"]

        # Update serve to net error
        upd_res = client.post(
            f"/api/serve-sessions/{sid}/serves/{serve_id}/update",
            json={"result": "net"},
            headers=hdr,
        )
        assert upd_res.status_code == 200
        assert upd_res.json()["summary"]["error_pct"] == 100.0
        assert upd_res.json()["summary"]["seam_count"] == 0

        # Delete serve
        del_s = client.post(f"/api/serve-sessions/{sid}/serves/{serve_id}/delete", headers=hdr)
        assert del_s.status_code == 200
        assert del_s.json()["summary"]["total"] == 0

        # Trigger auto-detect and wait for completion
        ad_res = client.post(f"/api/serve-sessions/{sid}/auto-detect", json={"server_name": "#7"}, headers=hdr)
        assert ad_res.status_code == 200
        deadline = time.time() + 10.0
        final_status = ad_res.json()["session"]["status"]
        while final_status == "scanning" and time.time() < deadline:
            time.sleep(0.1)
            final_status = client.get(f"/api/serve-sessions/{sid}").json()["session"]["status"]
        assert final_status == "ready"

        # List & delete serve session
        assert any(s["serve_session_id"] == sid for s in client.get("/api/serve-sessions").json())
        del_sess = client.post(f"/api/serve-sessions/{sid}/delete", headers=hdr)
        assert del_sess.status_code == 200
        assert client.get(f"/api/serve-sessions/{sid}").status_code == 404


def test_post_mutations_respect_server_filter_query(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        hdr = {"X-CSRF-Token": csrf}
        sid = client.post("/api/serve-sessions", json={"name": "Filter Test"}, headers=hdr).json()["session"]["serve_session_id"]
        client.post(f"/api/serve-sessions/{sid}/serves", json={"x_m": 2.0, "y_m": 6.0, "result": "in", "server_name": "#9"}, headers=hdr)

        # Adding a serve for #7 with ?server=%237 must return #7's filtered summary (total == 1, not 2)
        res = client.post(
            f"/api/serve-sessions/{sid}/serves?server=%237",
            json={"x_m": 3.15, "y_m": 6.3, "result": "ace", "server_name": "#7"},
            headers=hdr,
        )
        assert res.status_code == 200
        assert res.json()["summary"]["total"] == 1
        assert res.json()["summary"]["ace_pct"] == 100.0


def test_index_contains_tabs_and_serves_serve_tab_js(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        html = client.get("/").text
        assert 'data-tab="studio"' in html
        assert 'data-tab="serve"' in html
        assert 'id="serveTab"' in html
        assert 'id="serveCourtCanvas"' in html
        assert 'id="selectedServeResult"' in html
        assert "/static/serve_tab.js" in html
        js = client.get("/static/serve_tab.js")
        assert js.status_code == 200
