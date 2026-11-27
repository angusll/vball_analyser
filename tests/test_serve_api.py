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

        # Clear serves route
        client.post(f"/api/serve-sessions/{sid}/serves", json={"x_m": 2.0, "y_m": 6.0, "result": "in"}, headers=hdr)
        clr = client.post(f"/api/serve-sessions/{sid}/serves/clear", headers=hdr)
        assert clr.status_code == 200
        assert clr.json()["summary"]["total"] == 0

        # Trigger auto-detect and wait for completion
        ad_res = client.post(f"/api/serve-sessions/{sid}/auto-detect", json={"server_name": "#7"}, headers=hdr)
        assert ad_res.status_code == 200
        deadline = time.monotonic() + 30.0
        final_status = ad_res.json()["session"]["status"]
        while final_status == "scanning" and time.monotonic() < deadline:
            time.sleep(0.25)
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


def test_calibrate_court_from_net_endpoint(tmp_path: Path) -> None:
    client, cfg = _make_client(tmp_path)
    vid = cfg.allowed_local_dirs[0] / "net_cal.mp4"
    make_synthetic_baseline_video(vid, duration_s=1.5, fps=15)
    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        hdr = {"X-CSRF-Token": csrf}
        vid_id = client.post("/api/sessions/local", json={"path": str(vid), "skip_analysis": True}, headers=hdr).json()["session_id"]
        sid = client.post("/api/serve-sessions", json={"name": "Net Cal", "video_session_id": vid_id}, headers=hdr).json()["session"]["serve_session_id"]

        res = client.post(
            f"/api/serve-sessions/{sid}/calibration/from-net",
            json={
                "left_top": {"x": 215.0, "y": 252.0},
                "right_top": {"x": 978.0, "y": 260.0},
                "sideline_point": {"x": 40.0, "y": 515.0},
                "net_height_m": 2.43,
            },
            headers=hdr,
        )
        assert res.status_code == 200
        cal = res.json()["session"]["court_calibration"]
        assert cal["is_calibrated"] is True
        assert cal["method"] == "net"
        assert cal["net_left_top"]["x"] == 215.0
        assert cal["sideline_point"]["x"] == 40.0


def test_index_contains_tabs_and_serves_serve_tab_js(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        html = client.get("/").text
        assert 'data-tab="studio"' in html
        assert 'data-tab="serve"' in html
        assert 'id="serveTab"' in html
        assert 'id="serveCourtCanvas"' in html
        assert 'id="selectedServeResult"' in html
        assert 'id="toggleCalOverlayBtn"' in html
        assert 'id="deleteServeSessionBtn"' in html
        assert 'id="clearServesBtn"' in html
        assert 'class="topbar-left"' in html
        assert "/static/serve_tab.js" in html
        js = client.get("/static/serve_tab.js")
        assert js.status_code == 200


def test_duplicate_auto_detect_returns_409(tmp_path: Path) -> None:
    client, cfg = _make_client(tmp_path)
    vid = cfg.allowed_local_dirs[0] / "dup_scan.mp4"
    make_synthetic_baseline_video(vid, duration_s=1.5, fps=15)
    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        hdr = {"X-CSRF-Token": csrf}
        vid_id = client.post("/api/sessions/local", json={"path": str(vid), "skip_analysis": True}, headers=hdr).json()["session_id"]
        sid = client.post("/api/serve-sessions", json={"name": "Dup Scan", "video_session_id": vid_id}, headers=hdr).json()["session"]["serve_session_id"]
        client.post(
            f"/api/serve-sessions/{sid}/calibration",
            json={
                "front_left": {"x": 200.0, "y": 320.0},
                "front_right": {"x": 850.0, "y": 320.0},
                "attack_right": {"x": 940.0, "y": 420.0},
                "attack_left": {"x": 110.0, "y": 420.0},
            },
            headers=hdr,
        )
        # Simulate an active scanning state on disk
        from vball_analyser.serve_storage import ServeSessionStore

        sstore = ServeSessionStore(cfg.serve_sessions_dir)
        sess = sstore.load_session(sid)
        sess.status = "scanning"
        sstore.save_session(sess)

        dup = client.post(f"/api/serve-sessions/{sid}/auto-detect", json={"server_name": "#7"}, headers=hdr)
        assert dup.status_code == 409


def test_receive_quality_api_endpoints_and_server_filter(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        csrf = client.get("/api/csrf").json()["csrf_token"]
        hdr = {"X-CSRF-Token": csrf}

        # Create session returns empty receive_summary and default setter_target
        c_res = client.post("/api/serve-sessions", json={"name": "Receive API Test"}, headers=hdr)
        assert c_res.status_code == 200
        c_body = c_res.json()
        sid = c_body["session"]["serve_session_id"]
        assert c_body["session"]["setter_target"] == {"x": 6.0, "y": 0.8}
        assert c_body["receive_summary"]["scored_count"] == 0
        assert c_body["receive_summary"]["avg_rating"] == 0.0

        # CSRF enforced on update / add / serve update
        assert client.post(f"/api/serve-sessions/{sid}/update", json={"setter_target": {"x": 5.0, "y": 1.0}}).status_code == 403
        assert client.post(f"/api/serve-sessions/{sid}/serves", json={"x_m": 4.5, "y_m": 6.5}).status_code == 403

        # Chart a pass for #7 received by P1 near setter target -> score 3
        a1 = client.post(
            f"/api/serve-sessions/{sid}/serves",
            json={
                "x_m": 2.0,
                "y_m": 6.5,
                "result": "in",
                "server_name": "#7",
                "passer_label": "P1",
                "pass_x_m": 6.1,
                "pass_y_m": 1.0,
            },
            headers=hdr,
        )
        assert a1.status_code == 200
        s1_id = a1.json()["session"]["serves"][0]["serve_id"]
        assert a1.json()["session"]["serves"][0]["receive_score"] == 3
        assert a1.json()["receive_summary"]["scored_count"] == 1
        assert a1.json()["receive_summary"]["avg_rating"] == 3.0
        assert a1.json()["receive_summary"]["perfect_pct"] == 100.0
        assert a1.json()["receive_summary"]["by_passer"][0]["passer_label"] == "P1"

        # Chart an ace for #9 -> score 0
        a2 = client.post(
            f"/api/serve-sessions/{sid}/serves",
            json={"x_m": 7.5, "y_m": 7.5, "result": "ace", "server_name": "#9", "passer_label": "P3"},
            headers=hdr,
        )
        assert a2.status_code == 200
        assert a2.json()["receive_summary"]["scored_count"] == 2
        assert a2.json()["receive_summary"]["avg_rating"] == 1.5
        assert a2.json()["receive_summary"]["in_system_pct"] == 50.0

        # Server-filtered GET returns only #7's receive_summary
        g_filt = client.get(f"/api/serve-sessions/{sid}?server=%237")
        assert g_filt.status_code == 200
        assert g_filt.json()["receive_summary"]["scored_count"] == 1
        assert g_filt.json()["receive_summary"]["avg_rating"] == 3.0

        # Manual score override on s1 -> score 1, receive_score_manual=True
        u1 = client.post(
            f"/api/serve-sessions/{sid}/serves/{s1_id}/update?server=%237",
            json={"receive_score": 1},
            headers=hdr,
        )
        assert u1.status_code == 200
        assert u1.json()["session"]["serves"][0]["receive_score"] == 1
        assert u1.json()["session"]["serves"][0]["receive_score_manual"] is True
        assert u1.json()["receive_summary"]["avg_rating"] == 1.0

        # Clear manual score override -> reverts to auto score 3
        u2 = client.post(
            f"/api/serve-sessions/{sid}/serves/{s1_id}/update",
            json={"clear_receive_score": True},
            headers=hdr,
        )
        assert u2.status_code == 200
        assert u2.json()["session"]["serves"][0]["receive_score"] == 3
        assert u2.json()["session"]["serves"][0]["receive_score_manual"] is False

        # Move setter_target via /update -> clamps and recomputes s1 score to 0
        upd_sess = client.post(
            f"/api/serve-sessions/{sid}/update",
            json={"setter_target": {"x": 0.2, "y": 0.1}},
            headers=hdr,
        )
        assert upd_sess.status_code == 200
        assert upd_sess.json()["session"]["serves"][0]["receive_score"] == 0

        # Resize scoring rings via /update -> enlarges red/yellow/green rings and recomputes s1 score
        upd_rings = client.post(
            f"/api/serve-sessions/{sid}/update",
            json={"setter_target": {"x": 6.0, "y": 0.8}, "receive_ring_radii_m": [0.1, 0.2, 1.0]},
            headers=hdr,
        )
        assert upd_rings.status_code == 200
        assert upd_rings.json()["session"]["receive_ring_radii_m"] == [0.4, 0.7, 1.0]
        assert upd_rings.json()["session"]["serves"][0]["receive_score"] == 3

        # Restore setter_target to (0.5, 0.2) and default rings so subsequent assertions stay unchanged
        upd_sess = client.post(
            f"/api/serve-sessions/{sid}/update",
            json={"setter_target": {"x": 0.2, "y": 0.1}, "receive_ring_radii_m": [1.5, 3.2, 4.8]},
            headers=hdr,
        )
        assert upd_sess.status_code == 200
        assert upd_sess.json()["session"]["setter_target"] == {"x": 0.5, "y": 0.2}
        assert upd_sess.json()["session"]["serves"][0]["receive_score"] == 0
        assert upd_sess.json()["receive_summary"]["avg_rating"] == 0.0

        # Move pass destination near new setter_target (0.5, 0.2) and clear passer_label
        u3 = client.post(
            f"/api/serve-sessions/{sid}/serves/{s1_id}/update",
            json={"pass_x_m": 0.8, "pass_y_m": 0.5, "clear_passer_label": True},
            headers=hdr,
        )
        assert u3.status_code == 200
        assert u3.json()["session"]["serves"][0]["receive_score"] == 3
        assert u3.json()["session"]["serves"][0]["passer_label"] is None

        # Clear pass removes pass coords & score for "in" serve
        u4 = client.post(
            f"/api/serve-sessions/{sid}/serves/{s1_id}/update",
            json={"clear_pass": True},
            headers=hdr,
        )
        assert u4.status_code == 200
        assert u4.json()["session"]["serves"][0]["pass_x_m"] is None
        assert u4.json()["session"]["serves"][0]["receive_score"] is None
        assert u4.json()["receive_summary"]["scored_count"] == 1  # only the ace remains scored

        # Delete returns updated receive_summary
        d1 = client.post(f"/api/serve-sessions/{sid}/serves/{s1_id}/delete", headers=hdr)
        assert d1.status_code == 200
        assert "receive_summary" in d1.json()
        assert d1.json()["receive_summary"]["scored_count"] == 1
