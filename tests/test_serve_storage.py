"""Unit tests for disk-backed ServeSessionStore, snapshot isolation, and auto-serve merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vball_analyser.models import CourtCalibration, PasserPosition, Point2D, ServeEvent
from vball_analyser.serve_storage import ServeSessionStore


def test_serve_store_crud_and_snapshot_isolation(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    sess = store.create_session("Tuesday Practice")
    assert len(sess.default_passers) == 3

    # Add manual serve in seam between P1 and P2, plus a video-derived auto serve
    sess = store.add_serve(sess.serve_session_id, x_m=3.15, y_m=6.3, result="in", server_name="#7")
    auto_passers = [
        PasserPosition(label="P1", x_m=1.8, y_m=6.3),
        PasserPosition(label="P2", x_m=4.5, y_m=6.3),
    ]
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=3.15,
        y_m=6.3,
        result="in",
        server_name="#7",
        source="auto",
        timestamp_s=2.4,
        passers_snapshot=auto_passers,
    )
    assert len(sess.serves) == 2
    assert sess.serves[0].is_seam is True
    assert sess.serves[1].is_seam is True

    # Dragging default_passers updates pure manual serves but preserves video-derived auto snapshots
    on_serve_formation = [
        PasserPosition(label="P1", x_m=3.15, y_m=6.3),
        PasserPosition(label="P2", x_m=6.5, y_m=6.3),
    ]
    sess = store.update_session(sess.serve_session_id, default_passers=on_serve_formation)
    assert len(sess.default_passers) == 2
    assert sess.serves[0].is_seam is False
    assert sess.serves[1].is_seam is True
    assert [p.x_m for p in sess.serves[1].passers_snapshot] == [1.8, 4.5]

    # Editing a specific serve's passers_snapshot back to straddling recomputes its seam to True
    straddling = [
        PasserPosition(label="P1", x_m=1.8, y_m=6.3),
        PasserPosition(label="P2", x_m=4.8, y_m=6.3),
    ]
    sess = store.update_serve(sess.serve_session_id, sess.serves[0].serve_id, passers_snapshot=straddling)
    assert sess.serves[0].is_seam is True

    # Valid calibration saves; degenerate calibration fails fast
    cal = CourtCalibration(
        front_left=Point2D(x=200.0, y=320.0),
        front_right=Point2D(x=850.0, y=320.0),
        attack_right=Point2D(x=940.0, y=420.0),
        attack_left=Point2D(x=110.0, y=420.0),
    )
    sess = store.save_calibration(sess.serve_session_id, cal)
    assert sess.court_calibration is not None and sess.court_calibration.is_calibrated is True

    # Invalid IDs rejected
    with pytest.raises(ValueError):
        store.load_session("../escape")

    # Delete serve, clear remaining serves, and delete session
    sess = store.delete_serve(sess.serve_session_id, sess.serves[0].serve_id)
    assert len(sess.serves) == 1
    sess = store.clear_serves(sess.serve_session_id)
    assert sess.serves == []
    store.delete_session(sess.serve_session_id)
    with pytest.raises(FileNotFoundError):
        store.load_session(sess.serve_session_id)


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
        ServeEvent(
            serve_id="auto_new_1",
            index=1,
            source="auto",
            timestamp_s=12.0,
            x_m=7.5,
            y_m=7.0,
            result="in",
            zone="Zone 1",
            depth="deep",
            is_seam=False,
        ),
    ]
    sess = store.replace_auto_serves(sess.serve_session_id, fresh_auto)
    ids = [s.serve_id for s in sess.serves]
    assert edited_id in ids
    assert "auto_new_1" in ids
    assert len(sess.serves) == 3  # original manual + edited auto + 1 fresh auto
    assert [s.index for s in sess.serves] == [1, 2, 3]


def test_load_legacy_serve_session_json_without_receive_fields(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    legacy_dir = tmp_path / "serve_sessions" / "srv_legacy01"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "serve_session_id": "srv_legacy01",
        "name": "Legacy Practice",
        "created_at": "2026-01-15T10:00:00+00:00",
        "status": "ready",
        "progress_pct": 100.0,
        "default_passers": [{"label": "P1", "x_m": 1.8, "y_m": 6.2}],
        "servers": ["#4"],
        "serves": [
            {
                "serve_id": "s_old01",
                "index": 1,
                "source": "manual",
                "server_name": "#4",
                "x_m": 4.5,
                "y_m": 6.5,
                "result": "in",
                "zone": "Zone 6",
                "depth": "deep",
                "is_seam": False,
            }
        ],
    }
    (legacy_dir / "serve_session.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    loaded = store.load_session("srv_legacy01")
    assert loaded.setter_target == Point2D(x=6.0, y=0.8)
    assert len(loaded.serves) == 1
    ev = loaded.serves[0]
    assert ev.passer_label is None
    assert ev.pass_x_m is None
    assert ev.pass_y_m is None
    assert ev.pass_distance_m is None
    assert ev.receive_score is None
    assert ev.receive_score_manual is False


def test_add_serve_receive_invariants_and_setter_target_recomputation(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    sess = store.create_session("Receive Quality Storage")

    # 1) "in" serve with pass coords near default target (6.0, 0.8) -> auto score 3
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=2.0,
        y_m=6.5,
        result="in",
        passer_label="  P1  ",
        pass_x_m=6.2,
        pass_y_m=1.0,
    )
    s_auto = sess.serves[0]
    assert s_auto.passer_label == "P1"
    assert s_auto.pass_x_m == 6.2
    assert s_auto.pass_y_m == 1.0
    assert s_auto.receive_score == 3
    assert s_auto.receive_score_manual is False
    assert s_auto.pass_distance_m is not None and s_auto.pass_distance_m < 1.5

    # 2) "in" serve with pass coords AND explicit override score != auto_score -> manual override
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=4.5,
        y_m=6.5,
        result="in",
        passer_label="P2",
        pass_x_m=6.1,
        pass_y_m=0.9,
        receive_score=1,
    )
    s_manual_override = sess.serves[1]
    assert s_manual_override.receive_score == 1
    assert s_manual_override.receive_score_manual is True

    # 3) "in" serve with score only (no pass coords) -> manual override
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=7.0,
        y_m=6.5,
        result="in",
        receive_score=2,
    )
    assert sess.serves[2].pass_x_m is None
    assert sess.serves[2].receive_score == 2
    assert sess.serves[2].receive_score_manual is True

    # 4) "ace" serve ignores pass coords and forces score=0, manual=False
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=7.5,
        y_m=7.5,
        result="ace",
        pass_x_m=6.0,
        pass_y_m=0.8,
        receive_score=3,
    )
    assert sess.serves[3].pass_x_m is None
    assert sess.serves[3].pass_y_m is None
    assert sess.serves[3].pass_distance_m is None
    assert sess.serves[3].receive_score == 0
    assert sess.serves[3].receive_score_manual is False

    # 5) "out" and "net" serves clear pass coords and have receive_score=None
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=9.5,
        y_m=6.0,
        result="in",
        pass_x_m=6.0,
        pass_y_m=0.8,
        receive_score=3,
    )
    assert sess.serves[4].result == "out"
    assert sess.serves[4].pass_x_m is None
    assert sess.serves[4].receive_score is None

    sess = store.add_serve(
        sess.serve_session_id,
        x_m=4.5,
        y_m=1.0,
        result="net",
        pass_x_m=6.0,
        pass_y_m=0.8,
        receive_score=3,
    )
    assert sess.serves[5].result == "net"
    assert sess.serves[5].pass_x_m is None
    assert sess.serves[5].receive_score is None

    # Moving setter_target far left (and clamping x=0.1 -> 0.5, y=0.05 -> 0.2) recomputes s_auto
    # but preserves s_manual_override's manual score and ace's 0
    sess = store.update_session(sess.serve_session_id, setter_target=Point2D(x=0.1, y=0.05))
    assert sess.setter_target == Point2D(x=0.5, y=0.2)
    assert sess.serves[0].receive_score == 0  # (6.2, 1.0) is > 4.8m from (0.5, 0.2)
    assert sess.serves[0].pass_distance_m is not None and sess.serves[0].pass_distance_m > 4.8
    assert sess.serves[1].receive_score == 1  # manual override preserved
    assert sess.serves[1].receive_score_manual is True
    assert sess.serves[3].receive_score == 0  # ace unchanged


def test_update_serve_receive_edits_and_source_flip(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    sess = store.create_session("Update Receive Test")
    sess = store.add_serve(
        sess.serve_session_id,
        x_m=4.5,
        y_m=6.5,
        result="in",
        source="auto",
        timestamp_s=3.2,
        passer_label="P2",
        pass_x_m=6.0,
        pass_y_m=1.0,
    )
    sid = sess.serve_session_id
    ev_id = sess.serves[0].serve_id
    assert sess.serves[0].source == "auto"
    assert sess.serves[0].receive_score == 3

    # Manual score override flips source to manual and sets receive_score_manual=True
    sess = store.update_serve(sid, ev_id, receive_score=1)
    assert sess.serves[0].receive_score == 1
    assert sess.serves[0].receive_score_manual is True
    assert sess.serves[0].source == "manual"

    # Clearing manual override (receive_score=None) recomputes auto score from pass coords
    sess = store.update_serve(sid, ev_id, receive_score=None)
    assert sess.serves[0].receive_score == 3
    assert sess.serves[0].receive_score_manual is False

    # Moving pass coords recomputes score (yellow ring: ~2.2m away -> score 2)
    sess = store.update_serve(sid, ev_id, pass_x_m=6.0, pass_y_m=3.0)
    assert sess.serves[0].pass_x_m == 6.0
    assert sess.serves[0].pass_y_m == 3.0
    assert sess.serves[0].receive_score == 2
    assert sess.serves[0].receive_score_manual is False

    # Updating passer_label & clearing pass
    sess = store.update_serve(sid, ev_id, passer_label="P3", clear_pass=True)
    assert sess.serves[0].passer_label == "P3"
    assert sess.serves[0].pass_x_m is None
    assert sess.serves[0].pass_y_m is None
    assert sess.serves[0].pass_distance_m is None
    assert sess.serves[0].receive_score is None
    assert sess.serves[0].receive_score_manual is False

    # Changing result to ace sets receive_score=0; changing to net clears receive_score=None
    sess = store.update_serve(sid, ev_id, pass_x_m=6.0, pass_y_m=0.8)
    assert sess.serves[0].receive_score == 3
    sess = store.update_serve(sid, ev_id, result="ace")
    assert sess.serves[0].pass_x_m is None
    assert sess.serves[0].receive_score == 0
    sess = store.update_serve(sid, ev_id, result="net")
    assert sess.serves[0].receive_score is None


def test_slot_label_inference_manual_override_and_formation_update(tmp_path: Path) -> None:
    store = ServeSessionStore(tmp_path / "serve_sessions")
    sess = store.create_session("Slots")
    assert sess.session_mode == "drill"
    sid = sess.serve_session_id

    # Short manual "in" serve far ahead of the deep P1 stance still maps to Left (no distance cap)
    sess = store.add_serve(sid, x_m=2.0, y_m=3.0, result="in")
    sess = store.add_serve(sid, x_m=2.2, y_m=3.2, result="in")
    sess = store.add_serve(sid, x_m=2.1, y_m=3.1, result="in", source="auto", timestamp_s=4.0)
    s1, s2, video = sess.serves
    assert (s1.slot_label, s1.slot_label_manual) == ("Left", False)
    assert (s2.slot_label, s2.slot_label_manual) == ("Left", False)
    assert video.slot_label == "Left"

    # Manual override, then clear -> re-inferred
    sess = store.update_serve(sid, s1.serve_id, slot_label="Mid")
    assert (sess.serves[0].slot_label, sess.serves[0].slot_label_manual) == ("Mid", True)
    sess = store.update_serve(sid, s1.serve_id, clear_slot_label=True)
    assert (sess.serves[0].slot_label, sess.serves[0].slot_label_manual) == ("Left", False)
    sess = store.update_serve(sid, s1.serve_id, slot_label="Mid")

    # Formation drag re-infers only non-overridden pure manual serves
    sess = store.update_session(
        sid,
        default_passers=[
            PasserPosition(label="P1", x_m=0.5, y_m=6.0),
            PasserPosition(label="P2", x_m=2.5, y_m=6.0),
        ],
    )
    assert (sess.serves[0].slot_label, sess.serves[0].slot_label_manual) == ("Mid", True)
    assert sess.serves[1].slot_label == "Right"
    assert sess.serves[2].slot_label == "Left"  # video serve keeps its own snapshot slot

    # Serve errors are not received -> no slot, override dropped
    sess = store.update_serve(sid, s1.serve_id, result="net")
    assert (sess.serves[0].slot_label, sess.serves[0].slot_label_manual) == (None, False)

    reloaded = store.load_session(sid)
    assert reloaded.serves[1].slot_label == "Right"
