"""Unit tests for disk-backed ServeSessionStore, snapshot isolation, and auto-serve merge."""

from __future__ import annotations

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
    new_formation = [
        PasserPosition(label="P1", x_m=2.5, y_m=6.3),
        PasserPosition(label="P2", x_m=6.5, y_m=6.3),
    ]
    sess = store.update_session(sess.serve_session_id, default_passers=new_formation)
    assert len(sess.default_passers) == 2
    assert sess.serves[0].passers_snapshot == orig_snapshot
    assert sess.serves[0].is_seam is True

    # Editing a specific serve's passers_snapshot recomputes its seam
    shifted = [
        PasserPosition(label="P1", x_m=3.15, y_m=6.3),
        PasserPosition(label="P2", x_m=7.0, y_m=6.3),
    ]
    sess = store.update_serve(sess.serve_session_id, sess.serves[0].serve_id, passers_snapshot=shifted)
    assert sess.serves[0].is_seam is False

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

    # Delete serve and delete session
    sess = store.delete_serve(sess.serve_session_id, sess.serves[0].serve_id)
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
