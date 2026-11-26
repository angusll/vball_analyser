"""Tests for annotated MP4 export renderer."""

from __future__ import annotations

from pathlib import Path

from vball_analyser.calibration import auto_detect_net_tape
from vball_analyser.detector import SyntheticColorPredictor, analyse_play_window
from vball_analyser.renderer import export_annotated_play_mp4
from vball_analyser.video_io import iter_video_frames, probe_video


def test_export_annotated_play_mp4_with_contact_freeze(
    synthetic_clip: tuple[Path, dict], tmp_path: Path
) -> None:
    clip_path, _gt = synthetic_clip
    meta = probe_video(clip_path)
    frames = [bgr for _i, _t, bgr in iter_video_frames(clip_path, sample_fps=4.0)]
    cal = auto_detect_net_tape(frames)

    predictor = SyntheticColorPredictor()
    play, _ = analyse_play_window(
        clip_path, 1, 0.0, meta.duration_s, cal, predictor, meta, tmp_path / "sess"
    )

    no_freeze = export_annotated_play_mp4(
        clip_path, play, cal, tmp_path / "out_plain.mp4", freeze_at_contact_s=0.0
    )
    with_freeze = export_annotated_play_mp4(
        clip_path, play, cal, tmp_path / "out_freeze.mp4", freeze_at_contact_s=1.5
    )

    assert no_freeze.is_file()
    assert with_freeze.is_file()
    plain_meta = probe_video(no_freeze)
    freeze_meta = probe_video(with_freeze)
    assert freeze_meta.codec == "h264"
    expected_extra = int(round(1.5 * round(play.fps)))
    assert freeze_meta.frame_count >= plain_meta.frame_count + expected_extra - 2
