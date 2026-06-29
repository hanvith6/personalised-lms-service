"""Tests for stable-ID track assembly and live-feed windowed scoring (Module 9)."""
import pytest

from services.multi_person import assemble_tracks
from services.live_feed import (
    iter_windows,
    score_window,
    score_stream,
    keypoints_from_yolo,
)


def _kp(x, y):
    return {"nose": (x, y, 0.9), "left_eye": (x - 10, y - 5, 0.9),
            "right_eye": (x + 10, y - 5, 0.9),
            "left_shoulder": (x - 20, y + 70, 0.9),
            "right_shoulder": (x + 20, y + 70, 0.9)}


# ── assemble_tracks (stable-ID path) ─────────────────────────────────────────

def test_assemble_groups_by_stable_id():
    records = [{7: _kp(100, 50), 9: _kp(400, 60)},
               {7: _kp(101, 50), 9: _kp(401, 60)}]
    tracks = assemble_tracks(records)
    assert set(tracks) == {7, 9}
    assert all(len(fr) == 2 for fr in tracks.values())


def test_assemble_pads_absent_frames():
    # ID 9 only present in the middle frame.
    records = [{7: _kp(100, 50)}, {7: _kp(100, 50), 9: _kp(400, 60)},
               {7: _kp(100, 50)}]
    tracks = assemble_tracks(records)
    assert len(tracks[9]) == 3
    assert tracks[9][0]["keypoints"] == {}
    assert tracks[9][1]["keypoints"]
    assert tracks[9][2]["keypoints"] == {}


def test_assemble_min_frames_filter():
    records = [{7: _kp(100, 50)}, {7: _kp(100, 50), 9: _kp(400, 60)}]
    tracks = assemble_tracks(records, min_frames=2)
    assert set(tracks) == {7}  # ID 9 seen once, dropped


# ── iter_windows ─────────────────────────────────────────────────────────────

def test_tumbling_windows():
    assert list(iter_windows([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_sliding_windows_overlap():
    # Once a window reaches the final item, iteration stops (no [3, 4] tail).
    assert list(iter_windows([1, 2, 3, 4], 3, stride=1)) == [
        [1, 2, 3], [2, 3, 4]]


def test_trailing_partial_window_is_kept():
    assert list(iter_windows([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_empty_stream_yields_nothing():
    assert list(iter_windows([], 3)) == []


def test_invalid_window_or_stride_raises():
    with pytest.raises(ValueError):
        list(iter_windows([1, 2], 0))
    with pytest.raises(ValueError):
        list(iter_windows([1, 2], 2, stride=0))


# ── score_window / score_stream ──────────────────────────────────────────────

def test_score_window_scores_each_person():
    records = [{1: _kp(100, 50), 2: _kp(400, 60)} for _ in range(4)]
    scores = score_window(records)
    assert set(scores) == {1, 2}
    for s in scores.values():
        assert 0 <= s["eye_contact"] <= 10
        assert s["visible"] == 4


def test_score_stream_emits_one_reading_per_window():
    records = [{1: _kp(100, 50)} for _ in range(5)]
    out = list(score_stream(records, window=2))
    assert [o["window"] for o in out] == [0, 1, 2]  # 2,2,1 (trailing)
    assert all(o["presenter"] is not None for o in out)


def test_score_stream_presenter_is_most_visible():
    # ID 1 present every frame; ID 2 only once -> presenter is 1.
    records = [{1: _kp(100, 50), 2: _kp(400, 60)}] + [{1: _kp(100, 50)}] * 3
    (reading,) = list(score_stream(records, window=4))
    assert reading["presenter"]["visible"] == 4


def test_keypoints_from_yolo_maps_coco_order():
    row = [(i, i + 1, 0.5) for i in range(17)]
    kp = keypoints_from_yolo(row)
    assert kp["nose"] == (0.0, 1.0, 0.5)
    assert kp["right_ankle"] == (16.0, 17.0, 0.5)  # COCO index 16 (last)
