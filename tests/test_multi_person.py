"""Tests for the multi-person track splitter (Module 9, option C)."""
from services.multi_person import split_tracks, primary_track


def _person(x, y):
    """A minimal visible detection centred near (x, y)."""
    return {"keypoints": {
        "nose": (x, y, 0.9),
        "left_eye": (x - 10, y - 5, 0.9),
        "right_eye": (x + 10, y - 5, 0.9),
        "left_shoulder": (x - 20, y + 70, 0.9),
        "right_shoulder": (x + 20, y + 70, 0.9),
    }}


def test_no_frames_returns_empty():
    assert split_tracks([]) == {}


def test_frames_with_no_detections_yield_no_tracks():
    assert split_tracks([[], [], []]) == {}


def test_single_stable_person_is_one_track_spanning_all_frames():
    frames = [[_person(100, 50)] for _ in range(4)]
    tracks = split_tracks(frames)
    assert len(tracks) == 1
    (only,) = tracks.values()
    assert len(only) == 4
    assert all(f["keypoints"] for f in only)


def test_two_separated_people_become_two_tracks():
    frames = [[_person(80, 50), _person(400, 60)] for _ in range(3)]
    tracks = split_tracks(frames)
    assert len(tracks) == 2
    # Each track spans every frame and is consistently one of the two people.
    for fr in tracks.values():
        xs = [f["keypoints"]["nose"][0] for f in fr if f["keypoints"]]
        assert max(xs) - min(xs) < 50  # stayed with the same person


def test_late_entrant_track_is_backfilled_with_empty_frames():
    # Person B only appears from frame 2 onward.
    frames = [
        [_person(80, 50)],
        [_person(82, 50)],
        [_person(82, 50), _person(420, 60)],
        [_person(83, 50), _person(421, 60)],
    ]
    tracks = split_tracks(frames)
    assert len(tracks) == 2
    late = min(tracks.values(), key=lambda fr: sum(1 for f in fr if f["keypoints"]))
    assert len(late) == 4
    assert late[0]["keypoints"] == {}  # back-filled before entry
    assert late[1]["keypoints"] == {}
    assert late[2]["keypoints"]       # present from frame 2


def test_min_frames_filter_drops_blips():
    # A one-frame stray detection far from the main subject is noise.
    frames = [[_person(100, 50)], [_person(100, 50), _person(900, 700)],
              [_person(100, 50)]]
    tracks = split_tracks(frames, min_frames=2)
    assert len(tracks) == 1  # the blip track (1 frame) is filtered out


def test_primary_track_picks_the_most_visible_person():
    frames = [
        [_person(80, 50), _person(420, 60)],
        [_person(81, 50), _person(421, 60)],
        [_person(82, 50)],  # only person A remains
    ]
    tracks = split_tracks(frames)
    primary = primary_track(tracks)
    xs = [f["keypoints"]["nose"][0] for f in primary if f["keypoints"]]
    assert len(xs) == 3 and max(xs) < 100  # the person seen in all 3 frames


def test_primary_track_of_empty_is_empty():
    assert primary_track({}) == []
