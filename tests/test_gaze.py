"""Tests for gaze-based eye-contact scoring (Module 9)."""
from services.gaze import axis_ratio, score_gaze, DEFAULT_TOL_Y


def test_axis_ratio_centre_and_ends():
    assert axis_ratio(5, 0, 10) == 0.5
    assert axis_ratio(0, 0, 10) == 0.0
    assert axis_ratio(10, 0, 10) == 1.0


def test_axis_ratio_is_order_independent_and_clamped():
    assert axis_ratio(5, 10, 0) == 0.5
    assert axis_ratio(-5, 0, 10) == 0.0
    assert axis_ratio(99, 0, 10) == 1.0


def test_axis_ratio_zero_span_is_centre():
    assert axis_ratio(3, 3, 3) == 0.5


def test_no_frames_is_zero():
    assert score_gaze([]).score == 0.0


def test_no_face_visible_is_zero():
    assert score_gaze([{}, {}]).score == 0.0


def test_centred_gaze_scores_full():
    s = score_gaze([{"x": 0.5, "y": 0.5}] * 5)
    assert s.score == 10.0
    assert s.detail["forward_fraction"] == 1.0


def test_looking_down_at_notes_scores_low():
    # Head could be facing forward, but eyes are down (paper-reading).
    s = score_gaze([{"x": 0.5, "y": 0.85}] * 5)
    assert s.score == 0.0
    assert s.detail["down_fraction"] == 1.0


def test_looking_sideways_scores_low():
    s = score_gaze([{"x": 0.9, "y": 0.5}] * 5)
    assert s.score == 0.0


def test_partial_eye_contact_scores_between():
    frames = [{"x": 0.5, "y": 0.5}] * 3 + [{"x": 0.5, "y": 0.9}] * 2
    s = score_gaze(frames)
    assert s.score == 6.0  # 3 of 5 forward
    assert s.detail["frames_with_face"] == 5


def test_frames_without_face_are_not_counted():
    frames = [{"x": 0.5, "y": 0.5}, {}, {"x": 0.5, "y": 0.5}]
    s = score_gaze(frames)
    assert s.detail["frames_with_face"] == 2
    assert s.score == 10.0


def test_gaze_beats_head_pose_on_paper_reader():
    # The whole point: gaze flags a downward-looking reader that the pose
    # proxy (nose between eyes) would have passed.
    paper_reader = score_gaze([{"x": 0.5, "y": 0.5 + DEFAULT_TOL_Y + 0.1}] * 10)
    assert paper_reader.score < 3.0


def test_looking_up_is_not_penalised():
    # Iris in the upper portion of the eye (y < 0.5) means the speaker is
    # looking at the camera or screen above — still eye contact. Previously
    # this scored 0.0 because the symmetric Y gate failed below centre.
    # Calibrated against baselines/rW2r5uStgG0 whose gaze y values are all
    # 0.0-0.48 (upper eye) yet it is a clearly engaged speaker.
    looking_up = score_gaze([{"x": 0.5, "y": 0.2}] * 10)
    assert looking_up.score > 8.0
