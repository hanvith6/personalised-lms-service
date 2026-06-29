"""Tests for the A/B accuracy oracle (Module 9)."""
import pytest

from services.ab_test import compare_runs, assert_good_beats_bad


_BAD = {"eye_contact": 3.0, "posture": 4.0, "speech_pace": 4.0,
        "voice_stability": 3.5, "opening_closing_impact": 2.0}
_GOOD = {"eye_contact": 9.0, "posture": 8.5, "speech_pace": 8.0,
         "voice_stability": 7.5, "opening_closing_impact": 9.0}


def test_good_take_beats_bad_take_passes():
    r = compare_runs(_BAD, _GOOD)
    assert r["verdict"] == "PASS"
    assert r["n_improved"] == r["n_skills"] == 5
    assert r["mean_good"] > r["mean_bad"]
    assert r["mean_delta"] > 0


def test_per_skill_delta_and_direction():
    r = compare_runs(_BAD, _GOOD)
    assert r["per_skill"]["opening_closing_impact"]["delta"] == 7.0
    assert all(v["improved"] for v in r["per_skill"].values())


def test_average_key_is_ignored():
    bad = {**_BAD, "AVERAGE": 3.3}
    good = {**_GOOD, "AVERAGE": 8.4}
    r = compare_runs(bad, good)
    assert "AVERAGE" not in r["per_skill"]
    assert r["n_skills"] == 5


def test_only_common_skills_are_compared():
    bad = {"eye_contact": 3.0, "posture": 4.0}
    good = {"eye_contact": 9.0, "voice_stability": 8.0}  # posture absent in good
    r = compare_runs(bad, good)
    assert set(r["per_skill"]) == {"eye_contact"}


def test_no_common_skills_raises():
    with pytest.raises(ValueError):
        compare_runs({"eye_contact": 3.0}, {"posture": 8.0})


def test_regression_fails_verdict():
    # "good" take is actually worse — scorer should NOT pass it.
    r = compare_runs(_GOOD, _BAD)
    assert r["verdict"] == "FAIL"


def test_assert_helper_raises_on_failure():
    with pytest.raises(AssertionError, match="A/B oracle FAILED"):
        assert_good_beats_bad(_GOOD, _BAD)


def test_assert_helper_returns_result_on_success():
    r = assert_good_beats_bad(_BAD, _GOOD)
    assert r["verdict"] == "PASS"


def test_mixed_deltas_below_threshold_fails():
    # Only 1 of 5 sub-skills improves -> below the 60% threshold.
    bad = {"a": 5, "b": 5, "c": 5, "d": 5, "e": 5}
    good = {"a": 9, "b": 4, "c": 4, "d": 4, "e": 4}
    r = compare_runs(bad, good)
    assert r["verdict"] == "FAIL"
