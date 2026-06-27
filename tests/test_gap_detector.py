"""Tests for dimension-agnostic gap detection."""
from utils.taxonomy import SubSkillScore, ScoreSource
from services.gap_detector import detect_gaps, GAP_THRESHOLD


def _score(skill, value):
    return SubSkillScore(skill, value, ScoreSource.POSE)


def test_no_gaps_when_all_above_threshold():
    scores = [_score("a", 7.0), _score("b", 9.5), _score("c", 6.0)]  # 6.0 is NOT a gap
    assert detect_gaps(scores, {}) == []


def test_severity_and_priority_with_zero_boost():
    gaps = detect_gaps([_score("a", 4.0)], {"a": 0.0})
    assert len(gaps) == 1
    assert gaps[0].gap_severity == 2.0
    assert gaps[0].priority_score == 2.0


def test_boost_reorders_ranking():
    # small gap (severity 1) with high boost should outrank big gap (severity 3) with no boost.
    scores = [_score("small_highdemand", 5.0), _score("big_lowdemand", 3.0)]
    boosts = {"small_highdemand": 3.0, "big_lowdemand": 0.0}
    gaps = detect_gaps(scores, boosts)
    assert gaps[0].skill == "small_highdemand"   # priority 1*4=4 > 3*1=3
    assert gaps[1].skill == "big_lowdemand"


def test_threshold_boundary_excluded():
    assert detect_gaps([_score("a", GAP_THRESHOLD)], {}) == []
