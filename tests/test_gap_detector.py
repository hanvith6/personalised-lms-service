"""Tests for dimension-agnostic gap detection."""
from utils.taxonomy import SubSkillScore, ScoreSource
from services.gap_detector import (
    detect_gaps, summarize_gaps, severity_band, GAP_THRESHOLD,
)


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


# --- severity bands -----------------------------------------------------------

def test_severity_band_thresholds():
    assert severity_band(4.0) == "CRITICAL"   # score 2.0
    assert severity_band(3.0) == "CRITICAL"   # score 3.0 (boundary)
    assert severity_band(2.0) == "HIGH"       # score 4.0
    assert severity_band(1.5) == "HIGH"       # boundary
    assert severity_band(0.5) == "MODERATE"   # score 5.5


# --- summarize_gaps -----------------------------------------------------------

def test_summarize_gaps_counts_and_bands():
    scores = [
        _score("crit", 2.0),   # severity 4.0 -> CRITICAL
        _score("high", 4.0),   # severity 2.0 -> HIGH
        _score("mod", 5.5),    # severity 0.5 -> MODERATE
        _score("ok", 8.0),     # not a gap
    ]
    gaps = detect_gaps(scores, {})
    summary = summarize_gaps(gaps)
    assert summary["total_gaps"] == 3
    assert summary["critical_gaps"] == 1
    assert summary["by_band"] == {"CRITICAL": 1, "HIGH": 1, "MODERATE": 1}
    assert summary["top_gaps"][0] == "crit"   # highest priority first


def test_summarize_empty():
    assert summarize_gaps([]) == {
        "total_gaps": 0, "critical_gaps": 0, "top_gaps": [],
        "by_band": {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0},
    }
