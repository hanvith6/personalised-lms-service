"""Tests for the multi-reading aggregation service (Module 9, Step 13)."""
import pytest

from services.aggregate_profile import (
    aggregate_scores,
    reliability_flag,
    RELIABILITY_STABLE_MAX,
    RELIABILITY_NOISY_MAX,
)
from services.gap_detector import detect_gaps
from dimensions.public_speaking.taxonomy_ps import DEMAND_BOOSTS


def test_reliability_flag_bands():
    assert reliability_flag(0.0) == "stable"
    assert reliability_flag(RELIABILITY_STABLE_MAX) == "noisy"
    assert reliability_flag(RELIABILITY_NOISY_MAX) == "volatile"


def test_single_reading_is_rejected():
    with pytest.raises(ValueError):
        aggregate_scores([{"eye_contact": 8.0}])


def test_mean_is_computed_per_skill():
    runs = [{"eye_contact": 8.0, "posture": 6.0},
            {"eye_contact": 6.0, "posture": 4.0}]
    agg = aggregate_scores(runs)
    assert agg["per_skill"]["eye_contact"]["mean"] == 7.0
    assert agg["per_skill"]["posture"]["mean"] == 5.0
    assert agg["n_runs"] == 2


def test_average_column_is_ignored():
    runs = [{"eye_contact": 8.0, "AVERAGE": 8.0},
            {"eye_contact": 6.0, "AVERAGE": 6.0}]
    agg = aggregate_scores(runs)
    assert set(agg["per_skill"]) == {"eye_contact"}


def test_noisiest_skill_is_the_widest_spread():
    runs = [{"eye_contact": 9.0, "voice_stability": 2.0},
            {"eye_contact": 8.8, "voice_stability": 9.0}]
    agg = aggregate_scores(runs)
    assert agg["noisiest"] == "voice_stability"
    assert agg["per_skill"]["eye_contact"]["flag"] == "stable"
    assert agg["per_skill"]["voice_stability"]["flag"] == "volatile"


def test_missing_skill_in_some_runs_uses_available_values():
    runs = [{"eye_contact": 8.0, "posture": 6.0},
            {"eye_contact": 6.0}]  # posture missing here
    agg = aggregate_scores(runs)
    assert agg["per_skill"]["posture"]["n"] == 1
    assert agg["per_skill"]["eye_contact"]["n"] == 2


def test_aggregated_scores_feed_gap_detection():
    # Good vs bad A/B readings of one student -> stable profile -> gaps.
    runs = [
        {"eye_contact": 3.0, "posture": 4.0, "speech_pace": 3.5,
         "voice_stability": 3.0, "opening_closing_impact": 2.0,
         "slide_structure": 8.0, "audience_engagement": 8.0},
        {"eye_contact": 9.0, "posture": 9.0, "speech_pace": 8.5,
         "voice_stability": 8.0, "opening_closing_impact": 9.0,
         "slide_structure": 8.0, "audience_engagement": 8.0},
    ]
    agg = aggregate_scores(runs)
    gaps = detect_gaps(agg["scores"], DEMAND_BOOSTS)
    gap_skills = {g.skill for g in gaps}
    # Means of the weak sub-skills (~3-5.5) fall below GAP_THRESHOLD=6.0.
    assert "opening_closing_impact" in gap_skills
    assert "voice_stability" in gap_skills
    # Consistently strong sub-skills are not flagged.
    assert "slide_structure" not in gap_skills
