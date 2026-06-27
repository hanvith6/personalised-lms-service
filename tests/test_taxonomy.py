"""Tests for the cross-dimension contract and the PS taxonomy."""
import dataclasses

import pytest

from utils.taxonomy import SubSkillScore, ScoreSource, Gap, Resource, clamp_score
from dimensions.public_speaking.taxonomy_ps import PS_SUBSKILLS, DEMAND_BOOSTS

SEVEN = {"eye_contact", "posture", "speech_pace", "voice_stability",
         "opening_closing_impact", "slide_structure", "audience_engagement"}


def test_clamp_score_bounds():
    assert clamp_score(12) == 10.0
    assert clamp_score(-1) == 0.0
    assert clamp_score(5.5) == 5.5


def test_subskillscore_is_frozen():
    s = SubSkillScore("eye_contact", 7.0, ScoreSource.POSE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.score = 9.0  # type: ignore[misc]


def test_gap_and_resource_construct():
    g = Gap("posture", 4.0, 2.0, 2.8, "public_speaking")
    assert g.priority_score == 2.8
    r = Resource("id1", "t", "posture", "video", "beginner", "http://x", 10)
    assert r.resource_type == "video"


def test_ps_subskills_keys_and_sources():
    assert set(PS_SUBSKILLS) == SEVEN
    assert PS_SUBSKILLS["eye_contact"]["source"] is ScoreSource.POSE
    assert PS_SUBSKILLS["speech_pace"]["source"] is ScoreSource.AUDIO
    assert PS_SUBSKILLS["opening_closing_impact"]["source"] is ScoreSource.LLM
    assert PS_SUBSKILLS["slide_structure"]["source"] is ScoreSource.STUB
    assert PS_SUBSKILLS["audience_engagement"]["source"] is ScoreSource.STUB


def test_demand_boosts_in_range():
    assert set(DEMAND_BOOSTS) == SEVEN
    assert all(0.0 <= b <= 1.0 for b in DEMAND_BOOSTS.values())
