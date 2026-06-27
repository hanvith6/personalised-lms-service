"""Tests for the module stubs, resource library, and analytics emit."""
from utils.taxonomy import ScoreSource, Gap
from dimensions.public_speaking.stubs.module4_stub import stub_slide_structure
from dimensions.public_speaking.stubs.module8_stub import stub_audience_engagement
from dimensions.public_speaking.resource_seed_ps import PS_RESOURCES
from services.analytics_emit import emit_gap_profile_updated

SEVEN = {"eye_contact", "posture", "speech_pace", "voice_stability",
         "opening_closing_impact", "slide_structure", "audience_engagement"}


def test_stub_slide_structure_deterministic_and_stub():
    a = stub_slide_structure("stu-123")
    b = stub_slide_structure("stu-123")
    assert a.score == b.score
    assert 0.0 <= a.score <= 10.0
    assert a.source is ScoreSource.STUB
    assert a.detail["mock"] is True


def test_stub_audience_engagement_differs_by_salt():
    s4 = stub_slide_structure("stu-123").score
    s8 = stub_audience_engagement("stu-123").score
    # Same id, different module salt -> almost always different score.
    assert stub_audience_engagement("stu-123").score == s8
    assert isinstance(s8, float)
    assert (s4, s8) == (s4, s8)


def test_resource_library_well_formed():
    assert 12 <= len(PS_RESOURCES) <= 15
    ids = [r.id for r in PS_RESOURCES]
    assert len(ids) == len(set(ids))  # unique ids
    for r in PS_RESOURCES:
        assert r.skill in SEVEN
        assert r.resource_type in {"video", "article", "exercise", "quiz", "practice"}
        assert r.difficulty in {"beginner", "intermediate", "advanced"}
        assert r.url.startswith("http")


def test_emit_returns_and_prints_payload():
    captured = []
    gaps = [Gap("posture", 4.0, 2.0, 2.8, "public_speaking")]
    payload = emit_gap_profile_updated("stu-9", gaps, _print=captured.append)
    assert payload["event"] == "gap_profile_updated"
    assert payload["student_id"] == "stu-9"
    assert payload["gaps"][0]["skill"] == "posture"
    assert len(captured) == 1
