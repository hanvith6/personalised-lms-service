"""Tests for the real audience_engagement and slide_structure scorers (Module 9).

These replace the old mocks: real signals (audio reactions, on-screen slides),
and an honest not-applicable result instead of a fabricated number when the
signal is absent.
"""
from dimensions.public_speaking.scorer_ps import (
    score_audience_engagement, score_slide_structure)
from utils.taxonomy import ScoreSource


# --- audience_engagement ------------------------------------------------------

def test_engagement_no_audio_is_zero():
    s = score_audience_engagement(0, 0.0, 0.0)
    assert s.score == 0.0
    assert s.detail["reason"] == "no audio"


def test_engagement_no_reactions_scores_zero_not_mock():
    s = score_audience_engagement(0, 0.0, 180.0)
    assert s.score == 0.0
    assert s.source == ScoreSource.AUDIO
    assert s.detail["reaction_events"] == 0


def test_engagement_frequent_reactions_score_high():
    # 6 reactions + 12s of them over a 3-min talk -> strong engagement.
    s = score_audience_engagement(6, 12.0, 180.0)
    assert s.score > 8.0
    assert s.detail["reactions_per_min"] == 2.0


def test_engagement_is_monotonic_in_reactions():
    low = score_audience_engagement(1, 2.0, 180.0).score
    high = score_audience_engagement(5, 10.0, 180.0).score
    assert high > low


# --- slide_structure ----------------------------------------------------------

def test_slides_absent_is_not_applicable():
    # Speaker-only talk: slides in <30% of frames -> not applicable, excluded.
    s = score_slide_structure(frames_with_slide=2, total_frames=100,
                              n_transitions=0, mean_text_fill=0.0)
    assert s.detail["applicable"] is False
    assert s.source == ScoreSource.VISION


def test_slides_present_with_good_text_density_scores_high():
    s = score_slide_structure(frames_with_slide=80, total_frames=100,
                              n_transitions=10, mean_text_fill=0.22)
    assert s.detail["applicable"] is True
    assert s.score > 9.0


def test_slides_text_wall_scores_low():
    # Slides present but crammed with text (fill ~0.9) -> poor structure.
    s = score_slide_structure(frames_with_slide=80, total_frames=100,
                              n_transitions=5, mean_text_fill=0.9)
    assert s.detail["applicable"] is True
    assert s.score < 4.0


def test_slides_empty_scores_low():
    s = score_slide_structure(frames_with_slide=80, total_frames=100,
                              n_transitions=5, mean_text_fill=0.0)
    assert s.detail["applicable"] is True
    assert s.score < 4.0


def test_zero_total_frames_is_not_applicable():
    s = score_slide_structure(0, 0, 0, 0.0)
    assert s.detail["applicable"] is False
