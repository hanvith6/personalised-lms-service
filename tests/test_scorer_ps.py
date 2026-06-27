"""Hard-case tests for the real Public Speaking scorers."""
import numpy as np
import pytest

from utils.taxonomy import ScoreSource
from services.llm_json import LLMOutputError
from dimensions.public_speaking.scorer_ps import (
    score_eye_contact, score_posture, score_speech_pace,
    score_voice_stability, score_opening_closing,
)


def _frame(**kp):
    return {"keypoints": kp}


def _forward_frame():
    # nose between the eyes, all confident => facing camera
    return _frame(nose=(50, 30, 0.9), left_eye=(40, 25, 0.9), right_eye=(60, 25, 0.9))


# --- eye_contact ---------------------------------------------------------------

def test_eye_contact_no_person():
    s = score_eye_contact([])
    assert s.score == 0.0 and s.detail["reason"] == "no person"
    assert s.source is ScoreSource.POSE


def test_eye_contact_all_forward_high():
    assert score_eye_contact([_forward_frame()] * 5).score > 7


def test_eye_contact_turned_away_low():
    # eyes barely visible => no frame counts as facing
    turned = [_frame(nose=(50, 30, 0.2), left_eye=(40, 25, 0.2), right_eye=(60, 25, 0.2))] * 4
    assert score_eye_contact(turned).score < 4


def test_eye_contact_nose_outside_eyes_low():
    off = [_frame(nose=(90, 30, 0.9), left_eye=(40, 25, 0.9), right_eye=(60, 25, 0.9))] * 4
    assert score_eye_contact(off).score < 4


# --- posture -------------------------------------------------------------------

def test_posture_no_person():
    assert score_posture([]).score == 0.0


def test_posture_upright_high():
    good = _frame(
        left_shoulder=(40, 100, 0.9), right_shoulder=(60, 100, 0.9),
        left_hip=(42, 200, 0.9), right_hip=(58, 200, 0.9),
    )
    assert score_posture([good] * 3).score > 7


def test_posture_tilted_low():
    tilted = _frame(left_shoulder=(40, 100, 0.9), right_shoulder=(60, 140, 0.9))
    assert score_posture([tilted] * 3).score < 6


def test_posture_missing_hips_uses_shoulders():
    no_hips = _frame(left_shoulder=(40, 100, 0.9), right_shoulder=(60, 100, 0.9))
    s = score_posture([no_hips])
    assert s.score > 7  # level shoulders, no crash


# --- speech_pace ---------------------------------------------------------------

def test_pace_silence_zero():
    assert score_speech_pace(0, 60).score == 0.0
    assert score_speech_pace(10, 0).score == 0.0


def test_pace_ideal_high():
    assert score_speech_pace(130, 60).score > 8   # 130 wpm
    assert score_speech_pace(110, 60).score > 8   # 110 wpm (band edge)
    assert score_speech_pace(150, 60).score > 8   # 150 wpm (band edge)


def test_pace_too_fast_low():
    assert score_speech_pace(250, 60).score < 3   # 250 wpm


# --- voice_stability -----------------------------------------------------------

def test_voice_silence_zero():
    assert score_voice_stability(np.zeros(50), np.zeros(50)).score == 0.0


def test_voice_steady_high():
    f0 = np.full(50, 120.0)
    rms = np.full(50, 0.5)
    assert score_voice_stability(f0, rms).score > 8


def test_voice_unstable_low():
    rng = np.random.default_rng(0)
    f0 = np.abs(rng.normal(120, 80, 200)) + 1
    rms = np.abs(rng.normal(0.5, 0.6, 200)) + 1e-3
    assert score_voice_stability(f0, rms).score < 6


# --- opening_closing (LLM, injected) ------------------------------------------

class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_opening_closing_short_circuits_without_llm():
    fake = _FakeLLM(["{}"])
    s = score_opening_closing("too short", fake)
    assert s.score == 0.0
    assert fake.calls == 0  # LLM must NOT be called on a thin transcript


def test_opening_closing_valid():
    text = " ".join(["word"] * 40)
    fake = _FakeLLM(['{"score": 8.5, "justification": "Strong hook and call to action."}'])
    s = score_opening_closing(text, fake)
    assert s.score == 8.5
    assert "Strong hook" in s.detail["justification"]


def test_opening_closing_malformed_raises():
    text = " ".join(["word"] * 40)
    fake = _FakeLLM(["not json", "still not", "nope"])
    with pytest.raises(LLMOutputError):
        score_opening_closing(text, fake)
