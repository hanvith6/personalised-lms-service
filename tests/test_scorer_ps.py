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


def test_posture_swaying_scores_below_steady():
    # Same upright alignment; one speaker holds still, one drifts side to side.
    steady = [_frame(left_shoulder=(40, 100, 0.9), right_shoulder=(60, 100, 0.9),
                     left_hip=(42, 200, 0.9), right_hip=(58, 200, 0.9))] * 6
    swaying = [_frame(left_shoulder=(40 + 12 * i, 100, 0.9),
                      right_shoulder=(60 + 12 * i, 100, 0.9),
                      left_hip=(42 + 12 * i, 200, 0.9),
                      right_hip=(58 + 12 * i, 200, 0.9))
               for i in (0, 1, 0, -1, 0, 1)]
    steady_score = score_posture(steady).score
    sway_score = score_posture(swaying).score
    assert steady_score > sway_score
    assert score_posture(swaying).detail["sway"] > 0


# --- speech_pace ---------------------------------------------------------------

def test_pace_silence_zero():
    assert score_speech_pace(0, 60).score == 0.0
    assert score_speech_pace(10, 0).score == 0.0


def test_pace_ideal_high():
    assert score_speech_pace(130, 60).score > 8   # 130 wpm
    assert score_speech_pace(110, 60).score > 8   # 110 wpm (band edge)
    assert score_speech_pace(150, 60).score > 8   # 150 wpm (band edge)


def test_pace_too_fast_low():
    assert score_speech_pace(250, 60).score < 3   # 250 wpm (auctioneer)


def test_pace_fast_expressive_partial_credit():
    # A fast-but-intelligible "expressive" speaker (~190 wpm) must not score
    # like silence. Calibrated against baselines/Z0LhdQIhmzk (labeled
    # "fast+expressive"), which previously collapsed to 0.0.
    s190 = score_speech_pace(190, 60).score
    assert 4.0 < s190 < 7.0          # fast, but clearly passing — not zero
    assert s190 > score_speech_pace(0, 60).score   # strictly above silence


# --- voice_stability -----------------------------------------------------------

def test_voice_silence_zero():
    assert score_voice_stability(np.zeros(50), np.zeros(50)).score == 0.0


def test_voice_steady_high():
    # Fully voiced + constant energy -> max score
    f0 = np.full(50, 120.0)
    rms = np.full(50, 0.5)
    assert score_voice_stability(f0, rms).score > 8


def test_voice_expressive_pitch_not_penalised():
    # Expressive pitch variation (natural speech) should NOT drop below 6
    rng = np.random.default_rng(1)
    f0 = np.abs(rng.normal(150, 60, 200)) + 50   # wide pitch range — expressive
    rms = np.full(200, 0.4)                        # but constant energy
    s = score_voice_stability(f0, rms)
    assert s.score > 6, f"expressive speaker unfairly penalised: {s.score}"


def test_voice_low_voiced_ratio_scores_lower():
    # Mostly silent / mumbling (10% voiced) + constant energy -> low-medium score
    f0 = np.zeros(200)
    f0[:20] = 120.0          # only 10% voiced
    rms = np.full(200, 0.4)
    s = score_voice_stability(f0, rms)
    assert s.score < 6


def test_voice_wild_energy_scores_below_steady():
    # Whisper→yell swing should score meaningfully lower than a steady speaker
    f0 = np.full(200, 120.0)
    rms_wild = np.concatenate([np.full(100, 0.01), np.full(100, 5.0)])
    rms_steady = np.full(200, 0.5)
    wild  = score_voice_stability(f0, rms_wild).score
    steady = score_voice_stability(f0, rms_steady).score
    assert wild < steady
    assert wild < 7    # penalised, not catastrophic


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


# --- tolerance for how small models (Qwen2.5-3B) actually phrase the score ---

_TEXT = " ".join(["word"] * 40)


def test_opening_closing_alternate_key_rating():
    fake = _FakeLLM(['{"rating": 6, "justification": "Decent open, weak close."}'])
    assert score_opening_closing(_TEXT, fake).score == 6.0


def test_opening_closing_cased_key():
    fake = _FakeLLM(['{"Score": 9, "justification": "Memorable."}'])
    assert score_opening_closing(_TEXT, fake).score == 9.0


def test_opening_closing_string_score_with_slash():
    fake = _FakeLLM(['{"score": "7/10", "justification": "Good."}'])
    assert score_opening_closing(_TEXT, fake).score == 7.0


def test_opening_closing_nested_opening_closing_averaged():
    fake = _FakeLLM(['{"opening": {"score": 6}, "closing": {"score": 8}}'])
    assert score_opening_closing(_TEXT, fake).score == 7.0


def test_opening_closing_prose_fallback():
    # No JSON at all for 3 tries, then a prose number on the 4th call.
    fake = _FakeLLM(["no json here", "still none", "nope",
                     "I would give this a 7 out of 10 overall."])
    s = score_opening_closing(_TEXT, fake)
    assert s.score == 7.0


def test_opening_closing_clamps_out_of_range():
    fake = _FakeLLM(['{"score": 15}'])
    assert score_opening_closing(_TEXT, fake).score == 10.0
