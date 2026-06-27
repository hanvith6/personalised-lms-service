"""Real scorers for the Public Speaking dimension.

Module 9 §2.3-2.4, scoped to Public Speaking.

Design note: these functions deliberately do NOT import torch / ultralytics /
librosa / whisper. The heavy feature extraction (pose keypoints, audio f0/RMS,
transcript) happens in the Colab notebook; these functions accept the already-
extracted features. That keeps them pure, fast, and unit-testable on a laptop
with only numpy — and keeps the heavy stack off the local machine.
"""
from __future__ import annotations

import numpy as np

from utils.taxonomy import SubSkillScore, ScoreSource, clamp_score
from services.llm_json import parse_llm_json

# --- Pose helpers --------------------------------------------------------------

_CONF_MIN = 0.5  # keypoint confidence below this is treated as "not visible"


def _conf(keypoints: dict, name: str) -> float:
    kp = keypoints.get(name)
    return float(kp[2]) if kp else 0.0


def score_eye_contact(pose_frames) -> SubSkillScore:
    """Fraction of frames where the speaker faces the camera, scaled 0-10.

    Module 9 §2.3, scoped to Public Speaking. `pose_frames` is a list of
    ``{"keypoints": {name: (x, y, conf)}}`` using COCO names.
    """
    if not pose_frames:
        return SubSkillScore("eye_contact", 0.0, ScoreSource.POSE, {"reason": "no person"})

    facing = 0
    counted = 0
    for frame in pose_frames:
        kp = frame.get("keypoints", {})
        if _conf(kp, "nose") < _CONF_MIN or _conf(kp, "left_eye") < _CONF_MIN or _conf(kp, "right_eye") < _CONF_MIN:
            continue  # face not clearly visible this frame
        counted += 1
        nose_x = kp["nose"][0]
        lo, hi = sorted((kp["left_eye"][0], kp["right_eye"][0]))
        if lo <= nose_x <= hi:  # nose between the eyes => facing camera
            facing += 1

    if counted == 0:
        return SubSkillScore("eye_contact", 0.0, ScoreSource.POSE, {"reason": "face not visible"})

    frac = facing / counted
    return SubSkillScore(
        "eye_contact", clamp_score(frac * 10), ScoreSource.POSE,
        {"frames": len(pose_frames), "frames_with_face": counted, "facing_fraction": round(frac, 3)},
    )


def score_posture(pose_frames) -> SubSkillScore:
    """Shoulder-level symmetry blended with spine verticality, scaled 0-10.

    Module 9 §2.3, scoped to Public Speaking. Falls back to shoulder symmetry
    alone when the hips are not visible.
    """
    if not pose_frames:
        return SubSkillScore("posture", 0.0, ScoreSource.POSE, {"reason": "no person"})

    frame_scores: list[float] = []
    for frame in pose_frames:
        kp = frame.get("keypoints", {})
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if not ls or not rs or ls[2] < _CONF_MIN or rs[2] < _CONF_MIN:
            continue
        shoulder_w = abs(ls[0] - rs[0]) or 1.0
        level = 1.0 - min(1.0, abs(ls[1] - rs[1]) / shoulder_w)  # 1.0 = perfectly level

        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        if lh and rh and lh[2] >= _CONF_MIN and rh[2] >= _CONF_MIN:
            sh_mid_x, hip_mid_x = (ls[0] + rs[0]) / 2, (lh[0] + rh[0]) / 2
            sh_mid_y, hip_mid_y = (ls[1] + rs[1]) / 2, (lh[1] + rh[1]) / 2
            height = abs(sh_mid_y - hip_mid_y) or 1.0
            vert = 1.0 - min(1.0, abs(sh_mid_x - hip_mid_x) / height)  # 1.0 = vertical spine
            frame_scores.append(0.5 * level + 0.5 * vert)
        else:
            frame_scores.append(level)

    if not frame_scores:
        return SubSkillScore("posture", 0.0, ScoreSource.POSE, {"reason": "torso not visible"})

    avg = float(np.mean(frame_scores))
    return SubSkillScore("posture", clamp_score(avg * 10), ScoreSource.POSE,
                         {"frames_scored": len(frame_scores)})


# --- Audio scorers -------------------------------------------------------------

def score_speech_pace(word_count: int, duration_s: float) -> SubSkillScore:
    """Words-per-minute mapped to a delivery-quality band, scaled 0-10.

    Module 9 §2.3, scoped to Public Speaking. Full marks for ~115-145 wpm, with
    linear falloff to 0 at 70 wpm (too slow) and 190 wpm (too fast).
    """
    if word_count <= 0 or duration_s <= 0:
        return SubSkillScore("speech_pace", 0.0, ScoreSource.AUDIO, {"wpm": 0.0, "reason": "no speech"})

    wpm = word_count / (duration_s / 60.0)
    lo_ideal, hi_ideal = 115.0, 145.0
    if lo_ideal <= wpm <= hi_ideal:
        score = 10.0
    elif wpm < lo_ideal:
        score = 10.0 * (wpm - 70.0) / (lo_ideal - 70.0)
    else:
        score = 10.0 * (190.0 - wpm) / (190.0 - hi_ideal)
    return SubSkillScore("speech_pace", clamp_score(score), ScoreSource.AUDIO, {"wpm": round(wpm, 1)})


def score_voice_stability(f0_series, rms_series) -> SubSkillScore:
    """Inverse of pitch + energy variation, scaled 0-10 (steadier = higher).

    Module 9 §2.3, scoped to Public Speaking. `f0_series` is per-frame
    fundamental frequency (0 where unvoiced); `rms_series` is per-frame energy.
    """
    f0 = np.asarray(f0_series, dtype=float)
    rms = np.asarray(rms_series, dtype=float)
    f0_voiced = f0[f0 > 0]

    if f0_voiced.size == 0 or not np.any(rms > 1e-9):
        return SubSkillScore("voice_stability", 0.0, ScoreSource.AUDIO, {"reason": "silence"})

    def _cv(a):
        mean = float(np.mean(a))
        return float(np.std(a) / mean) if mean > 1e-9 else 1.0

    mean_cv = 0.5 * _cv(f0_voiced) + 0.5 * _cv(rms[rms > 1e-9])
    score = 10.0 * (1.0 - min(1.0, mean_cv))
    return SubSkillScore("voice_stability", clamp_score(score), ScoreSource.AUDIO, {"cv": round(mean_cv, 3)})


# --- LLM scorer ----------------------------------------------------------------

_MIN_WORDS = 12  # below this the transcript is too thin to judge impact


def _sanitize_segment(text: str, max_chars: int = 500) -> str:
    """Truncate and bracket untrusted transcript text to contain prompt injection."""
    return f"<transcript>{text[:max_chars]}</transcript>"


def _impact_prompt(opening: str, closing: str) -> str:
    return (
        "You are a public-speaking coach. Rate the IMPACT of a talk's opening and "
        "closing on a 0-10 scale (10 = memorable, audience-grabbing).\n"
        "Return ONLY a JSON object, no markdown, no prose: "
        '{"score": <number 0-10>, "justification": "<one sentence>"}\n\n'
        f"OPENING: {_sanitize_segment(opening)}\n\nCLOSING: {_sanitize_segment(closing)}"
    )


def score_opening_closing(transcript: str, llm_generate) -> SubSkillScore:
    """Rate opening + closing impact via an injected LLM.

    Module 9 §2.4, scoped to Public Speaking. Short transcripts short-circuit to
    0.0 without calling the LLM. `llm_generate` is a ``(prompt) -> str`` callable.
    """
    words = transcript.split()
    if len(words) < _MIN_WORDS:
        return SubSkillScore("opening_closing_impact", 0.0, ScoreSource.LLM,
                             {"reason": "transcript too short", "words": len(words)})

    quarter = max(1, len(words) // 4)
    opening = " ".join(words[:quarter])
    closing = " ".join(words[-quarter:])

    def _validate(obj):
        if not isinstance(obj, dict) or "score" not in obj:
            raise ValueError("missing 'score'")

    obj = parse_llm_json(llm_generate, _impact_prompt(opening, closing), attempts=3, validate=_validate)
    return SubSkillScore(
        "opening_closing_impact", clamp_score(float(obj["score"])), ScoreSource.LLM,
        {"justification": obj.get("justification", "")},
    )
