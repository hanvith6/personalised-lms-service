"""Gaze-based eye-contact scoring (Module 9, accuracy upgrade for eye_contact).

The pose-based ``score_eye_contact`` only checks head orientation ("nose
between the eyes") — a coarse proxy that a paper-reading speaker still passes,
because the head can face forward while the eyes look down. This module scores
*actual gaze* from iris position relative to the eye corners and eyelids
(e.g. MediaPipe FaceMesh with iris refinement), so:

  - looking left/right  -> horizontal ratio drifts from centre
  - looking DOWN at notes -> vertical ratio drifts below centre

Pure geometry — no MediaPipe import here, so it is unit-tested without the
model. The notebook extracts the landmarks and feeds per-frame ratios in.
"""
from __future__ import annotations

from utils.taxonomy import ScoreSource, SubSkillScore, clamp_score

# How far the iris may sit from eye-centre (as a fraction of the eye span) and
# still count as eye contact. Horizontal eyes move more freely than vertical.
DEFAULT_TOL_X = 0.22
DEFAULT_TOL_Y = 0.22


def axis_ratio(p: float, a: float, b: float) -> float:
    """Position of ``p`` between landmarks ``a`` and ``b``, clamped to [0, 1].
    0.5 means centred. Order-independent."""
    lo, hi = (a, b) if a <= b else (b, a)
    span = hi - lo
    if span == 0:
        return 0.5
    return max(0.0, min(1.0, (p - lo) / span))


def score_gaze(
    frames: list[dict],
    tol_x: float = DEFAULT_TOL_X,
    tol_y: float = DEFAULT_TOL_Y,
) -> SubSkillScore:
    """Fraction of face-visible frames where gaze is directed forward, 0-10.

    Each frame is ``{"x": ratio, "y": ratio}`` (0.5 = centred; ``y`` > 0.5 means
    looking down) or ``{}`` when no face was detected that frame. Returns an
    ``eye_contact`` score so it is a drop-in replacement for the pose proxy when
    iris landmarks are available.
    """
    if not frames:
        return SubSkillScore("eye_contact", 0.0, ScoreSource.POSE, {"reason": "no person"})

    forward = 0
    counted = 0
    looking_down = 0
    for f in frames:
        if "x" not in f or "y" not in f:
            continue
        counted += 1
        x_ok = abs(f["x"] - 0.5) <= tol_x
        # Y: only penalise downward gaze (at notes/floor). Looking up at the
        # camera or screen is still eye contact — symmetric Y gate is wrong.
        down = f["y"] > 0.5 + tol_y
        if down:
            looking_down += 1
        if x_ok and not down:
            forward += 1

    if counted == 0:
        return SubSkillScore("eye_contact", 0.0, ScoreSource.POSE, {"reason": "face not visible"})

    frac = forward / counted
    return SubSkillScore(
        "eye_contact", clamp_score(frac * 10), ScoreSource.POSE,
        {"method": "gaze", "frames_with_face": counted,
         "forward_fraction": round(frac, 3),
         "down_fraction": round(looking_down / counted, 3)})
