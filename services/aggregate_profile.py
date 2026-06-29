"""Aggregate several score readings into one reliable profile (Module 9).

A single video is a noisy snapshot. A student's true skill is better estimated
by the *mean across several readings* — multiple videos of the same student, or
(with option C) the several people scored within one classroom video.

This module holds the math that previously lived only inside the notebook's
Step 13 cell, so it is unit-tested and reusable by the notebook and the API.
Pure-python (stdlib ``statistics``); no model imports.
"""
from __future__ import annotations

import statistics

from utils.taxonomy import ScoreSource, SubSkillScore

# Spread (population std) thresholds that label how trustworthy a single
# reading is. Wider spread across readings => trust one reading less.
RELIABILITY_STABLE_MAX = 0.75
RELIABILITY_NOISY_MAX = 1.5


def reliability_flag(std: float) -> str:
    """Map a sub-skill's across-reading spread to a reliability label."""
    if std < RELIABILITY_STABLE_MAX:
        return "stable"
    if std < RELIABILITY_NOISY_MAX:
        return "noisy"
    return "volatile"


def aggregate_scores(
    runs: list[dict[str, float]],
    source: ScoreSource = ScoreSource.STUB,
) -> dict:
    """Combine per-reading ``{skill: score}`` dicts into a stable profile.

    ``runs`` is a list of readings (one dict per video or per person). Keys that
    are not real sub-skills (e.g. an ``"AVERAGE"`` summary column) are ignored.

    Returns a dict with:
      - ``per_skill``: ``{skill: {mean, std, min, max, n, flag}}``
      - ``scores``:    ``list[SubSkillScore]`` carrying the mean per skill
                       (ready to feed ``gap_detector.detect_gaps``)
      - ``noisiest``:  the skill with the widest spread (or ``None``)
      - ``n_runs``:    how many readings were aggregated

    Raises ``ValueError`` when fewer than two readings are supplied — a single
    reading cannot produce a reliability estimate.
    """
    if len(runs) < 2:
        raise ValueError("aggregation needs at least 2 readings")

    skills = [k for k in runs[0] if k != "AVERAGE"]
    per_skill: dict[str, dict] = {}
    scores: list[SubSkillScore] = []

    for sk in skills:
        vals = [r[sk] for r in runs if sk in r]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        per_skill[sk] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": len(vals),
            "flag": reliability_flag(std),
        }
        scores.append(SubSkillScore(
            sk, mean, source, {"n": len(vals), "std": round(std, 2)}))

    noisiest = max(per_skill, key=lambda s: per_skill[s]["std"]) if per_skill else None
    return {"per_skill": per_skill, "scores": scores,
            "noisiest": noisiest, "n_runs": len(runs)}
