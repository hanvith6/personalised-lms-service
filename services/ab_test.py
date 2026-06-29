"""A/B accuracy oracle for the Public Speaking scorer (Module 9).

The cheapest ground truth available without hand-labelling: take the *same
speaker* delivering the *same content* once poorly and once well (e.g. the
"Good Presentation VS Bad Presentation" clip, or two takes of one student).
A trustworthy scorer must rank the good take above the bad take. If it does
not, the metric does not track reality — that is the signal this module exists
to surface.

Pure-python, no model imports — unit-testable without a GPU.
"""
from __future__ import annotations

# Fraction of comparable sub-skills that must improve for the oracle to PASS.
DEFAULT_MIN_IMPROVED_FRACTION = 0.6


def compare_runs(run_bad: dict[str, float], run_good: dict[str, float]) -> dict:
    """Compare a known-bad reading against a known-good reading of one speaker.

    Each run is ``{skill: score}``. The ``"AVERAGE"`` summary key, if present,
    is ignored for the per-skill breakdown. Returns:

      - ``per_skill``: ``{skill: {bad, good, delta, improved}}`` where
        ``delta = good - bad`` and ``improved`` is ``delta > 0``
      - ``mean_bad`` / ``mean_good`` / ``mean_delta`` over comparable skills
      - ``n_improved`` / ``n_skills``
      - ``verdict``: ``"PASS"`` when the good take beats the bad take overall
        and a majority of sub-skills improve, else ``"FAIL"``
    """
    skills = [k for k in run_bad if k != "AVERAGE" and k in run_good]
    if not skills:
        raise ValueError("no common sub-skills to compare")

    per_skill: dict[str, dict] = {}
    for sk in skills:
        bad, good = run_bad[sk], run_good[sk]
        delta = good - bad
        per_skill[sk] = {
            "bad": round(bad, 2),
            "good": round(good, 2),
            "delta": round(delta, 2),
            "improved": delta > 0,
        }

    n = len(skills)
    n_improved = sum(1 for v in per_skill.values() if v["improved"])
    mean_bad = sum(run_bad[s] for s in skills) / n
    mean_good = sum(run_good[s] for s in skills) / n
    passed = (mean_good > mean_bad
              and n_improved / n >= DEFAULT_MIN_IMPROVED_FRACTION)

    return {
        "per_skill": per_skill,
        "mean_bad": round(mean_bad, 2),
        "mean_good": round(mean_good, 2),
        "mean_delta": round(mean_good - mean_bad, 2),
        "n_improved": n_improved,
        "n_skills": n,
        "verdict": "PASS" if passed else "FAIL",
    }


def assert_good_beats_bad(
    run_bad: dict[str, float],
    run_good: dict[str, float],
    min_improved_fraction: float = DEFAULT_MIN_IMPROVED_FRACTION,
) -> dict:
    """Run the oracle and raise ``AssertionError`` when the scorer fails to
    rank the good take above the bad take. Returns the comparison on success."""
    result = compare_runs(run_bad, run_good)
    frac = result["n_improved"] / result["n_skills"]
    if result["mean_good"] <= result["mean_bad"] or frac < min_improved_fraction:
        raise AssertionError(
            f"A/B oracle FAILED: good={result['mean_good']} bad={result['mean_bad']}, "
            f"{result['n_improved']}/{result['n_skills']} sub-skills improved "
            f"(need >= {min_improved_fraction:.0%}). The scorer does not track "
            f"the quality delta — investigate before trusting the metric.")
    return result
