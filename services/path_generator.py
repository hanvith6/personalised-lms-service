"""Dimension-agnostic learning-path generation + adaptive sequencing.

Module 9 §3.3-3.4, dimension-agnostic.
"""
from __future__ import annotations

import copy
from collections import Counter

from services.llm_json import parse_llm_json  # noqa: F401  (re-exported for callers)

_RESOURCE_TYPES = {"video", "article", "exercise", "quiz", "practice"}
_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
_TIER = ["beginner", "intermediate", "advanced"]
_REQUIRED = {
    "title", "dimension", "skill_addressed", "resource_type", "difficulty",
    "estimated_minutes", "reason", "prerequisite_step_index",
}

# Adaptive-sequencing thresholds (percent of max score on a completion event).
_ACCELERATE_PCT = 85
_PREREQ_PCT = 50
_PREREQ_REPEATS = 2


def _validate_steps(obj) -> None:
    """Reject anything that is not a non-empty array of well-formed steps."""
    if not isinstance(obj, list) or not obj:
        raise ValueError("expected a non-empty JSON array")
    for i, step in enumerate(obj):
        if not isinstance(step, dict) or not _REQUIRED.issubset(step):
            raise ValueError(f"step {i} missing required fields")
        if step["resource_type"] not in _RESOURCE_TYPES:
            raise ValueError(f"step {i} has invalid resource_type")
        if step["difficulty"] not in _DIFFICULTIES:
            raise ValueError(f"step {i} has invalid difficulty")


def _build_prompt(target_role, top_gaps, avg_completion_hours, pace, preferred_resource_types) -> str:
    lines = [
        "You are a learning-path designer. Return ONLY a JSON array, no markdown, no prose.",
        f"Target role: {target_role}. Learner pace: {pace}. "
        f"Average completion hours available: {avg_completion_hours}.",
        f"Preferred resource types: {', '.join(preferred_resource_types)}.",
        "Address these skill gaps (skill: current score out of 10):",
    ]
    for g in top_gaps:
        lines.append(f"- {g.skill}: {g.score}/10 (priority {g.priority_score})")
    lines.append(
        "Each array item is an object with keys: title, dimension, skill_addressed, "
        "resource_type (video/article/exercise/quiz/practice), difficulty "
        "(beginner/intermediate/advanced), estimated_minutes (integer), reason "
        "(one sentence tied to the learner's actual score), prerequisite_step_index "
        "(null or integer index of a step that must come first)."
    )
    return "\n".join(lines)


def generate_learning_path(target_role, top_gaps, avg_completion_hours, pace,
                           preferred_resource_types, llm_generate) -> list[dict]:
    """Generate an ordered learning path via an injected LLM.

    Module 9 §3.3, dimension-agnostic. Wraps the LLM call in a 3-attempt
    parse-and-validate loop; raises LLMOutputError if it never returns valid JSON.
    """
    prompt = _build_prompt(target_role, top_gaps, avg_completion_hours, pace, preferred_resource_types)
    return parse_llm_json(llm_generate, prompt, attempts=3, validate=_validate_steps)


def adapt_path(path, completion_events) -> list[dict]:
    """Re-sequence a path from completion events. Pure: never mutates input.

    Module 9 §3.4, dimension-agnostic.
      * a completed step scoring > 85% bumps the NEXT step up one difficulty tier;
      * a skill scoring < 50% on two events gets a simpler prerequisite inserted before it.
    """
    new = copy.deepcopy(path)

    # Accelerate: bump the difficulty of the step after a high-scoring completion.
    for ev in completion_events:
        if ev.get("score_pct", 0) > _ACCELERATE_PCT:
            idx = ev.get("step_index")
            if idx is not None and 0 <= idx + 1 < len(new):
                cur = new[idx + 1].get("difficulty")
                if cur in _TIER:
                    bumped = _TIER[min(_TIER.index(cur) + 1, len(_TIER) - 1)]
                    new[idx + 1] = {**new[idx + 1], "difficulty": bumped}

    # Insert prerequisite: skills that struggled (<50%) on two events.
    struggled = Counter(
        ev["skill"] for ev in completion_events
        if "skill" in ev and ev.get("score_pct", 100) < _PREREQ_PCT
    )
    inserted: set = set()
    result: list[dict] = []
    for step in new:
        skill = step.get("skill_addressed")
        if skill in struggled and struggled[skill] >= _PREREQ_REPEATS and skill not in inserted:
            result.append({
                "title": f"Prerequisite primer: {skill}",
                "dimension": step.get("dimension", "public_speaking"),
                "skill_addressed": skill,
                "resource_type": "exercise",
                "difficulty": "beginner",
                "estimated_minutes": 15,
                "reason": f"Inserted because {skill} scored below {_PREREQ_PCT}% on two attempts.",
                "prerequisite_step_index": None,
            })
            inserted.add(skill)
        result.append(step)
    return result
