"""Tests for learning-path generation, schema validation, and adaptive sequencing."""
import json

import pytest

from utils.taxonomy import Gap
from services.llm_json import LLMOutputError
from services.path_generator import generate_learning_path, adapt_path


def _step(skill="eye_contact", difficulty="beginner"):
    return {
        "title": f"Improve {skill}",
        "dimension": "public_speaking",
        "skill_addressed": skill,
        "resource_type": "video",
        "difficulty": difficulty,
        "estimated_minutes": 20,
        "reason": "Tied to a low score.",
        "prerequisite_step_index": None,
    }


def _gen(responses):
    state = {"i": 0}

    def generate(prompt):
        state["last_prompt"] = prompt
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r

    generate.state = state
    return generate


GAPS = [Gap("eye_contact", 4.0, 2.0, 2.8, "public_speaking"),
        Gap("posture", 3.0, 3.0, 4.2, "public_speaking")]


def test_generate_valid_path():
    payload = json.dumps([_step("eye_contact"), _step("posture")])
    steps = generate_learning_path("Data Analyst", GAPS, 5, "moderate", ["video"], _gen([payload]))
    assert len(steps) == 2
    assert steps[0]["skill_addressed"] == "eye_contact"


def test_prompt_includes_role_and_scores():
    payload = json.dumps([_step()])
    g = _gen([payload])
    generate_learning_path("Data Analyst", GAPS, 5, "moderate", ["video"], g)
    prompt = g.state["last_prompt"]
    assert "Data Analyst" in prompt
    assert "eye_contact: 4.0/10" in prompt


def test_generate_fenced_then_valid_retry():
    bad = "here you go:"
    good = "```json\n" + json.dumps([_step()]) + "\n```"
    steps = generate_learning_path("X", GAPS, 5, "fast", ["video"], _gen([bad, good]))
    assert len(steps) == 1


def test_generate_always_malformed_raises():
    with pytest.raises(LLMOutputError):
        generate_learning_path("X", GAPS, 5, "slow", ["video"], _gen(["nope", "nope", "nope"]))


def test_generate_rejects_bad_schema():
    broken = json.dumps([{"title": "missing fields"}])
    with pytest.raises(LLMOutputError):
        generate_learning_path("X", GAPS, 5, "slow", ["video"], _gen([broken, broken, broken]))


def test_generate_rejects_bad_difficulty():
    bad_step = _step()
    bad_step["difficulty"] = "expert"  # not in the allowed set
    payload = json.dumps([bad_step])
    with pytest.raises(LLMOutputError):
        generate_learning_path("X", GAPS, 5, "slow", ["video"], _gen([payload, payload, payload]))


# --- adapt_path ----------------------------------------------------------------

def test_adapt_accelerates_next_step():
    path = [_step("eye_contact", "beginner"), _step("posture", "beginner")]
    events = [{"step_index": 0, "skill": "eye_contact", "score_pct": 92}]
    out = adapt_path(path, events)
    assert out[1]["difficulty"] == "intermediate"   # next step bumped
    assert path[1]["difficulty"] == "beginner"        # input not mutated


def test_adapt_inserts_prerequisite_after_two_low_scores():
    path = [_step("posture", "intermediate")]
    events = [
        {"skill": "posture", "score_pct": 40},
        {"skill": "posture", "score_pct": 45},
    ]
    out = adapt_path(path, events)
    assert out[0]["title"].startswith("Prerequisite primer")
    assert out[0]["difficulty"] == "beginner"
    assert out[1]["skill_addressed"] == "posture"


def test_adapt_no_change_when_thresholds_not_met():
    path = [_step("eye_contact", "beginner"), _step("posture", "beginner")]
    events = [{"step_index": 0, "skill": "eye_contact", "score_pct": 70}]
    assert adapt_path(path, events) == path
