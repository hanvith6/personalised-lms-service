"""Tests for the retrying LLM JSON parser."""
import pytest

from services.llm_json import parse_llm_json, LLMOutputError


def _gen(responses):
    state = {"i": 0}

    def generate(_prompt):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r

    return generate


def test_clean_json():
    assert parse_llm_json(_gen(['{"a": 1}']), "p") == {"a": 1}


def test_fenced_json():
    fenced = '```json\n{"a": 2}\n```'
    assert parse_llm_json(_gen([fenced]), "p") == {"a": 2}


def test_plain_fence_no_lang():
    fenced = '```\n[1, 2, 3]\n```'
    assert parse_llm_json(_gen([fenced]), "p") == [1, 2, 3]


def test_malformed_then_valid_retries():
    g = _gen(["garbage", '{"ok": true}'])
    assert parse_llm_json(g, "p", attempts=3) == {"ok": True}


def test_always_malformed_raises_after_attempts():
    calls = {"n": 0}

    def generate(_p):
        calls["n"] += 1
        return "nope"

    with pytest.raises(LLMOutputError):
        parse_llm_json(generate, "p", attempts=3)
    assert calls["n"] == 3


def test_validate_rejects_then_raises():
    def validate(obj):
        if "score" not in obj:
            raise ValueError("missing score")

    with pytest.raises(LLMOutputError):
        parse_llm_json(_gen(['{"x": 1}']), "p", attempts=2, validate=validate)


# --- Prose-wrapped output (real Qwen2.5-3B behaviour) -------------------------

def test_json_object_wrapped_in_prose():
    """Qwen often prepends/appends a sentence around the JSON object."""
    raw = 'Sure! Here is the rating:\n{"score": 7.5, "justification": "strong open"}\nHope this helps!'
    assert parse_llm_json(_gen([raw]), "p") == {"score": 7.5, "justification": "strong open"}


def test_json_array_wrapped_in_prose():
    raw = 'Here is your learning path: [{"title": "A"}, {"title": "B"}] — enjoy.'
    assert parse_llm_json(_gen([raw]), "p") == [{"title": "A"}, {"title": "B"}]


def test_prose_then_fenced_json():
    raw = 'Certainly.\n```json\n{"score": 9}\n```\nLet me know if you need more.'
    assert parse_llm_json(_gen([raw]), "p") == {"score": 9}


def test_braces_inside_string_not_miscounted():
    """A '}' inside a string value must not close the object early."""
    raw = 'Result: {"justification": "use a closing like }{ for emphasis", "score": 6}'
    out = parse_llm_json(_gen([raw]), "p")
    assert out["score"] == 6 and "}{" in out["justification"]


def test_prose_wrapped_passes_validate():
    def validate(obj):
        if "score" not in obj:
            raise ValueError("missing 'score'")

    raw = 'My assessment: {"score": 8, "justification": "clear arc"}. Thanks!'
    out = parse_llm_json(_gen([raw]), "p", attempts=3, validate=validate)
    assert out["score"] == 8


def test_skips_first_block_without_key_finds_later_one():
    """Qwen sometimes echoes the prompt object first, then the real answer."""
    def validate(obj):
        if "score" not in obj:
            raise ValueError("missing 'score'")

    raw = ('Echoing input: {"opening": "Good morning", "closing": "thank you"}\n'
           'My rating: {"score": 7, "justification": "solid"}')
    out = parse_llm_json(_gen([raw]), "p", attempts=3, validate=validate)
    assert out["score"] == 7
