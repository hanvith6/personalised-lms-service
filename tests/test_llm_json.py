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
