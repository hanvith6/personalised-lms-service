"""Robust JSON extraction from small-LLM output.

Module 9 §3.2, dimension-agnostic. Small instruct models (e.g. Qwen2.5-3B)
occasionally wrap JSON in markdown fences or trail a sentence, so we strip
fences, json.loads, optionally validate, and retry up to `attempts` times.

SECURITY: we NEVER eval/exec model output — JSON parsing only.
"""
from __future__ import annotations

import json
import re

# Matches a leading ```/```json fence or a trailing ``` fence.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class LLMOutputError(Exception):
    """Raised when the LLM never returns parseable, valid JSON."""


def _strip(raw: str) -> str:
    """Remove surrounding markdown fences and whitespace."""
    return _FENCE.sub("", raw.strip()).strip()


def parse_llm_json(generate, prompt, attempts: int = 3, validate=None):
    """Call ``generate(prompt)`` and parse strict JSON, retrying on failure.

    Module 9 §3.2, dimension-agnostic.

    Args:
        generate: callable ``(prompt: str) -> str`` (the injected LLM).
        prompt: the prompt to send each attempt.
        attempts: max attempts before giving up (default 3).
        validate: optional ``(obj) -> None`` that raises if the object is invalid.

    Returns:
        The parsed JSON object.

    Raises:
        LLMOutputError: if no attempt yields parseable, valid JSON.
    """
    last_error = None
    for _ in range(attempts):
        raw = generate(prompt)
        try:
            obj = json.loads(_strip(raw))
            if validate is not None:
                validate(obj)
            return obj
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc
            continue
    raise LLMOutputError(
        f"failed to parse valid JSON after {attempts} attempts: {last_error}"
    )
