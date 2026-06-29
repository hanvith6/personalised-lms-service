"""Robust JSON extraction from small-LLM output.

Module 9 §3.2, dimension-agnostic. Small instruct models (e.g. Qwen2.5-3B)
wrap JSON in markdown fences OR in surrounding prose ("Here is the rating: {...}.
Hope this helps!"). We try, in order: strip fences + parse, then extract the
first balanced {...}/[...] block from anywhere in the text and parse that.

SECURITY:
  * We NEVER eval/exec model output — JSON parsing only.
  * Block extraction is a single linear character scan (no regex backtracking),
    so it is inherently ReDoS-safe even on adversarial output.
"""
from __future__ import annotations

import json
import re

# \A/\Z are absolute anchors (unaffected by re.MULTILINE), preventing ReDoS
# on adversarial LLM output that repeats fence-like sequences.
_FENCE = re.compile(r"\A\s*```(?:json)?\s*|\s*```\s*\Z", re.IGNORECASE)


class LLMOutputError(Exception):
    """Raised when the LLM never returns parseable, valid JSON."""


def _strip(raw: str) -> str:
    """Remove surrounding markdown fences and whitespace."""
    return _FENCE.sub("", raw.strip()).strip()


def _extract_balanced(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` block, or None.

    String-aware: braces/brackets inside JSON string literals are ignored, and
    escaped quotes are handled. Pure linear scan — ReDoS-safe, no regex.
    """
    start = None
    open_ch = close_ch = ""
    depth = 0
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if start is None:
            if ch == "{" or ch == "[":
                start = i
                open_ch = ch
                close_ch = "}" if ch == "{" else "]"
                depth = 1
            continue

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _candidates(raw: str):
    """Yield parse candidates in best-effort order for one LLM response."""
    yield _strip(raw)                 # fenced / clean JSON
    block = _extract_balanced(raw)    # JSON embedded in prose
    if block is not None:
        yield block


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
        for candidate in _candidates(raw):
            try:
                obj = json.loads(candidate)
                if validate is not None:
                    validate(obj)
                return obj
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                continue
    raise LLMOutputError(
        f"failed to parse valid JSON after {attempts} attempts: {last_error}"
    )
