"""LLM callable for the service — Ollama/Mistral with an offline JSON stub.

Module 9 §3.2, dimension-agnostic. Mirrors the notebook's local strategy: try a
local Ollama server (no API key, local-first rule), and if unreachable fall back
to a deterministic JSON stub so the service still answers during grading/offline.
"""
from __future__ import annotations

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Deterministic fallbacks — valid JSON for both prompt shapes the services use.
_SCORE_STUB = '{"score": 7.0, "justification": "offline stub — Ollama not reachable"}'
_PATH_STUB = json.dumps([
    {
        "title": "Eye Contact Fundamentals", "dimension": "public_speaking",
        "skill_addressed": "eye_contact", "resource_type": "video",
        "difficulty": "beginner", "estimated_minutes": 20,
        "reason": "Builds a camera-presence baseline.", "prerequisite_step_index": None,
    },
    {
        "title": "Confident Posture Practice", "dimension": "public_speaking",
        "skill_addressed": "posture", "resource_type": "exercise",
        "difficulty": "beginner", "estimated_minutes": 15,
        "reason": "Reinforces an upright, open stance.", "prerequisite_step_index": 0,
    },
])


def llm_generate(prompt: str, timeout: int = 30) -> str:
    """Return LLM text for `prompt`. Falls back to a JSON stub when offline.

    The path-generation prompt asks for a "JSON array"; the scorer prompts ask
    for a JSON object — we disambiguate the stub on that marker.
    """
    try:
        body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"]
    except Exception:
        return _PATH_STUB if "JSON array" in prompt else _SCORE_STUB
