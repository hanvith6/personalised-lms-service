"""Emit a gap-profile-updated event to the shared analytics module (Module 5).

Module 9 §3.5, dimension-agnostic. STUB: builds and prints the payload it WOULD
send to Module 5 — no real network call is made.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass


def emit_gap_profile_updated(student_id, gaps, _print=print) -> dict:
    """Build, print, and return the Module-5 analytics payload.

    Module 9 §3.5, dimension-agnostic. `_print` is injectable so tests can assert
    on it without capturing stdout.
    """
    payload = {
        "event": "gap_profile_updated",
        "student_id": student_id,
        "gaps": [asdict(g) if is_dataclass(g) else g for g in gaps],
    }
    _print("[analytics -> Module 5] " + json.dumps(payload))
    return payload
