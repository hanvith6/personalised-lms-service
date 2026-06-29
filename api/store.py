"""In-memory per-student state for the thin service.

Module 9 §4 (tables) reduced to RAM for a single-dimension demo. Honours the
project immutability rule: stored lists are never mutated in place — each setter
replaces the entry with a fresh list, and getters hand back copies.

This is intentionally NOT a database. The full Module 9 microservice (§4) uses
PostgreSQL; here we only need enough state to demonstrate the AC-09 loop:
scores -> gaps -> path -> completion events -> adapted path.
"""
from __future__ import annotations

from utils.taxonomy import Gap, SubSkillScore

_scores: dict[str, list[SubSkillScore]] = {}
_gaps: dict[str, list[Gap]] = {}
_paths: dict[str, list[dict]] = {}
_events: dict[str, list[dict]] = {}


def set_scores(student_id: str, scores: list[SubSkillScore]) -> None:
    _scores[student_id] = list(scores)


def get_scores(student_id: str) -> list[SubSkillScore]:
    return list(_scores.get(student_id, []))


def set_gaps(student_id: str, gaps: list[Gap]) -> None:
    _gaps[student_id] = list(gaps)


def get_gaps(student_id: str) -> list[Gap]:
    return list(_gaps.get(student_id, []))


def set_path(student_id: str, path: list[dict]) -> None:
    _paths[student_id] = list(path)


def get_path(student_id: str) -> list[dict]:
    return list(_paths.get(student_id, []))


def append_events(student_id: str, events: list[dict]) -> list[dict]:
    """Append completion events and return the full (copied) event log."""
    merged = list(_events.get(student_id, [])) + list(events)
    _events[student_id] = merged
    return list(merged)


def get_events(student_id: str) -> list[dict]:
    return list(_events.get(student_id, []))


def reset() -> None:
    """Clear all state — used by tests for isolation."""
    _scores.clear()
    _gaps.clear()
    _paths.clear()
    _events.clear()
