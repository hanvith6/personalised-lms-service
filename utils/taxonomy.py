"""Cross-dimension data contract for the Personalised LMS Engine (Module 9).

Module 9 §2.1, dimension-agnostic — every dimension (Public Speaking, Technical,
Communication, Cognitive & Soft Skills) emits these same shapes so the shared
services (gap_detector, path_generator) work unchanged across all four. Frozen
dataclasses enforce the project immutability rule: never mutate, always rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScoreSource(Enum):
    """Where a sub-skill score came from."""

    POSE = "pose"      # YOLOv8-pose keypoints
    AUDIO = "audio"    # Librosa / Whisper features
    LLM = "llm"        # transcript analysis by an injected LLM
    VISION = "vision"  # frame analysis other than pose (e.g. on-screen slides)
    STUB = "stub"      # a module this dimension does not own (mocked)


def clamp_score(x: float) -> float:
    """Clamp any raw value into the canonical 0.0-10.0 score range.

    Module 9 §2.1, dimension-agnostic.
    """
    return float(max(0.0, min(10.0, x)))


@dataclass(frozen=True)
class SubSkillScore:
    """One scored sub-skill. The universal unit every dimension produces."""

    skill: str
    score: float
    source: ScoreSource
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Gap:
    """A sub-skill scoring below threshold, ranked for remediation."""

    skill: str
    score: float
    gap_severity: float
    priority_score: float
    dimension: str


@dataclass(frozen=True)
class Resource:
    """A learning resource in the seed library."""

    id: str
    title: str
    skill: str
    resource_type: str   # video | article | exercise | quiz | practice
    difficulty: str       # beginner | intermediate | advanced
    url: str
    est_minutes: int
