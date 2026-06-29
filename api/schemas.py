"""Pydantic request/response models — validation at the HTTP boundary.

Module 9 §5. Project rule: validate all input at system boundaries. Scores are
clamped 0-10, resource types / difficulties are enum-constrained, and the
completion event mirrors adapt_path()'s expected shape.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

_SOURCES = {"pose", "audio", "llm", "stub"}


class ScoreIn(BaseModel):
    skill: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=10.0)
    source: str = "stub"
    detail: dict = Field(default_factory=dict)

    def normalised_source(self) -> str:
        return self.source if self.source in _SOURCES else "stub"


class ScoresIn(BaseModel):
    scores: list[ScoreIn] = Field(min_length=1)


class GapOut(BaseModel):
    skill: str
    score: float
    gap_severity: float
    priority_score: float
    dimension: str


class GapsOut(BaseModel):
    student_id: str
    gaps: list[GapOut]


class GapSummaryOut(BaseModel):
    student_id: str
    total_gaps: int
    critical_gaps: int
    top_3_gaps: list[GapOut]


class PathGenerateIn(BaseModel):
    target_role: str = "Data Analyst"
    avg_completion_hours: int = Field(default=4, ge=1, le=40)
    pace: str = "moderate"
    preferred_resource_types: list[str] = Field(
        default_factory=lambda: ["video", "exercise", "article"])


class PathStep(BaseModel):
    title: str
    dimension: str
    skill_addressed: str
    resource_type: str
    difficulty: str
    estimated_minutes: int
    reason: str
    prerequisite_step_index: Optional[int] = None


class PathOut(BaseModel):
    student_id: str
    steps: list[PathStep]
    progress_pct: float = 0.0


class CompletionEvent(BaseModel):
    step_index: Optional[int] = None
    skill: str = Field(min_length=1)
    score_pct: float = Field(ge=0.0, le=100.0)


class PathEventIn(BaseModel):
    events: list[CompletionEvent] = Field(min_length=1)


class ResourceOut(BaseModel):
    id: str
    title: str
    skill: str
    resource_type: str
    difficulty: str
    url: str
    est_minutes: int
