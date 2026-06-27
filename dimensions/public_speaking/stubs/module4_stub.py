"""=============================================================================
  MOCK — NOT MY DIMENSION.
  `slide_structure` is produced by Module 4 (AI Auto-Video-Content Generation),
  which I do NOT own. This stub returns a deterministic placeholder score so the
  end-to-end pipeline runs. Replace with the real Module 4 integration later.
=============================================================================
Module 9 §2.5, scoped to Public Speaking (integration seam to Module 4).
"""
from __future__ import annotations

import hashlib

from utils.taxonomy import SubSkillScore, ScoreSource


def _seeded_score(student_id: str) -> float:
    """Deterministic 0-10 from the student id (stable across runs)."""
    digest = hashlib.sha256(str(student_id).encode()).hexdigest()
    return round((int(digest[:8], 16) % 1001) / 100.0, 2)  # 0.00-10.00


def stub_slide_structure(student_id: str) -> SubSkillScore:
    """MOCK slide-structure score (owned by Module 4)."""
    return SubSkillScore(
        "slide_structure", _seeded_score(student_id), ScoreSource.STUB,
        {"mock": True, "owner": "Module 4 (Auto-Video-Content)"},
    )
