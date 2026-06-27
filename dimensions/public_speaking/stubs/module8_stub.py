"""=============================================================================
  MOCK — NOT MY DIMENSION.
  `audience_engagement` is produced by Module 8 (Community & Group Discussion),
  which I do NOT own. This stub returns a deterministic placeholder score so the
  end-to-end pipeline runs. Replace with the real Module 8 integration later.
=============================================================================
Module 9 §2.5, scoped to Public Speaking (integration seam to Module 8).
"""
from __future__ import annotations

import hashlib

from utils.taxonomy import SubSkillScore, ScoreSource


def _seeded_score(student_id: str) -> float:
    """Deterministic 0-10 from the student id (different salt from Module 4)."""
    digest = hashlib.sha256(("m8:" + str(student_id)).encode()).hexdigest()
    return round((int(digest[:8], 16) % 1001) / 100.0, 2)  # 0.00-10.00


def stub_audience_engagement(student_id: str) -> SubSkillScore:
    """MOCK audience-engagement score (owned by Module 8)."""
    return SubSkillScore(
        "audience_engagement", _seeded_score(student_id), ScoreSource.STUB,
        {"mock": True, "owner": "Module 8 (Community & Group Discussion)"},
    )
