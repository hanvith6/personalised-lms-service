"""Dimension-agnostic gap detection.

Module 9 §3.1, dimension-agnostic. Consumes a list of SubSkillScore from ANY
dimension and ranks the below-threshold ones by remediation priority.
"""
from __future__ import annotations

from utils.taxonomy import SubSkillScore, Gap

GAP_THRESHOLD = 6.0  # sub-skills scoring below 6/10 are treated as gaps


def detect_gaps(scores, demand_boosts, dimension: str = "public_speaking") -> list[Gap]:
    """Return below-threshold skills as Gaps, sorted by priority descending.

    Module 9 §3.1, dimension-agnostic.

    priority_score = gap_severity * (1 + industry_demand_boost), where
    gap_severity = GAP_THRESHOLD - score (only computed when score < threshold).
    """
    gaps: list[Gap] = []
    for s in scores:
        if s.score < GAP_THRESHOLD:
            severity = GAP_THRESHOLD - s.score
            boost = float(demand_boosts.get(s.skill, 0.0))
            priority = severity * (1.0 + boost)
            gaps.append(Gap(s.skill, s.score, round(severity, 4), round(priority, 4), dimension))
    return sorted(gaps, key=lambda g: g.priority_score, reverse=True)
