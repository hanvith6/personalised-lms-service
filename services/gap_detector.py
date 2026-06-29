"""Dimension-agnostic gap detection.

Module 9 §3.1, dimension-agnostic. Consumes a list of SubSkillScore from ANY
dimension and ranks the below-threshold ones by remediation priority.
"""
from __future__ import annotations

from utils.taxonomy import SubSkillScore, Gap

GAP_THRESHOLD = 6.0  # sub-skills scoring below 6/10 are treated as gaps

# Severity bands within the gap range (severity = GAP_THRESHOLD - score).
# Mirrors the dashboard heatmap: a deeper gap is more urgent.
CRITICAL_SEVERITY = 3.0   # score <= 3.0  -> CRITICAL
HIGH_SEVERITY = 1.5       # 3.0 < score <= 4.5 -> HIGH; else MODERATE


def severity_band(severity: float) -> str:
    """Map a gap_severity to a human-readable urgency band.

    Module 9 §3.1, dimension-agnostic. severity = GAP_THRESHOLD - score.
    """
    if severity >= CRITICAL_SEVERITY:
        return "CRITICAL"
    if severity >= HIGH_SEVERITY:
        return "HIGH"
    return "MODERATE"


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


def summarize_gaps(gaps, top_n: int = 3) -> dict:
    """Build a presentation-ready summary of a ranked gap list.

    Module 9 §2.7 (dashboard) and §5 (/gaps/:id/summary), dimension-agnostic.
    Assumes `gaps` is already priority-sorted (as detect_gaps returns it).
    Returns counts, the top-N skills, and a per-band tally.
    """
    by_band = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0}
    for g in gaps:
        by_band[severity_band(g.gap_severity)] += 1
    return {
        "total_gaps": len(gaps),
        "critical_gaps": by_band["CRITICAL"],
        "top_gaps": [g.skill for g in gaps[:top_n]],
        "by_band": by_band,
    }
