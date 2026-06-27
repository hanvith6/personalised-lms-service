"""Public Speaking dimension taxonomy — the 7 sub-skills and their metadata.

Module 9 §2.2, scoped to Public Speaking.
"""
from __future__ import annotations

from utils.taxonomy import ScoreSource

# The 7 Public Speaking sub-skills. `source` declares how each is produced.
PS_SUBSKILLS: dict[str, dict] = {
    "eye_contact": {"label": "Eye Contact", "source": ScoreSource.POSE},
    "posture": {"label": "Posture & Body Language", "source": ScoreSource.POSE},
    "speech_pace": {"label": "Speech Pace", "source": ScoreSource.AUDIO},
    "voice_stability": {"label": "Voice Stability", "source": ScoreSource.AUDIO},
    "opening_closing_impact": {"label": "Opening & Closing Impact", "source": ScoreSource.LLM},
    # Owned by other modules — mocked in this dimension (see stubs/).
    "slide_structure": {"label": "Slide Structure", "source": ScoreSource.STUB},
    "audience_engagement": {"label": "Audience Engagement", "source": ScoreSource.STUB},
}

# Stubbed industry-demand boosts (0.0-1.0). In production these would come from a
# labour-market signal feed; here they are fixed so gap ranking is demonstrable.
DEMAND_BOOSTS: dict[str, float] = {
    "eye_contact": 0.50,
    "posture": 0.40,
    "speech_pace": 0.45,
    "voice_stability": 0.35,
    "opening_closing_impact": 0.70,
    "slide_structure": 0.60,
    "audience_engagement": 0.65,
}
