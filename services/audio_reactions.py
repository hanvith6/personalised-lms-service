"""Detect audience reactions (applause / laughter) in a talk's audio (Module 9).

Used to drive ``score_audience_engagement``. The naive "loud + noise-like frame"
test counts speech fricatives (s, sh, f are broadband, high spectral-flatness)
as reactions — a 90s solo talk produced 13 bogus 0.05s "events". Real applause
and laughter are *sustained* (≈1s+), so this requires a minimum run length,
collapsing transient consonants while keeping genuine reactions.

Pure-python (numpy in, no librosa/audio I/O) — unit-testable without audio.
"""
from __future__ import annotations

import numpy as np

# A reaction must be this loud (percentile of RMS) AND this noise-like
# (spectral flatness) AND last at least this long to count.
DEFAULT_LOUD_PCT = 75.0
DEFAULT_FLATNESS = 0.25
DEFAULT_MIN_EVENT_S = 1.0


def detect_reactions(rms, flatness, frame_rate: float,
                     loud_pct: float = DEFAULT_LOUD_PCT,
                     flatness_min: float = DEFAULT_FLATNESS,
                     min_event_s: float = DEFAULT_MIN_EVENT_S) -> tuple[int, float]:
    """Return ``(event_count, total_seconds)`` of sustained audience reactions.

    ``rms`` and ``flatness`` are per-frame arrays (e.g. from librosa);
    ``frame_rate`` is frames per second (``sr / hop_length``). A frame is a
    candidate when it is both loud (above ``loud_pct`` of RMS) and noise-like
    (flatness above ``flatness_min``); only *runs* of candidates lasting at
    least ``min_event_s`` are counted, which rejects brief speech fricatives.
    """
    rms = np.asarray(rms, dtype=float)
    flat = np.asarray(flatness, dtype=float)
    n = min(len(rms), len(flat))
    if n == 0 or frame_rate <= 0:
        return 0, 0.0
    rms, flat = rms[:n], flat[:n]

    candidate = (rms > np.percentile(rms, loud_pct)) & (flat > flatness_min)
    min_frames = max(1, int(round(min_event_s * frame_rate)))

    events = 0
    total_frames = 0
    run = 0
    for c in candidate:
        if c:
            run += 1
        else:
            if run >= min_frames:
                events += 1
                total_frames += run
            run = 0
    if run >= min_frames:           # trailing run
        events += 1
        total_frames += run

    return events, total_frames / frame_rate
