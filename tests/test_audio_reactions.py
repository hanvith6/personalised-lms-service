"""Tests for sustained audience-reaction detection (Module 9)."""
import numpy as np

from services.audio_reactions import detect_reactions


def _series(loud_flat_runs, n, frame_rate=31.25):
    """Build (rms, flatness) where listed (start, length) runs are loud+noisy."""
    rms = np.full(n, 0.01)       # quiet baseline
    flat = np.full(n, 0.05)      # harmonic baseline (speech-like)
    for start, length in loud_flat_runs:
        rms[start:start + length] = 1.0
        flat[start:start + length] = 0.6
    return rms, flat, frame_rate


def test_empty_input():
    assert detect_reactions([], [], 31.25) == (0, 0.0)


def test_sustained_burst_counts_as_one_event():
    # One ~2s applause in a 16s clip (reactions stay a small, sparse fraction so
    # the relative-loudness percentile sits in the speech baseline).
    rms, flat, fr = _series([(50, 62)], 500)
    events, secs = detect_reactions(rms, flat, fr, min_event_s=1.0)
    assert events == 1
    assert 1.8 < secs < 2.2


def test_brief_fricatives_are_rejected():
    # Many 1-frame (~0.03s) noisy blips — speech consonants, not reactions.
    runs = [(i, 1) for i in range(0, 480, 12)]
    rms, flat, fr = _series(runs, 500)
    events, secs = detect_reactions(rms, flat, fr, min_event_s=1.0)
    assert events == 0
    assert secs == 0.0


def test_two_separate_applauses():
    rms, flat, fr = _series([(20, 40), (200, 50)], 500)
    events, _ = detect_reactions(rms, flat, fr, min_event_s=1.0)
    assert events == 2


def test_quiet_noise_not_counted():
    # Noise-like but quiet (HVAC hum) must not count — reactions are loud.
    rms = np.full(200, 0.01)
    flat = np.full(200, 0.8)
    events, secs = detect_reactions(rms, flat, 31.25)
    assert events == 0
