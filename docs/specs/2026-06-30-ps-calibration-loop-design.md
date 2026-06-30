# PS Calibration Loop — Design (Notebook 1)

**Date:** 2026-06-30
**Module:** 9 (Personalised LMS) — Public Speaking dimension
**Status:** approved design, ready for implementation plan

## Goal (plain)

Use the real data from the 15 catalog videos to find where the Public Speaking
scorers disagree with reality, fix the worst disagreement, re-run, and repeat —
until all 15 clips score sensibly. No invented numbers; the data tells us what
to fix next.

## Success bar (grounded in the Module 9 DPR)

`Module9_Personalised_LMS_Engine.docx` gives a gap threshold and relative pillar
mappings, but **no absolute per-clip target scores**. So the bar is:

1. **Objective rule (strongest):** every "good" A/B clip must **beat** its "bad"
   pair on the relevant sub-skills. An A/B inversion is an objective bug.
2. **Directional labels:** the catalog labels are relative expectations
   (e.g. "slow+rigid" → lower speech_pace; "fast+expressive" → do NOT over-penalise
   voice_stability — the known over-reading).
3. **Boundary:** gap threshold = 6.0 (already in `services/gap_detector.py`).

DPR-faithful means we do NOT fabricate exact bands per clip.

## The 15-video catalog (already in `notebook_demo.ipynb` cell 5)

- `baselines` (5): single-speaker cadence/energy edge cases.
- `ab_pairs` (5): same-speaker good-vs-bad → the A/B oracle.
- `multi_person` (5): 2–6 people in frame → per-person tracking.

## Architecture — one run feeds every loop (interlinked, not separate)

```
run 15 clips ──► results store (per-skill, per-person, per-window)
                      │
        ┌─────────────┼─────────────────────┐
        ▼             ▼                     ▼
  baseline        A/B oracle          multi-person
  calibration   (good > bad)         per-person profiles
        └─────────────┴─────────────────────┘
                      │
              disagreement table
                      │
        fix the single worst scorer → re-run → git push
```

- A **single batch run** scores all 15 once and writes a structured results file
  (reuse `run_pipeline.py` + the unit-tested `services/`; multi_person clips
  scored per-person via `assemble_tracks`/`split_tracks`).
- The **same dataset** drives baseline calibration, A/B ranking, and multi-person
  profiles — these are turns of one loop, not three projects.
- Each turn changes **one** scorer threshold/logic, re-runs, and is committed +
  pushed so the notebook/Colab side can pull.

## Cross-module liveness (flagged, later turn)

Per the DPR, PS scores should ultimately flow **from Module 1's `/reel` endpoint**
(Module 9 orchestrates; it does not own a separate scoring pipeline). Today `/reel`
emits the **5 assessment pillars**, not the **7 PS sub-skills** — reconciling that
mapping (pillars ⇄ sub-skills) is its own loop turn, after the scorers are
calibrated. Not in scope for turn 1.

## Edge-case hardening (cross-cutting)

Every notebook cell must survive real-data edge cases already seen:
no-gaps (fixed in cell 36), no-slides (honest exclusion), no-audience-reactions
(honest 0.0). Hardening happens continuously as runs surface breakage.

## Tooling

- **Jupyter** (`quantum-ps` kernel; standalone server on 127.0.0.1:8888 token
  `quantumps`, or MCP when connected) for notebook edits + runs.
- **git commit + push** per improvement as the update channel.
- Env: `quantum-ps` conda env (numpy-2 stack), local Ollama (`dolphin-phi`).

## Out of scope (own specs later)

- Full Module 1 `/reel` orchestration wiring (cross-module).
- Live webcam/RTSP field testing UI (the `live_feed` service exists and is
  unit-tested; productionising a live UI is separate).
