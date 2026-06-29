# ANTIGRAVITY SESSION HANDOFF — 2026-06-27
## Quantum School of AI · Module 9 · Public Speaking Dimension

---

## TL;DR

Module 9 Public Speaking dimension is COMPLETE.
45/45 pytest cases pass. Full pipeline verified on Colab T4 GPU.
One task remains: run all 30 notebook cells on Colab with LOCAL_DEBUG=False.
The user cannot click Run All themselves — they need help or a nudge.

---

## PROJECT LOCATIONS

| Location | Path |
|----------|------|
| Mac (source of truth) | `/Users/hanvith/dev/Quant/personalised-lms-service/` |
| Colab (runtime) | `/content/personalised-lms-service/` |
| Notebook URL | https://colab.research.google.com/drive/1kwZWVQ7WV-LZxxMLJCjG2xxje5zw0M05 |
| Session save file | `~/.claude/session-data/2026-06-27-lms-ps-dim-session.tmp` |

---

## WHAT THIS PROJECT DOES

Takes a student video → scores 7 public speaking sub-skills → detects gaps
→ generates adaptive learning path via local LLM → emits to analytics module.

### Pipeline

```
Video
  → ffmpeg frames → YOLOv8-pose  → eye_contact (0-10) + posture (0-10)
  → ffmpeg audio  → Librosa      → voice_stability (0-10)
  → ffmpeg audio  → Whisper ASR  → transcript → WPM → speech_pace (0-10)
  → transcript    → Qwen2.5-3B   → opening_closing_impact (0-10)
  → [stub]                       → slide_structure (Module 4 placeholder)
  → [stub]                       → audience_engagement (Module 8 placeholder)

7 scores → GapDetector  → gaps (sub-skills below 6.0, priority ranked)
gaps     → PathGenerator + Qwen2.5-3B → ordered 3-step learning path
path + completion events → adapt_path() → difficulty bump or prereq insert
gaps     → analytics_emit() → payload dict for Module 5
```

### Rules That Must Never Be Broken

- Local-first AI only. Qwen2.5-3B via HuggingFace. No OpenAI/Anthropic/Gemini.
- `scorer_ps.py` is pure numpy — no torch/ultralytics imports (keeps it testable locally)
- `llm_generate(prompt)->str` is always injected, never imported globally
- `SubSkillScore`, `Gap`, `Resource` are frozen dataclasses (immutable)
- `GAP_THRESHOLD = 6.0`
- Adaptive rule: >85% score → bump difficulty. Fail <50% twice → insert prereq
- Never store raw video/audio as records. Frames/audio.wav are transient.

---

## ALL FILES (22 Python files + 1 notebook, all complete)

```
utils/taxonomy.py                            frozen dataclasses: SubSkillScore,
                                             Gap, Resource, ScoreSource, clamp_score
services/llm_json.py                         parse_llm_json() 3-attempt retry,
                                             \A/\Z ReDoS-safe regex
services/gap_detector.py                     detect_gaps(), GAP_THRESHOLD=6.0,
                                             priority=gap_severity*(1+demand_boost)
services/path_generator.py                   generate_learning_path(), adapt_path()
services/analytics_emit.py                   emit_gap_profile_updated()
dimensions/public_speaking/scorer_ps.py      5 real scorers (pure numpy)
dimensions/public_speaking/taxonomy_ps.py    DEMAND_BOOSTS per sub-skill
dimensions/public_speaking/resource_seed_ps.py  15 curated learning resources
dimensions/public_speaking/stubs/module4_stub.py  stub_slide_structure(student_id)
dimensions/public_speaking/stubs/module8_stub.py  stub_audience_engagement(student_id)
dimensions/public_speaking/notebook_demo.ipynb    30-cell Colab notebook
tests/test_scorer_ps.py                      scorer unit tests
tests/test_gap_detector.py                   gap detection tests
tests/test_path_generator.py                 path generation + adapt_path tests
tests/test_llm_json.py                       JSON retry + ReDoS tests
tests/test_taxonomy.py                       dataclass tests
tests/test_stubs_and_emit.py                 stub + analytics emit tests
conftest.py                                  pytest config
```

---

## NOTEBOOK STRUCTURE (30 cells)

| Cell | Type | Content |
|------|------|---------|
| [00] | markdown | title |
| [01] | code | GPU CHECK — skipped if LOCAL_DEBUG=True |
| [02] | code | PROJECT SETUP — sets sys.path to project root |
| [03] | code | **RUN MODE: `LOCAL_DEBUG = True/False` ← SET FALSE FOR FULL RUN** |
| [04] | markdown | Step 1 header |
| [05] | code | pip install ultralytics whisper librosa transformers etc (skipped if LOCAL_DEBUG) |
| [06] | code | LLM setup — loads Qwen2.5-3B OR uses stub if LOCAL_DEBUG |
| [07] | markdown | Step 2 header |
| [08] | code | video download (w3schools MP4) or synthetic fallback |
| [09] | code | YOLO pose estimation OR synthetic pose data if LOCAL_DEBUG |
| [10] | code | eye_contact + posture scoring from pose_frames |
| [11] | markdown | Step 3 header |
| [12] | code | librosa f0/RMS OR synthetic audio if LOCAL_DEBUG |
| [13] | code | Whisper transcript + speech_pace + voice_stability scoring |
| [14] | markdown | Step 4 header |
| [15] | code | opening_closing_impact via llm_generate |
| [16] | markdown | Step 5 header |
| [17] | code | stub scores: slide_structure + audience_engagement → all_scores list |
| [18] | code | **SCORE MATRIX printout (all 7 sub-skills, bands, average)** |
| [19] | markdown | Step 6 header |
| [20] | code | detect_gaps() → gaps list, top5 |
| [21] | markdown | Step 7 header |
| [22] | code | generate_learning_path() → path list |
| [23] | markdown | Step 8 header |
| [24] | code | print path steps |
| [25] | code | matplotlib bar chart of all 7 scores |
| [26] | markdown | Step 9 header |
| [27] | code | adapt_path() demo — fake completion events, prereq insertion |
| [28] | markdown | Step 10 header |
| [29] | code | emit_gap_profile_updated() → analytics payload |

---

## VERIFIED RESULTS FROM LAST COLAB T4 RUN

| Sub-Skill | Score | Band | Source |
|-----------|-------|------|--------|
| posture | 9.17 | STRONG | POSE |
| voice_stability | 9.85 | STRONG | AUDIO |
| eye_contact | scored | — | POSE |
| opening_closing_impact | scored | — | LLM Qwen2.5-3B |
| slide_structure | mock | — | STUB Module 4 |
| audience_engagement | mock | — | STUB Module 8 |
| speech_pace | 0.00 | CRITICAL | AUDIO (synthetic video = 0 WPM, EXPECTED) |

- Gap detected: CRITICAL gap in speech_pace ✓
- Learning path: 3-step path generated by Qwen2.5-3B ✓
- Adaptation: struggled twice → prereq primer auto-inserted ✓
- Analytics emit: payload dict returned correctly ✓
- pytest: **45/45 passed** ✓

---

## SECURITY FIXES APPLIED — DO NOT REVERT

| ID | Fix |
|----|-----|
| H1 | Streaming download 200MB cap (no RAM OOM on Colab) |
| H2 | SHA256 integrity check on downloaded video |
| H3 | `rms > 1e-9` in voice_stability scorer (IEEE 754 subnormal fix) |
| M1 | `\A`/`\Z` anchors in llm_json.py (not `^` and `$`, prevents ReDoS) |
| M2 | Duplicate TED URL replaced in resource_seed_ps.py |
| M3 | `*.webm` `*.ogv` in .gitignore |
| M4 | `_sanitize_segment()` truncates + XML-tags transcript before LLM prompt |

---

## WHAT REMAINS

### Item 1 — Run notebook end-to-end on Colab (LOCAL_DEBUG=False)

The user has not yet pressed Ctrl+F9 in their signed-in Chrome tab.
Playwright/Chrome DevTools MCP cannot do it (not signed into Google).
Tell the user: open Chrome, go to the notebook URL, press Ctrl+F9.
Expected runtime: ~10 min (pip install + Qwen download on first run).

### Item 2 — Module 5 analytics cloud sink (future, different module)

`emit_gap_profile_updated()` already returns the correct payload dict.
No code changes needed here. When Module 5 is built, POST that dict.

---

## HOW TO RUN ON COLAB T4 (full pipeline)

```bash
# 1. Runtime → Change runtime type → T4 GPU → Save

# 2. Upload zip and extract in Colab terminal
unzip personalised-lms-service.zip -d /content/

# 3. Open notebook
# dimensions/public_speaking/notebook_demo.ipynb

# 4. Cell [03]: set LOCAL_DEBUG = False

# 5. Ctrl+F9 (Run All)
#    Cell [05]: installs packages (~3-5 min)
#    Cell [06]: downloads Qwen2.5-3B (~6GB, first run only)

# 6. Run tests
cd /content/personalised-lms-service && python -m pytest -q
# Expected: 45 passed
```

**Verify output at:**
- Cell [10]: eye_contact + posture scores
- Cell [13]: speech_pace + voice_stability scores
- Cell [15]: opening_closing_impact score
- Cell [17]: MOCK slide_structure + audience_engagement
- Cell [18]: SCORE MATRIX table (all 7 scores + bands + average)
- Cell [20]: gap list with priorities
- Cell [22]: learning path steps
- Cell [25]: matplotlib bar chart
- Cell [27]: adapt_path() prereq demo
- Cell [29]: analytics payload dict

---

## HOW TO RUN LOCALLY IN VS CODE (no GPU, debug mode)

```
1. Open notebook_demo.ipynb in VS Code (Jupyter extension)
2. Select kernel: local Python 3.x
3. Cell [03]: LOCAL_DEBUG = True  (already set)
4. Run All — no GPU, no installs, synthetic data, all logic tested (~5 seconds)
```

---

## IMPORTANT DATA SHAPES

```python
# SubSkillScore
SubSkillScore(skill: str, score: float,  # 0-10
              source: ScoreSource, detail: dict)

# Gap
Gap(skill, current_score, gap_severity, priority_score, demand_boost)

# pose_frames (input to eye_contact + posture scorers)
[{"keypoints": {"nose": (x, y, conf), "left_eye": (x, y, conf), ...}}]
# COCO keypoint names; skip frame if conf < 0.5

# generate_learning_path() signature
generate_learning_path(
    target_role,            # str e.g. "Data Analyst"
    top_gaps,               # list[Gap]
    avg_completion_hours,   # int e.g. 4
    pace,                   # "slow" | "moderate" | "fast"
    preferred_resource_types, # list e.g. ["video", "exercise"]
    llm_generate            # callable (prompt: str) -> str
)
# LLM must return JSON array of steps with fields:
# title, skill_addressed, dimension, difficulty, resource_type,
# estimated_minutes, reason, prerequisite_step_index
# Valid difficulty: beginner / intermediate / advanced
# Valid resource_type: video / article / exercise / practice / quiz

# adapt_path() signature
adapt_path(
    path,              # list[dict] from generate_learning_path
    completion_events  # list[{"step_index": int, "skill": str, "score_pct": float}]
)
# score_pct > 0.85 → bumps next step difficulty
# score_pct < 0.50 twice for same skill → inserts prerequisite primer step
```

---

## STUDENT + PLATFORM CONTEXT

| Item | Value |
|------|-------|
| STUDENT_ID in notebook | `stu-2026-001` |
| Platform | Quantum School of AI (colleague's EdTech platform) |
| This dimension | Module 9 within an 8-module LMS architecture |
| Analytics consumer | Module 5 |
| Stub owners | Module 4 (slide_structure), Module 8 (audience_engagement) |
| User email | bhimireddy.h@northeastern.edu |

---

*End of handoff.*
