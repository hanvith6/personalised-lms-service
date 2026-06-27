# Personalised LMS Engine (Module 9) — Public Speaking dimension

One dimension of a 4-dimension **Personalised LMS Engine**. This repo implements the
**Public Speaking & Presentation** dimension end-to-end: it scores 7 sub-skills from a
short presentation video, detects skill gaps, and generates an adaptive, LLM-authored
learning path. The other three dimensions (Technical, Communication, Cognitive & Soft
Skills) are owned by teammates and plug into the **same** shared services unchanged.

## What's real vs. stubbed

| Sub-skill | Source | Status |
|---|---|---|
| `eye_contact` | YOLOv8-pose keypoints | **Real** (`scorer_ps.py`) |
| `posture` | YOLOv8-pose keypoints | **Real** (`scorer_ps.py`) |
| `speech_pace` | Whisper words ÷ Librosa duration | **Real** (`scorer_ps.py`) |
| `voice_stability` | Librosa f0 + RMS variation | **Real** (`scorer_ps.py`) |
| `opening_closing_impact` | LLM (Qwen2.5-3B-Instruct) on transcript | **Real** (`scorer_ps.py`) |
| `slide_structure` | Module 4 (Auto-Video-Content) | **Mock** (`stubs/module4_stub.py`) |
| `audience_engagement` | Module 8 (Group Discussion) | **Mock** (`stubs/module8_stub.py`) |

Gap detection and learning-path generation (`services/`) are **dimension-agnostic** — they
consume a list of `SubSkillScore` and never know which dimension produced them.

## Storage rule (important on an 8GB Mac)

- **The heavy stack runs on Google Colab only.** `torch / ultralytics / openai-whisper /
  transformers` and the model weights are multi-GB — never `pip install -r requirements.txt`
  on the laptop.
- **Locally you only run the tests**, which use injected fakes + small numpy arrays and need
  only `pytest` + `numpy` (already installed).

## Run the demo on Colab (driven from VS Code)

1. Install the **Google Colab** extension in VS Code and connect to a Colab GPU runtime
   (Runtime → Change runtime type → **T4 GPU**).
2. Open `dimensions/public_speaking/notebook_demo.ipynb` against that runtime.
3. **Run all cells, top to bottom.** Cell 1 installs deps and loads `Qwen2.5-3B-Instruct`;
   cell 2 downloads a license-clean sample clip (with a synthetic fallback if offline).
4. The notebook ends with the gap-severity chart, the adaptive-sequencing demo, and the
   `gap_profile_updated` payload it would emit to Module 5.

> No API keys. All intelligence is local/open-weight (Quantum "local-first AI" rule).

## Run the tests locally (storage-safe)

```bash
cd personalised-lms-service
python3 -m pytest -q          # 45 tests, ~0.2s
python3 -m pytest --cov=.     # ~99% on pure logic
```

## How a teammate plugs in their dimension

1. Produce a `list[SubSkillScore]` (from `utils/taxonomy.py`) for your sub-skills.
2. Provide a `demand_boosts` dict (`skill -> 0.0..1.0`).
3. Call `detect_gaps(scores, demand_boosts, dimension="technical")` and
   `generate_learning_path(...)` — unchanged. The services are already dimension-agnostic.

## Layout

```
utils/taxonomy.py                     shared contract (SubSkillScore, Gap, Resource)
dimensions/public_speaking/
  taxonomy_ps.py                      the 7 sub-skills + demand boosts
  scorer_ps.py                        real pose/audio/LLM scorers
  stubs/module4_stub.py               MOCK slide_structure
  stubs/module8_stub.py               MOCK audience_engagement
  resource_seed_ps.py                 ~14 tagged public resources
  notebook_demo.ipynb                 10-step Colab demo
services/
  llm_json.py                         retrying JSON parser (no eval/exec)
  gap_detector.py                     dimension-agnostic gap ranking
  path_generator.py                   LLM path + adaptive sequencing
  analytics_emit.py                   Module-5 emit stub
tests/                                pytest suite (fakes only)
```
