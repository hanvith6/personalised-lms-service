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

## Run the thin HTTP service (optional — exposes this dimension as an API)

A small FastAPI wrapper (`api/`) exposes the dimension over REST, demonstrating
the **AC-09 loop** (scores → gaps → path → completion events → adapted path) over
HTTP — not just in the notebook. In-memory, local-first, no API keys.

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8400      # docs at http://localhost:8400/docs
```

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/health` | liveness |
| `POST` | `/scores/{student_id}` | ingest 7 sub-skill scores → detect gaps → emit Module-5 event |
| `GET`  | `/gaps/{student_id}` | ranked gaps |
| `GET`  | `/gaps/{student_id}/summary` | totals + top-3 + critical count |
| `POST` | `/path/generate/{student_id}` | LLM learning path from stored gaps |
| `GET`  | `/path/{student_id}` | current path + progress % |
| `POST` | `/path/event/{student_id}` | completion events → adapt path (accelerate / insert prereq) |
| `GET`  | `/resources` | filtered seed library (`?skill=&difficulty=&resource_type=`) |

The LLM uses local Ollama/Mistral (`:11434`) with a deterministic JSON-stub
fallback, so the service answers even offline. Tests: `tests/test_api.py`
(FastAPI `TestClient`, no live server needed).

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
api/
  main.py                             FastAPI app (8 endpoints, §5)
  schemas.py                          Pydantic request/response models
  store.py                            in-memory per-student state
  llm.py                              Ollama wrapper + offline JSON stub
tests/                                pytest suite (fakes only) + test_api.py
```
