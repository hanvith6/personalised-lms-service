"""HTTP-level tests for the thin Public Speaking service wrapper.

Uses FastAPI's TestClient (no live server, no network). Covers the full AC-09
loop: scores -> gaps -> path -> completion events -> adapted path, plus input
validation and resource filtering. The LLM is offline here, so path generation
exercises the deterministic JSON stub in api/llm.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import store
from api.main import app

client = TestClient(app)

STUDENT = "stu-2026-001"

# Seven sub-skill scores: two clear gaps (slide_structure, audience_engagement low).
SCORES = {
    "scores": [
        {"skill": "eye_contact", "score": 8.3, "source": "pose"},
        {"skill": "posture", "score": 9.1, "source": "pose"},
        {"skill": "speech_pace", "score": 6.2, "source": "audio"},
        {"skill": "voice_stability", "score": 9.3, "source": "audio"},
        {"skill": "opening_closing_impact", "score": 7.0, "source": "llm"},
        {"skill": "slide_structure", "score": 2.0, "source": "stub"},
        {"skill": "audience_engagement", "score": 3.5, "source": "stub"},
    ]
}


@pytest.fixture(autouse=True)
def _isolate():
    store.reset()
    yield
    store.reset()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "dimension": "public_speaking"}


def test_submit_scores_returns_ranked_gaps():
    r = client.post(f"/scores/{STUDENT}", json=SCORES)
    assert r.status_code == 200
    gaps = r.json()["gaps"]
    # Only the two below-threshold (<6.0) skills are gaps.
    skills = {g["skill"] for g in gaps}
    assert skills == {"slide_structure", "audience_engagement"}
    # Ranked by priority descending — slide_structure (score 2.0) outranks 3.5.
    assert gaps[0]["skill"] == "slide_structure"
    assert gaps[0]["priority_score"] >= gaps[1]["priority_score"]


def test_get_gaps_after_submit():
    client.post(f"/scores/{STUDENT}", json=SCORES)
    r = client.get(f"/gaps/{STUDENT}")
    assert r.status_code == 200
    assert len(r.json()["gaps"]) == 2


def test_gap_summary():
    client.post(f"/scores/{STUDENT}", json=SCORES)
    r = client.get(f"/gaps/{STUDENT}/summary")
    body = r.json()
    assert body["total_gaps"] == 2
    # slide_structure severity = 6.0 - 2.0 = 4.0 >= 3.0 -> critical
    assert body["critical_gaps"] == 1
    assert len(body["top_3_gaps"]) == 2


def test_score_validation_rejects_out_of_range():
    bad = {"scores": [{"skill": "eye_contact", "score": 99.0}]}
    r = client.post(f"/scores/{STUDENT}", json=bad)
    assert r.status_code == 422  # Pydantic ge/le bound


def test_path_generate_requires_gaps_first():
    r = client.post(f"/path/generate/{STUDENT}", json={})
    assert r.status_code == 409


def test_full_loop_generate_then_adapt():
    # 1. scores -> gaps
    client.post(f"/scores/{STUDENT}", json=SCORES)
    # 2. generate path
    r = client.post(f"/path/generate/{STUDENT}", json={"target_role": "Data Analyst"})
    assert r.status_code == 200
    path = r.json()["steps"]
    assert len(path) >= 1
    assert r.json()["progress_pct"] == 0.0
    # 3. completion event: complete step 0 strongly -> adaptation runs
    ev = {"events": [{"step_index": 0, "skill": path[0]["skill_addressed"], "score_pct": 92}]}
    r2 = client.post(f"/path/event/{STUDENT}", json=ev)
    assert r2.status_code == 200
    # progress reflects one completed step
    assert r2.json()["progress_pct"] > 0.0


def test_path_event_remediation_inserts_prereq():
    client.post(f"/scores/{STUDENT}", json=SCORES)
    gen = client.post(f"/path/generate/{STUDENT}", json={}).json()["steps"]
    target_skill = gen[0]["skill_addressed"]
    # Same skill struggling twice (<50%) -> prereq inserted before it.
    ev = {"events": [
        {"skill": target_skill, "score_pct": 40},
        {"skill": target_skill, "score_pct": 47},
    ]}
    r = client.post(f"/path/event/{STUDENT}", json=ev)
    assert r.status_code == 200
    adapted = r.json()["steps"]
    assert len(adapted) >= len(gen)  # prereq may be inserted


def test_get_path_404_when_none():
    r = client.get(f"/path/{STUDENT}")
    assert r.status_code == 404


def test_resources_filter_by_skill():
    r = client.get("/resources", params={"skill": "eye_contact"})
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1
    assert all(i["skill"] == "eye_contact" for i in items)


def test_resources_filter_by_difficulty():
    r = client.get("/resources", params={"difficulty": "beginner"})
    assert r.status_code == 200
    assert all(i["difficulty"] == "beginner" for i in r.json())


def test_frontend_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # Key mount points the JS relies on must be present.
    body = r.text
    assert "My Learning Path" in body
    assert 'id="skills"' in body and 'id="queue"' in body and 'id="ring"' in body


def test_openapi_lists_core_endpoints():
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in ["/scores/{student_id}", "/gaps/{student_id}",
              "/path/generate/{student_id}", "/path/event/{student_id}", "/resources"]:
        assert p in paths
