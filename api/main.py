"""FastAPI app — thin HTTP wrapper for the Public Speaking dimension.

Module 9 §5, scoped to one dimension. Wires the existing dimension-agnostic
services to REST endpoints and demonstrates the AC-09 loop end-to-end over HTTP:

    POST /scores/{id}   -> store 7 sub-skill scores, run gap detection, emit event
    GET  /gaps/{id}     -> ranked gaps
    POST /path/generate -> LLM-generated learning path from stored gaps
    GET  /path/{id}     -> current path + progress
    POST /path/event    -> ingest completion events, adapt the path (accel/prereq)
    GET  /resources     -> filtered seed library

Run:  uvicorn api.main:app --reload --port 8400
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import store
from api.llm import llm_generate
from api.schemas import (
    GapOut, GapsOut, GapSummaryOut, PathEventIn, PathGenerateIn, PathOut,
    ResourceOut, ScoresIn,
)
from dimensions.public_speaking.resource_seed_ps import PS_RESOURCES
from dimensions.public_speaking.taxonomy_ps import DEMAND_BOOSTS
from services.analytics_emit import emit_gap_profile_updated
from services.gap_detector import detect_gaps, summarize_gaps
from services.path_generator import adapt_path, generate_learning_path
from utils.taxonomy import Gap, ScoreSource, SubSkillScore, clamp_score

DIMENSION = "public_speaking"  # "critical" band lives in services.gap_detector

_FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")

app = FastAPI(
    title="Personalised LMS — Public Speaking dimension",
    description="Thin Module 9 service wrapper (one dimension). Local-first, no API keys.",
    version="1.0.0",
)

# Permissive CORS for the local demo frontend (served same-origin below, but this
# also allows opening index.html from a different localhost port during dev).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8400", "http://127.0.0.1:8400", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the single-file 'My Learning Path' widget at the root."""
    return FileResponse(_FRONTEND)


def _gap_to_out(g: Gap) -> GapOut:
    return GapOut(skill=g.skill, score=g.score, gap_severity=g.gap_severity,
                  priority_score=g.priority_score, dimension=g.dimension)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dimension": DIMENSION}


@app.post("/scores/{student_id}", response_model=GapsOut)
def submit_scores(student_id: str, payload: ScoresIn) -> GapsOut:
    """Ingest sub-skill scores, run gap detection, emit the Module-5 event.

    Mirrors §3.3 step 5: store gaps + emit `gap_profile_updated`.
    """
    scores = [
        SubSkillScore(s.skill, clamp_score(s.score),
                      ScoreSource(s.normalised_source()), s.detail)
        for s in payload.scores
    ]
    store.set_scores(student_id, scores)

    gaps = detect_gaps(scores, DEMAND_BOOSTS, dimension=DIMENSION)
    store.set_gaps(student_id, gaps)
    emit_gap_profile_updated(student_id, gaps)

    return GapsOut(student_id=student_id, gaps=[_gap_to_out(g) for g in gaps])


@app.get("/gaps/{student_id}", response_model=GapsOut)
def get_gaps(student_id: str) -> GapsOut:
    gaps = store.get_gaps(student_id)
    return GapsOut(student_id=student_id, gaps=[_gap_to_out(g) for g in gaps])


@app.get("/gaps/{student_id}/summary", response_model=GapSummaryOut)
def get_gap_summary(student_id: str) -> GapSummaryOut:
    gaps = store.get_gaps(student_id)
    s = summarize_gaps(gaps, top_n=3)  # shared, dimension-agnostic helper
    return GapSummaryOut(
        student_id=student_id,
        total_gaps=s["total_gaps"],
        critical_gaps=s["critical_gaps"],
        top_3_gaps=[_gap_to_out(g) for g in gaps[:3]],
    )


@app.post("/path/generate/{student_id}", response_model=PathOut)
def generate_path(student_id: str, payload: PathGenerateIn) -> PathOut:
    """Generate a learning path from the student's stored gaps via the LLM."""
    gaps = store.get_gaps(student_id)
    if not gaps:
        raise HTTPException(status_code=409,
                            detail="No gaps for student — POST /scores first.")
    path = generate_learning_path(
        target_role=payload.target_role,
        top_gaps=gaps[:5],
        avg_completion_hours=payload.avg_completion_hours,
        pace=payload.pace,
        preferred_resource_types=payload.preferred_resource_types,
        llm_generate=llm_generate,
    )
    store.set_path(student_id, path)
    return PathOut(student_id=student_id, steps=path, progress_pct=0.0)


@app.get("/path/{student_id}", response_model=PathOut)
def get_path(student_id: str) -> PathOut:
    path = store.get_path(student_id)
    if not path:
        raise HTTPException(status_code=404, detail="No path — POST /path/generate first.")
    events = store.get_events(student_id)
    completed = {e["step_index"] for e in events if e.get("step_index") is not None}
    progress = round(100.0 * len(completed) / len(path), 1) if path else 0.0
    return PathOut(student_id=student_id, steps=path, progress_pct=progress)


@app.post("/path/event/{student_id}", response_model=PathOut)
def path_event(student_id: str, payload: PathEventIn) -> PathOut:
    """Ingest completion events and adapt the path (§2.5 accelerate / remediate)."""
    path = store.get_path(student_id)
    if not path:
        raise HTTPException(status_code=404, detail="No path — POST /path/generate first.")

    events = [e.model_dump() for e in payload.events]
    all_events = store.append_events(student_id, events)
    adapted = adapt_path(path, all_events)
    store.set_path(student_id, adapted)

    completed = {e["step_index"] for e in all_events if e.get("step_index") is not None}
    progress = round(100.0 * len(completed) / len(adapted), 1) if adapted else 0.0
    return PathOut(student_id=student_id, steps=adapted, progress_pct=progress)


@app.get("/resources", response_model=list[ResourceOut])
def list_resources(
    skill: str | None = Query(default=None),
    difficulty: str | None = Query(default=None),
    resource_type: str | None = Query(default=None),
) -> list[ResourceOut]:
    """Filtered seed resource library (§2.4)."""
    out = []
    for r in PS_RESOURCES:
        if skill and r.skill != skill:
            continue
        if difficulty and r.difficulty != difficulty:
            continue
        if resource_type and r.resource_type != resource_type:
            continue
        out.append(ResourceOut(
            id=r.id, title=r.title, skill=r.skill, resource_type=r.resource_type,
            difficulty=r.difficulty, url=r.url, est_minutes=r.est_minutes))
    return out
