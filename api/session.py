"""Session evaluation endpoint — video in, gaps + learning path out.

Module 9 §2.6.3 + §3.1. Ties the calibrated PS pipeline (run_pipeline) to
the Module 9 gap detection and path generation loop in a single HTTP call.

    POST /session/evaluate
        multipart: video file
        form fields: student_id, student_name, student_email, topic,
                     target_role (optional), upload_to_host (optional bool)

Flow:
  1. Save uploaded video to a temp file
  2. Run calibrated PS pipeline (run_pipeline.score_one) — no 60s trim for sessions
  3. Convert SubSkillScores → POST /scores/{student_id} (gap detection)
  4. GET /gaps/{student_id}
  5. POST /path/generate/{student_id} (LLM learning path)
  6. Optionally upload to host for A/B baseline (upload_to_host=true)
  7. Return: pipeline scores + gaps + learning path steps
"""
from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from pydantic import BaseModel

from api import store
from api.llm import llm_generate
from api.schemas import GapOut, PathStep
from dimensions.public_speaking.taxonomy_ps import DEMAND_BOOSTS
from services.gap_detector import detect_gaps, summarize_gaps
from services.path_generator import generate_learning_path
from utils.taxonomy import ScoreSource, SubSkillScore, clamp_score, Gap

router = APIRouter(prefix="/session", tags=["session"])

DIMENSION = "public_speaking"


def _gap_to_out(g: Gap) -> GapOut:
    return GapOut(skill=g.skill, score=g.score, gap_severity=g.gap_severity,
                  priority_score=g.priority_score, dimension=g.dimension)


class SessionResult(BaseModel):
    student_id: str
    scores: dict[str, float]
    gaps: list[GapOut]
    gap_summary: dict
    path: list[dict]
    host_submission_id: int | None = None
    pipeline_meta: dict = {}


@router.post("/evaluate", response_model=SessionResult)
async def evaluate_session(
    video: UploadFile = File(...),
    student_id: str = Form(...),
    student_name: str = Form(default="Student"),
    student_email: str = Form(default=""),
    topic: str = Form(default="Presentation"),
    target_role: str = Form(default="Data Analyst"),
    upload_to_host: bool = Form(default=False),
) -> SessionResult:
    """Full session pipeline: video → calibrated scores → gaps → learning path."""

    # ── 1. Save upload to temp file ───────────────────────────────────────────
    suffix = os.path.splitext(video.filename or "session.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(video.file, tmp)
        tmp.flush()
        mp4_path = tmp.name
    finally:
        tmp.close()

    try:
        # ── 2. Run calibrated PS pipeline ─────────────────────────────────────
        from services.model_loader import MODELS
        from run_pipeline import score_one

        if not MODELS.ready:
            raise HTTPException(status_code=503,
                                detail=f"Models not ready: {MODELS.errors}")
        try:
            result = score_one(mp4_path, MODELS.pose, MODELS.asr,
                               MODELS.mesh, MODELS.llm, trim=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

        # score_one returns a flat dict: {skill: float, ..., "_meta": {...}}
        meta: dict = result.get("_meta", {})
        raw_scores: dict = {k: v for k, v in result.items() if not k.startswith("_")}

        # ── 3. Store scores + run gap detection ───────────────────────────────
        sub_scores = [
            SubSkillScore(skill, clamp_score(float(score)), ScoreSource.STUB, {})
            for skill, score in raw_scores.items()
        ]
        store.set_scores(student_id, sub_scores)

        gaps = detect_gaps(sub_scores, DEMAND_BOOSTS, dimension=DIMENSION)
        store.set_gaps(student_id, gaps)

        gap_summary = summarize_gaps(gaps, top_n=3)

        # ── 4. Generate learning path via LLM ─────────────────────────────────
        path: list[dict] = []
        if gaps:
            try:
                path = generate_learning_path(
                    target_role=target_role,
                    top_gaps=gaps[:5],
                    avg_completion_hours=4,
                    pace="moderate",
                    preferred_resource_types=["video", "exercise", "practice"],
                    llm_generate=llm_generate,
                )
                store.set_path(student_id, path)
            except Exception:
                path = []  # LLM unavailable — gaps still returned

        # ── 5. Optional: upload to host for A/B baseline ──────────────────────
        host_submission_id: int | None = None
        if upload_to_host:
            try:
                from services.backend_client import upload_and_evaluate
                host_submission_id, _ = upload_and_evaluate(
                    mp4_path, student_id, student_name, student_email, topic
                )
            except Exception:
                pass  # host unavailable is non-fatal

    finally:
        os.unlink(mp4_path)

    return SessionResult(
        student_id=student_id,
        scores=raw_scores,
        gaps=[_gap_to_out(g) for g in gaps],
        gap_summary=gap_summary,
        path=path,
        host_submission_id=host_submission_id,
        pipeline_meta=meta,
    )


@router.get("/health")
def session_health() -> dict:
    """Check pipeline and host connectivity."""
    pipeline_ok = False
    try:
        import run_pipeline  # noqa: F401
        pipeline_ok = True
    except Exception:
        pass

    from services.backend_client import health as host_health
    return {
        "pipeline": "ok" if pipeline_ok else "unavailable",
        "host": "ok" if host_health() else "unreachable",
        "host_url": os.getenv("QUANTUM_HOST_URL", "http://192.168.0.226:8000"),
    }
