"""HTTP client for the Windows host backend (quantum-reel-dashboard FastAPI).

Module 9 §3.1 integration bridge. Connects this Mac-side calibrated pipeline
to the host at 192.168.0.226:8000 for:
  - Uploading student videos (to trigger Module 1 baseline evaluation)
  - Reading Module 1 scores (for A/B comparison)

The host currently has no PATCH endpoint, so calibrated scores live in our
own Module 9 store — not written back to the host submission record.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

HOST_URL = os.getenv("QUANTUM_HOST_URL", "http://192.168.0.226:8000")
TIMEOUT = 60.0  # seconds — video upload may be large


def _client() -> httpx.Client:
    return httpx.Client(base_url=HOST_URL, timeout=TIMEOUT)


def health() -> bool:
    """Return True if the host FastAPI is reachable."""
    try:
        with _client() as c:
            r = c.get("/health")
            return r.status_code == 200
    except Exception:
        return False


def upload_video(
    mp4_path: str | Path,
    student_id: str,
    student_name: str,
    student_email: str,
    topic: str,
    script_id: int | None = None,
) -> int:
    """Upload a video to the host and return the submission_id.

    Module 9 §9.1: PS practice videos flow through Module 1's /reel endpoint.
    """
    mp4 = Path(mp4_path)
    with _client() as c:
        with open(mp4, "rb") as fh:
            r = c.post(
                "/api/videos/upload",
                data={
                    "student_id": student_id,
                    "student_name": student_name,
                    "student_email": student_email,
                    "topic": topic,
                    **({"script_id": str(script_id)} if script_id else {}),
                },
                files={"video": (mp4.name, fh, "video/mp4")},
            )
    r.raise_for_status()
    return int(r.json()["id"])


def trigger_evaluate(submission_id: int) -> dict:
    """Trigger Module 1's 8-stage pipeline on the host for a submission."""
    with _client() as c:
        r = c.post(f"/api/videos/evaluate/{submission_id}")
    r.raise_for_status()
    return r.json()


def get_submission(submission_id: int) -> dict:
    """Fetch the full submission record from the host (includes Module 1 scores)."""
    with _client() as c:
        r = c.get(f"/api/videos/{submission_id}")
    r.raise_for_status()
    return r.json()


def list_submissions(limit: int = 50) -> list[dict]:
    """List recent submissions from the host (Module 5 dashboard source)."""
    with _client() as c:
        r = c.get("/api/submissions/", params={"limit": limit})
    r.raise_for_status()
    return r.json()


def upload_and_evaluate(
    mp4_path: str | Path,
    student_id: str,
    student_name: str,
    student_email: str,
    topic: str,
) -> tuple[int, dict]:
    """Upload + trigger + wait for result. Returns (submission_id, result_dict).

    Convenience wrapper for the session endpoint. Use for A/B baseline only —
    Module 9 calibrated scores are stored in our own API, not here.
    """
    sid = upload_video(mp4_path, student_id, student_name, student_email, topic)
    result = trigger_evaluate(sid)
    return sid, result
