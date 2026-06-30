"""Load heavy AI models once at process startup.

All models are module-level singletons — loaded on first import of this module.
The session endpoint imports `MODELS` and passes each into score_one().
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Models:
    pose: Any = None
    asr: Any = None
    mesh: Any = None
    llm: Any = None
    ready: bool = False
    errors: dict[str, str] = field(default_factory=dict)


def load() -> Models:
    m = Models()

    # YOLO pose
    try:
        from ultralytics import YOLO
        m.pose = YOLO("yolov8n-pose.pt")
    except Exception as e:
        m.errors["pose"] = str(e)

    # Whisper ASR
    try:
        import whisper
        m.asr = whisper.load_model("base")
    except Exception as e:
        m.errors["asr"] = str(e)

    # MediaPipe FaceMesh (optional — gaze scorer)
    try:
        import mediapipe as mp
        try:
            FM = mp.solutions.face_mesh.FaceMesh
        except AttributeError:
            from mediapipe.python.solutions.face_mesh import FaceMesh as FM
        m.mesh = FM(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
    except Exception as e:
        m.errors["mesh"] = str(e)  # non-fatal: gaze falls back to YOLO

    # Ollama LLM wrapper
    try:
        import run_pipeline as rp
        m.llm = rp.make_llm()
    except Exception as e:
        m.errors["llm"] = str(e)  # non-fatal: opening_closing skipped

    m.ready = m.pose is not None and m.asr is not None
    return m


# Singleton — loaded once when this module is first imported
MODELS: Models = Models()  # empty until initialise() is called


def initialise() -> Models:
    """Load all models into the singleton. Call once at app startup."""
    global MODELS
    MODELS = load()
    return MODELS
