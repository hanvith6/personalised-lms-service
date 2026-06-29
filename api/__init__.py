"""Thin FastAPI wrapper exposing the Public Speaking dimension over HTTP.

Module 9 §5 (API Endpoints), scoped to ONE dimension. This is a demo-grade,
in-memory service that wraps the existing dimension-agnostic services
(gap_detector, path_generator) and the Public Speaking taxonomy/resources.

It deliberately does NOT pull in torch/ultralytics/whisper — it accepts
already-extracted sub-skill scores (exactly as scorer_ps does), so it runs on
the laptop with only fastapi + numpy. The heavy scoring stays in the notebook.
"""
