#!/usr/bin/env python3
"""PS calibration harness — turn engine for the calibration loop.

Scores a bucket of the 15-clip catalog with the real pipeline (loads models
ONCE, loops the clips), writes structured results, and prints a disagreement
table judged against (a) the catalog labels and (b) the A/B objective rule.

    python calibrate.py ab_pairs          # 5 good/bad clips (strongest signal)
    python calibrate.py baselines         # 5 single-speaker edge cases
    python calibrate.py multi_person      # 5 group clips (per-person)
    python calibrate.py all --trim 60     # everything

Reuses run_pipeline.score_one + the unit-tested services/. No mock data.
Per the Module 9 DPR: gap threshold 6.0, relative labels (no invented bands).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_pipeline as rp

# ── 15-clip catalog (mirrors notebook_demo.ipynb cell 5 TEST_VIDEOS) ──────────
CATALOG = {
    "multi_person": [
        ("ZNTBAKAT_WQ", "MIT OCW Local vs Expert (2-4 ppl)"),
        ("Ll8zzImnYOE", "Religious Right Panel (4-5 ppl)"),
        ("dlC_f2jfDqI", "Health Literacy Panel (3-5 ppl)"),
        ("depjUykp_wo", "STEAM Expo Panel (2-3 ppl)"),
        ("_rmYgmJN1-M", "Prop 62 News Desk (2 ppl)"),
    ],
    "ab_pairs": [
        ("V8eLdbKXGzk", "Project IDEA Good vs Bad (within-clip A/B)"),
        ("S5c1susCPAE", "Husain Good/Bad cuts"),
        ("8ghfgC_K0Fo", "Managing Nervousness"),
        ("tEF2vNP3S9A", "Do's and Don'ts Slides"),
        ("WXe2KE5C9ag", "Bad PPT exaggerated"),
    ],
    "baselines": [
        ("Z0LhdQIhmzk", "Gifted Labeling — fast+expressive"),
        ("wbftlDzIALA", "Effects of Lying — slow+rigid"),
        ("rW2r5uStgG0", "Power of Reading — structured"),
        ("OMbNoo4mCcI", "Education For All — younger presenter"),
        ("r01KsFLKdO4", "MIT OCW Cost-Benefit — natural cadence"),
    ],
}
SKILLS = ["eye_contact", "posture", "speech_pace", "voice_stability",
          "audience_engagement", "opening_closing_impact", "slide_structure"]


def fetch(vid: str, cookies: str = "") -> str:
    """Download one clip by id -> dl_<id>.mp4 (skip if present)."""
    out = f"dl_{vid}.mp4"
    if os.path.exists(out):
        return out
    ck = ["--cookies", cookies] if cookies and os.path.exists(cookies) else []
    url = f"https://www.youtube.com/watch?v={vid}"
    r = subprocess.run(["yt-dlp", *ck, "-f", "mp4[height<=480]/best[height<=480]/best",
                        "--no-warnings", "-o", out, url], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "yt-dlp failed").strip().splitlines()[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bucket", choices=[*CATALOG, "all"])
    ap.add_argument("--trim", type=int, default=60)
    ap.add_argument("--out", default="calibration_results.json")
    ap.add_argument("--cookies", default="")
    args = ap.parse_args()

    buckets = list(CATALOG) if args.bucket == "all" else [args.bucket]

    # load heavy models ONCE
    from ultralytics import YOLO
    import whisper
    pose = YOLO("yolov8n-pose.pt")
    asr = whisper.load_model("base")
    try:
        import mediapipe as mp
        try:
            FM = mp.solutions.face_mesh.FaceMesh
        except AttributeError:
            from mediapipe.python.solutions.face_mesh import FaceMesh as FM
        mesh = FM(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
    except Exception as e:
        print("gaze disabled:", e); mesh = None
    llm = rp.make_llm()

    store = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for bucket in buckets:
        for vid, label in CATALOG[bucket]:
            key = f"{bucket}/{vid}"
            print(f"\n=== {key} :: {label} ===", flush=True)
            try:
                mp4 = fetch(vid, args.cookies)
                scores = rp.score_one(mp4, pose, asr, mesh, llm, args.trim)
                scores["_label"] = label
                scores["_bucket"] = bucket
                store[key] = scores
                print("  ", {k: scores[k] for k in SKILLS if k in scores}, flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                store[key] = {"_error": str(e), "_label": label, "_bucket": bucket}
            json.dump(store, open(args.out, "w"), indent=2)  # checkpoint each clip

    # ── disagreement table ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'clip':<22}{'eye':>6}{'post':>6}{'pace':>6}{'voice':>6}{'aud':>6}{'open':>6}{'slide':>6}")
    print("-" * 78)
    for key, s in store.items():
        if "_error" in s:
            print(f"{key:<22}  ERROR: {s['_error'][:40]}"); continue
        row = "".join(f"{s.get(k,'-'):>6}" if not isinstance(s.get(k), float)
                      else f"{s[k]:>6.1f}" for k in SKILLS)
        print(f"{key:<22}{row}")
    print(f"\nwrote {args.out} ({len([k for k in store if '_error' not in store[k]])} scored)")


if __name__ == "__main__":
    main()
