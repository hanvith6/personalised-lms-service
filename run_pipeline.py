#!/usr/bin/env python3
"""Headless Public Speaking scorer — runs the real pipeline anywhere (EC2 GPU,
local CPU/MPS), no notebook required.

    python run_pipeline.py "<youtube-url>"            # one video
    python run_pipeline.py /path/to/videos            # a folder of *.mp4
    python run_pipeline.py clip.mp4 --trim 120        # a local file

Reuses the unit-tested services/ + dimension scorers. Device-aware: Qwen on
CUDA, else Ollama for the LLM (skips opening_closing honestly if neither). On a
datacenter IP (Colab/EC2) YouTube may block yt-dlp — prefer local .mp4 files
or pass --cookies cookies.txt.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.multi_person import split_tracks, primary_track, assemble_tracks
from services.gaze import axis_ratio, score_gaze
from services.audio_reactions import detect_reactions
from services.aggregate_profile import aggregate_scores
from dimensions.public_speaking.scorer_ps import (
    score_eye_contact, score_posture, score_speech_pace, score_voice_stability,
    score_opening_closing, score_audience_engagement, score_slide_structure)

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def _kp(kp):
    return {COCO[i]: (float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])) for i in range(len(COCO))}


def make_llm():
    """Return a (prompt)->str LLM, or None. Qwen on CUDA, else Ollama if up."""
    try:
        import torch
        if torch.cuda.is_available():
            from transformers import AutoModelForCausalLM, AutoTokenizer
            mid = "Qwen/Qwen2.5-3B-Instruct"
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)

            def gen(prompt):
                msgs = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ins = tok(text, return_tensors="pt").to(mdl.device)
                out = mdl.generate(**ins, max_new_tokens=512, do_sample=False)
                return tok.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True)
            print("LLM: Qwen2.5-3B on CUDA")
            return gen
    except Exception as e:
        print("Qwen unavailable:", e)
    import urllib.request
    def ollama(prompt):
        body = json.dumps({"model": "mistral", "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["response"]
    try:
        ollama("ping")
        print("LLM: Ollama/mistral")
        return ollama
    except Exception:
        print("LLM: none (opening_closing will be skipped, not faked)")
        return None


def score_one(video, pose_model, asr, mesh, llm, trim):
    import cv2, librosa, torch
    tr = (["-to", str(trim)] if trim else [])
    os.makedirs("frames", exist_ok=True)
    for p in glob.glob("frames/*.jpg"):
        os.remove(p)
    subprocess.run(["ffmpeg", "-y", "-i", video, *tr, "-vf", "fps=2", "frames/f_%04d.jpg"],
                   check=True, capture_output=True)
    paths = sorted(glob.glob("frames/*.jpg"))

    # pose + tracking (ByteTrack, greedy fallback)
    recs = []
    try:
        for res in pose_model.track(paths, persist=True, tracker="bytetrack.yaml",
                                    verbose=False, stream=True):
            rec = {}
            if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
                for tid, k in zip([int(i) for i in res.boxes.id.tolist()], res.keypoints.data.cpu().numpy()):
                    rec[tid] = _kp(k)
            recs.append(rec)
        tracks = assemble_tracks(recs, min_frames=5)
    except Exception:
        tracks = {}
    if not tracks:
        multi = []
        for res in pose_model(paths, verbose=False, stream=True):
            ps = []
            if res.keypoints is not None and res.keypoints.data.shape[0] > 0:
                for k in res.keypoints.data.cpu().numpy():
                    ps.append({"keypoints": _kp(k)})
            multi.append(ps)
        tracks = split_tracks(multi)
    pf = primary_track(tracks) or [{"keypoints": {}}]
    s_eye, s_post = score_eye_contact(pf), score_posture(pf)

    # gaze override
    if mesh is not None:
        gf = []
        for fp in paths:
            img = cv2.imread(fp); h, w = img.shape[:2]
            r = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not r.multi_face_landmarks:
                gf.append({}); continue
            lm = r.multi_face_landmarks[0].landmark
            X = lambda i: lm[i].x * w; Y = lambda i: lm[i].y * h
            lx = axis_ratio(X(468), X(33), X(133)); ly = axis_ratio(Y(468), Y(159), Y(145))
            rx = axis_ratio(X(473), X(362), X(263)); ry = axis_ratio(Y(473), Y(386), Y(374))
            gf.append({"x": (lx + rx) / 2, "y": (ly + ry) / 2})
        if any(g for g in gf):
            s_eye = score_gaze(gf)

    # audio
    subprocess.run(["ffmpeg", "-y", "-i", video, *tr, "-ac", "1", "-ar", "16000", "audio.wav"],
                   check=True, capture_output=True)
    y, sr = librosa.load("audio.wav", sr=16000)
    dur = float(librosa.get_duration(y=y, sr=sr))
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr); f0 = np.nan_to_num(f0)
    rms = librosa.feature.rms(y=y)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    transcript = asr.transcribe("audio.wav").get("text", "").strip()
    wc = len(transcript.split())
    s_pace = score_speech_pace(wc, dur)
    s_voice = score_voice_stability(f0, rms)
    rev, rsec = detect_reactions(rms, flat, sr / 512)
    s_aud = score_audience_engagement(rev, rsec, dur)

    # slides
    fws, fills, prev, trans = 0, [], None, 0
    for fp in paths:
        g = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY)
        ed = (cv2.Canny(g, 100, 200) > 0).mean()
        if (g > 200).mean() > 0.40 and ed > 0.05:
            fws += 1; fills.append(ed)
        if prev is not None and np.mean(np.abs(g.astype(int) - prev)) > 25:
            trans += 1
        prev = g.astype(int)
    s_slide = score_slide_structure(fws, len(paths), trans, float(np.mean(fills)) if fills else 0.0)

    scores = {"eye_contact": round(s_eye.score, 2), "posture": round(s_post.score, 2),
              "speech_pace": round(s_pace.score, 2), "voice_stability": round(s_voice.score, 2),
              "audience_engagement": round(s_aud.score, 2)}
    if wc >= 12 and llm is not None:
        scores["opening_closing_impact"] = round(score_opening_closing(transcript, llm).score, 2)
    if s_slide.detail.get("applicable"):
        scores["slide_structure"] = round(s_slide.score, 2)
    scores["_meta"] = {"words": wc, "duration_s": round(dur, 1), "reactions": rev,
                       "tracks": len(tracks)}
    return scores


def build_jobs(source, cookies):
    if os.path.isdir(source):
        return [(os.path.splitext(os.path.basename(f))[0], f)
                for f in sorted(glob.glob(os.path.join(source, "*.mp4")))]
    if str(source).startswith("http"):
        out = "dl.mp4"
        ck = ["--cookies", cookies] if cookies and os.path.exists(cookies) else []
        r = subprocess.run(["yt-dlp", *ck, "-f", "mp4[height<=480]/best[height<=480]/best",
                            "--no-warnings", "-o", out, source], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("yt-dlp: " + ((r.stderr or "").strip().splitlines() or ["failed"])[-1])
        label = source.split("v=")[-1][:11] if "v=" in source else "video"
        return [(label, out)]
    return [(os.path.splitext(os.path.basename(source))[0], source)]


def main():
    ap = argparse.ArgumentParser(description="Headless Public Speaking scorer")
    ap.add_argument("source", help="YouTube URL, a .mp4, or a directory of .mp4s")
    ap.add_argument("--trim", type=int, default=90, help="seconds per clip (0 = whole)")
    ap.add_argument("--out", default="scores.json")
    ap.add_argument("--cookies", default="")
    args = ap.parse_args()

    from ultralytics import YOLO
    import whisper
    pose_model = YOLO("yolov8n-pose.pt")
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
    llm = make_llm()

    store = {}
    for label, video in build_jobs(args.source, args.cookies):
        try:
            store[label] = score_one(video, pose_model, asr, mesh, llm, args.trim or None)
            print(f"  ✅ {label}: {store[label]}")
        except Exception as e:
            print(f"  ❌ {label} FAILED: {e}")

    real = [{k: v for k, v in s.items() if k != "_meta"} for s in store.values()]
    if len(real) >= 2:
        agg = aggregate_scores(real)
        store["_aggregate"] = agg["per_skill"]
        print("\naggregate (mean across", agg["n_runs"], "videos):")
        for sk, st in agg["per_skill"].items():
            print(f"  {sk:24s} mean={st['mean']:.2f} std={st['std']:.2f} [{st['flag']}]")
    json.dump(store, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out} ({len(store)} entr{'y' if len(store)==1 else 'ies'})")


if __name__ == "__main__":
    main()
