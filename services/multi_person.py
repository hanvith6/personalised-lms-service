"""Split multi-person pose detections into per-person tracks (Module 9, option C).

The YOLOv8-pose model returns 0..N persons per frame. The Public Speaking
scorers (`scorer_ps.score_eye_contact`, `score_posture`) each expect a list of
*single-person* frames: ``[{"keypoints": {name: (x, y, conf)}}, ...]``.

This module sits between the two. It takes the raw multi-person stream and
associates detections across frames into stable tracks, so each person can be
scored independently. Pure-python, no torch/ultralytics import — testable
without a GPU (project rule: heavy I/O stays in the notebook).

Contract
--------
Input  ``frames``: ``list[list[dict]]`` — one entry per video frame; each entry
        is a list of person dicts ``{"keypoints": {name: (x, y, conf)}}`` (may
        be empty when no person is detected).
Output ``dict[int, list[dict]]`` — ``track_id -> pose_frames`` where each
        ``pose_frames`` is the single-person format the scorers already accept,
        padded with empty-keypoint frames so every track spans all video frames.
"""
from __future__ import annotations

import math

_CONF_MIN = 0.5          # keypoint below this confidence is "not visible"
_EMPTY = {"keypoints": {}}

# Defaults tuned for upper-body presentation framing at fps=2.
DEFAULT_MAX_DIST = 120.0  # px: centroid jump beyond this is a different person
DEFAULT_MAX_GAP = 5       # frames a track may vanish before it is closed
DEFAULT_MIN_FRAMES = 1    # tracks seen fewer times than this are dropped


def _centroid(keypoints: dict) -> tuple[float, float] | None:
    """Best available centre for a detection: the nose if visible, else the
    mean of all visible keypoints. Returns None when nothing is visible."""
    nose = keypoints.get("nose")
    if nose and nose[2] >= _CONF_MIN:
        return (nose[0], nose[1])
    pts = [(x, y) for (x, y, c) in keypoints.values() if c >= _CONF_MIN]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def split_tracks(
    frames: list[list[dict]],
    max_dist: float = DEFAULT_MAX_DIST,
    max_gap: int = DEFAULT_MAX_GAP,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> dict[int, list[dict]]:
    """Associate per-frame detections into per-person tracks.

    Greedy nearest-centroid matching: within each frame, candidate
    (track, detection) pairs are matched in ascending-distance order, each used
    at most once, accepting a match only within ``max_dist``. Unmatched
    detections start new tracks (back-filled with empty frames). Tracks unseen
    for more than ``max_gap`` frames are closed. Deterministic.
    """
    n = len(frames)
    tracks: list[dict] = []  # {id, centroid, last_seen, frames}

    for fi, detections in enumerate(frames):
        dets = [d for d in detections if _centroid(d.get("keypoints", {}))]
        det_centroids = [_centroid(d["keypoints"]) for d in dets]
        active = [t for t in tracks if fi - t["last_seen"] <= max_gap]

        # Build candidate pairs, match greedily by smallest distance.
        pairs = sorted(
            ((_dist(t["centroid"], det_centroids[di]), ti, di)
             for ti, t in enumerate(active) for di in range(len(dets))),
            key=lambda p: p[0],
        )
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for d, ti, di in pairs:
            if d > max_dist or ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            t = active[ti]
            t["frames"].append(dets[di])
            t["centroid"] = det_centroids[di]
            t["last_seen"] = fi

        # Tracks with no match this frame get an empty placeholder.
        matched = {id(active[ti]) for ti in used_tracks}
        for t in tracks:
            if id(t) not in matched:
                t["frames"].append(dict(_EMPTY))

        # Unmatched detections become new tracks, back-filled to current frame.
        for di in range(len(dets)):
            if di in used_dets:
                continue
            tracks.append({
                "id": len(tracks),
                "centroid": det_centroids[di],
                "last_seen": fi,
                "frames": [dict(_EMPTY)] * fi + [dets[di]],
            })

    # Pad every track to full length and apply the min-frames filter.
    result: dict[int, list[dict]] = {}
    for t in tracks:
        fr = t["frames"] + [dict(_EMPTY)] * (n - len(t["frames"]))
        seen = sum(1 for f in fr if f.get("keypoints"))
        if seen >= min_frames:
            result[t["id"]] = fr
    return result


def primary_track(tracks: dict[int, list[dict]]) -> list[dict]:
    """The presenter: the track visible in the most frames (option A from a
    multi-person split). Returns an empty list when there are no tracks."""
    if not tracks:
        return []
    return max(
        tracks.values(),
        key=lambda fr: sum(1 for f in fr if f.get("keypoints")),
    )
