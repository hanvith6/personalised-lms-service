"""Live-feed (streaming) Public Speaking scoring (Module 9).

The batch pipeline scores a whole downloaded file. A live feed — a webcam, a
classroom camera, or an RTSP/HLS stream — must instead be scored *as it plays*,
in rolling windows, so feedback appears while the speaker is still talking.

This module separates the streaming concern into testable pieces:

  - :func:`iter_windows`   pure windowing of any per-frame sequence
  - :func:`score_window`   per-person pose scores for one window of records
  - :class:`FrameSource`   the I/O edge: webcam index / file path / stream URL
  - :func:`stream_pose_records`  glue that drives a YOLO tracker over a source

Only the I/O edge needs a camera / model. Everything else is unit-tested. The
pose path reuses the same scorers as the batch pipeline, so a live score and a
batch score of the same footage are computed identically.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from services.multi_person import assemble_tracks, primary_track
from dimensions.public_speaking.scorer_ps import score_eye_contact, score_posture

# COCO-17 keypoint order emitted by YOLOv8-pose.
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def iter_windows(
    items: Iterable,
    window: int,
    stride: int | None = None,
) -> Iterator[list]:
    """Yield successive windows of ``window`` items, advancing by ``stride``.

    ``stride`` defaults to ``window`` (non-overlapping/tumbling windows); a
    smaller stride gives overlapping/sliding windows for smoother live output.
    A trailing partial window is yielded so the tail of a stream is not lost.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    stride = window if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    buf = list(items)
    if not buf:
        return
    i = 0
    while i < len(buf):
        chunk = buf[i:i + window]
        yield chunk
        if i + window >= len(buf):
            break
        i += stride


def score_window(frame_records: list[dict[int, dict]]) -> dict[int, dict]:
    """Score every tracked person in one window of stable-ID frame records.

    ``frame_records[i]`` maps ``track_id -> {keypoints}``. Returns
    ``{track_id: {"eye_contact", "posture", "visible"}}``.
    """
    tracks = assemble_tracks(frame_records)
    out: dict[int, dict] = {}
    for tid, frames in tracks.items():
        out[tid] = {
            "eye_contact": round(score_eye_contact(frames).score, 2),
            "posture": round(score_posture(frames).score, 2),
            "visible": sum(1 for f in frames if f["keypoints"]),
        }
    return out


def score_stream(
    records: Iterable[dict[int, dict]],
    window: int = 20,
    stride: int | None = None,
) -> Iterator[dict]:
    """Roll over a stream of frame records, emitting one score dict per window.

    Each yielded item: ``{"window", "per_person", "presenter"}`` where
    ``presenter`` is the most-visible track's scores in that window. This is
    what a live dashboard consumes — a fresh reading every ``stride`` frames.
    """
    for wi, chunk in enumerate(iter_windows(list(records), window, stride)):
        per_person = score_window(chunk)
        presenter_id = None
        if per_person:
            presenter_id = max(per_person, key=lambda t: per_person[t]["visible"])
        yield {
            "window": wi,
            "per_person": per_person,
            "presenter": per_person.get(presenter_id) if presenter_id is not None else None,
        }


def keypoints_from_yolo(kp_row) -> dict:
    """Convert one YOLO keypoint row (``(17, 3)`` array-like) to the
    ``{name: (x, y, conf)}`` dict the scorers expect."""
    return {COCO[i]: (float(kp_row[i][0]), float(kp_row[i][1]), float(kp_row[i][2]))
            for i in range(len(COCO))}


class FrameSource:
    """Source-agnostic frame reader (the I/O edge — needs OpenCV at runtime).

    ``src`` accepts anything ``cv2.VideoCapture`` does:
      - ``0`` (or another int) — a local webcam
      - a file path — replay a recording as a stream
      - an ``rtsp://`` / ``http(s)://`` URL — a network camera / live stream
    """

    def __init__(self, src):
        self.src = src
        self._cap = None

    def __enter__(self):
        import cv2  # imported lazily so unit tests never require OpenCV
        self._cap = cv2.VideoCapture(self.src)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open frame source: {self.src!r}")
        return self

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()

    def frames(self, every: int = 1) -> Iterator:
        """Yield decoded frames, keeping one in every ``every`` (frame stride
        to match the batch pipeline's ``fps=2`` sampling on a faster stream)."""
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if idx % every == 0:
                yield frame
            idx += 1


def stream_pose_records(source: FrameSource, model, every: int = 1) -> Iterator[dict[int, dict]]:
    """Drive a YOLO-pose tracker over a :class:`FrameSource`, yielding one
    stable-ID record per kept frame. Runtime glue (needs the model + a source);
    the windowing/scoring it feeds is unit-tested separately."""
    for frame in source.frames(every=every):
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        rec: dict[int, dict] = {}
        if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
            ids = [int(i) for i in res.boxes.id.tolist()]
            data = res.keypoints.data.cpu().numpy()
            for tid, kp in zip(ids, data):
                rec[tid] = keypoints_from_yolo(kp)
        yield rec
