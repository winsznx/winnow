"""SumMe (Gygli et al., ECCV 2014): fixed-length segments labelled per user.

Each 2-second segment is one clip. A user "keeps" a segment when their summary
covers more than half of the segment's frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

SEGMENT_SECONDS = 2.0
COVERAGE_THRESHOLD = 0.5
DATASET_NAME = "SumMe"
DATASET_URL = "https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip"
DATASET_LICENSE = (
    "Videos: not owned by the authors, provided for non-commercial research purposes only. "
    "Annotations: free to use for research purposes. Cite Gygli et al., ECCV 2014."
)


@dataclass(frozen=True)
class SummeVideo:
    name: str
    path: Path
    duration: float
    labels: np.ndarray  # (n_segments, n_users) in {0, 1}

    @property
    def n_segments(self) -> int:
        return self.labels.shape[0]

    def segment_start(self, i: int) -> float:
        return i * SEGMENT_SECONDS


def segment_labels(user_score: np.ndarray, fps: float, n_segments: int) -> np.ndarray:
    """Per-user keep label for each segment from frame-level summary membership."""
    keep = user_score > 0
    labels = np.zeros((n_segments, keep.shape[1]), dtype=np.int8)
    for i in range(n_segments):
        lo = int(round(i * SEGMENT_SECONDS * fps))
        hi = min(int(round((i + 1) * SEGMENT_SECONDS * fps)), keep.shape[0])
        labels[i] = keep[lo:hi].mean(axis=0) > COVERAGE_THRESHOLD
    return labels


def load_video(root: Path, name: str) -> SummeVideo:
    gt = sio.loadmat(root / "GT" / f"{name}.mat")
    user_score = gt["user_score"]
    duration = float(gt["video_duration"].item())
    fps = user_score.shape[0] / duration
    n_segments = int(duration // SEGMENT_SECONDS)
    return SummeVideo(
        name=name,
        path=root / "videos" / f"{name}.mp4",
        duration=duration,
        labels=segment_labels(user_score, fps, n_segments),
    )


def video_names(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "GT").glob("*.mat"))
