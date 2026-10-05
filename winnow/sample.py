"""Extract frames per clip at fixed fractions of its duration, cached on disk."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

FRACTIONS = (0.2, 0.5, 0.8)


def clip_key(path: Path, start: float = 0.0, length: float | None = None) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{start:.3f}|{length}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def sample_frames(
    path: Path,
    cache_dir: Path,
    duration: float,
    start: float = 0.0,
    length: float | None = None,
) -> list[Path]:
    """Frames at 20/50/80% of [start, start+length). Cached: reruns do no work."""
    length = duration if length is None else length
    key = clip_key(path, start, length)
    out_dir = cache_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, frac in enumerate(FRACTIONS):
        target = out_dir / f"{key}_{i}.jpg"
        if not target.exists():
            ts = start + frac * length
            cmd = [
                "ffmpeg", "-v", "error", "-y", "-ss", f"{ts:.3f}", "-i", str(path),
                "-frames:v", "1", "-vf", "scale=-2:384", "-q:v", "3", str(target),
            ]
            subprocess.run(cmd, check=False, capture_output=True)
        if target.exists():
            frames.append(target)
    return frames
