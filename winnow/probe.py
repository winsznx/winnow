"""ffprobe wrapper: duration, fps, start timecode, camera tag."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    duration: float
    fps: float
    start_timecode: str | None
    camera: str | None
    creation_time: str | None


def probe_clip(path: Path) -> ClipInfo | None:
    """Return None for unreadable clips so callers can count and skip them."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        return None
    try:
        data = json.loads(run.stdout)
        video = next(s for s in data["streams"] if s.get("codec_type") == "video")
        duration = float(data["format"].get("duration") or video.get("duration"))
        fps = float(Fraction(video.get("avg_frame_rate") or video["r_frame_rate"]))
    except (StopIteration, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError):
        return None
    tags = {**data["format"].get("tags", {}), **video.get("tags", {})}
    lowered = {k.lower(): v for k, v in tags.items()}
    camera = lowered.get("com.apple.quicktime.model") or lowered.get("model") or lowered.get("encoder")
    return ClipInfo(
        path=path,
        duration=duration,
        fps=fps,
        start_timecode=lowered.get("timecode"),
        camera=camera,
        creation_time=lowered.get("creation_time"),
    )


def probe_all(paths: list[Path]) -> tuple[list[ClipInfo], int]:
    infos, skipped = [], 0
    for p in paths:
        info = probe_clip(p)
        if info is None:
            skipped += 1
        else:
            infos.append(info)
    return infos, skipped
