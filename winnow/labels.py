"""Turn a final timeline export into per-clip used/unused labels.

Supported: FCPXML, Premiere XML (xmeml) and CMX3600 EDL. A clip counts as used
when any timeline event references its source file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import etree

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".mts", ".braw", ".r3d"}


@dataclass(frozen=True)
class TimelineUse:
    name: str
    reel: str | None
    in_s: float
    out_s: float


@dataclass
class LabelResult:
    labels: dict[str, int]
    ranges: dict[str, list[tuple[float, float]]]
    unmatched_sources: list[str] = field(default_factory=list)
    n_timeline_sources: int = 0

    @property
    def unmatched_fraction(self) -> float:
        if not self.n_timeline_sources:
            return 1.0
        return len(self.unmatched_sources) / self.n_timeline_sources

    def report(self) -> str:
        used = sum(self.labels.values())
        lines = [
            f"clips: {len(self.labels)}  used: {used}  unused: {len(self.labels) - used}",
            f"timeline sources: {self.n_timeline_sources}  "
            f"unmatched: {len(self.unmatched_sources)} ({self.unmatched_fraction:.1%})",
        ]
        lines += [f"  unmatched: {name}" for name in sorted(self.unmatched_sources)]
        return "\n".join(lines)


def parse_timeline(path: str | Path, fps: float = 24.0) -> list[TimelineUse]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".edl":
        return _parse_edl(path.read_text(errors="replace"), fps)
    root = etree.parse(str(path)).getroot()
    if root.tag == "fcpxml":
        return _parse_fcpxml(root)
    if root.tag == "xmeml":
        return _parse_xmeml(root)
    raise ValueError(f"unrecognised timeline format: {path}")


def _basename(src: str) -> str:
    parsed = urlparse(src)
    raw = unquote(parsed.path) if parsed.scheme else src
    return raw.replace("\\", "/").rsplit("/", 1)[-1]


def _rational(value: str | None) -> float:
    if not value:
        return 0.0
    value = value.rstrip("s")
    return float(Fraction(value))


def _parse_fcpxml(root) -> list[TimelineUse]:
    assets: dict[str, str] = {}
    for asset in root.iter("asset"):
        src = asset.get("src") or next(
            (rep.get("src") for rep in asset.iter("media-rep") if rep.get("src")), None
        )
        assets[asset.get("id")] = _basename(src) if src else (asset.get("name") or "")
    spine_roots = list(root.iter("sequence")) or [root]
    uses: list[TimelineUse] = []
    for scope in spine_roots:
        for el in scope.iter("asset-clip", "clip", "video", "audio", "ref-clip"):
            if el.getparent() is not None and el.getparent().tag == "resources":
                continue
            ref = el.get("ref")
            name = assets.get(ref)
            if not name:
                continue
            start = _rational(el.get("start"))
            uses.append(TimelineUse(name, None, start, start + _rational(el.get("duration"))))
    return uses


def _parse_xmeml(root) -> list[TimelineUse]:
    files: dict[str, str] = {}
    for f in root.iter("file"):
        fid = f.get("id")
        pathurl = f.findtext("pathurl")
        name = _basename(pathurl) if pathurl else f.findtext("name")
        if fid and name and fid not in files:
            files[fid] = name
    uses: list[TimelineUse] = []
    for item in root.iter("clipitem"):
        f = item.find("file")
        if f is None:
            continue
        name = files.get(f.get("id")) or f.findtext("name")
        if not name:
            continue
        timebase = float(item.findtext("rate/timebase") or f.findtext("rate/timebase") or 24)
        in_f = float(item.findtext("in") or 0)
        out_f = float(item.findtext("out") or in_f)
        uses.append(TimelineUse(name, None, in_f / timebase, out_f / timebase))
    return uses


_EDL_EVENT = re.compile(
    r"^\d+\s+(\S+)\s+\S+\s+\S+\s+(\d\d:\d\d:\d\d[:;]\d\d)\s+(\d\d:\d\d:\d\d[:;]\d\d)\s+"
)
_EDL_CLIP_NAME = re.compile(r"^\*\s*FROM CLIP NAME:\s*(.+?)\s*$", re.I)


def _tc_seconds(tc: str, fps: float) -> float:
    h, m, s, f = (int(x) for x in re.split(r"[:;]", tc))
    return h * 3600 + m * 60 + s + f / fps


def _parse_edl(text: str, fps: float) -> list[TimelineUse]:
    uses: list[TimelineUse] = []
    pending: tuple[str, float, float] | None = None
    clip_name: str | None = None

    def flush() -> None:
        if pending:
            reel, in_s, out_s = pending
            uses.append(TimelineUse(clip_name or reel, reel, in_s, out_s))

    for line in text.splitlines():
        line = line.strip()
        event = _EDL_EVENT.match(line)
        if event:
            flush()
            clip_name = None
            reel = event.group(1)
            pending = (reel, _tc_seconds(event.group(2), fps), _tc_seconds(event.group(3), fps))
            continue
        name = _EDL_CLIP_NAME.match(line)
        if name:
            clip_name = name.group(1)
    flush()
    return [u for u in uses if u.reel != "BL"]


def build_labels(
    uses: list[TimelineUse], clip_files: list[Path]
) -> LabelResult:
    """Match timeline sources to clip files by filename stem, then reel name."""
    by_stem = {p.stem.lower(): p.name for p in clip_files}
    labels = {p.name: 0 for p in clip_files}
    ranges: dict[str, list[tuple[float, float]]] = {p.name: [] for p in clip_files}
    unmatched: set[str] = set()
    sources: set[str] = set()
    for use in uses:
        sources.add(use.name)
        stem = Path(use.name).stem.lower()
        match = by_stem.get(stem) or (by_stem.get(use.reel.lower()) if use.reel else None)
        if match is None:
            unmatched.add(use.name)
            continue
        labels[match] = 1
        ranges[match].append((use.in_s, use.out_s))
    return LabelResult(labels, ranges, sorted(unmatched), len(sources))


def label_clips(
    clips_dir: str | Path, timeline: str | Path, fps: float = 24.0, max_unmatched: float = 0.10
) -> LabelResult:
    clip_files = sorted(
        p for p in Path(clips_dir).rglob("*") if p.suffix.lower() in VIDEO_EXTS
    )
    result = build_labels(parse_timeline(timeline, fps), clip_files)
    if result.unmatched_fraction >= max_unmatched:
        raise GateFailure(
            f"Gate 2 failed: {result.unmatched_fraction:.1%} of timeline sources unmatched "
            f"(limit {max_unmatched:.0%})\n{result.report()}"
        )
    return result


class GateFailure(RuntimeError):
    pass
