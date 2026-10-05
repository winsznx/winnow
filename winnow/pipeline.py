"""winnow run: rank one real shoot from the editor's final timeline."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .evaluate import DEPTHS, Split, evaluate_split, file_sha256, write_results
from .export import write_markers_csv, write_markers_fcpxml, write_report
from .features import TECH_COLUMNS, ClipFrames, build_features
from .labels import VIDEO_EXTS, label_clips
from .probe import ClipInfo, probe_all
from .rank import MIN_SEED_POSITIVES, fit_ranker
from .sample import clip_key, sample_frames
from .split import time_split

PASS_POINTS = 0.10


def _start_seconds(info: ClipInfo) -> float | None:
    tc = info.start_timecode
    if not tc or not re.fullmatch(r"\d\d[:;]\d\d[:;]\d\d[:;]\d\d", tc):
        return None
    h, m, s, f = (int(x) for x in re.split(r"[:;]", tc))
    return h * 3600 + m * 60 + s + f / (info.fps or 25)


def time_key(info: ClipInfo) -> tuple:
    """Start timecode, then container creation time, then filename."""
    return (_start_seconds(info) if _start_seconds(info) is not None else float("inf"),
            info.creation_time or "", info.path.name)


def run_pipeline(clips_dir: Path, timeline: Path, seed_fraction: float, fps: float, project: Path) -> Path:
    labelled = label_clips(clips_dir, timeline, fps)
    print(labelled.report())
    paths = sorted(p for p in Path(clips_dir).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    infos, unreadable = probe_all(paths)
    print(f"probed {len(infos)} clips, skipped {unreadable} unreadable")
    infos.sort(key=time_key)

    cache = project / "cache"
    clips = [ClipFrames(clip_key(i.path), sample_frames(i.path, cache, i.duration), i.duration) for i in infos]
    table = build_features(clips, cache / "features.parquet").set_index("key")
    ordered = table.loc[[c.key for c in clips]]
    emb = np.stack(ordered["emb"].to_numpy())
    tech = ordered[TECH_COLUMNS].to_numpy(dtype=float)
    y = np.array([labelled.labels[i.path.name] for i in infos])

    seed_idx, held_idx = time_split(len(infos), seed_fraction)
    y_seed, y_held = y[seed_idx], y[held_idx]
    if y_held.sum() == 0:
        raise RuntimeError("no selects in the held-out block, nothing to evaluate")
    ranker = fit_ranker(emb[seed_idx], tech[seed_idx], y_seed, MIN_SEED_POSITIVES)
    names = [i.path.name for i in infos]
    out = project / "out"
    if ranker is None:
        print(f"only {int(y_seed.sum())} positive seeds (need {MIN_SEED_POSITIVES}): "
              "falling back to chronological order, no learned arms evaluated")
        scores, recall = -np.arange(len(infos), dtype=float), None
    else:
        ranker.save(out / "model.pkl")
        scores = np.empty(len(infos))
        scores[seed_idx] = ranker.score(emb[seed_idx], tech[seed_idx])
        scores[held_idx] = ranker.score(emb[held_idx], tech[held_idx])
        split = Split(emb[seed_idx], tech[seed_idx], y_seed, emb[held_idx], tech[held_idx], y_held, TECH_COLUMNS)
        recall = evaluate_split(split)
    manifest = hashlib.sha256("\n".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode()).hexdigest()
    results = {
        "n_clips": len(infos), "n_selects": int(y.sum()), "n_seed": len(seed_idx), "n_held": len(held_idx),
        "n_unmatched": len(labelled.unmatched_sources), "unmatched_sources": labelled.unmatched_sources,
        "n_unreadable": unreadable, "protocol": {"seed_fraction": seed_fraction, "depths": list(DEPTHS)},
        "inputs": {"timeline_sha256": file_sha256(timeline), "clip_manifest_sha256": manifest},
        "fallback_chronological": recall is None,
        "arms": {}, "verdicts": {},
    }
    if recall is not None:
        diff = recall["tabpfn"]["20"] - recall["knn"]["20"]
        results["arms"] = {a: {d: {"mean": v, "n": 1} for d, v in cells.items()} for a, cells in recall.items()}
        results["verdicts"] = {"beats_similarity": {
            "rule": f"tabpfn minus knn at 20% depth >= {PASS_POINTS}, single split, no CI",
            "result": {"mean": diff}, "verdict": "PASS" if diff >= PASS_POINTS else "NULL"}}
    path = write_results(project / "runs", results)
    write_markers_csv(out / "markers.csv", names, [i.duration for i in infos], scores)
    write_markers_fcpxml(out / "markers.fcpxml", names, [i.path for i in infos], [i.duration for i in infos], scores)
    write_report(out / "report.md", {**results, "timestamp_utc": path.parent.name}, labelled.unmatched_sources)
    print(f"results: {path}")
    for name, v in results["verdicts"].items():
        print(f"verdict {name}: {v['verdict']} ({v['result']['mean']:+.3f})")
    return path
