"""Ranked markers for an editor, plus the human-readable report."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

TIERS = (("Green", 1 / 3), ("Yellow", 2 / 3), ("Red", 1.0))
FPS = 25


def tier_colors(n: int) -> list[str]:
    """Colour by rank tercile: top third green, middle yellow, bottom red."""
    colors = []
    for rank in range(n):
        frac = (rank + 1) / n
        colors.append(next(name for name, bound in TIERS if frac <= bound + 1e-9))
    return colors


def timecode(seconds: float, fps: int = FPS) -> str:
    frames = int(round(seconds * fps))
    h, rem = divmod(frames, 3600 * fps)
    m, rem = divmod(rem, 60 * fps)
    s, f = divmod(rem, fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def write_markers_csv(path: Path, names: list[str], durations: list[float], scores: np.ndarray) -> None:
    """One row per clip, best first. Marker name is the score. Timecodes are cumulative
    positions in a ranked-order sequence, so the list reads as a review queue."""
    order = np.argsort(-scores, kind="stable")
    colors = tier_colors(len(order))
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Marker Name", "Source Clip", "Timecode In", "Color", "Rank"])
        for rank, i in enumerate(order, start=1):
            w.writerow([f"{scores[i]:.3f}", names[i], timecode(cursor), colors[rank - 1], rank])
            cursor += durations[i]


def write_markers_fcpxml(path: Path, names: list[str], paths: list[Path], durations: list[float],
                         scores: np.ndarray) -> None:
    order = np.argsort(-scores, kind="stable")
    colors = tier_colors(len(order))
    frame = f"1/{FPS}s"
    assets, clips, cursor = [], [], 0
    for rank, i in enumerate(order, start=1):
        frames = max(1, int(round(durations[i] * FPS)))
        dur = f"{frames}/{FPS}s"
        assets.append(
            f'<asset id="a{i}" name={quoteattr(names[i])} start="0s" duration="{dur}" '
            f'hasVideo="1" format="r1"><media-rep kind="original-media" src={quoteattr(paths[i].resolve().as_uri())}/></asset>'
        )
        clips.append(
            f'<asset-clip ref="a{i}" offset="{cursor}/{FPS}s" name={quoteattr(names[i])} start="0s" duration="{dur}">'
            f'<marker start="0s" duration="{frame}" value="{scores[i]:.3f} {colors[rank - 1]}"/></asset-clip>'
        )
        cursor += frames
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.9"><resources>'
        f'<format id="r1" name="FFVideoFormat1080p25" frameDuration="{frame}" width="1920" height="1080"/>'
        + "".join(assets)
        + '</resources><library><event name="Winnow ranked"><project name="Winnow ranked">'
        f'<sequence format="r1" duration="{cursor}/{FPS}s" tcStart="0s" tcFormat="NDF"><spine>'
        + "".join(clips)
        + "</spine></sequence></project></event></library></fcpxml>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)


def _cell(entry) -> str:
    if isinstance(entry, dict):
        lo, hi = entry.get("ci95", (None, None))
        text = f"{entry['mean']:.2f}"
        return text if lo is None else f"{text} [{lo:.2f}, {hi:.2f}]"
    return f"{entry:.2f}"


def results_table(results: dict) -> str:
    if not results["arms"]:
        return "No arms evaluated (fallback to chronological order)."
    depths = sorted(next(iter(results["arms"].values())), key=int)
    head = "| Arm | " + " | ".join(f"Recall @{d}%" for d in depths) + " |"
    rule = "| --- |" + " --- |" * len(depths)
    rows = [f"| {arm} | " + " | ".join(_cell(cells[d]) for d in depths) + " |"
            for arm, cells in results["arms"].items()]
    return "\n".join([head, rule, *rows])


def verdict_lines(results: dict) -> list[str]:
    out = []
    for name, v in results["verdicts"].items():
        r = v["result"]
        ci = f", 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" if "ci95" in r else ""
        by_video = (f", by-video CI [{r['ci95_by_video'][0]:+.3f}, {r['ci95_by_video'][1]:+.3f}]"
                    if "ci95_by_video" in r else "")
        out.append(f"- {name}: **{v['verdict']}** ({v['rule']}; difference {r['mean']:+.3f}{ci}{by_video})")
    return out


def write_report(path: Path, results: dict, unmatched: list[str] | None = None) -> None:
    lines = ["# Winnow report", "", f"Run: {results['timestamp_utc']}", ""]
    if "dataset" in results:
        d = results["dataset"]
        lines += [f"Dataset: {d['name']} ({results['n_videos']} videos, {results['n_editors']} editors, "
                  f"{results['n_pairs']} pairs used, {results['n_pairs_skipped']} skipped)", ""]
    else:
        lines += [f"Clips: {results['n_clips']}, selects: {results['n_selects']}, "
                  f"unmatched timeline sources: {results['n_unmatched']}", ""]
    lines += ["## Recall of held-out picks", "", results_table(results), "", "## Verdicts", ""]
    lines += verdict_lines(results)
    if unmatched is not None:
        lines += ["", "## Unmatched timeline sources", ""]
        lines += [f"- {u}" for u in unmatched] or ["None"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def report_from_file(results_path: Path, out_path: Path) -> None:
    write_report(out_path, json.loads(results_path.read_text()))
