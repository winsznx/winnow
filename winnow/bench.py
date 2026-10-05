"""Per-video, per-editor benchmark on SumMe segments, leave-one-video-out for generic taste."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .datasets import summe
from .evaluate import DEPTHS, Split, evaluate_split, file_sha256, new_stamp, summarize, write_results
from .features import TECH_COLUMNS, ClipFrames, build_features
from .rank import PCA_DIMS
from .sample import clip_key, sample_frames
from .split import time_split

MIN_SEED_POSITIVES = 3
RANDOM_SEED_VARIANT = "random_seed_secondary"

PREREGISTRATION = {
    "written_before_run": True,
    "primary_split": "first 25% of segments by time is seed, rest held out",
    "secondary_split": "random 25% of segments is seed (content shift not blocked), labeled secondary",
    "arms_added": {
        "generic": "logistic regression on all users' labels from the other 24 videos, no labels from this video. "
                   "Logreg replaces TabPFN because the stacked training set (~33k rows) exceeds TabPFN's 10k-row limit. "
                   "PCA(64) fit on the other videos' embeddings, C=1, no tuning.",
        "hybrid": "TabPFN on this editor's seed rows with the generic score (probability) as an extra feature",
        "personal": "the existing tabpfn arm",
    },
    "renamed": {"crowd": "crowd_oracle (not deployable)"},
    "verdict": {
        "personal_adds_value": "hybrid minus generic at 20% depth, 95% bootstrap CI lower bound > 0 -> PASS, else NULL",
    },
    "also_reported": ["generic minus chronological at 20% depth"],
    "unchanged": ["learns_personal_taste", "beats_similarity", "skip rule: fewer than 3 seed positives"],
}


def video_features(video: summe.SummeVideo, cache_dir: Path):
    clips = []
    for i in range(video.n_segments):
        start = video.segment_start(i)
        frames = sample_frames(video.path, cache_dir, video.duration, start, summe.SEGMENT_SECONDS)
        clips.append(ClipFrames(clip_key(video.path, start, summe.SEGMENT_SECONDS), frames, summe.SEGMENT_SECONDS))
    table = build_features(clips, cache_dir / "features.parquet").set_index("key")
    ordered = table.loc[[c.key for c in clips]]
    return np.stack(ordered["emb"].to_numpy()), ordered[TECH_COLUMNS].to_numpy(dtype=float)


def generic_scores(target: str, data: dict) -> np.ndarray:
    """Probability a segment is kept by a typical user, learned only from the other videos."""
    others = [n for n in data if n != target]
    emb = np.vstack([data[n]["emb"] for n in others])
    tech = np.vstack([data[n]["tech"] for n in others])
    pca = PCA(n_components=PCA_DIMS, random_state=0).fit(emb)
    scaler = StandardScaler().fit(np.hstack([pca.transform(emb), tech]))
    users = [data[n]["video"].labels.shape[1] for n in others]
    x = np.concatenate([np.tile(np.hstack([pca.transform(data[n]["emb"]), data[n]["tech"]]), (u, 1))
                        for n, u in zip(others, users)])
    y = np.concatenate([data[n]["video"].labels[:, u] for n in others for u in range(data[n]["video"].labels.shape[1])])
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(x), y)
    own = np.hstack([pca.transform(data[target]["emb"]), data[target]["tech"]])
    return model.predict_proba(scaler.transform(own))[:, 1]


def split_indices(n: int, seed_fraction: float, variant: str, video_seed: int):
    if variant == "primary":
        return time_split(n, seed_fraction)
    cut = max(1, int(n * seed_fraction))
    perm = np.random.default_rng(video_seed).permutation(n)
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def run_video(video, emb, tech, generic, seed_fraction, variant, video_seed):
    seed_idx, held_idx = split_indices(video.n_segments, seed_fraction, variant, video_seed)
    pairs, skipped = [], {"few_seed_positives": 0, "no_seed_negatives": 0, "no_heldout_positives": 0}
    for editor in range(video.labels.shape[1]):
        y = video.labels[:, editor]
        y_seed, y_held = y[seed_idx], y[held_idx]
        if y_seed.sum() < MIN_SEED_POSITIVES:
            skipped["few_seed_positives"] += 1
        elif y_seed.sum() == len(y_seed):
            skipped["no_seed_negatives"] += 1
        elif y_held.sum() == 0:
            skipped["no_heldout_positives"] += 1
        else:
            others = np.delete(video.labels, editor, axis=1)
            split = Split(
                emb[seed_idx], tech[seed_idx], y_seed, emb[held_idx], tech[held_idx], y_held,
                TECH_COLUMNS, crowd_held=others[held_idx].mean(axis=1),
                generic_seed=generic[seed_idx], generic_held=generic[held_idx],
            )
            pairs.append({"video": video.name, "editor": editor, "n_seed": len(seed_idx),
                          "n_held": len(held_idx), "seed_positives": int(y_seed.sum()),
                          "held_positives": int(y_held.sum()), "recall": evaluate_split(split, seed=editor)})
    return pairs, skipped


def format_video_line(name: str, variant: str, pairs: list[dict], skipped: dict) -> str:
    arms = sorted(pairs[0]["recall"]) if pairs else []
    body = "  ".join(f"{a.split(' ')[0]}={np.mean([p['recall'][a]['20'] for p in pairs]):.2f}" for a in arms)
    return f"[{variant}] {name}: {len(pairs)} pairs, skipped {sum(skipped.values())}  recall@20% {body}"


def _block(pairs: list[dict], skipped: dict) -> dict:
    summary = summarize(pairs)
    return {"n_videos": summary["n_videos"], "n_pairs": summary["n_pairs"],
            "n_pairs_skipped": sum(skipped.values()), "skipped_breakdown": skipped,
            "arms": summary["arms"], "verdicts": summary["verdicts"], "comparisons": summary["comparisons"]}


def run_bench(root: Path, cache_dir: Path, runs_dir: Path, seed_fraction: float = 0.25,
              only: list[str] | None = None, deadline_utc: str | None = None) -> Path:
    stamp = new_stamp()
    run_dir = runs_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preregistration.json").write_text(json.dumps(PREREGISTRATION, indent=2))

    names = summe.video_names(root)
    data = {}
    for n in names:
        video = summe.load_video(root, n)
        emb, tech = video_features(video, cache_dir)
        data[n] = {"video": video, "emb": emb, "tech": tech}
    targets = only or names

    pairs = {"primary": [], RANDOM_SEED_VARIANT: []}
    skipped = {"primary": {}, RANDOM_SEED_VARIANT: {}}
    done, partial = [], False
    for idx, name in enumerate(targets):
        if deadline_utc and datetime.now(timezone.utc).strftime("%H:%M") >= deadline_utc:
            partial = True
            break
        d = data[name]
        generic = generic_scores(name, data)
        for variant, key in (("primary", "primary"), ("random", RANDOM_SEED_VARIANT)):
            p, sk = run_video(d["video"], d["emb"], d["tech"], generic, seed_fraction, variant, idx)
            pairs[key] += p
            for k, v in sk.items():
                skipped[key][k] = skipped[key].get(k, 0) + v
            print(format_video_line(name, key, p, sk), flush=True)
        done.append(name)

    primary = _block(pairs["primary"], skipped["primary"])
    payload = {
        "dataset": {"name": summe.DATASET_NAME, "url": summe.DATASET_URL, "license": summe.DATASET_LICENSE,
                    "archive_sha256": file_sha256(root / "SumMe.zip") if (root / "SumMe.zip").exists() else None,
                    "segment_seconds": summe.SEGMENT_SECONDS, "coverage_threshold": summe.COVERAGE_THRESHOLD},
        "protocol": {"seed_fraction": seed_fraction, "depths": list(DEPTHS), "min_seed_positives": MIN_SEED_POSITIVES},
        "preregistration": PREREGISTRATION,
        "partial": partial, "videos_completed": done, "videos_not_run": [n for n in targets if n not in done],
        "n_editors": int(sum(data[n]["video"].labels.shape[1] for n in done)),
        **primary,
        RANDOM_SEED_VARIANT: _block(pairs[RANDOM_SEED_VARIANT], skipped[RANDOM_SEED_VARIANT]),
        "pairs": pairs["primary"],
        f"pairs_{RANDOM_SEED_VARIANT}": pairs[RANDOM_SEED_VARIANT],
    }
    return write_results(runs_dir, payload, stamp)
