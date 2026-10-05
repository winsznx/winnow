"""Run every arm on one split, score recall at review depths, aggregate with bootstrap CIs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .rank import FeatureMap, tabpfn_scores

DEPTHS = (0.10, 0.20, 0.30)
CROWD = "crowd_oracle (not deployable)"
ARMS = (
    "tabpfn", CROWD, "knn", "chronological", "sharpness_stability",
    "logreg", "tech_only_tabpfn", "shuffled_tabpfn", "duration_only", "generic", "hybrid",
)
KNN_K = 5
BOOTSTRAP_DRAWS = 10_000


@dataclass
class Split:
    emb_seed: np.ndarray
    tech_seed: np.ndarray
    y_seed: np.ndarray
    emb_held: np.ndarray
    tech_held: np.ndarray
    y_held: np.ndarray
    tech_columns: list[str]
    crowd_held: np.ndarray | None = None
    generic_seed: np.ndarray | None = None
    generic_held: np.ndarray | None = None


def recall_at(scores: np.ndarray, y_held: np.ndarray, depth: float, rng: np.random.Generator) -> float:
    """Share of held-out positives found in the top `depth` of the ranking. Ties break randomly."""
    k = max(1, math.ceil(depth * len(scores)))
    order = np.lexsort((rng.random(len(scores)), -scores))
    return float(y_held[order[:k]].sum() / y_held.sum())


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def arm_scores(s: Split, seed: int = 0) -> dict[str, np.ndarray]:
    cols = {c: i for i, c in enumerate(s.tech_columns)}
    fmap = FeatureMap.fit(s.emb_seed)
    x_seed, x_held = fmap.transform(s.emb_seed, s.tech_seed), fmap.transform(s.emb_held, s.tech_held)
    n_held = len(s.y_held)

    positives = s.emb_seed[s.y_seed == 1]
    sims = np.sort(s.emb_held @ positives.T, axis=1)[:, ::-1]
    knn = sims[:, : min(KNN_K, positives.shape[0])].mean(axis=1)

    scaler = StandardScaler().fit(x_seed)
    logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(x_seed), s.y_seed)

    shuffled = np.random.default_rng(seed).permutation(s.y_seed)
    scores = {
        "tabpfn": tabpfn_scores(x_seed, s.y_seed, x_held, seed),
        "knn": knn,
        "chronological": -np.arange(n_held, dtype=float),
        "sharpness_stability": _zscore(s.tech_held[:, cols["sharpness"]]) - _zscore(s.tech_held[:, cols["motion"]]),
        "logreg": logreg.predict_proba(scaler.transform(x_held))[:, 1],
        "tech_only_tabpfn": tabpfn_scores(s.tech_seed, s.y_seed, s.tech_held, seed),
        "shuffled_tabpfn": tabpfn_scores(x_seed, shuffled, x_held, seed) if 0 < shuffled.sum() < len(shuffled) else np.zeros(n_held),
        "duration_only": s.tech_held[:, cols["duration"]].astype(float),
    }
    if s.crowd_held is not None:
        scores[CROWD] = s.crowd_held.astype(float)
    if s.generic_held is not None:
        scores["generic"] = s.generic_held
        scores["hybrid"] = tabpfn_scores(
            np.hstack([x_seed, s.generic_seed[:, None]]), s.y_seed,
            np.hstack([x_held, s.generic_held[:, None]]), seed,
        )
    return scores


def evaluate_split(s: Split, seed: int = 0) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    return {
        arm: {f"{int(d * 100)}": recall_at(sc, s.y_held, d, rng) for d in DEPTHS}
        for arm, sc in arm_scores(s, seed).items()
    }


def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None = None, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean. With groups, resample whole groups (e.g. videos) instead of rows."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        means = values[idx].mean(axis=1)
    else:
        uniq = np.unique(groups)
        by_group = [values[groups == g] for g in uniq]
        picks = rng.integers(0, len(uniq), size=(BOOTSTRAP_DRAWS, len(uniq)))
        sums = np.array([g.sum() for g in by_group])
        counts = np.array([len(g) for g in by_group])
        means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(pairs: list[dict], depth: str = "20") -> dict:
    """pairs: [{'video': str, 'editor': int, 'recall': {arm: {depth: value}}}]."""
    if not pairs:
        raise ValueError("no editor-video pairs survived the seed-positive filter")
    groups = np.array([p["video"] for p in pairs])
    arms = sorted({a for p in pairs for a in p["recall"]})
    table = {}
    for arm in arms:
        table[arm] = {}
        for d in (f"{int(x * 100)}" for x in DEPTHS):
            v = np.array([p["recall"][arm][d] for p in pairs if arm in p["recall"]])
            lo, hi = bootstrap_ci(v)
            table[arm][d] = {"mean": float(v.mean()), "ci95": [lo, hi], "n": len(v)}

    def diff(a: str, b: str) -> dict:
        both = [p for p in pairs if a in p["recall"] and b in p["recall"]]
        v = np.array([p["recall"][a][depth] - p["recall"][b][depth] for p in both])
        g = np.array([p["video"] for p in both])
        lo, hi = bootstrap_ci(v)
        glo, ghi = bootstrap_ci(v, g)
        return {"mean": float(v.mean()), "ci95": [lo, hi], "ci95_by_video": [glo, ghi], "n": len(v)}

    taste, sim = diff("tabpfn", CROWD), diff("tabpfn", "knn")
    verdicts = {
        "learns_personal_taste": {
            "rule": f"tabpfn minus {CROWD} at {depth}% depth, 95% CI lower bound > 0",
            "result": taste,
            "verdict": "PASS" if taste["ci95"][0] > 0 else "NULL",
        },
        "beats_similarity": {
            "rule": f"tabpfn minus knn at {depth}% depth, mean >= 0.10",
            "result": sim,
            "verdict": "PASS" if sim["mean"] >= 0.10 else "NULL",
        },
    }
    comparisons = {}
    if "generic" in arms and "hybrid" in arms:
        added = diff("hybrid", "generic")
        verdicts["personal_adds_value"] = {
            "rule": f"hybrid minus generic at {depth}% depth, 95% bootstrap CI lower bound > 0",
            "result": added,
            "verdict": "PASS" if added["ci95"][0] > 0 else "NULL",
        }
        comparisons["generic_vs_chronological"] = diff("generic", "chronological")
    return {"arms": table, "verdicts": verdicts, "comparisons": comparisons, "n_pairs": len(pairs), "n_videos": len(set(groups))}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    names = ["tabpfn", "torch", "open_clip_torch", "scikit-learn", "numpy", "pandas", "opencv-python", "lxml"]
    return {n: importlib.metadata.version(n) for n in names}


def new_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_results(runs_dir: Path, payload: dict, stamp: str | None = None) -> Path:
    stamp = stamp or new_stamp()
    out = runs_dir / stamp / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": stamp, "versions": package_versions(), **payload}
    out.write_text(json.dumps(payload, indent=2))
    return out
