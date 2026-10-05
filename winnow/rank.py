"""TabPFN v2 ranker over PCA'd embeddings plus technical features."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

PCA_DIMS = 64
MIN_SEED_POSITIVES = 5


@dataclass
class FeatureMap:
    """PCA fit on seed rows only, so held-out data never shapes the projection."""

    pca: PCA

    @classmethod
    def fit(cls, emb_seed: np.ndarray, dims: int = PCA_DIMS) -> "FeatureMap":
        n_components = min(dims, emb_seed.shape[0] - 1, emb_seed.shape[1])
        return cls(PCA(n_components=n_components, random_state=0).fit(emb_seed))

    def transform(self, emb: np.ndarray, tech: np.ndarray) -> np.ndarray:
        return np.hstack([self.pca.transform(emb), tech])


def tabpfn_scores(
    x_seed: np.ndarray, y_seed: np.ndarray, x_held: np.ndarray, seed: int = 0
) -> np.ndarray:
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device="cpu", random_state=seed, ignore_pretraining_limits=True)
    clf.fit(x_seed, y_seed)
    classes = list(clf.classes_)
    return clf.predict_proba(x_held)[:, classes.index(1)]


def can_train(y_seed: np.ndarray, min_positives: int) -> bool:
    positives = int(y_seed.sum())
    return positives >= min_positives and positives < len(y_seed)


@dataclass
class Ranker:
    feature_map: FeatureMap
    x_seed: np.ndarray
    y_seed: np.ndarray

    def score(self, emb: np.ndarray, tech: np.ndarray) -> np.ndarray:
        """TabPFN is in-context: the 'fitted model' is the seed set plus the PCA."""
        return tabpfn_scores(self.x_seed, self.y_seed, self.feature_map.transform(emb, tech))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Ranker":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_ranker(
    emb_seed: np.ndarray, tech_seed: np.ndarray, y_seed: np.ndarray, min_positives: int = MIN_SEED_POSITIVES
) -> Ranker | None:
    """None means the caller must fall back to chronological order and say why."""
    if not can_train(y_seed, min_positives):
        return None
    fmap = FeatureMap.fit(emb_seed)
    return Ranker(fmap, fmap.transform(emb_seed, tech_seed), y_seed)
