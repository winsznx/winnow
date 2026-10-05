"""Per-clip features: SigLIP embedding (mean of frames) plus technical stats."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"
TECH_COLUMNS = ["sharpness", "motion", "luma", "duration"]
EMBED_BATCH = 32


@dataclass(frozen=True)
class ClipFrames:
    key: str
    frames: list[Path]
    duration: float


def technical_features(frames: list[Path], duration: float) -> dict[str, float]:
    grays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY) for p in frames]
    sharpness = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]))
    luma = float(np.mean([g.mean() for g in grays]))
    small = [cv2.resize(g, (128, 72)).astype(np.float32) for g in grays]
    diffs = [np.abs(a - b).mean() for a, b in zip(small, small[1:])]
    motion = float(np.mean(diffs)) if diffs else 0.0
    return {"sharpness": sharpness, "motion": motion, "luma": luma, "duration": duration}


class Embedder:
    """SigLIP image embeddings via open_clip, loaded on first use."""

    def __init__(self) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), EMBED_BATCH):
                batch = self.torch.stack(
                    [self.preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + EMBED_BATCH]]
                ).to(self.device)
                vec = self.model.encode_image(batch)
                vec = vec / vec.norm(dim=-1, keepdim=True)
                out.append(vec.float().cpu().numpy())
        return np.concatenate(out)


def build_features(clips: list[ClipFrames], cache_path: Path) -> pd.DataFrame:
    """Compute features for clips missing from cache_path, then return all requested rows."""
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    have = set(cached["key"]) if len(cached) else set()
    todo = [c for c in clips if c.key not in have and c.frames]
    if todo:
        embedder = Embedder()
        flat = [p for c in todo for p in c.frames]
        vecs = embedder.embed_paths(flat)
        rows, cursor = [], 0
        for c in todo:
            clip_vecs = vecs[cursor : cursor + len(c.frames)]
            cursor += len(c.frames)
            mean = clip_vecs.mean(axis=0)
            rows.append(
                {"key": c.key, "emb": (mean / np.linalg.norm(mean)).astype(np.float32).tolist(),
                 **technical_features(c.frames, c.duration)}
            )
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path)
    wanted = {c.key for c in clips}
    return cached[cached["key"].isin(wanted)].reset_index(drop=True)
