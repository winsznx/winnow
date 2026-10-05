"""Time-ordered seed/held-out split. Nothing crosses the boundary."""
from __future__ import annotations

import math

import numpy as np


def time_split(n_clips: int, seed_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Clips must already be sorted by time. First block is seed, the rest is held out."""
    if not 0 < seed_fraction < 1:
        raise ValueError("seed_fraction must be in (0, 1)")
    cut = max(1, math.floor(n_clips * seed_fraction))
    idx = np.arange(n_clips)
    return idx[:cut], idx[cut:]
