import numpy as np

from winnow.datasets.summe import segment_labels
from winnow.evaluate import CROWD, bootstrap_ci, recall_at, summarize
from winnow.export import tier_colors, timecode
from winnow.split import time_split


def test_time_split_keeps_order_and_boundary():
    seed, held = time_split(20, 0.25)
    assert list(seed) == [0, 1, 2, 3, 4]
    assert held[0] == 5 and held[-1] == 19


def test_segment_labels_use_strict_majority_coverage():
    fps = 10
    score = np.zeros((60, 1), dtype=int)
    score[0:10, 0] = 1   # covers half of segment 0 (20 frames): not > 50%
    score[20:31, 0] = 1  # covers 11/20 of segment 1: kept
    labels = segment_labels(score, fps, 3)
    assert labels[:, 0].tolist() == [0, 1, 0]


def test_recall_at_perfect_and_worst_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    rng = np.random.default_rng(0)
    assert recall_at(np.array([9, 8, 1, 1, 1, 1, 1, 1, 1, 1.0]), y, 0.2, rng) == 1.0
    assert recall_at(-np.array([9, 8, 1, 1, 1, 1, 1, 1, 1, 1.0]), y, 0.2, rng) == 0.0


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(1).normal(0.3, 0.1, 200)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi


def test_summarize_verdicts():
    def pair(video, tab, crowd, knn, generic=0.2, hybrid=0.2):
        cell = lambda x: {"10": x, "20": x, "30": x}
        return {"video": video, "editor": 0, "recall": {
            "tabpfn": cell(tab), CROWD: cell(crowd), "knn": cell(knn), "generic": cell(generic),
            "hybrid": cell(hybrid), "chronological": cell(0.1)}}

    better = summarize([pair(f"v{i}", 0.6, 0.3, 0.4) for i in range(10)])["verdicts"]
    assert better["learns_personal_taste"]["verdict"] == "PASS"
    assert better["beats_similarity"]["verdict"] == "PASS"
    worse = summarize([pair(f"v{i}", 0.3, 0.6, 0.29) for i in range(10)])["verdicts"]
    assert worse["learns_personal_taste"]["verdict"] == "NULL"
    assert worse["beats_similarity"]["verdict"] == "NULL"


def test_tier_colors_and_timecode():
    assert tier_colors(6) == ["Green", "Green", "Yellow", "Yellow", "Red", "Red"]
    assert timecode(61.2) == "00:01:01:05"


def test_personal_adds_value_needs_ci_above_zero():
    def pair(video, hybrid, generic):
        cell = lambda x: {"10": x, "20": x, "30": x}
        return {"video": video, "editor": 0, "recall": {
            "tabpfn": cell(0.3), CROWD: cell(0.5), "knn": cell(0.2), "generic": cell(generic),
            "hybrid": cell(hybrid), "chronological": cell(0.1)}}

    gain = summarize([pair(f"v{i}", 0.5 + 0.01 * i, 0.3) for i in range(10)])
    assert gain["verdicts"]["personal_adds_value"]["verdict"] == "PASS"
    assert gain["comparisons"]["generic_vs_chronological"]["mean"] > 0
    flat = summarize([pair(f"v{i}", 0.3 + (0.05 if i % 2 else -0.05), 0.3) for i in range(10)])
    assert flat["verdicts"]["personal_adds_value"]["verdict"] == "NULL"
