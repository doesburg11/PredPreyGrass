"""Tests for analyze_band_scatter.py: group classification and the summary on synthetic data."""
import numpy as np

from predpreygrass.non_evolutionary.predator_bands import analyze_band_scatter as sc


def test_group_of_classification():
    moved_at = {"a": 100, "b": 100, "c": 100}
    assert sc.group_of("x", "couple_male", moved_at, 500) == "founder"
    assert sc.group_of("y", "born", moved_at, 500) == "born"
    assert sc.group_of("a", "born", moved_at, 110) == "moved <25"          # 10 steps since the move
    assert sc.group_of("b", "single_female", moved_at, 150) == "moved 25-100"
    assert sc.group_of("c", "child", moved_at, 300) == "moved >100"
    assert sc.group_of("a", "born", moved_at, 125) == "moved 25-100"       # boundary: 25 steps since the move
    assert sc.group_of("a", "born", moved_at, 200) == "moved 25-100"       # boundary: 100 steps since the move


def test_summarize_reports_every_group_and_handles_empty_ones():
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(6):
        acc = {g: np.zeros(4) for g in sc.GROUPS}
        far = {g: 0 for g in sc.GROUPS}
        acc["founder"] = np.array([10.0, 30.0, 10.0, 4.0])  # n, sum centroid distance, n with a band-mate, n out of range
        far["founder"] = 5
        episodes.append({"acc": acc, "far": far})
    lines = sc.summarize("t", episodes, 8.0, rng)
    text = "\n".join(lines)
    assert "founder" in text and "no observations" in text
    founder = next(l for l in lines if l.strip().startswith("founder"))
    assert "+3.00" in founder and "+0.400" in founder and "+0.500" in founder  # 30/10, 4/10, 5/10
