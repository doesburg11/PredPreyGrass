"""swarm_metrics on hand-made configurations."""

import math

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import swarm_metrics as sm

W = H = 800.0


def test_distances_wrap_across_the_edges():
    positions = np.array([[5.0, 400.0], [795.0, 400.0], [400.0, 400.0]])
    d = sm.torus_distances(positions, W, H)
    assert d[0, 1] == pytest.approx(10.0)
    assert d[0, 2] == pytest.approx(395.0)
    assert math.isinf(d[0, 0])
    np.testing.assert_allclose(sm.nearest_neighbour_distances(positions, W, H), [10, 10, 395])


def test_random_null_matches_clark_evans_for_many_animals():
    n = 200
    expected = 0.5 / math.sqrt(n / (W * H))  # Clark & Evans (1954), no edges on a torus
    assert sm.random_nearest_neighbour_mean(n, W, H, samples=50) == pytest.approx(expected, rel=0.05)


def test_alignment_parallel_opposite_and_lone():
    positions = np.array([[100.0, 100.0], [130.0, 100.0], [600.0, 600.0]])
    same = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(sm.local_alignment(positions, same, W, H, 50), [1.0, 1.0])  # the lone one is left out
    opposite = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(sm.local_alignment(positions, opposite, W, H, 50), [-1.0, -1.0])
    focal = np.array([True, False, False])
    np.testing.assert_allclose(sm.local_alignment(positions, same, W, H, 50, focal=focal), [1.0])
    eligible = np.array([True, False, True])
    assert len(sm.local_alignment(positions, same, W, H, 50, eligible)) == 0


def test_shuffled_alignment_is_near_zero_for_random_headings():
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 200, size=(60, 2))
    angles = rng.uniform(0, 2 * math.pi, 60)
    velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    values = sm.shuffled_alignment(positions, velocities, W, H, 80, np.ones(60, bool), rng)
    assert abs(values.mean()) < 0.1


def test_shuffling_keeps_alignment_of_a_uniform_drift():
    """All moving the same way is not local coordination: the null keeps it."""
    rng = np.random.default_rng(2)
    positions = rng.uniform(0, 800, size=(30, 2))
    velocities = np.tile([1.0, 0.0], (30, 1))
    values = sm.shuffled_alignment(positions, velocities, W, H, 200, np.ones(30, bool), rng)
    assert values.mean() == pytest.approx(1.0)


def test_polarization():
    assert sm.polarization(np.array([[1.0, 0.0], [3.0, 0.0]])) == pytest.approx(1.0)
    assert sm.polarization(np.array([[1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(0.0)
    assert math.isnan(sm.polarization(np.empty((0, 2))))


def test_clusters_chain_and_wrap():
    positions = np.array([[790.0, 10.0], [10.0, 10.0], [70.0, 10.0], [400.0, 400.0]])
    labels = sm.clusters(positions, W, H, link=61)
    assert labels[0] == labels[1] == labels[2] != labels[3]
    np.testing.assert_array_equal(sm.group_sizes(labels), [3, 3, 3, 1])


def test_pair_tracker_spells_exclusion_and_censoring():
    tracker = sm.PairTracker(excluded=lambda a, b: {a, b} == {"a", "c"})
    near = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    far = np.array([[0.0, 0.0], [300.0, 0.0], [20.0, 0.0]])
    for positions in (near, near, far, near):
        tracker.update(["a", "b", "c"], positions, W, H, 50)
    finished, censored = tracker.close()
    # a-b: 2 decisions, parted; b-c the same; a-c excluded; a-b and b-c again at the end.
    assert sorted(finished) == [2, 2]
    assert sorted(censored) == [1, 1]


def test_pair_tracker_censors_a_spell_ended_by_death():
    tracker = sm.PairTracker()
    tracker.update(["a", "b"], np.array([[0.0, 0.0], [10.0, 0.0]]), W, H, 50)
    tracker.update(["a", "b"], np.array([[0.0, 0.0], [10.0, 0.0]]), W, H, 50)
    tracker.update(["a"], np.array([[0.0, 0.0]]), W, H, 50)  # b died
    assert tracker.close() == ([], [2])


def test_kaplan_meier_with_and_without_censoring():
    assert sm.km_median([1, 2, 3, 4, 5], []) == 3
    assert sm.km_share_lasting([1, 2, 3, 4], [], 3) == pytest.approx(0.5)
    # Spells cut short at 1 still count as lasting at least 1: the median rises.
    assert sm.km_median([2, 3, 4], [1, 1, 1, 1]) == 3
    assert sm.km_median([1, 2, 3, 4], []) <= sm.km_median([1, 2, 3, 4], [10, 10, 10])
    assert math.isnan(sm.km_median([1], [5, 5, 5]))


def test_group_tracker_continues_on_overlap_and_ends():
    tracker = sm.GroupTracker(min_size=3, min_overlap=0.5)
    tracker.update(["a", "b", "c", "d"], [0, 0, 0, 1])  # group abc
    tracker.update(["a", "b", "c", "d"], [0, 0, 0, 0])  # abcd: Jaccard 3/4, continues
    tracker.update(["a", "b", "c", "d"], [0, 1, 2, 3])  # gone
    tracker.update(["x", "y", "z"], [5, 5, 5])  # a new group
    finished, censored = tracker.close()
    assert finished == [2]
    assert censored == [1]
