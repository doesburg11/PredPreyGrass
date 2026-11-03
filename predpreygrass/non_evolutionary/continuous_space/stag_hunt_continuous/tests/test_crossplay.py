"""Tests of the cross-play assignment of policies."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.crossplay import cell_modules, cells, paired_ratio, same_episode, same_world, summarise_cell
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import MOVES, SPECIES, StagHuntContinuous, new_hunt_stats, species_of
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.prey_switching import new_bins, play_episode


def test_every_assignment_once():
    assert len(cells()) == 2 ** len(SPECIES) == len(set(cells()))


def test_cell_takes_each_species_from_its_source():
    modules = {source: {species: (source, species) for species in SPECIES} for source in "AB"}
    for cell in cells():
        chosen = cell_modules(modules, cell)
        assert chosen == {species: (source, species) for species, source in zip(SPECIES, cell)}


def test_worlds_must_match():
    assert same_world({"a": 1}, {"a": 1}) == {"a": 1}
    with pytest.raises(ValueError, match="differ in .'a'"):
        same_world({"a": 1}, {"a": 2})


def row(seed, caught, exposure):
    bins = new_bins()
    bins[0].update(catches=caught, rabbit_hunter_steps=exposure)
    return {"seed": seed, "bins": bins}


def test_paired_ratio_pools_episodes_and_needs_matching_seeds():
    a = [row(0, 2, 10), row(1, 4, 10)]
    b = [row(0, 1, 10), row(1, 2, 10)]
    ratio, (low, high) = paired_ratio(a, b, "catches", "rabbit_hunter_steps", 0)
    assert ratio == pytest.approx(2.0) and low == pytest.approx(2.0) and high == pytest.approx(2.0)
    with pytest.raises(ValueError, match="seeds"):
        paired_ratio(a, b[::-1], "catches", "rabbit_hunter_steps", 0)


def test_paired_ratio_without_exposure_is_not_a_number():
    ratio, interval = paired_ratio([row(0, 0, 0)], [row(0, 0, 0)], "catches", "rabbit_hunter_steps", 0)
    assert ratio != ratio and interval is None


def test_same_episode_flags_differing_plays():
    played = {"seed": 3, "outcome": "coexisting", "hunt_stats": {"rabbits_caught": 4}}
    same_episode(played, {"outcome": "coexisting", "hunt_stats": {"rabbits_caught": 4}})
    with pytest.raises(ValueError, match="seed 3"):
        same_episode(played, {"outcome": "coexisting", "hunt_stats": {"rabbits_caught": 5}})


def test_summary_of_a_cell_counts_survivors_and_food():
    env = StagHuntContinuous({"predator_count": 12, "mammoth_count": 6, "max_time_steps": 300})
    rng = np.random.default_rng(0)

    def choose(obs):
        return {a: (np.array([rng.integers(MOVES), rng.integers(2)]) if species_of(a) == "predator" else int(rng.integers(MOVES))) for a in obs}

    measured = [play_episode(env, choose, 0)]
    stats = dict(new_hunt_stats(), mammoth_energy=30.0, rabbit_energy=10.0, rabbits_caught=7)
    played = [{"seed": 0, "outcome": "only_mammoth", "extinct_at": {"predator": 5, "rabbit": 9}, "hunt_stats": stats, "tracked": {"encounter_episodes": 0}, "null": {"encounter_episodes": 0}, "physics_steps": 9}]
    summary = summarise_cell(("A", "B", "A"), played, measured)
    assert summary["cell"] == {"predator": "A", "rabbit": "B", "mammoth": "A"}
    assert summary["survived"] == {"predator": 0, "rabbit": 0, "mammoth": 1}
    assert summary["mammoth_food_share"] == pytest.approx(0.75)
    assert summary["rabbits_caught_per_episode"] == 7
    assert summary["joint_arrival_rate"] is None
