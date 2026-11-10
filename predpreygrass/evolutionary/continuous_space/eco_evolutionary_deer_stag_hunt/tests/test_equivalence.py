"""With the gene inputs removed, an episode here is deer_stag_hunt's, given
the same seed and the same actions (environment.py, module docstring):
positions, velocities, energies, cooldowns, agents, rewards, end flags and
observations. Skipped without deer_stag_hunt."""

import numpy as np
import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import DEFAULT_CONFIG, EvolvingDeerStagHunt
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.genes import GENES

base = pytest.importorskip("predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment")


# deer_stag_hunt's gradual-grass settings, which this module does not have
# (its grass is always instant, deer_stag_hunt's default).
GRASS_ONLY_THERE = {"grass_regrowth": "instant", "grass_growth_rate": 0.0187, "grass_root_reserve": 0.01, "grass_bite": 0.1, "grass_stubble": 0.2, "grass_dieback_delay": 400}


def base_config(config):
    return {**GRASS_ONLY_THERE, **{k: v for k, v in config.items() if not k.startswith("gene_")}}


def state(env):
    return [(a.id, a.position.x, a.position.y, a.velocity.x, a.velocity.y, a.energy, a.catch_ready_at, a.orientation) for s in ("predator", "deer") for a in env.animals[s]]


def strip(agent, obs):
    return obs[: -len(GENES)] if agent.startswith("predator") else obs


def run_both(config, seed, decisions):
    new, old = EvolvingDeerStagHunt(config), base.DeerStagHunt(base_config(config))
    new_obs, _ = new.reset(seed=seed)
    old_obs, _ = old.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(decisions):
        assert state(new) == state(old)
        assert new.active == old.active and sorted(new_obs) == sorted(old_obs)
        for agent, value in new_obs.items():
            np.testing.assert_array_equal(strip(agent, value), old_obs[agent])
        moves = {agent: int(rng.integers(48)) for agent in sorted(new.active)}
        new_obs, *new_rest = new.step(moves)
        old_obs, *old_rest = old.step(moves)
        for ours, theirs in zip(new_rest[:3], old_rest[:3]):
            assert ours == theirs
        if new_rest[1]["__all__"] or new_rest[2]["__all__"]:
            break
    return new, old


def test_defaults_are_deer_stag_hunts():
    assert base_config(DEFAULT_CONFIG) == base.DEFAULT_CONFIG


@pytest.mark.parametrize("seed", [0, 1])
def test_same_episode(seed):
    new, old = run_both(dict(DEFAULT_CONFIG), seed, 600)
    assert new.time_step == old.time_step > 300


def test_same_long_episode_with_births():
    config = dict(DEFAULT_CONFIG, energy_predator_initial=11.5, max_time_steps=4000)
    new, old = run_both(config, 3, 2000)
    assert new.births == old.births and new.births["predator"] > 0
    assert new.stats == old.stats


CROWDED = dict(
    width=300,
    height=300,
    predator_view_distance=150,
    deer_view_distance=150,
    predator_count=14,
    deer_count=20,
    energy_deer_initial=3.0,
    reproduction_deer_cost=3.0,
    grass_cluster_count=4,
    grass_count=40,
)


@pytest.mark.parametrize("seed", [0, 1])
def test_same_episode_touch_rule(seed):
    new, old = run_both(dict(DEFAULT_CONFIG, deer_catch_rule="touch"), seed, 600)
    assert new.time_step == old.time_step > 300


def test_same_crowded_episode_touch_rule_with_adult_and_shared_young_catches():
    new, old = run_both(dict(DEFAULT_CONFIG, deer_catch_rule="touch", **CROWDED), 7, 1500)
    assert new.stats == old.stats and new.stats["adults_caught"] > 0 and new.stats["young_caught_party"] > 0


def test_same_crowded_episode_with_adult_catches():
    config = dict(
        DEFAULT_CONFIG,
        width=300,
        height=300,
        predator_view_distance=150,
        deer_view_distance=150,
        predator_count=14,
        deer_count=20,
        energy_deer_initial=3.0,
        reproduction_deer_cost=3.0,
        grass_cluster_count=4,
        grass_count=40,
    )
    new, old = run_both(config, 7, 1500)
    assert new.stats == old.stats and new.stats["adults_caught"] > 0
