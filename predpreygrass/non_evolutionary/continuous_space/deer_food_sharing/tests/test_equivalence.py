"""With adult_share_fraction 0 this world is deer_stag_hunt's: with the same
settings, seed and movement actions, the same episode (positions,
velocities, energies, cooldowns, agents, observations, rewards, end flags,
catch counters and births), under both catch rules. And with sharing on but
no bystander ever in reach (radius 0), still the same."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing import environment as new_env
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import environment as old_env

SHARING_KEYS = ("adult_share_fraction", "adult_share_radius")


def test_the_defaults_are_deer_stag_hunt_s_plus_sharing_off():
    assert {k: v for k, v in new_env.DEFAULT_CONFIG.items() if k not in SHARING_KEYS} == old_env.DEFAULT_CONFIG
    assert new_env.DEFAULT_CONFIG["adult_share_fraction"] == 0.0


def state(env):
    return [(a.id, a.position.x, a.position.y, a.velocity.x, a.velocity.y, a.energy, a.catch_ready_at, a.orientation) for s in env.animals for a in env.animals[s]]


def run_both(old_config, sharing, seed, decisions):
    new = new_env.DeerStagHunt(dict(old_config, **sharing))
    old = old_env.DeerStagHunt(old_config)
    new_obs, _ = new.reset(seed=seed)
    old_obs, _ = old.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(decisions):
        assert state(new) == state(old)
        assert new.active == old.active and sorted(new_obs) == sorted(old_obs)
        for agent, value in new_obs.items():
            np.testing.assert_array_equal(value, old_obs[agent])
        moves = {agent: int(rng.integers(new_env.MOVES)) for agent in sorted(new.active)}
        new_obs, new_r, new_t, new_tr, _ = new.step(moves)
        old_obs, old_r, old_t, old_tr, _ = old.step(moves)
        assert (new_r, new_t, new_tr) == (old_r, old_t, old_tr)
        assert {k: v for k, v in new.stats.items() if k not in ("adults_shared", "share_bystanders", "energy_adult_to_bystanders")} == old.stats
        if new_t["__all__"] or new_tr["__all__"]:
            break
    assert new.births == old.births and new.time_step == old.time_step
    return new, old


CROWDED = dict(width=300, height=300, predator_view_distance=150, deer_view_distance=150, predator_count=14, deer_count=20)


@pytest.mark.parametrize("rule", ["party", "touch"])
@pytest.mark.parametrize("seed", [0, 1])
def test_same_episode_as_deer_stag_hunt_without_sharing(rule, seed):
    config = dict(old_env.DEFAULT_CONFIG, deer_catch_rule=rule)
    new, _ = run_both(config, {}, seed, 600)
    assert new.time_step > 300


@pytest.mark.parametrize("rule", ["party", "touch"])
def test_same_crowded_episode_with_adult_catches_without_sharing(rule):
    config = dict(old_env.DEFAULT_CONFIG, deer_catch_rule=rule, **CROWDED)
    new, _ = run_both(config, {}, 4, 1500)
    assert new.stats["adults_caught"] > 0


def test_sharing_with_no_bystander_in_reach_is_still_deer_stag_hunt():
    config = dict(old_env.DEFAULT_CONFIG, deer_catch_rule="touch", **CROWDED)
    new, _ = run_both(config, {"adult_share_fraction": 0.5, "adult_share_radius": 0.0}, 4, 1500)
    assert new.stats["adults_caught"] > 0 and new.stats["adults_shared"] == 0


def test_sharing_changes_a_crowded_episode():
    config = dict(old_env.DEFAULT_CONFIG, deer_catch_rule="touch", **CROWDED)
    new = new_env.DeerStagHunt(dict(config, adult_share_fraction=0.5, adult_share_radius=128.0))
    new.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(1500):
        _, _, t, tr, _ = new.step({agent: int(rng.integers(new_env.MOVES)) for agent in sorted(new.active)})
        if t["__all__"] or tr["__all__"]:
            break
    assert new.stats["adults_shared"] > 0 and new.stats["energy_adult_to_bystanders"] > 0
