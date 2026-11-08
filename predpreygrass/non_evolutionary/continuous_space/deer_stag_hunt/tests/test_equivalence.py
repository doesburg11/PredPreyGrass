"""The deer world is stag_hunt_continuous's world of its RESULTS.md section 8
(arm lean_fat_r7_signal) without calls: same defaults, and with the same seed
and the same movement actions, the same episode (positions, velocities,
energies, cooldowns, agents, rewards, end flags and observations, the
removed inputs aside)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment import DEFAULT_CONFIG, SLOT_SIZE, DeerStagHunt

# stag_hunt_continuous now lives in PredPreyGrass-archive (top-level package
# stag_hunt_continuous): found in $PREDPREYGRASS_ARCHIVE, or in a checkout
# next to this repository; skipped if neither is there.
ARCHIVE = Path(os.environ.get("PREDPREYGRASS_ARCHIVE", Path(__file__).resolve().parents[6] / "PredPreyGrass-archive"))
if (ARCHIVE / "stag_hunt_continuous").is_dir() and str(ARCHIVE) not in sys.path:
    sys.path.append(str(ARCHIVE))
old_env = pytest.importorskip("stag_hunt_continuous.environment", reason=f"stag_hunt_continuous not found in {ARCHIVE}")
calibration = pytest.importorskip("stag_hunt_continuous.launchers.calibration", reason=f"stag_hunt_continuous not found in {ARCHIVE}")

RENAMED = {
    "deer_fawn_energy": "rabbit_lean_energy",
    "deer_adult_energy": "rabbit_fat_energy",
    "deer_party_radius": "rabbit_party_radius",
    "deer_catch_cooldown": "rabbit_catch_cooldown",
    "deer_touch_window": "rabbit_touch_window",
}


# Settings of this module without a counterpart there, with the value that
# is the old world (deer_catch_rule "party": the old catching rules).
NO_COUNTERPART = {"deer_catch_rule": "party"}


def old_key(key):
    return RENAMED.get(key) or key.replace("deer", "rabbit")


def old_config(new):
    """stag_hunt_continuous's settings for a deer world: the passing arm
    without calls, with the deer settings carried over."""
    config = dict(calibration.ARMS["lean_fat_r7_signal"], hunter_calls=False)
    for key, value in NO_COUNTERPART.items():
        assert new.get(key, value) == value, f"{key}={new[key]!r} has no counterpart in stag_hunt_continuous"
    config.update({old_key(key): value for key, value in new.items() if key not in NO_COUNTERPART})
    # Mammoths are absent, but their view distance is still checked.
    config["mammoth_view_distance"] = min(old_env.DEFAULT_CONFIG["mammoth_view_distance"], config["width"] / 2, config["height"] / 2)
    return config


def old_id(agent):
    return agent.replace("deer", "rabbit")


def test_defaults_are_the_passing_world():
    """Every deer setting equals its counterpart in the passing arm (calls
    off); the settings dropped are those of the removed features, at values
    that switch them off."""
    old = dict(old_env.DEFAULT_CONFIG, **calibration.ARMS["lean_fat_r7_signal"])
    for key, value in DEFAULT_CONFIG.items():
        if key in NO_COUNTERPART:
            assert value == NO_COUNTERPART[key], key
            continue
        assert old[old_key(key)] == value, key
    assert old["mammoth_count"] == 0 and old["burrow_count"] == 0
    assert old["grass_season_amplitude"] == 0 and old["reproduction_rabbit_interval"] == 0 and old["reproduction_predator_interval"] == 0
    assert old["observe_rabbit_condition"] and old["rabbit_fat_touch_only"] and not old["truncate_on_prey_extinction"]


def old_observation(obs, agent, env):
    """An old observation without the inputs the deer world dropped: the
    mammoth slots and the predators' hunt cooldown."""
    if not agent.startswith("predator"):
        return obs
    start = 4 + SLOT_SIZE * env.observe[("predator", "predator")]
    mammoths = SLOT_SIZE * env.observe[("predator", "mammoth")]
    obs = np.delete(obs, np.arange(start, start + mammoths))
    # Tail after the deer slots and food: energy, hunt cooldown, ...
    cooldown = start + SLOT_SIZE * env.observe[("predator", "rabbit")] + 4 + 1
    return np.delete(obs, cooldown)


def state(env, species):
    return [
        (a.id.replace("rabbit", "deer"), a.position.x, a.position.y, a.velocity.x, a.velocity.y, a.energy, a.catch_ready_at, a.orientation)
        for s in species
        for a in env.animals[s]
    ]


def run_both(config, seed, decisions):
    new = DeerStagHunt(config)
    old = old_env.StagHuntContinuous(old_config(config))
    new_obs, _ = new.reset(seed=seed)
    old_obs, _ = old.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(decisions):
        assert state(new, ("predator", "deer")) == state(old, ("predator", "rabbit"))
        assert new.active == [a.replace("rabbit", "deer") for a in old.active]
        assert sorted(new_obs) == sorted(a.replace("rabbit", "deer") for a in old_obs)
        for agent, value in new_obs.items():
            np.testing.assert_array_equal(value, old_observation(old_obs[old_id(agent)], old_id(agent), old))
        moves = {agent: int(rng.integers(48)) for agent in sorted(new.active)}
        new_obs, new_r, new_t, new_tr, _ = new.step(moves)
        old_obs, old_r, old_t, old_tr, _ = old.step({old_id(a): (np.array([m, 0]) if a.startswith("predator") else m) for a, m in moves.items()})
        for ours, theirs in ((new_r, old_r), (new_t, old_t), (new_tr, old_tr)):
            assert ours == {a.replace("rabbit", "deer"): v for a, v in theirs.items()}
        if new_t["__all__"] or new_tr["__all__"]:
            break
    return new, old


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_episode_as_the_passing_world(seed):
    """Random actions end these episodes early (hunters starve)."""
    new, old = run_both(dict(DEFAULT_CONFIG), seed, 600)
    assert new.time_step == old.time_step > 300
    assert new.births == {"predator": old.births["predator"], "deer": old.births["rabbit"]}


def test_same_long_episode_with_births_of_both_species():
    """Well-fed hunters live long enough under random actions for births,
    cluster collapses and the episode's end."""
    config = dict(DEFAULT_CONFIG, energy_predator_initial=11.5, max_time_steps=4000)
    new, old = run_both(config, 3, 2000)
    assert new.time_step == old.time_step
    assert new.births["predator"] > 0 and new.births["deer"] > 0
    assert new.grass.total_collapses == old.food["grass"].total_collapses


def test_same_episode_in_a_crowded_world_with_adult_catches():
    """A small arena with many predators, so adults are caught together
    under random actions and every catch branch is compared."""
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
    ours, theirs = new.stats, old.hunt_stats
    assert ours["adults_caught"] > 0 and ours["adult_failures"] > 0
    assert ours["young_caught_solo"] > 0 and ours["young_caught_party"] > 0 and ours["young_failures"] > 0
    assert ours["adults_caught"] == theirs["rabbits_caught_touching"]
    assert ours["young_caught_party"] == theirs["rabbits_caught_party"] - theirs["rabbits_caught_touching"]
    assert ours["young_caught_solo"] == theirs["rabbits_caught_solo"]
    assert ours["adult_failures"] + ours["young_failures"] == theirs["rabbit_catch_failures"]
    assert ours["deer_caught"] == theirs["rabbits_caught"]
    assert sum(ours[f"energy_from_{kind}"] for kind in ("fawn", "middle", "adult")) == pytest.approx(theirs["rabbit_energy"])
