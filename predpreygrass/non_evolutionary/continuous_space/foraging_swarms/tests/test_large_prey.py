"""capture_assist_radius / capture_assist_threshold: a large prey needs a
helper nearby to be caught; observe_energy_in_slots adds the observed
animal's energy to each slot. Defaults are covered by test_regression."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import (
    FIXES,
    SELF_SIZE,
    SLOT_SIZE,
    PredPreyGrassContinuous,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.geometry import Vector

STOP = 32
ASSIST = dict(FIXES, capture_assist_radius=64.0, capture_assist_threshold=5.0, observe_energy_in_slots=True)


def staged(prey_energy, helper_at, extra=None, **settings):
    """predator_0 touches a prey at (100, 100); predator_1 at helper_at; a
    spare prey far away keeps the episode going."""
    spots = ((120.0, 100.0), helper_at) + tuple(extra or ())
    env = PredPreyGrassContinuous(dict(ASSIST, predator_count=len(spots), prey_count=2, **settings))
    env.reset(seed=0)
    prey, spare = env.prey
    prey.position, prey.velocity, prey.energy = Vector(100.0, 100.0), Vector(0.0, 0.0), prey_energy
    spare.position = Vector(600.0, 600.0)
    for predator, (x, y) in zip(env.predators, spots):
        predator.position, predator.velocity, predator.energy = Vector(x, y), Vector(0.0, 0.0), 5.0
    env.step({agent: STOP for agent in env.active})
    return env, prey.id


def test_lone_predator_cannot_catch_a_large_prey():
    env, prey = staged(6.0, (400.0, 400.0))
    assert env.catch_log == []
    assert prey in {a.id for a in env.prey}


def test_helper_nearby_makes_the_catch_and_shares_it():
    env, prey = staged(6.0, (100.0, 150.0))  # helper 50 units from the prey
    (log,) = env.catch_log
    assert log["prey"] == prey and log["catcher"] == "predator_0" and log["assisted"]
    assert log["fed"] == ["predator_0", "predator_1"]
    assert log["gain"] == pytest.approx(0.5 * 6.0 / 2)


def test_small_prey_is_caught_alone():
    env, prey = staged(4.0, (400.0, 400.0))
    (log,) = env.catch_log
    assert log["fed"] == ["predator_0"] and not log["assisted"]


def test_threshold_zero_makes_every_prey_large():
    env, _ = staged(1.0, (400.0, 400.0), capture_assist_threshold=0.0)
    assert env.catch_log == []


def test_energy_in_slots():
    env = PredPreyGrassContinuous(dict(ASSIST, predator_count=6, prey_count=30))
    base = PredPreyGrassContinuous(dict(FIXES))
    n_slots = env.observe_count["predator"] + env.observe_count["prey"]
    assert env.slot_size == SLOT_SIZE + 1
    assert env.obs_size == base.obs_size + n_slots
    obs, _ = env.reset(seed=3)
    by_id = {a.id: a for a in env.predators + env.prey}
    checked = 0
    for agent, o in obs.items():
        observer = by_id[agent]
        for k in range(n_slots):
            slot = o[SELF_SIZE + env.slot_size * k : SELF_SIZE + env.slot_size * (k + 1)]
            if slot[0] <= 0:
                assert not slot.any()
                continue
            species = "predator" if k < env.observe_count["predator"] else "prey"
            reach = observer.view_distance
            seen = min(
                (a for a in (env.predators if species == "predator" else env.prey) if a is not observer),
                key=lambda a: abs(env.torus.offset(observer.position, a.position)[0] - slot[1] * reach)
                + abs(env.torus.offset(observer.position, a.position)[1] - slot[2] * reach),
            )
            assert slot[6] == pytest.approx(seen.energy / env.max_energy[species], abs=1e-5)
            checked += 1
    assert checked > 10


@pytest.mark.parametrize("key,bad", [("capture_assist_radius", -1.0), ("capture_assist_threshold", float("nan")),
                                     ("observe_energy_in_slots", 1)])
def test_validation(key, bad):
    with pytest.raises(ValueError, match=key):
        PredPreyGrassContinuous({key: bad})


def test_measurements_run_with_energy_slots():
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import measure_predators as mp
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import measure_swarms as ms
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.tests.test_measure_swarms import Uniform

    env = PredPreyGrassContinuous(dict(ASSIST, predator_count=12, prey_count=40, max_time_steps=800))
    modules = {"predator": Uniform(), "prey": Uniform()}
    e = mp.play(env, modules, 0)
    mp.format_summary("t", mp.summarize([e]))
    ms.format_summary("t", ms.summarize([ms.play(env, modules, 0, np.random.default_rng(0))], modules, ms.layout(env)))


@pytest.mark.parametrize("simultaneous", [True, False])
def test_assisted_capture_under_both_movement_orders(simultaneous):
    lone, _ = staged(6.0, (400.0, 400.0), fix_simultaneous_movement=simultaneous)
    helped, prey = staged(6.0, (100.0, 150.0), fix_simultaneous_movement=simultaneous)
    assert lone.catch_log == []
    assert [c["prey"] for c in helped.catch_log] == [prey] and helped.catch_log[0]["assisted"]


def test_helpers_and_shared_carcass_are_not_counted_twice():
    """Helper within both radii, a bystander only within the share radius."""
    env, _ = staged(6.0, (100.0, 150.0), extra=((100.0, 20.0),), energy_catch_share_radius=96.0)
    (log,) = env.catch_log
    assert sorted(log["fed"]) == ["predator_0", "predator_1", "predator_2"]
    assert log["gain"] == pytest.approx(0.5 * 6.0 / 3)


def test_no_stale_helpers_across_physics_steps():
    """A helper that leaves after an assisted step does not enable a later lone catch."""
    env, _ = staged(6.0, (400.0, 400.0))
    env._assists["prey_0"] = ["stale"]  # leftovers must not survive into the next physics step
    env.step({agent: 32 for agent in env.active})
    assert env.catch_log == [] and env._assists == {}


def test_warm_start_checkpoints_exist():
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.launchers.launch import WARM_START

    for path in WARM_START.values():
        assert path.is_dir(), path
        assert "continuous_space/base_environment_continuous/runs" in str(path)
        for policy in ("predator_policy", "prey_policy"):
            assert (path / "learner_group" / "learner" / "rl_module" / policy / "module_state.pkl").is_file()
