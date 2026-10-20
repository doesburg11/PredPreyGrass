"""energy_catch_share_radius: a catch's energy is split among the catcher
and the predators within the radius; 0 keeps the catcher-takes-all rule."""

import pytest

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import FIXES, PredPreyGrassContinuous
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.geometry import Vector

STOP = 32  # speed index 2: stand still


def staged_catch(radius):
    """One prey at (100, 100); predator_0 touching it, predator_1 50 units
    away, predator_2 far away. Returns (env, energies after one decision)."""
    env = PredPreyGrassContinuous(dict(FIXES, predator_count=3, prey_count=1, energy_catch_share_radius=radius))
    env.reset(seed=0)
    prey = env.prey[0]
    prey.position, prey.velocity = Vector(100.0, 100.0), Vector(0.0, 0.0)
    prey.energy = 6.0
    for predator, (x, y) in zip(env.predators, ((120.0, 100.0), (150.0, 100.0), (500.0, 500.0))):
        predator.position, predator.velocity = Vector(x, y), Vector(0.0, 0.0)
        predator.energy = 5.0
    env.step({agent: STOP for agent in env.active})
    return env, {a.id: a.energy for a in env.predators}


def test_shared_catch_splits_the_energy():
    _, alone = staged_catch(0.0)
    env, shared = staged_catch(96.0)
    (log,) = env.catch_log
    assert log["catcher"] == "predator_0" and log["prey"] == "prey_0"
    assert log["fed"] == ["predator_0", "predator_1"]
    assert log["gain"] == pytest.approx(0.5 * 6.0 / 2)
    # same movement and costs in both runs: the differences are the shares
    assert alone["predator_0"] - shared["predator_0"] == pytest.approx(log["gain"])
    assert shared["predator_1"] - alone["predator_1"] == pytest.approx(log["gain"])
    assert shared["predator_2"] == pytest.approx(alone["predator_2"])


def test_no_sharing_feeds_the_catcher_alone():
    env, _ = staged_catch(0.0)
    (log,) = env.catch_log
    assert log["fed"] == ["predator_0"] and log["gain"] == pytest.approx(3.0)


def test_small_radius_excludes_the_bystander():
    env, _ = staged_catch(40.0)
    assert env.catch_log[0]["fed"] == ["predator_0"]


@pytest.mark.parametrize("bad", [-1.0, float("inf"), True, "96"])
def test_radius_validation(bad):
    with pytest.raises(ValueError, match="energy_catch_share_radius"):
        PredPreyGrassContinuous({"energy_catch_share_radius": bad})


def test_sharing_wraps_around_the_torus_and_feeds_several():
    """Catch at the left edge; bystanders across the right edge are within the radius."""
    env = PredPreyGrassContinuous(dict(FIXES, predator_count=4, prey_count=1, energy_catch_share_radius=96.0))
    env.reset(seed=0)
    prey = env.prey[0]
    prey.position, prey.velocity, prey.energy = Vector(10.0, 300.0), Vector(0.0, 0.0), 8.0
    spots = ((30.0, 300.0), (760.0, 300.0), (790.0, 330.0), (600.0, 300.0))  # 20, 50, ~36, 210 units away
    for predator, (x, y) in zip(env.predators, spots):
        predator.position, predator.velocity, predator.energy = Vector(x, y), Vector(0.0, 0.0), 5.0
    env.step({agent: STOP for agent in env.active})
    (log,) = env.catch_log
    assert log["fed"] == ["predator_0", "predator_1", "predator_2"]
    assert log["gain"] == pytest.approx(0.5 * 8.0 / 3)


def test_catch_log_is_cleared_at_the_next_decision_and_reset():
    env = PredPreyGrassContinuous(dict(FIXES, predator_count=2, prey_count=2, energy_catch_share_radius=96.0))
    env.reset(seed=0)
    prey, spare = env.prey
    prey.position, prey.velocity = Vector(100.0, 100.0), Vector(0.0, 0.0)
    spare.position = Vector(600.0, 600.0)  # keeps the episode going after the catch
    for predator, (x, y) in zip(env.predators, ((120.0, 100.0), (400.0, 400.0))):
        predator.position, predator.velocity = Vector(x, y), Vector(0.0, 0.0)
    env.step({agent: STOP for agent in env.active})
    assert len(env.catch_log) == 1
    assert env.active
    env.step({agent: STOP for agent in env.active})
    assert env.catch_log == []
    env.catch_log.append("stale")
    env.reset(seed=1)
    assert env.catch_log == []


def test_counterfactual_fed_outcome_under_sharing():
    """A bystander within the radius is fed from a catch it did not make."""
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.counterfactual_hunting import rollout
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.tests.test_measure_swarms import Uniform

    for radius, expected in ((96.0, ["predator_0", "predator_1"]), (0.0, ["predator_0"])):
        env = PredPreyGrassContinuous(dict(FIXES, predator_count=3, prey_count=2, energy_catch_share_radius=radius))
        env.reset(seed=0)
        prey, spare = env.prey
        prey.position, prey.velocity = Vector(100.0, 100.0), Vector(0.0, 0.0)
        spare.position = Vector(600.0, 600.0)
        for predator, (x, y) in zip(env.predators, ((120.0, 100.0), (150.0, 100.0), (400.0, 100.0))):
            predator.position, predator.velocity = Vector(x, y), Vector(0.0, 0.0)
        actions = {agent: STOP for agent in env.active}
        catcher, k, fed = rollout(env, {"predator": Uniform(), "prey": Uniform()}, actions, {}, prey.id, 0)
        assert (catcher, k, fed) == ("predator_0", 0, expected)


def test_launch_conditions_change_only_what_they_name():
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "launchers" / "launch.py").read_text()
    tree = ast.parse(source)
    conditions = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "CONDITIONS"
    )
    assert conditions["S1"] == {"energy_catch_share_radius": 96.0}
    assert conditions["S2"] == {"energy_catch_share_radius": 96.0, "predator_max_velocity": 4.5}
    PredPreyGrassContinuous(dict(FIXES, **conditions["S2"]))  # a valid world
