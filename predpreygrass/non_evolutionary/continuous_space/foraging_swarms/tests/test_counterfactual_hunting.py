"""counterfactual_hunting with stand-in policies (no checkpoint)."""

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import counterfactual_hunting as ch
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import FIXES, PredPreyGrassContinuous
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_predators import intended_velocity, toward
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.tests.test_measure_swarms import Uniform

WORLD = dict(FIXES, predator_count=15, prey_count=60, max_time_steps=1600)


def test_factual_branch_reproduces_the_episode():
    env = PredPreyGrassContinuous(dict(WORLD))
    old = (ch.CAP, ch.REPLAY_SHARE)
    ch.CAP, ch.REPLAY_SHARE = 200, 1.0
    try:
        episodes = [ch.play(env, {"predator": Uniform(), "prey": Uniform()}, s) for s in (0, 1)]
    finally:
        ch.CAP, ch.REPLAY_SHARE = old
    checked = sum(e["counts"].get("determinism_checked", 0) for e in episodes)
    assert checked > 20
    assert sum(e["counts"].get("determinism_mismatch", 0) for e in episodes) == 0
    for e in episodes:  # outcomes partition
        for key, n in e["counts"].items():
            if key.count("|") == 2:
                assert e["sums"].get(f"{key}|anyone", 0) == e["sums"].get(f"{key}|self", 0) + e["sums"].get(f"{key}|other", 0)
                assert e["sums"].get(f"{key}|anyone", 0) <= n
    summary = ch.paired_differences(episodes, np.random.default_rng(0), resamples=50)
    assert summary
    ch.format_summary("test", {"episodes": 2, "determinism_checked": checked, "determinism_mismatch": 0, "differences": summary})
    assert env._move.__name__ == "_move"


def test_rollout_does_not_touch_the_original():
    env = PredPreyGrassContinuous(dict(WORLD))
    rng = np.random.default_rng(0)
    env.reset(seed=4)
    for _ in range(5):
        env.step({a: int(rng.integers(48)) for a in env.active})
    before = [(a.id, a.position.x, a.position.y, a.energy) for a in env.predators + env.prey]
    actions = {a: int(rng.integers(48)) for a in env.active}
    target = env.prey[0].id
    ch.rollout(env, {"predator": Uniform(), "prey": Uniform()}, actions, {}, target, 1)
    assert [(a.id, a.position.x, a.position.y, a.energy) for a in env.predators + env.prey] == before


def test_away_action_points_away():
    env = PredPreyGrassContinuous(dict(WORLD))
    env.reset(seed=0)
    helper = env.predators[0]
    prey_position = Vector((helper.position.x + 50) % env.width, helper.position.y)  # prey to the +x side
    a = ch.away_action(env, helper, prey_position)
    assert toward(*intended_velocity(a), 50.0, 0.0) < -0.99
    assert a // 16 == 0  # full speed
    assert not intended_velocity(ch.STOP_ACTION).any()
