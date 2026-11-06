"""Tests of hunters' calls (hunter_calls) and of fat rabbits caught only by
hunters touching them at once (rabbit_fat_touch_only)."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import CALL_SIZE, StagHuntContinuous
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.tests.test_environment import STOP, random_actions, stopped_actions, world
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.tests.test_lean_fat import rabbit_beside


def calling_world(**config):
    env = world(mammoth_count=0, hunter_calls=True, **config)
    env.rabbits[0].position = Vector(700, 100)  # out of the way: not caught
    env.rabbits[0].velocity = Vector(0, 0)
    return env


def actions(env, callers=()):
    return {a: (np.array([STOP, int(a in callers)]) if a.startswith("predator") else STOP) for a in env.active}


def test_a_call_is_heard_in_any_direction_within_the_radius():
    env = calling_world(predator_fov=10)
    listener, caller, far = env.predators[:3]
    listener.position = Vector(400, 400)
    caller.position = Vector(400 - 150, 400)  # behind, outside a 10 degree cone
    far.position = Vector(400, 400 + 250)  # beyond the radius
    env.predators[3].position = Vector(100, 700)
    env.action_repeat = 1
    obs, *_ = env.step(actions(env, callers={caller.id, far.id}))
    heard = obs[listener.id][-CALL_SIZE:]
    assert heard[0] == 1.0 and heard[1] == pytest.approx(-150 / 200, abs=1e-5) and heard[2] == pytest.approx(0.0, abs=1e-5)
    assert heard[3] == pytest.approx(150 / 200, abs=1e-5)
    # A caller does not hear itself, and the far caller is beyond its range.
    assert list(obs[caller.id][-CALL_SIZE:]) == [0.0] * CALL_SIZE
    assert env.hunt_stats["calls"] == 2


def test_the_nearest_caller_is_the_one_heard_and_silence_is_zeros():
    env = calling_world()
    listener, near, farther = env.predators[:3]
    listener.position = Vector(400, 400)
    near.position = Vector(400 + 60, 400)
    farther.position = Vector(400 + 120, 400)
    env.predators[3].position = Vector(100, 700)
    env.action_repeat = 1
    obs, *_ = env.step(actions(env, callers={near.id, farther.id}))
    assert obs[listener.id][-CALL_SIZE + 3] == pytest.approx(60 / 200, abs=1e-5)
    obs, *_ = env.step(actions(env))
    assert list(obs[listener.id][-CALL_SIZE:]) == [0.0] * CALL_SIZE


def test_calls_add_only_the_call_observation():
    env, plain = calling_world(), world(mammoth_count=0)
    assert env.obs_size["predator"] == plain.obs_size["predator"] + CALL_SIZE
    assert env.obs_size["rabbit"] == plain.obs_size["rabbit"]
    obs, _ = env.reset(seed=1)
    assert all(env.observation_spaces[a].contains(o) for a, o in obs.items())


def test_a_fat_rabbit_is_caught_by_two_touching_and_shared_by_them_only():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_touch_only=True)
    rabbit = rabbit_beside(env, 0, 2, energy=7.0)  # predator 2 near, not touching
    toucher = env.predators[1]
    toucher.position = Vector(rabbit.position.x + 20, rabbit.position.y)
    before = [p.energy for p in env.predators[:3]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_touching"] == 1 and env.hunt_stats["rabbits_caught_party"] == 1
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * 7.0 / 2)
    assert env.predators[2].energy == pytest.approx(before[2])


def test_a_fat_rabbit_escapes_one_toucher_even_with_a_partner_near():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_touch_only=True)
    rabbit = rabbit_beside(env, 0, 1, energy=7.0)
    env.step(stopped_actions(env))
    assert rabbit.alive and env.hunt_stats["rabbit_catch_failures"] == 1 and env.hunt_stats["rabbits_caught"] == 0
    assert env.predators[0].catch_ready_at > env.time_step - 1


def test_lean_and_middle_rabbits_follow_the_old_rule_under_touch_only():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_touch_only=True)
    rabbit_beside(env, 0, 1, energy=4.0)  # middle, a partner near: caught, shared
    before = [p.energy for p in env.predators[:2]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_touching"] == 0
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * 4.0 / 2)


def test_off_by_default_and_unchanged():
    def run(**config):
        env = StagHuntContinuous(dict(mammoth_count=0, max_time_steps=1500, rabbit_fat_energy=6.0, **config))
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        trace = []
        while True:
            obs, rewards, terms, truncs, _ = env.step(random_actions(env, rng))
            trace.append((sorted(rewards.items()), [(a.id, round(a.position.x, 9), round(a.energy, 9)) for a in env.living()]))
            if terms["__all__"] or truncs["__all__"]:
                return trace, env

    plain, env = run()
    explicit, _ = run(hunter_calls=False, call_radius=200.0, rabbit_fat_touch_only=False, rabbit_touch_window=0)
    assert plain == explicit
    assert env.hunt_stats["calls"] == 0 and env.hunt_stats["rabbits_caught_touching"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"hunter_calls": 1},
        {"hunter_calls": True},  # mammoths in the default world
        {"mammoth_count": 0, "hunter_calls": True, "call_radius": 0.0},
        {"rabbit_fat_touch_only": True},  # no lean and fat rabbits
        {"rabbit_fat_energy": 6.0, "rabbit_fat_touch_only": "yes"},
        {"rabbit_touch_window": -1},
        {"rabbit_touch_window": 2.5},
    ],
)
def test_settings_are_validated(settings):
    with pytest.raises(ValueError):
        StagHuntContinuous(settings)


def test_the_signal_arm_is_r7_with_calls_and_touching():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calibration import ARMS, INIT_FROM

    arm = dict(ARMS["lean_fat_r7_signal"])
    assert (arm.pop("hunter_calls"), arm.pop("call_radius"), arm.pop("rabbit_fat_touch_only"), arm.pop("rabbit_touch_window")) == (True, 200.0, True, 8)
    assert arm == ARMS["lean_fat_r7"] and "lean_fat_r7_signal" not in INIT_FROM
    StagHuntContinuous(dict(ARMS["lean_fat_r7_signal"]))


def test_calls_summary():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calls import summarise

    keys = ("decisions", "calls", "alone_at_fat", "alone_at_fat_calls", "call_moments", "call_arrivals", "silent_moments", "silent_arrivals", "censored")
    empty = dict.fromkeys(keys, 0)
    assert summarise([empty])["calls_followed_by_partner"] is False
    rows = [dict(empty, decisions=100, calls=20, alone_at_fat=10, alone_at_fat_calls=5, call_moments=5, call_arrivals=4, silent_moments=5, silent_arrivals=1) for _ in range(5)]
    s = summarise(rows)
    assert s["arrival_after_call"] == 0.8 and s["arrival_after_silence"] == 0.2 and s["calls_followed_by_partner"] is True
    assert s["call_rate_alone_at_fat"] == 0.5 and s["call_rate_otherwise"] == pytest.approx(15 / 90)


def _fat_rabbit_with_two_hunters(window, gap):
    """Hunter 0 touches a fat rabbit; hunter 1, far at first, touches it gap
    physics steps later. Returns (env, rabbit, energies before)."""
    env = world(rabbit_fat_energy=6.0, rabbit_fat_touch_only=True, rabbit_touch_window=window, rabbit_catch_cooldown=0)
    rabbit = rabbit_beside(env, 0, energy=7.0)
    second = env.predators[1]
    second.position = Vector(rabbit.position.x + 300, rabbit.position.y)
    before = [p.energy for p in env.predators[:2]]
    for _ in range(gap):
        env.step(stopped_actions(env))
        env.predators[0].position = Vector(rabbit.position.x - 300, rabbit.position.y)  # away until the catch test
    second.position = Vector(rabbit.position.x + 20, rabbit.position.y)
    env.step(stopped_actions(env))
    return env, rabbit, before


def test_two_touchers_within_the_window_catch_and_share_the_fat_rabbit():
    env, rabbit, before = _fat_rabbit_with_two_hunters(window=8, gap=3)
    assert not rabbit.alive and env.hunt_stats["rabbits_caught_touching"] == 1
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * 7.0 / 2)


def test_touches_further_apart_than_the_window_do_not_catch():
    env, rabbit, _ = _fat_rabbit_with_two_hunters(window=2, gap=3)
    assert rabbit.alive and env.hunt_stats["rabbits_caught_touching"] == 0
    assert env.hunt_stats["rabbit_catch_failures"] == 2


def test_window_zero_needs_touching_in_the_same_step():
    env, rabbit, _ = _fat_rabbit_with_two_hunters(window=0, gap=1)
    assert rabbit.alive and env.hunt_stats["rabbits_caught_touching"] == 0


def _run_calls(env, script):
    """Drive launchers.calls.count with a scripted episode: script is a list
    of callables (env) -> action dict, one per decision."""
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers import calls

    def play(env, modules, seed):
        for make in script:
            env.step(make(env))

    original = calls.evaluate.play
    calls.evaluate.play = play
    try:
        return calls.count(env, None, 0)
    finally:
        calls.evaluate.play = original


def _alone_at_fat():
    env = world(mammoth_count=0, hunter_calls=True, rabbit_fat_energy=6.0, energy_rabbit_resting_metabolic_cost=0.0, energy_rabbit_speed_cost=0.0)
    rabbit = env.rabbits[0]
    rabbit.energy, rabbit.velocity = 7.0, Vector(0, 0)  # fat, below the breeding threshold (8)
    rabbit.position = Vector(400, 400)
    hunter = env.predators[0]
    hunter.position = Vector(400 - 50, 400)  # near, not touching
    for k, p in enumerate(env.predators[1:]):
        p.position = Vector(100 + 30 * k, 750)
    env.action_repeat = 1
    return env, rabbit, hunter


def test_a_partner_reaching_the_rabbit_after_a_call_counts_once():
    env, rabbit, hunter = _alone_at_fat()
    partner = env.predators[1]

    def call(env):
        return actions(env, callers={hunter.id})

    def arrive(env):
        partner.position = Vector(rabbit.position.x + 50, rabbit.position.y)
        return actions(env)

    c = _run_calls(env, [call, call, arrive])
    assert c["call_moments"] == 1 and c["call_arrivals"] == 1 and c["silent_moments"] == 0


def test_a_partner_meeting_the_hunter_away_from_the_rabbit_is_no_arrival():
    env, rabbit, hunter = _alone_at_fat()
    partner = env.predators[1]

    def silent(env):
        return actions(env)

    def meet_elsewhere(env):
        hunter.position = Vector(400 - 60, 400)
        partner.position = Vector(400 - 120, 400)  # near the hunter, 120 from the rabbit
        return actions(env)

    c = _run_calls(env, [silent, meet_elsewhere, silent, silent, silent, silent])
    assert c["silent_moments"] == 1 and c["silent_arrivals"] == 0


def test_a_rabbit_leaving_censors_the_moment():
    env, rabbit, hunter = _alone_at_fat()

    def silent(env):
        return actions(env)

    def gone(env):
        rabbit.in_burrow = True
        return actions(env)

    env.burrow_count = 0  # in_burrow set by hand only
    c = _run_calls(env, [silent, gone])
    assert c["silent_moments"] == 0 and c["censored"] == 1


def test_touches_are_forgotten_when_the_rabbit_stops_being_fat():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_touch_only=True, rabbit_touch_window=8, rabbit_catch_cooldown=0)
    rabbit = rabbit_beside(env, 0, energy=7.0)
    env.step(stopped_actions(env))
    assert rabbit.id in env.fat_touches
    env.predators[0].position = Vector(rabbit.position.x - 300, rabbit.position.y)
    rabbit.energy = 5.0  # no longer fat
    env.step(stopped_actions(env))
    assert rabbit.id not in env.fat_touches
    rabbit.energy = 7.0  # fat again: hunter 1 alone does not catch with hunter 0's old touch
    env.predators[1].position = Vector(rabbit.position.x + 20, rabbit.position.y)
    env.step(stopped_actions(env))
    assert rabbit.alive and env.hunt_stats["rabbits_caught_touching"] == 0


def test_a_moment_is_censored_when_the_rabbit_stops_being_fat():
    env, rabbit, hunter = _alone_at_fat()
    partner = env.predators[1]

    def silent(env):
        return actions(env)

    def thin_then_partner(env):
        rabbit.energy = 4.0
        partner.position = Vector(rabbit.position.x + 50, rabbit.position.y)
        return actions(env)

    c = _run_calls(env, [silent, thin_then_partner])
    assert c["silent_moments"] == 0 and c["censored"] == 1


def test_a_partner_passing_the_rabbit_within_a_decision_counts():
    env, rabbit, hunter = _alone_at_fat()
    env.action_repeat = 2
    partner = env.predators[1]
    original = env._physics_step
    calls_made = []

    def physics(moves):
        # In the first physics step of the second decision the partner is
        # beside the rabbit, then gone again before the decision ends.
        calls_made.append(1)
        if len(calls_made) == 3:
            partner.position = Vector(rabbit.position.x + 50, rabbit.position.y)
        result = original(moves)
        if len(calls_made) == 3:
            partner.position = Vector(100, 750)
        return result

    env._physics_step = physics

    def silent(env):
        return actions(env)

    c = _run_calls(env, [silent, silent, silent])
    assert c["silent_moments"] == 1 and c["silent_arrivals"] == 1
