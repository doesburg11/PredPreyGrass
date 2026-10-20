"""measure_predators end to end with stand-in policies (no checkpoint)."""

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import measure_predators as mp
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import FIXES, PredPreyGrassContinuous
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.tests.test_measure_swarms import Uniform

WORLD = dict(FIXES, predator_count=12, prey_count=40, max_time_steps=1200)


def count_caught(env, seed):
    """Prey caught according to the environment's own flags (terminated, not starved)."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    caught = 0
    while True:
        prey = [a.id for a in env.prey]
        obs, _, terms, truncs, infos = env.step({a: int(rng.integers(48)) for a in env.active})
        caught += sum(bool(terms.get(a)) and not infos.get(a, {}).get("starved") for a in prey)
        if terms["__all__"] or truncs["__all__"]:
            return caught


def test_every_catch_is_attributed_once():
    env = PredPreyGrassContinuous(dict(WORLD))
    modules = {"predator": Uniform(), "prey": Uniform()}
    episodes = [mp.play(env, modules, seed) for seed in (0, 1, 2)]
    for e in episodes:
        c = e["counts"]
        # every catch is counted once: by an old enough catcher, or left out as young
        assert c.get("catch_joint", 0) + c.get("catch_by_young", 0) == e["catches"]
        assert sum(c.get(f"catch|g{g}", 0) for g in range(4)) == c.get("catch_joint", 0)
        assert sum(c.get(f"enc_success|g{g}", 0) for g in range(4)) == c.get("enc_joint", 0)
    assert sum(e["catches"] for e in episodes) > 0
    summary = mp.summarize(episodes)
    mp.format_summary("test", summary)
    assert env._move.__name__ == "_move"  # the recording wrapper is removed again


def test_catch_count_matches_the_environment():
    """The wrapper counts the same catches as the environment's flags, under
    the same random actions."""
    env = PredPreyGrassContinuous(dict(WORLD))
    caught = count_caught(env, 5)
    seen = []
    move = env._move

    def recording(animal, action, catcher, aways, catches):
        if catcher is not None:
            seen.append(animal.id)
        move(animal, action, catcher, aways, catches)

    env._move = recording
    assert count_caught(env, 5) == caught == len(seen)


def test_helpers():
    assert mp.bin_of(5, mp.PREY_EDGES) == 0 and mp.bin_of(6, mp.PREY_EDGES) == 1 and mp.bin_of(31, mp.PREY_EDGES) == 3
    assert mp.toward(1.0, 0.0, 3.0, 0.0) == 1.0 and mp.toward(1.0, 0.0, -2.0, 0.0) == -1.0
    np.testing.assert_allclose(mp.intended_velocity(0), [0.0, -1.0], atol=1e-12)  # direction 0 points to -y
    np.testing.assert_allclose(mp.intended_velocity(16 + 4), [0.5, 0.0], atol=1e-12)  # half speed, +x
    assert not mp.intended_velocity(32).any()  # stopped


def test_encounter_target_outcomes_add_up():
    env = PredPreyGrassContinuous(dict(WORLD))
    e = mp.play(env, {"predator": Uniform(), "prey": Uniform()}, 3)
    s, c = e["sums"], e["counts"]
    for j in (0, 1):
        anyone = s.get(f"enc_target_caught|joint{j}", 0)
        assert anyone == s.get(f"enc_target_by_self|joint{j}", 0) + s.get(f"enc_target_by_other|joint{j}", 0)
        # the d-split adds up to the total
        assert sum(c.get(f"enc_target_caught|joint{j}|d{d}", 0) for d in (0, 1)) == c.get(f"enc_target_caught|joint{j}", 0)
    # success is target-specific: the same events as "caught by self"
    assert sum(s.get(f"enc_success|g{g}", 0) for g in range(4)) == sum(
        s.get(f"enc_target_by_self|joint{j}", 0) for j in (0, 1))
