"""Seeded reproducibility across RLlib auto-resets (reset(seed=None)) in the
9 discrete_space modules whose reset() used to recreate self.rng from the
seed unconditionally: after reset(seed=s), later seedless resets must
continue one seeded stream (identical for two envs with the same seed) and
not repeat the first episode's layout."""

import importlib

import numpy as np
import pytest

BASE = "predpreygrass.non_evolutionary.discrete_space."
MODULES = {
    "base_environment_seasonal": "config_env",
    "project_cooperation.network_reciprocity": "config.config_env",
    "red_queen": "config.config_env_train",
    "walls_occlusion": "config.config_env_train",
    "project_reward_shaping.base_environment_dense_rewards": "config_env",
    "project_reward_shaping.base_environment_dense_rewards_additive": "config_env",
    "project_reward_shaping.base_environment_sparse_rewards": "config_env",
    "project_reward_shaping.base_environment_sparse_rewards_plus_eating": "config_env",
    "project_reward_shaping.base_environment_sparse_rewards_plus_kickback": "config_env",
}


def make(module):
    env_class = importlib.import_module(BASE + module + ".predpreygrass_rllib_env").PredPreyGrass
    config = importlib.import_module(BASE + module + "." + MODULES[module]).config_env
    return env_class(dict(config))


def fingerprint(obs):
    return tuple((agent, np.asarray(o).tobytes()) for agent, o in sorted(obs.items()))


def episodes(env, seed, n=3):
    first, _ = env.reset(seed=seed)
    out = [fingerprint(first)]
    for _ in range(n - 1):
        obs, _ = env.reset()  # what RLlib's auto-reset calls
        out.append(fingerprint(obs))
    return out


@pytest.mark.parametrize("module", list(MODULES))
def test_same_seed_same_episode_sequence_across_auto_resets(module):
    a = episodes(make(module), 7)
    b = episodes(make(module), 7)
    assert a == b
    assert a[1] != a[0] and a[2] != a[1]  # the stream advances; layouts aren't repeated
    assert episodes(make(module), 8)[0] != a[0]


@pytest.mark.parametrize("module", list(MODULES))
def test_explicit_reseed_restarts_the_stream(module):
    env = make(module)
    first = episodes(env, 7)
    assert episodes(env, 7) == first  # after advancing, reset(seed=7) starts over


@pytest.mark.parametrize("module", list(MODULES))
def test_spawn_fallback_ignores_global_numpy_random(module):
    # occupy every cell around the parent so the helper falls back to a random
    # free cell; two identically seeded envs must agree whatever the global
    # np.random state is
    def draws(global_seed):
        env = make(module)
        env.reset(seed=7)
        np.random.seed(global_seed)
        centre = (env.grid_size // 2, env.grid_size // 2)
        occupied = {(centre[0] + dy, centre[1] + dx) for dy in range(-2, 3) for dx in range(-2, 3)}
        return [env._find_available_spawn_position(centre, occupied) for _ in range(20)]

    a, b = draws(1), draws(2)
    assert a == b and len(set(a)) > 1


@pytest.mark.parametrize("module", ["red_queen", "walls_occlusion"])
def test_a_first_seedless_reset_uses_the_config_seed(module):
    env_class = importlib.import_module(BASE + module + ".predpreygrass_rllib_env").PredPreyGrass
    config = dict(importlib.import_module(BASE + module + "." + MODULES[module]).config_env, seed=11)
    seedless, _ = env_class(config).reset()
    seeded, _ = env_class(config).reset(seed=11)
    assert fingerprint(seedless) == fingerprint(seeded)
