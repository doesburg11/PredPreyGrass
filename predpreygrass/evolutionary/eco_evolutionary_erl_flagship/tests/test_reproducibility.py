"""Regression tests for two reproducibility bugs found during Stage-0 calibration
(see README.md's Status section): (1) driver.reset() wasn't passing a seed to
env.reset(), so founder agent/grass placement drew fresh OS entropy every run
regardless of --seed; (2) FrozenPredatorPolicy sampled from PyTorch's unseeded
global RNG. Both are exercised here with the REAL PredPreyGrass env (not a fake)
since the bug was specifically about env.reset()'s own seeding contract.
"""

from pathlib import Path

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import (
    DEFAULT_PREDATOR_CHECKPOINT_DIR,
    config_env_flagship,
    config_erl_flagship,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import Trial13Driver
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass


def _cfg(seed):
    cfg = dict(config_erl_flagship)
    cfg["seed"] = seed
    return cfg


def test_reset_with_same_seed_gives_identical_founder_positions():
    env_a = PredPreyGrass(dict(config_env_flagship))
    driver_a = Trial13Driver(env_a, predator_policy=None, cfg=_cfg(7), rng=np.random.default_rng(7))
    driver_a.reset()

    env_b = PredPreyGrass(dict(config_env_flagship))
    driver_b = Trial13Driver(env_b, predator_policy=None, cfg=_cfg(7), rng=np.random.default_rng(7))
    driver_b.reset()

    assert env_a.agent_positions == env_b.agent_positions
    assert env_a.grass_positions == env_b.grass_positions


def test_reset_with_different_seeds_gives_different_founder_positions():
    env_a = PredPreyGrass(dict(config_env_flagship))
    driver_a = Trial13Driver(env_a, predator_policy=None, cfg=_cfg(1), rng=np.random.default_rng(1))
    driver_a.reset()

    env_b = PredPreyGrass(dict(config_env_flagship))
    driver_b = Trial13Driver(env_b, predator_policy=None, cfg=_cfg(2), rng=np.random.default_rng(2))
    driver_b.reset()

    assert env_a.agent_positions != env_b.agent_positions


@pytest.mark.skipif(
    not Path(DEFAULT_PREDATOR_CHECKPOINT_DIR).is_dir(),
    reason="Real predator checkpoint not present on this machine.",
)
def test_full_run_with_same_seed_is_bit_for_bit_reproducible():
    """End-to-end: same --seed, two independent driver instances (including the
    frozen predator's own action sampling), must produce identical population
    trajectories for a real number of steps."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.predator_policy import FrozenPredatorPolicy

    def run(seed, steps=60):
        env = PredPreyGrass(dict(config_env_flagship))
        predator_policy = FrozenPredatorPolicy(DEFAULT_PREDATOR_CHECKPOINT_DIR, deterministic=False, seed=seed)
        driver = Trial13Driver(env, predator_policy, _cfg(seed), np.random.default_rng(seed))
        driver.reset()
        trajectory = []
        for _ in range(steps):
            driver.step()
            trajectory.append(driver.population_counts())
            if trajectory[-1]["prey"] == 0 or trajectory[-1]["predator"] == 0:
                break
        return trajectory

    assert run(3) == run(3)
