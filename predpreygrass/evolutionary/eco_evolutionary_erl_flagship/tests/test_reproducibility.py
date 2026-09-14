"""Regression tests for two reproducibility bugs found during Stage-0 calibration
(see README.md's Status section): (1) driver.reset() wasn't passing a seed to
env.reset(), so founder agent/grass placement drew fresh OS entropy every run
regardless of --seed; (2) the predator policy in use at the time
(predator_policy.FrozenPredatorPolicy) sampled from PyTorch's unseeded global
RNG. Both are exercised here with the REAL PredPreyGrass env (not a fake) since
bug (1) was specifically about env.reset()'s own seeding contract. The
end-to-end test now uses centralized_predator.CentralizedPredatorPolicy (the
current default -- see config.py's "Predator strategy" note), since
Trial13Driver.step() only drives predators through that stateful interface;
FrozenPredatorPolicy/RuleBasedPredatorPolicy remain usable standalone (see
their own test suites) but not through the driver.
"""

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_predator import CentralizedPredatorPolicy
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import (
    N_ACTIONS,
    PREDATOR_OBS_DIM,
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


def test_full_run_with_same_seed_is_bit_for_bit_reproducible():
    """End-to-end: same --seed, two independent driver instances (including the
    shared predator policy's own initialization, action sampling, and online
    updates), must produce identical population trajectories for a real number
    of steps."""

    def run(seed, steps=60):
        env = PredPreyGrass(dict(config_env_flagship))
        cfg = _cfg(seed)
        rng = np.random.default_rng(seed)
        predator_policy = CentralizedPredatorPolicy(
            PREDATOR_OBS_DIM, N_ACTIONS, rng,
            init_std=cfg["predator_founder_weight_std"],
            lr_positive=cfg["predator_lr_positive"],
            lr_negative=cfg["predator_lr_negative"],
        )
        driver = Trial13Driver(env, predator_policy, cfg, rng)
        driver.reset()
        trajectory = []
        for _ in range(steps):
            driver.step()
            trajectory.append(driver.population_counts())
            if trajectory[-1]["prey"] == 0 or trajectory[-1]["predator"] == 0:
                break
        return trajectory

    assert run(3) == run(3)
