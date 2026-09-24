"""
Build the environment a saved run was TRAINED in, for rollout-based analysis.

Why this exists: `tune_ppo_fixed_prey_density.py` trains `FixedPreyDensityEnv` (prey floor) or, when
`predator_density_target` is set, `FixedPredatorDensityEnv`; runs with a `predator_population_cap` are
FixedPreyDensityEnv plus the cap (which lives in the base class). The rollout analysis scripts originally
built the plain base `PredPreyGrass` from the saved config, which silently ignores the prey floor and the
density target -- so for those runs the policies were evaluated in a different ecology than they were
trained in (measured on SUCCESSONLY uncapped: 2 of 3 base-env episodes collapsed to zero prey after
~110-140 steps, against 1000 steps with prey held at 20 in the trained env; on the density-target runs the
population was no longer pinned at the target). Iterations 12-15's rollout-based energy-share tables and
Iteration 14's coordination tests were computed that way and were re-run after this fix.

The env class is chosen from `config_env` in the run's `run_config.json`:
  predator_density_target set  -> FixedPredatorDensityEnv
  prey_density_floor present   -> FixedPreyDensityEnv   (includes predator_population_cap runs)
  otherwise                    -> PredPreyGrass         (all runs before Iteration 11)
"""
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_predator_density_env import (
    FixedPredatorDensityEnv,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_prey_density_env import FixedPreyDensityEnv
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass


def env_class_for(config):
    if config.get("predator_density_target") is not None:
        return FixedPredatorDensityEnv
    if "prey_density_floor" in config:
        return FixedPreyDensityEnv
    return PredPreyGrass


def make_env(config):
    return env_class_for(config)(config)
