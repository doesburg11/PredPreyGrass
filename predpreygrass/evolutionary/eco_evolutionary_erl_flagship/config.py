"""Simulation parameters for eco_evolutionary_erl_flagship (Trial 13).

Two deliberate deviations from flagship's shipped `config_env` (base_environment/
config_env.py), both required for a long, non-resetting multi-generational run --
see this module's README.md for the full reasoning:

  - max_steps: flagship's default (1000) truncates the "episode" and resets all
    agents to their initial positions/counts. Overridden here to an effectively
    unbounded value so extinction is the only natural stopping condition, matching
    Trial 12's design.
  - n_possible_prey / n_possible_predators: flagship allocates agent IDs
    (`prey_<i>`) monotonically and NEVER recycles them on death (confirmed by
    reading predpreygrass_rllib_env.py directly: `_next_prey_idx` only increments).
    Reproduction is silently and permanently gated off once cumulative births hit
    this cap (2000 by default) -- normally invisible because flagship's own
    max_steps=1000 resets the counter every episode. With max_steps overridden
    away, this cap becomes a hard ceiling on TOTAL lifetime births for the whole
    run, not concurrent population, and would silently stall reproduction long
    before any real result forms. Raised well above what Stage 3's full-scale
    step budget could plausibly produce; validate at Stage 0 that this is
    actually large enough (see tests/test_newborn_parent_pairing.py and
    run_trial13_simulation.py's own --steps run).

Founding-population sizes and initial_energy_predator are FLAGSHIP'S STOCK
VALUES, not tuned -- an earlier version of this file tuned them upward
(initial_energy_predator 5.0->11.0, n_initial_active_prey 8->16,
n_initial_active_predator 6->4) to compensate for predators starving out fast
against genome-driven prey. That tuning turned out to be compensating for two
things that no longer apply: (1) an undertrained predator checkpoint (see
DEFAULT_PREDATOR_CHECKPOINT_DIR below -- iteration 110 vs. the now-used 1000),
and (2) a since-fixed reproducibility bug (see driver.py's reset() and
predator_policy.py -- Trial 13 runs were NOT actually reproducible by seed
before that fix, so the "seeds" that earlier tuning was validated against
weren't real repeatable trials). Re-tested against the correct checkpoint with
working reproducibility, the tuned values showed no clear improvement over
stock -- one tuned-config seed even crashed prey down to a single individual,
worse than the stock config's equivalent seed. Reverted rather than kept as an
unjustified deviation. See README.md's status section for the full history.
"""

from predpreygrass.global_config import RAY_RESULTS_DIR
from predpreygrass.non_evolutionary.base_environment.config_env import config_env as _flagship_config_env

# Flagship's env config, with the overrides above applied. A plain dict copy,
# not a reference -- Trial 13 must never mutate base_environment's shared config_env.
config_env_flagship = dict(_flagship_config_env)
config_env_flagship["max_steps"] = 10_000_000
config_env_flagship["n_possible_prey"] = 500_000
config_env_flagship["n_possible_predators"] = 500_000

# A converged PPO predator_policy checkpoint from a prior base_environment
# tournament run (see master_tournament_matrix.py). checkpoint_000099 is
# training_iteration 1000 -- the run's final, most-converged checkpoint.
# checkpoint_000010 (iteration 110) was used originally on the theory that an
# early, less-converged predator would be a gentler adversary; in practice it
# behaved erratically (sometimes barely hunting at all, sometimes wiping prey
# out almost immediately) while iteration 1000 produced clearly more sensible,
# legible boom-bust predator-prey dynamics (real growth from successful
# hunting, gradual decline, not chaotic swings) -- see README.md's status
# section.
DEFAULT_PREDATOR_CHECKPOINT_RUN_DIR = (
    RAY_RESULTS_DIR / "master_tournament_2026-09-06" / "PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45"
)
DEFAULT_PREDATOR_CHECKPOINT_DIR = DEFAULT_PREDATOR_CHECKPOINT_RUN_DIR / "checkpoint_000099"

# --- Prey genome architecture ---
OBS_DIM = 8  # energy_norm, predator_dx/dy/proximity, food_dx/dy/proximity, local_grass_density (features.py)
N_ACTIONS = 9  # flagship's Discrete(9) Moore-neighborhood action space (predpreygrass_rllib_env.py)

config_erl_flagship = {
    "seed": 41,

    # --- Genome mutation (asexual, mutation-only -- see genome.py, README.md) ---
    "mutation_rate": 0.05,
    "mutation_std": 0.05,
    "founder_weight_std": 0.5,

    # --- Local reinforcement learning (within-lifetime; live action network only) ---
    # Unchanged from Trial 12 -- same REINFORCE-style approximation, same tiny
    # linear-network scale, no reason to expect flagship's ecology needs different
    # learning rates a priori. Revisit if Stage 1 shows action weights swinging
    # wildly or not moving at all.
    "lr_positive": 0.05,
    "lr_negative": 0.02,

    # --- Predator (frozen PPO checkpoint, inference-only -- see predator_policy.py) ---
    "predator_checkpoint_dir": str(DEFAULT_PREDATOR_CHECKPOINT_DIR),
    "predator_deterministic": False,  # sample from the policy's action distribution, not argmax -- an
    # argmax-only predator is a fixed function of observation alone and can be
    # exploited by memorizing one evasion pattern; sampling keeps it a genuine,
    # somewhat-unpredictable threat, consistent with Trial 12's own carnivore
    # (a hard-coded FSA, not a fixed lookup either).
}
