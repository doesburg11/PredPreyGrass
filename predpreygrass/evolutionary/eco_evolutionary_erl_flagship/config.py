"""Simulation parameters for eco_evolutionary_erl_flagship (Trial 13).

Several deliberate deviations from flagship's shipped `config_env` (base_environment/
config_env.py) -- see this module's README.md for the full reasoning:

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
  - initial_energy_predator / n_initial_active_prey / n_initial_active_predator:
    a Stage-0 calibration fix, not a cosmetic default. At flagship's own values
    (5.0 / 8 / 6), predators consistently starved out within ~100-150 steps
    against genome-driven prey, even though the SAME checkpoint thrived (6->19+)
    against its own co-trained PPO prey_policy under identical conditions --
    verified directly, ruling out a loading/inference bug. Root cause: a
    predator only has ~33 steps of energy runway before starving
    (initial_energy_predator / energy_loss_per_step_predator), and flagship's
    PPO prey_policy is a fully-trained forager that reproduces fast (8->32 prey
    by step 20) -- giving predators abundant targets almost immediately. Genome
    prey start with RANDOM, untrained linear policies, so their population
    ramps much slower early on, and predators were starving before genome-prey
    density ever caught up to what the checkpoint was calibrated to expect.
    initial_energy_predator=11.0 keeps predators strictly BELOW
    predator_creation_energy_threshold (12.0) -- deliberately, so predators
    still must actually hunt to reproduce, unlike 15.0+ which was tried and
    rejected: it let founders reproduce for free at spawn with zero hunting (an
    artifact, not a fix). Combined with more founding prey (16, was 8) and
    fewer founding predators (4, was 6) -- more early targets per predator --
    this produced real, sustained hunting/reproduction/cycling for hundreds of
    steps across multiple seeds (one ran 700+ steps with predators cycling
    1-12 and prey 12-46) instead of a ~150-step collapse. Eventual predator
    extinction in some seeds was still observed at longer horizons -- accepted
    as normal finite-population stochastic dynamics, not something further
    tuned away; see README.md's status section.
"""

from predpreygrass.global_config import RAY_RESULTS_DIR
from predpreygrass.non_evolutionary.base_environment.config_env import config_env as _flagship_config_env

# Flagship's env config, with the overrides above applied. A plain dict copy,
# not a reference -- Trial 13 must never mutate base_environment's shared config_env.
config_env_flagship = dict(_flagship_config_env)
config_env_flagship["max_steps"] = 10_000_000
config_env_flagship["n_possible_prey"] = 500_000
config_env_flagship["n_possible_predators"] = 500_000
config_env_flagship["initial_energy_predator"] = 11.0  # was 5.0 -- see module docstring
config_env_flagship["n_initial_active_prey"] = 16  # was 8
config_env_flagship["n_initial_active_predator"] = 4  # was 6

# A converged, non-trivial PPO predator_policy checkpoint that already exists on
# disk from a prior base_environment tournament run (see master_tournament_matrix.py).
# checkpoint_000010 is iteration 110 -- deliberately an EARLY, less-converged
# checkpoint (not the run's most-trained one) for Stage 0/1's weaker adversary; see
# README.md's "Predator handling" section and the plan's risk #1 (a fully-converged
# predator is likely lethal enough to prevent any prey genome from ever reproducing).
DEFAULT_PREDATOR_CHECKPOINT_RUN_DIR = (
    RAY_RESULTS_DIR / "master_tournament_2026-09-06" / "PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45"
)
DEFAULT_PREDATOR_CHECKPOINT_DIR = DEFAULT_PREDATOR_CHECKPOINT_RUN_DIR / "checkpoint_000010"

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
