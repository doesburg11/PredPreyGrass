# Malthusian RL in PredPreyGrass

This module is a dedicated copy of `rllib/walls_occlusion` under:

`src/predpreygrass/rllib/malthusian_rl`

Its purpose is to provide a clean base for **Malthusian Reinforcement Learning (MRL)** experiments in Predator-Prey-Grass (PPG), where walls can be used to create spatially isolated "islands" (demes) inside a single gridworld.

## What Malthusian RL Means Here

In Malthusian RL, selection pressure is shaped by **population pressure across environments** rather than only by single-episode rewards. A practical mapping for this codebase is:

- each walled compartment = one island,
- local interaction and reproduction happen inside each island,
- per-island fitness is measured per species (for example from reproduction, survival, or energy growth),
- population allocation over islands is updated between epochs.

Conceptually:

- `phi[species, island]`: local fitness signal,
- `mu[species, island]`: fraction of that species allocated to each island for the next epoch.

## Leibo Malthusian RL Explained

Leibo et al. (2019) introduce Malthusian RL as a two-timescale learning process:

1. Behavioral timescale (inside episodes):
   - Agents execute policies in an environment.
   - Standard RL learning happens from trajectories and returns.

2. Ecological timescale (between episodes):
   - For each species and island, compute a fitness signal `phi[s, i]`.
   - Update species distribution over islands `mu[s, i]` so islands with higher `phi` receive a larger share of that species in later episodes.

The key idea is that adaptation pressure is not only from immediate rewards, but from a population-allocation feedback loop.

In simplified form:

- Run episode(s) with current `mu`.
- Measure per-species island fitness `phi`.
- Update allocation with a multiplicative/softmax-like rule:
  `mu_next[s, i] ∝ mu[s, i] * exp(eta * phi[s, i])` (often normalized/stabilized in practice).
- Use `mu_next` to choose where each species is instantiated in the next training episodes.

Important conceptual points from the paper:

- Species are policy-sharing groups: one policy per species.
- Malthusian pressure is primarily inter-episode allocation pressure.
- Multi-island structure is required for this pressure to be meaningful.
- Heterogeneous species (different policies) are central in the stronger coexistence/specialization experiments.

## Comparison: Leibo vs This Codebase

The table below is the practical mapping from paper concepts to this repository.

| Dimension | Leibo (paper concept) | Current implementation here | Match |
|---|---|---|---|
| Archipelago structure | Multiple islands/environments | Hard islands from wall-defined connected components on one grid | Strong |
| Species as policy groups | One policy per species | `type_1_predator`, `type_2_predator`, `type_1_prey`, `type_2_prey` each map to separate policies | Strong |
| Inter-episode allocation | `mu` controls species allocation over islands | Reset placement samples per-species counts per island from `mu` | Strong |
| Fitness signal `phi` | Per-species, per-island performance signal | Per-agent ecology-weighted score aggregated by species+spawn island | Strong generalization |
| Allocation update | Multiplicative/softmax-like update from `phi` | Exponentiated/logit update with normalization and optional floor | Strong |
| Episode horizon | Often fixed horizons | Fixed horizon via `max_steps`; no extinction-based `__all__` termination | Strong |
| Globality of `mu` during training | Single ecological process | Enforced with `num_env_runners=1`, `num_envs_per_env_runner=1` | Strong |
| Within-episode demography | Not the central mechanism | Explicit birth/death/reproduction dynamics are active | Divergence (intentional generalization) |
| Migration | Task-dependent | Hard islands by default (no migration gates) | Compatible |

What this means in practice:

- This module now captures the core Malthusian control loop (measure `phi`, update `mu`, reallocate next episode).
- It is a Leibo-inspired generalization rather than a strict reproduction, mainly because explicit within-episode reproduction/death dynamics are included and `phi` is ecology-weighted by default.
- If strict paper-style ablation is desired, configure a "replication mode" by minimizing/turning off within-episode reproduction effects and simplifying `phi` to return-based fitness.

## Current Status of This Module

This module currently includes:

- static/manual walls,
- default hard-island map (4 disconnected equal-size islands on 25x25),
- default heterogeneous species setup enabled (4 policy species: `type_1_predator`, `type_2_predator`, `type_1_prey`, `type_2_prey`),
- LOS-aware observations and movement constraints,
- local movement and local spawn near parent,
- multi-policy RLlib training/evaluation scripts,
- fixed-horizon episode handling (`max_steps`),
- episode-end Malthusian scaffold:
  - `phi[species,island]` computed from explicit ecology metrics per agent (offspring, survival, foraging/captures, relative energy change, death indicator),
  - `mu[species,island]` updated with an exponentiated update rule,
  - reset-time initial agent placement sampled per-island from `mu`,
  - strict single-env training default (`num_env_runners=1`, `num_envs_per_env_runner=1`) so `mu` is not split across worker-local env copies.

Current default `phi` scoring (from `config_env.py`) is:

`phi_agent = 2.0*offspring + 1.0*survival + 0.5*foraging + 0.25*energy_delta_rel - 1.0*death + 0.0*reward`

## Why This Is a Good Base

- The environment already supports wall-defined compartments (`manual_wall_positions`).
- LOS masking can reduce cross-compartment informational leakage.
- Reproduction and spawn logic are centralized and easy to extend with island constraints.
- Existing tune/eval scripts allow quick iteration while adding island metrics.

## Quick Start

Train:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/tune_ppo_malthusian_rl.py
```

Random rollout viewer:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/random_policy.py
```

Checkpoint evaluation:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/evaluate_ppo_from_checkpoint_debug.py
```

## Recommended Hard-Island Setup

For clean Malthusian experiments, start with:

- equal-size chambers,
- no migration gates,
- LOS masking enabled,
- movement constrained by walls,
- local reproduction only.

Typical config knobs to control:

- `wall_placement_mode`
- `manual_wall_positions`
- `mask_observation_with_visibility`
- `respect_los_for_movement`
- spawn/reproduction placement logic in `predpreygrass_rllib_env.py`

## Suggested Next Additions

1. Add trainer-side logging/plots for `phi` and `mu` trajectories per species and island.
2. If scaling back to parallel env runners, add explicit global `mu` synchronization across workers at episode boundaries.
3. Add optional migration controls (rare gates or transfer budget) to move from hard-island to soft-island experiments.
4. Tune `malthusian_phi_weights` against your exact research target (for example reducing direct reward term to zero or adjusting death penalty).

## References

1. Leibo, J. Z., Hughes, E., Lanctot, M., et al. (2019). *Malthusian Reinforcement Learning*. AAMAS 2019.  
   https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf
2. Malthus, T. R. (1798). *An Essay on the Principle of Population*.  
   https://www.gutenberg.org/ebooks/4239
3. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
   http://incompleteideas.net/book/the-book-2nd.html
4. Hardin, G. (1968). *The Tragedy of the Commons*. Science, 162(3859), 1243-1248.  
   https://doi.org/10.1126/science.162.3859.1243
