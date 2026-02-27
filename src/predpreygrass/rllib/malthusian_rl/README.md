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

## Current Status of This Module

This module currently preserves the walls/occlusion environment mechanics as the starting point:

- static/manual walls,
- LOS-aware observations and movement constraints,
- local movement and local spawn near parent,
- multi-policy RLlib training/evaluation scripts.

The explicit archipelago update (`phi -> mu`) is not yet wired into the training loop by default. This module is intended as the place to add it.

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

1. Precompute `island_id` for every free grid cell (flood fill over non-wall cells).
2. Track each living agent's island from its position.
3. Log per-island species stats each epoch (births, deaths, energy delta, captures).
4. Compute `phi[species, island]` from those stats.
5. Update `mu[species, island]` at epoch boundaries.
6. Use `mu` to bias initial spawn / replenishment by island.

## References

1. Leibo, J. Z., Hughes, E., Lanctot, M., et al. (2019). *Malthusian Reinforcement Learning*. AAMAS 2019.  
   https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf
2. Malthus, T. R. (1798). *An Essay on the Principle of Population*.  
   https://www.gutenberg.org/ebooks/4239
3. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
   http://incompleteideas.net/book/the-book-2nd.html
4. Hardin, G. (1968). *The Tragedy of the Commons*. Science, 162(3859), 1243-1248.  
   https://doi.org/10.1126/science.162.3859.1243
