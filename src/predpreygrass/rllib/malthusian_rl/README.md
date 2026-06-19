# Malthusian RL in PredPreyGrass

This module is a dedicated copy of `rllib/walls_occlusion` under:

`src/predpreygrass/rllib/malthusian_rl`

Its purpose is to provide a clean base for **Malthusian Reinforcement Learning (MRL)** experiments in Predator-Prey-Grass (PPG), where walls can be used to create spatially isolated "islands" (demes) inside a single gridworld.

For strict paper-parity work tracking, see:

- `EXACT_REPRODUCTION_CHECKLIST.md`
- `EXACT_DEVIATIONS.md`
- `EXACT_CITATION_MAP.md`

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
| Fitness signal `phi` | Per-species, per-island performance signal | Strict mode: return-only fitness; generalized mode: ecology-weighted score | Strong |
| Allocation update | Multiplicative/softmax-like update from `phi` | Strict mode: multiplicative update; generalized mode: z-score/logit update | Strong |
| Episode horizon | Often fixed horizons | Fixed horizon via `max_steps`; no extinction-based `__all__` termination | Strong |
| Globality of `mu` during training | Single ecological process | Enforced with `num_env_runners=1`, `num_envs_per_env_runner=1` | Strong |
| Within-episode demography | Not the central mechanism | Strict mode disables within-episode reproduction; generalized mode can enable it | Strong in strict mode |
| Migration | Task-dependent | Hard islands by default (no migration gates) | Compatible |

What this means in practice:

- This module now captures the core Malthusian control loop (measure `phi`, update `mu`, reallocate next episode).
- Default config now targets strict paper-style replication (`malthusian_replication_mode="strict"`).
- The previous ecology-heavy setup remains available via `malthusian_replication_mode="generalized"`.

## Current Status of This Module

This module currently includes:

- static/manual walls,
- default hard-island map (4 disconnected equal-size islands on 25x25),
- default heterogeneous species setup enabled (4 policy species: `type_1_predator`, `type_2_predator`, `type_1_prey`, `type_2_prey`),
- LOS-aware observations and movement constraints,
- local movement and local spawn near parent,
   - multi-policy RLlib training/evaluation scripts,
   - APPO/V-trace training path with LSTM-capable module specs,
- fixed-horizon episode handling (`max_steps`),
- episode-end Malthusian scaffold:
   - strict default: `phi[species,island]` computed from per-agent episode return,
   - strict default: `mu[species,island]` updated with multiplicative update,
   - optional generalized mode: ecology-weighted `phi` and z-score/logit update,
  - reset-time initial agent placement sampled per-island from `mu`,
  - strict single-env training default (`num_env_runners=1`, `num_envs_per_env_runner=1`) so `mu` is not split across worker-local env copies,
  - callback logging of Malthusian diagnostics (`mu`, `phi`, and `phi` components) into RLlib metrics,
  - RLlib-safe dynamic-agent handling: agent IDs are not reused within the same episode after termination,
   - strict default disables within-episode reproduction (`enable_within_episode_reproduction=False`).

Current strict default (from `config_env.py`) is:

`phi_agent = cumulative_reward`

Generalized optional mode keeps ecology-weighted scoring:

`phi_agent = 2.0*offspring + 1.0*survival + 0.5*foraging + 0.25*energy_delta_rel - 1.0*death + 0.0*reward`

## Logged Malthusian Metrics

During training, the callback reads episode-end `infos["__all__"]` and logs:

- `malthusian/mu/<species>/island_<id>`
- `malthusian/phi/<species>/island_<id>`
- `malthusian/phi_component/<component>/<species>/island_<id>`
- `malthusian/count/<species>/island_<id>`
- `malthusian/count_total/<species>`
- `malthusian/count_total/predators`
- `malthusian/count_total/prey`

This makes it possible to inspect ecological dynamics over time in Tune results and TensorBoard.

## Nature vs Nurture (Simple)

In this setup:

- **Nature** = policy/species differences (`type_1_*` vs `type_2_*` policies).
- **Nurture** = island-specific ecology (local opponents, local resources, local crowding).

How to interpret your Malthusian curves:

- If `mu` stays uniform, nurture is weak (or `phi` has little island contrast).
- If `mu` becomes species-specific and non-uniform, the ecology is sorting behaviors into niches.
- If a species concentrates on one island (`mu` near 1.0 there), that island is currently the best niche for that species under your `phi` definition.

So the mechanism is usually **nature x nurture**:

- policy differences create different behavioral tendencies,
- island ecology selects among those tendencies,
- `phi -> mu` turns that selection into population reallocation next episodes.

## Current Results Snapshot (2026-02-27)

Run inspected:

- `~/Dropbox/02_marl_results/predpreygrass_results/ray_results/PPO_MALTHUSIAN_HARD_ISLANDS_2026-02-27_18-23-00`
- Trial: `PPO_PredPreyGrass_f741c_00000_0_2026-02-27_18-23-00`
- Snapshot point: training iteration 56

Observed metrics at iteration 56:

- `env_runners/episode_return_mean`: `420.0`
- total counts: predators `6` (type_1 `3`, type_2 `3`), prey `50` (type_1 `25`, type_2 `25`)
- `mu` (allocation) is non-uniform for all species:
  - `type_1_predator`: `[0.256, 0.234, 0.290, 0.220]` (highest island 2)
  - `type_2_predator`: `[0.185, 0.246, 0.114, 0.456]` (highest island 3)
  - `type_1_prey`: `[0.450, 0.140, 0.127, 0.284]` (highest island 0)
  - `type_2_prey`: `[0.097, 0.361, 0.097, 0.444]` (highest island 3)
- per-island prey counts show niche concentration:
  - `type_1_prey` counts: `[23, 0, 0, 2]`
  - `type_2_prey` counts: `[0, 10, 0, 15]`

Conclusions from this snapshot:

- The Malthusian loop is active: `mu` is adapting (not uniform) and species allocations diverge across islands.
- Prey species show clearer niche separation than predators at this point.
- Predator fitness signals are sparse/noisier (small populations, many `phi` values near 0 or death-penalty levels), so predator `mu` is less stable.
- No global collapse is visible in this snapshot (both prey species at full configured mass; predators persist at baseline mass).

Notes:

- This is a point-in-time snapshot, not a final claim about convergence.
- Re-check over longer horizons to confirm whether prey specialization persists and whether predator allocations stabilize.

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

Strict exact APPO run:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/tune_appo_malthusian_exact.py
```

Strict exact APPO multi-seed run:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/scripts/run_exact_reproduction_seeds.py --seeds 0 1 2
```

Article-task Allelopathy reconstruction:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/scripts/run_article_reproduction_seeds.py --task allelopathy --variant biased --condition allelopathy_biased_heterogeneous_dynamic --seeds 0 1 2
```

Article-task Clamity reconstruction:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/scripts/run_article_reproduction_seeds.py --task clamity --condition clamity_dynamic_population --seeds 0 1 2
```

Full article-condition matrix reconstruction:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/scripts/run_article_condition_matrix.py --seeds 0 1 2
```

Paper-like evaluation for exact runs:

```bash
PYTHONPATH=src python src/predpreygrass/rllib/malthusian_rl/evaluate_exact_reproduction.py
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

1. Implement the paper's Clamity and Allelopathy games directly if the goal is exact article-figure reproduction.
2. Scale the exact protocol from the current PPG four-island mapped analogue to the article's K=960, NI=60 Allelopathy setting once the paper game exists.
3. If scaling back to parallel env runners, add explicit global `mu` synchronization across workers at episode boundaries.
4. Add optional migration controls (rare gates or transfer budget) to move from hard-island to soft-island experiments.

## Exact Reproduction Boundary

Implemented now:

- frozen mapped paper protocol: `config/config_paper_protocol.py`,
- text-grounded article-task reconstructions: `article_tasks.py`,
- article-task protocol configs: `config/config_article_protocol.py`,
- cited APPO/V-trace learner config: `config/config_appo_exact.py`,
- exact trainer: `tune_appo_malthusian_exact.py`,
- article-task trainer: `tune_appo_article_exact.py`,
- named article-condition presets for dynamic, fixed-population, single-agent, and solitary-evaluation protocol conditions,
- multi-seed harness: `scripts/run_exact_reproduction_seeds.py`,
- article-task multi-seed harness: `scripts/run_article_reproduction_seeds.py`,
- full article-condition matrix harness: `scripts/run_article_condition_matrix.py`,
- paper-like evaluator and plots: `evaluate_exact_reproduction.py`,
- condition summaries plus Figure 2/Figure 3 family CSV and PNG summaries,
- article-condition coverage checks in acceptance reports,
- metadata-integrity checks for git status, package versions, and config checksum validation,
- mapped-protocol acceptance bands in `config/config_paper_protocol.py`.

Still not claimable:

- exact Clamity reproduction,
- exact Allelopathy reproduction,
- matching article Figure 2 or Figure 3 outcomes.

Reason:

- Allelopathy and Clamity now exist as runnable text-grounded reconstructions.
- Exact article reproduction is still blocked because the paper does not publish enough environment constants for literal Figure 2/Figure 3 reproduction.
- DeepMind Melting Pot contains a related official `allelopathic_harvest` substrate, but it is a later non-identical task, not the 2019 two-shrub Malthusian Allelopathy task.
- DeepMind Lab2D contains the simulator platform, but no public Clamity or 2019 two-shrub Allelopathy task implementation was found there.
- The missing constants are tracked in `config/config_article_protocol.py` under `ARTICLE_EXACT_BLOCKERS`.

## References

1. Leibo, J. Z., Hughes, E., Lanctot, M., et al. (2019). *Malthusian Reinforcement Learning*. AAMAS 2019.  
   https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf
2. Malthus, T. R. (1798). *An Essay on the Principle of Population*.  
   https://www.gutenberg.org/ebooks/4239
3. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
   http://incompleteideas.net/book/the-book-2nd.html
4. Hardin, G. (1968). *The Tragedy of the Commons*. Science, 162(3859), 1243-1248.  
   https://doi.org/10.1126/science.162.3859.1243
