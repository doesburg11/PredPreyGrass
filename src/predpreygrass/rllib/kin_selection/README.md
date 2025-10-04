# Kin Selection (Predator–Prey–Grass)

This module is a kin-selection-focused variant of the Predator–Prey–Grass multi-agent environment built on Ray RLlib (new API). It extends the walls/occlusion experiments with heritable types, an optional Tier‑1 “Selfish Gene” lineage reward, and a minimal helping mechanic (energy sharing) with line‑of‑sight (LOS) constraints. The goal is to make cooperation measurable and reproducible during training and evaluation.

The earlier `selfish_gene` path tried to detect “emergent cooperation” indirectly, mainly via spatial clustering and lineage counts. In practice, those signals were too coarse and confounded: clustering can arise from walls, food gradients, or identical policy priors; lineage can spike simply from lucky foraging dynamics. We could not separate “agents helping kin” from “agents co‑located with kin” or “agents reproducing more because the map made it easy.” After several runs with optimistic hyperparameters, the online metrics remained flat and post‑hoc analyses (CIs over shuffled baselines) didn’t move the needle.

Kin Selection tackles that gap with an explicit, learnable helping action. By adding a SHARE action (with eligibility, LOS, and energy budgets) and logging attempts vs. successes in real time, we turn a fuzzy proxy into a clear behavioral readout. Prey start with a modest, feasible sharing option; action masking exposes when SHARE is actually allowed, and a kin‑energy feature provides local context to learn when helping makes sense.

## Objectives

- Model kin selection pressures via lineage-based rewards and heritable types.
- Provide a simple, learnable helping action (SHARE) with eligibility and sensible costs.
- Surface cooperation online (during training) and offline (post‑hoc) through robust metrics.
- Keep experiments reproducible with versioned configs and stable logging artifacts.

## What’s new vs walls_occlusion

Compared to `rllib/walls_occlusion`:

1. Lineage reward (Tier‑1 Selfish Gene)
   - Toggleable via `lineage_reward_enabled` and `reproduction_reward_enabled`.
   - Counts living offspring born within a sliding step window (configurable globally and per species):
     - `lineage_reward_window`, `lineage_reward_window_predator`, `lineage_reward_window_prey`.
   - When enabled with `reproduction_reward_enabled=False`, direct reproduction rewards are replaced by lineage counts.

2. Helping/sharing mechanic (SHARE)
   - Extra discrete action appended to eligible roles (default: prey only) controlled by `share_enabled` and `share_roles`.
   - Transfer rules controlled by thresholds and LOS:
     - `share_amount`, `share_efficiency`, `share_donor_min`, `share_donor_safe`, `share_cooldown`, `share_radius`,
       `share_respect_los`, `share_kin_only`.
   - Action masking: `action_mask_enabled=True` adds an `action_mask` to observations; SHARE is masked out when ineligible.
   - Optional kin-energy feature channel to inform policies about nearby kin energy: `include_kin_energy_channel=True`,
     `kin_energy_respect_los`.

3. Observation and movement with LOS
   - Inherits walls/occlusion LOS features; optionally appends a visibility channel and/or masks dynamic channels by LOS.
   - Movement can forbid corner cutting and LOS-blocked moves (`respect_los_for_movement`).

4. Online cooperation metrics (training‑time)
   - HelpingMetricsCallback logs to Tune/Ray result JSON and TensorBoard:
     - `custom_metrics/helping_rate` – share successes per step (donor-side)
     - `custom_metrics/received_share_mean` – average received energy per step
     - `custom_metrics/shares_per_episode` – total shares per episode
     - `custom_metrics/share_attempt_rate` – attempt frequency per step (success + failure)
   - Combined safely with an EpisodeReturn callback via `CombinedCallbacks`.

5. Safer callbacks
   - Handles list/dict shaped `infos` from env runners and guards sub-callbacks with try/except to avoid worker crashes.

## Code map

- `predpreygrass_rllib_env.py`
  - Env core with walls, LOS, energy/reproduction, lineage reward, SHARE action, action masks, kin‑energy channel.
  - Key methods:
    - `_attempt_share`, `_build_action_mask`, `_can_share_now`, `_has_share_recipient`
    - `_compute_lineage_reward`, `_compute_kin_energy_feature`
    - `_line_of_sight_clear`, `_get_observation`, `_get_move`
- `tune_ppo_kin_selection.py`
  - New‑API PPO trainer: builds multi‑agent module spec, attaches `CombinedCallbacks`, writes `run_config.json`.
- `utils/`
  - `helping_metrics_callback.py` – logs helping metrics online.
  - `combined_callbacks.py` – composes EpisodeReturn + HelpingMetrics safely.
  - `networks.py` – builds RLlib `MultiAgentRLModuleSpec`, supports Dict observations with `action_mask`.
  - `matplot_renderer.py`, `pygame_grid_renderer_rllib.py` – visualization utilities.
- `config/`
  - `config_env_perimeter_four_gaps_walls.py` – base env with manual walls + LOS features.
  - `config_env_kin_selection.py` – lineage + sharing variant; gentle thresholds for early feasibility.
  - `config_env_base.py` – general defaults (energy, reproduction, ranges).
  - `config_ppo_cpu.py`, `config_ppo_gpu.py` – PPO defaults sized for typical CPU/GPU setups.
- `analysis/`
  - `eval_checkpoints_helping.py`, `plot_helping_time_series.py` – offline cooperation evaluation and plots.
- `test/`
  - LOS/corner-cutting sanity tests and utilities.

## Config highlights

Environment (in `config/config_env_kin_selection.py`):
- Lineage reward
  - `lineage_reward_enabled=True`
  - `reproduction_reward_enabled=False` (to use lineage instead of direct reproduction reward)
  - `lineage_reward_window=150` (override per species as needed)
- Sharing (helping)
  - `share_enabled=True`, `share_roles=["prey"]`
  - `share_radius=1`, `share_respect_los=True`, `share_kin_only=True`
  - `share_amount=1.0`, `share_efficiency=0.8`
  - `share_donor_min=4.0`, `share_donor_safe=2.0`, `share_cooldown=2`
- Observation helpers
  - `action_mask_enabled=True`
  - `include_kin_energy_channel=True`, `kin_energy_respect_los=True`

PPO (in `config/config_ppo_cpu.py`)
- New API stack with Torch; exploration nudged via `entropy_coeff=0.01` (tune if SHARE discovery is slow).
- Env runners sized by CPU; see file for details.

## How to run

- Install the project (editable):

```bash
pip install -e .
```

- Launch kin selection training (CPU defaults):

```bash
python -u src/predpreygrass/rllib/kin_selection/tune_ppo_kin_selection.py
```

Artifacts:
- Ray results under `~/Dropbox/02_marl_results/predpreygrass_results/ray_results/` with experiment folder `PPO_KIN_SELECTION_<timestamp>`.
- Per‑run `run_config.json` storing env + PPO configs.
- TensorBoard metrics include `custom_metrics/*` above.

Optional quick visualization (random policy):

```bash
python src/predpreygrass/rllib/kin_selection/random_policy.py
```

## Metrics: interpretation and debugging

- `custom_metrics/share_attempt_rate` near zero → SHARE not discoverable/eligible. Try:
  - Increase `share_radius` (1 → 2)
  - Loosen thresholds: `share_donor_min` (4 → 3), `share_donor_safe` (2 → 1)
  - Temporarily increase `entropy_coeff` (0.01 → 0.02) for early exploration
- `custom_metrics/helping_rate` low but attempts > 0 → many attempts fail eligibility mid‑step (cooldown/thresholds/LOS). Inspect `share_reason` in env `infos` if debugging live.
- `received_share_mean` low → either fewer recipients in LOS/radius or energy caps quickly saturate; consider slightly larger radius or lower `share_amount` to distribute.

## Design notes and gotchas

- Action spaces differ by role:
  - Predator: Discrete(9) (3x3 movement)
  - Prey: Discrete(10) (3x3 movement + SHARE at last index)
  - To enable predator sharing too, set `share_roles=["prey","predator"]`.
- Action masking: Only SHARE is masked; movement remains unmasked.
- Kin-energy feature: Single scalar channel broadcast across the window; LOS-aware if configured.
- Lineage reward uses living offspring count within a window; direct reproduction rewards should be disabled to avoid double-counting when testing kin selection hypotheses.
- LOS: Movement blocks diagonal corner cutting; visibility masking can be used without appending a visibility channel if you need a stable channel count.
- Reproducibility: Don’t modify older version folders; this module is a separate variant.

## Next steps

- Add per-type helping breakdown and survival uplift metrics offline (recipient lifetime vs. baseline).
- Explore 2×2 experiments: lineage on/off × share on/off to isolate effects.
- Hyperparameter sweeps for SHARE thresholds and `entropy_coeff` to accelerate emergence.

If you want, I can also add a minimal “quickstart” script that loads a checkpoint and prints the cooperation metrics summary over N evaluation episodes.
