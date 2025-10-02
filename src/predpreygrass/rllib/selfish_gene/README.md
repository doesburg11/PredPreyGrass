# PredPreyGrass – Selfish Gene experiment

This experiment variant introduces a biologically inspired Tier‑1 Selfish Gene reward and an optional observation‑only kin‑density channel on top of the standard PredPreyGrass multi‑agent gridworld.

## What’s new
- Tier‑1 Selfish Gene reward: replaces direct reproduction reward for predators and prey with a windowed lineage reward equal to the count of living offspring born within a configurable window of steps.
- Kin‑density observation channel (optional): adds a per‑agent scalar channel containing a normalized count of same‑policy neighbors within a local radius. This only changes observations (no reward shaping).
- Repro/lineage logging: each reproduction event logs parent/offspring unique IDs and the lineage reward to `reproduction_events.log`.

## Key files
- Environment: `predpreygrass_rllib_env.py`
- Default env config: `config/config_env_selfish_gene.py`
- Training entrypoint: `tune_ppo_selfish_gene.py`
- Evaluation (RLModule loader): `evaluate_ppo_from_checkpoint_debug.py`

## Environment config (relevant keys)
- Lineage reward
  - `lineage_reward_window` (int, default 50): number of steps for the window used when counting living offspring for the reward.
  - `lineage_reward_window_predator` (int, optional): species-specific override for predators; falls back to `lineage_reward_window`.
  - `lineage_reward_window_prey` (int, optional): species-specific override for prey; falls back to `lineage_reward_window`.
- Observation features
  - `include_visibility_channel` (bool): appends an LOS visibility mask as a channel; dynamic observation channels can also be masked via `mask_observation_with_visibility`.
  - `mask_observation_with_visibility` (bool): multiplies dynamic channels by LOS mask (does not change channel count).
  - `include_kin_density_channel` (bool): appends normalized same‑policy neighbor count as last channel.
  - `kin_density_los_aware` (bool): when True, kin-density counts only kin that are LOS-visible (blocked by walls otherwise).
  - `kin_density_radius` (int): Chebyshev radius R for kin counting (neighbors where max(|dx|,|dy|) ≤ R).
  - `kin_density_norm_cap` (int/float): kin count is divided by this cap and clipped to [0,1].
- Episode length
  - `max_steps` (int): episode step cap (see recommendations below).

Notes on channel ordering
- Base channels: walls + dynamic layers (predators, prey, grass) → `num_obs_channels`.
- Optional visibility (if `include_visibility_channel=True`) is appended next.
- Optional kin‑density (if `include_kin_density_channel=True`) is appended last.

Compatibility
- Enabling extra channels changes the observation shape. Checkpoints trained without kin‑density or visibility must be evaluated with the same flags they were trained with. The evaluator reads `run_config.json` and merges observation‑critical keys to avoid mismatches.

## Recommendations
- Lineage window
  - Start with `lineage_reward_window = 100–150` (baseline: 150). You can override per-species via `lineage_reward_window_predator` and `lineage_reward_window_prey` if their timescales diverge.
  - Rationale: your grass refill is ≈50 steps (`2.0 / 0.04`); a 2–3× window captures ecologically connected offspring while avoiding myopic credit.
  - If rewards feel sparse/slow: 80–100. If you want stronger emphasis on survival: 200–250.
- Max steps
  - With `lineage_reward_window ≈ 150`: set `max_steps = 700–900` (baseline: 800).
  - Fast debug: 200–400. Long‑horizon dynamics: 1200–2000.
  - Rule of thumb: `max_steps ≈ 4–6 × lineage_reward_window` or `≈ 10–20 × grass refill cycle`.
- Kin‑density channel
  - Feature is observation‑only; no reward change. Safe to try enabling for richer inputs.
  - Default `kin_density_radius = 2`, `kin_density_norm_cap = 8` are a good start.
  - If your maps have corridors/occlusion and you want realistic perception, set `kin_density_los_aware = True`.

### Choosing the kin-density radius R

- Start with R = 2 for most runs. It’s local enough to avoid washing out the signal but large enough to catch clusters.
- If LOS-aware is enabled, keep R within the observation windows (prey range 5 → practical R ≈ 2; predator range 7 → R up to 3). Using a single R=2 works well for both species.
- Adjustments:
  - Very sparse populations or heavy occlusion/corridors: R = 3
  - Very dense populations: R = 1–2 to keep the measure local
- Quick sanity check: expected same-policy neighbors ≈ ρ_g × A_R, where A_R = (2R+1)^2 − 1 and ρ_g is group density (agents of that policy / free cells). Set `kin_density_norm_cap` ≈ 1.5–2× that expectation to avoid early saturation and keep the channel informative.

## How to run
- Train (PPO, new RLlib API):
  - Update `config/config_env_selfish_gene.py` for your desired flags (e.g., enable kin channel and set lineage window).
  - Run the training launcher:
    - `python src/predpreygrass/rllib/selfish_gene/tune_ppo_selfish_gene.py`
- Evaluate a checkpoint (RLModule loader + visualization):
  - Adjust `evaluate_ppo_from_checkpoint_debug.py` paths to your checkpoint.
  - The script auto‑merges observation‑critical keys from `run_config.json`.

Artifacts & logging
- Ray results (training): `~/Dropbox/02_marl_results/predpreygrass_results/ray_results/` (subfolder per run).
- Env one‑shot event log: `reproduction_events.log` (step, parent UID, child UID, lineage reward, living‑offspring list).

## Implementation details
- Tier‑1 Selfish Gene reward is computed by `_windowed_lineage_reward(agent_id, window)` and applied during predator/prey reproduction handlers for the parent. The per‑agent lineage window is configurable via `lineage_reward_window`.
  - When species-specific keys are present, predators use `lineage_reward_window_predator` and prey use `lineage_reward_window_prey`; otherwise both fall back to `lineage_reward_window`.
- Unique lineage tracking maintains `unique_agents` and per‑unique stats (birth/death steps, parent UID, offspring count, fitness aggregates) to support downstream analysis.
- Kin‑density channel uses same‑policy prefix (e.g. `type_1_predator`) and Chebyshev radius. When `kin_density_los_aware=True`, the kin count is restricted to LOS‑visible kin by multiplying the kin mask with the visibility mask before counting; this does not change the observation shape.

## Optional extensions
- Directional kin‑density: split the kin‑density signal into sectors (e.g., forward/left/right or octants) for richer spatial cues.
- Smoother credit: replace hard cutoff with exponential decay per living offspring age (credit shaping with a half‑life).

## Troubleshooting
- Observation shape mismatch when loading a checkpoint:
  - Ensure `include_visibility_channel` and `include_kin_density_channel` match training. The evaluator attempts to merge keys from `run_config.json`, but explicit overrides in your current config may still differ.
- Sparse lineage reward signal:
  - Reduce `lineage_reward_window` to 80–100, or increase prey/grass abundance for easier reproduction.
- Early population collapse:
  - Shorten `max_steps` to 400–600 for more resets, or lower movement/step energy losses.

## Cooperation logging and analysis

You can enable lightweight per-episode logging for post-hoc cooperation analysis without changing rewards or training.

Enable in `config/config_env_selfish_gene.py`:

- `"enable_coop_logging": true`
- `"coop_log_dir": "output/coop_logs"`

Each episode writes one JSON file, containing:

- Static metadata: `episode_index`, `seed`, `max_steps`, subset of config (incl. `kin_density_radius`), and `walls` when present (used by LOS-aware analysis).
- Steps: for each step, per-agent fields: `unique_id`, `policy_group`, `root_ancestor`, `position`, `age`, `energy`, `offspring_count`.

The environment flushes logs on reset, at max-steps truncation, and on `close()`.

Run analysis:

```
python src/predpreygrass/rllib/selfish_gene/analysis/coop_metrics.py --log-dir output/coop_logs
```

Options:

- `--los-aware`: Use wall LOS blocking when counting neighbors.
- `--bootstrap N`: Run N bootstrap iterations to report 95% confidence intervals.
- `--seed S`: Seed for shuffling and bootstrap.

Outputs JSON blocks for AI and KPA:

- AI: Mean fraction of same-root neighbors minus a shuffled baseline (ten shuffles per step). With `--bootstrap`, returns `ci95` as well.
- KPA: Difference in reproduction probability when at least one same-root neighbor is within radius vs none; optional `ci95` with `--bootstrap`.

Tips:

- Use longer episodes and multiple runs to accumulate signal. Short debug runs typically yield values near zero.
- `--los-aware` requires the episode logs to include a `walls` array (automatically added by the env when logging is enabled).

### One-shot harness (generate + analyze)

To streamline experimentation, use the helper script to both generate episodes with logging and compute metrics in one command:

```
python src/predpreygrass/rllib/selfish_gene/analysis/quick_run_and_analyze.py \
  --episodes 20 \
  --max-steps 800 \
  --los-aware \
  --bootstrap 1000 \
  --by-policy
```

Override env config at runtime with repeated `--config-override key=value` flags, e.g.:

```
--config-override kin_density_radius=3 \
--config-override include_kin_density_channel=true \
--config-override wall_placement_mode=manual
```

Notes:

- The harness enables `enable_coop_logging` internally and uses `coop_log_dir` from the env config unless `--log-dir` is provided.
- LOS-aware analysis uses the `walls` stored in each episode log.
- For consistency, prefer running multiple episodes (≥20) with `max_steps` aligned to your lineage windows.

