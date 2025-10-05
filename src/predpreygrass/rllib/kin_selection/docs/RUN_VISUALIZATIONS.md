# Visualizations run guide

### 1) View metrics in TensorBoard

Point TensorBoard at the Ray results root (from Section 2):

```bash
tensorboard --logdir "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
```

Look under `ray/tune/env_runners/custom_metrics/*` for:
- Global
  - helping_rate, share_attempt_rate, received_share_mean, shares_per_episode
- Per-type
  - helping_rate_type_1_prey, helping_rate_type_2_prey, ... (same for predators)
  - share_attempt_rate_<group>, received_share_mean_<group>
- Routing and population
  - shares_to_same_type_rate, shares_to_other_type_rate
  - fraction_type_2_prey, fraction_type_2_predator

### 2) Metric glossary (quick)

- helping_rate: donor-side successful shares per step
- share_attempt_rate: share attempts per step (success + failure)
- received_share_mean: average received energy per step (recipient-side)
- shares_per_episode: total shares in an episode (donor-side)
- shares_to_same_type_rate / shares_to_other_type_rate: donor→recipient routing rates per step
- fraction_type_2_prey / fraction_type_2_predator: average fraction of type_2 present across steps in an episode

### 3) Save evidence screenshots

Save your TensorBoard screenshots to the repo’s root assets folder so markdown renders them:

- Helping rate: `assets/images/kin_selection/helping_rate_<tag>.png`
- Shares per episode: `assets/images/kin_selection/shares_per_episode_<tag>.png`

You can embed them from module docs with relative paths like:

```markdown
![Helping](../../../assets/images/kin_selection/helping_rate_<tag>.png)
```

### 4) Offline evaluation and plotting

You can evaluate saved checkpoints and produce time-series plots without running training.

1) Generate a CSV over all checkpoints in a results directory (with CIs)

Replace `<RESULTS_DIR>` with your experiment folder, e.g.
`~/Dropbox/02_marl_results/predpreygrass_results/ray_results/PPO_KIN_SELECTION_YYYY-MM-DD_HH-MM-SS`.

```bash
python -u src/predpreygrass/rllib/kin_selection/analysis/eval_checkpoints_helping.py \
  --results-dir "<RESULTS_DIR>" \
  --out "output/parameter_variation/helping_eval.csv" \
  --n-episodes 10 \
  --bootstrap 200 \
  --confidence 0.95
```

Options:
- Add `--stochastic` to include exploration during evaluation.
- Pass `--env-config path/to/overrides.json` to merge env overrides for eval.

2) Plot helping rate (with CI band) and type counts over iterations

```bash
python -u src/predpreygrass/rllib/kin_selection/analysis/plot_helping_time_series.py \
  --csv "output/parameter_variation/helping_eval.csv" \
  --out "assets/images/kin_selection/helping_time_series.png" \
  --ema 5
```

The CSV schema includes: iteration, helping_rate (+CI), received_share_mean (+CI), and per-type counts.
