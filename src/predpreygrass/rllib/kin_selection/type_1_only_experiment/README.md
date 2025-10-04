# Kin Selection – Type_1-only Experiment Results

Date: 2025-10-04

## Setup

- Environment: kin_selection variant (`config/config_env_kin_selection.py`)
  - lineage_reward_enabled: True
  - reproduction_reward_enabled: False
  - lineage_reward_window: 150
  - share_enabled: True; share_roles: ["prey"]
  - share_radius: 1; share_respect_los: True; share_kin_only: True
  - share_amount: 1.0; share_efficiency: 0.8; share_donor_min: 4.0; share_donor_safe: 2.0; share_cooldown: 2
  - action_mask_enabled: False (stability with RLlib new API); include_kin_energy_channel: True
- Policies present: type_1 only (predator and prey). type_2 initial actives were set to 0, so no type_2 policies were built for this run.
- Trainer: `tune_ppo_kin_selection.py` (RLlib PPO new API, Torch)
- Observations to RLModule: Box-only (no Dict obs to avoid encoder permute issues)

## Evidence of Cooperation

Tensorboard screenshots:

Helping metrics increased steadily during training, indicating learned use of SHARE by prey:

Helping rate: `custom_metrics/helping_rate` (per-step donor-side successes)

![Helping rate](../../../../assets/images/kin_selection/helping_rate_type1_only.png)

Shares per episode: `custom_metrics/shares_per_episode`

![Shares per episode](../../../../assets/images/kin_selection/shares_per_episode_type1_only.png)

## Interpretation

- With only one type in the population, kin-only eligibility is trivially satisfied: every neighbor is kin.
- The upward trends in helping_rate and shares_per_episode demonstrate that the SHARE mechanic is discoverable and beneficial under the configured thresholds and LOS constraints.
- This is cooperation, not kin discrimination. Because there is no out-group type, we cannot infer a bias toward kin vs non-kin from this run.

## Sanity checks observed

- No action_mask in obs to avoid Dict encoder issues; training remained stable.
- Intermittent env-runner restarts observed previously in other runs did not prevent progress; helping metrics continued to climb.

## Next steps for kin selection tests

- Introduce type_2: set `n_initial_active_type_2_prey` and/or `n_initial_active_type_2_predator` > 0 (and ensure corresponding `n_possible_type_2_*` > 0), then start a new Tune experiment so policies for type_2 are created.
- Keep `share_kin_only=True` and examine per-type SHARE patterns (same-type vs other-type recipients) online and offline.
- Optionally relax thresholds or increase `share_radius` if `share_attempt_rate` is low.
- For robustness, add offline plots splitting SHARE counts by donor/recipient type and bootstrap CIs vs shuffled baselines.

## Reproducibility notes

- Run configs are stored per experiment in `run_config.json` under the Ray results directory.
- Resume utility: use `resume_tune_ppo_kin_selection.py` to continue the exact Tune experiment from its folder.
