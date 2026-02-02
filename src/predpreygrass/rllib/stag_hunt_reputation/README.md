# Stag Hunt Reputation

This module extends the stag-hunt defection environment with a lightweight
predator reputation signal. It **does not** change capture rules; it only
tracks join/defect history and exposes it in observations and metrics.

## What this adds

- Per-predator reputation score in [0, 1] based on join decisions.
- Optional spatial reputation channel and/or per-agent reputation summary.
- Extra metrics for conditional cooperation (join rate vs partner reputation).

All reputation features are opt-in and default to off.

## How reputation works (summary)

- Each predator gets a reputation score based on join decisions.
- Update rule:
  - Rolling window: average of last `reputation_window` join decisions.
  - EMA: if `reputation_window <= 0`, use `reputation_ema_alpha`.
- Optionally only update on "opportunity" steps (prey in Moore neighborhood).
- Neutral default (0.5) is used until `reputation_min_samples` are observed.
- Optional observation noise via `reputation_noise_std`.

For the full design rationale, see `reputation.md`.

## Config knobs

In `config/config_env_stag_hunt_reputation.py`:

- `reputation_enabled` (bool)
- `reputation_window` (int, <= 0 switches to EMA)
- `reputation_ema_alpha` (float)
- `reputation_opportunity_only` (bool)
- `reputation_min_samples` (int)
- `reputation_noise_std` (float)
- `include_reputation_channel` (bool)
- `include_reputation_summary` (bool)
- `reputation_visibility_range` (int or None)

## Files and structure

Key files:

- `predpreygrass_rllib_env.py`: reputation-enabled environment.
- `config/config_env_stag_hunt_reputation.py`: default env config.
- `tune_ppo.py`: training entry point.
- `evaluate_ppo_from_checkpoint_debug.py`: single-run eval + visuals.
- `evaluate_ppo_from_checkpoint_multi_runs.py`: batch eval + aggregates.
- `utils/reputation_metrics.py`: reputation metric helpers.
- `reputation.md`: design notes and expectations.

## Notes

This module keeps the same action semantics and capture logic as the defection
environment. If you disable reputation features, behavior is identical to the
baseline defection setup (same ecology, same action space, same rewards).
