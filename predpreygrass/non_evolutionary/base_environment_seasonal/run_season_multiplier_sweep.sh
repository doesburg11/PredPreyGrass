#!/bin/bash
# Seasonal grass-regrowth multiplier sweep for base_environment_seasonal.
# 6 regimes, from "no seasonality" (1.0/1.0, equivalent to base_environment --
# see test_season_disabled_reproduces_flat_baseline) up to the module's
# committed default (1.5/0.5), 500 iterations each, sequential (single GPU).
# Each run gets its own experiment name (BASE_ENV_SEASONAL_HIGH<h>_LOW<l>_<ts>)
# via --season-high/--season-low, so results land in distinct ~/ray_results
# directories. Primary metric of interest: predator_births/prey_births
# (logged each episode from the env's own _next_predator_idx/_next_prey_idx
# counters), plus predator_count_end/prey_count_end/grass_count_end/
# grass_energy_mean_end and the built-in episode_len_mean/episode_return_mean.
#
# Usage: bash predpreygrass/non_evolutionary/base_environment_seasonal/run_season_multiplier_sweep.sh

set -e
cd "$(dirname "$0")/../../.."  # repo root

REGIMES=(
  "1.0 1.0"
  "1.1 0.9"
  "1.2 0.8"
  "1.3 0.7"
  "1.4 0.6"
  "1.5 0.5"
)
MAX_ITERS=500

for regime in "${REGIMES[@]}"; do
  read -r HIGH LOW <<< "$regime"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting regime high=$HIGH low=$LOW ==="
  python predpreygrass/non_evolutionary/base_environment_seasonal/tune_ppo_base_environment_seasonal.py \
    --season-high "$HIGH" --season-low "$LOW" --max-iters "$MAX_ITERS" \
    2>&1 | tee "/tmp/season_sweep_high${HIGH}_low${LOW}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished regime high=$HIGH low=$LOW ==="
done

echo "=== All 6 sweep regimes complete ==="
