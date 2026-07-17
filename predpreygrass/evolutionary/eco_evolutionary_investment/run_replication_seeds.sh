#!/bin/bash
# R7 multi-seed replication: runs 6 seeded runs (3 real + 3 control) sequentially,
# one at a time, since they share one GPU. Safe to Ctrl-C between runs -- each is
# a separate process and already-finished runs are unaffected.
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_investment/run_replication_seeds.sh
# Logs go to /tmp/investment_replication_<condition>_seed<seed>.log so you can tail them.

set -e
cd "$(dirname "$0")/../../.."  # repo root

SEEDS=(42 43 44)
MAX_ITERS=1000
RAY_RESULTS_DIR="$HOME/ray_results"

# Copies a run's console log into its own ray_results experiment directory --
# /tmp is not permanent (cleared on reboot); ray_results is.
archive_console_log() {
  local glob_prefix="$1" log_file="$2"
  local latest_dir
  latest_dir=$(ls -td "${RAY_RESULTS_DIR}/${glob_prefix}"* 2>/dev/null | head -1)
  if [ -n "$latest_dir" ]; then
    cp "$log_file" "$latest_dir/console_output.log"
  else
    echo "WARNING: no ray_results dir matching ${glob_prefix}* -- console log only in $log_file"
  fi
}

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting REAL run, seed $seed ==="
  LOG="/tmp/investment_replication_real_seed${seed}.log"
  python predpreygrass/evolutionary/eco_evolutionary_investment/tune_ppo_investment.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "$LOG"
  archive_console_log "PPO_ECO_EVOLUTION_INVESTMENT_SEED${seed}_" "$LOG"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished REAL run, seed $seed ==="
done

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting CONTROL run, seed $seed ==="
  LOG="/tmp/investment_replication_control_seed${seed}.log"
  python predpreygrass/evolutionary/eco_evolutionary_investment/tune_ppo_investment_neutral_control.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "$LOG"
  archive_console_log "PPO_ECO_EVOLUTION_INVESTMENT_NEUTRAL_CONTROL_SEED${seed}_" "$LOG"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished CONTROL run, seed $seed ==="
done

echo "=== All 6 replication runs complete ==="
echo "Run: python predpreygrass/evolutionary/eco_evolutionary_investment/analyze_replication_seeds.py"
