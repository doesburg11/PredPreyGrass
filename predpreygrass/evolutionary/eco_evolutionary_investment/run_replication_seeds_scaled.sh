#!/bin/bash
# Trial 6 (population scaling) multi-seed replication, remaining runs.
#
# Seed 42 real finished cleanly (2026-07-18/19, 1000/1000 iterations).
# Seed 42 neutral-control was interrupted by an accidental shutdown on
# 2026-07-20 at iteration 964/1000 -- rather than resuming from its
# checkpoint, it is relaunched here from scratch, followed by seeds 43/44
# real, then seeds 43/44 control -- mirroring the ordering convention used
# by run_replication_seeds.sh (the original, unscaled R7 replication).
#
# Safe to Ctrl-C between runs -- each is a separate process and already-finished
# runs are unaffected.
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_investment/run_replication_seeds_scaled.sh
# Logs go to /tmp/investment_scaled_replication_<condition>_seed<seed>.log so you can tail them.

set -e
cd "$(dirname "$0")/../../.."  # repo root

REAL_SEEDS=(43 44)
CONTROL_SEEDS=(42 43 44)
MAX_ITERS=1000
RAY_RESULTS_DIR="$HOME/ray_results"
PYTHON_BIN="$(pwd)/.conda/bin/python"

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

for seed in "${REAL_SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting SCALED REAL run, seed $seed ==="
  LOG="/tmp/investment_scaled_replication_real_seed${seed}.log"
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" predpreygrass/evolutionary/eco_evolutionary_investment/tune_ppo_investment_scaled.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "$LOG"
  archive_console_log "PPO_ECO_EVOLUTION_INVESTMENT_SCALED_SEED${seed}_" "$LOG"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished SCALED REAL run, seed $seed ==="
done

for seed in "${CONTROL_SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting SCALED CONTROL run, seed $seed ==="
  LOG="/tmp/investment_scaled_replication_control_seed${seed}.log"
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" predpreygrass/evolutionary/eco_evolutionary_investment/tune_ppo_investment_neutral_control_scaled.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "$LOG"
  archive_console_log "PPO_ECO_EVOLUTION_INVESTMENT_SCALED_NEUTRAL_CONTROL_SEED${seed}_" "$LOG"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished SCALED CONTROL run, seed $seed ==="
done

echo "=== All remaining Trial 6 (scaled) replication runs complete ==="
echo "Combined with the already-finished seed 42 real run, all 6 seeded runs are done."
