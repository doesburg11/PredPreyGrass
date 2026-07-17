#!/bin/bash
# R4 step 2 -- reverse-leg fitness sweep: freeze offspring_investment_fraction at
# several fixed values (genome_enabled=False, no inheritance/mutation/variation)
# and compare fitness outcomes across them. Tests whether the trait affects
# fitness at all, independent of any selection/drift question -- the check
# eco_evolutionary_metabolic_rate skipped from Iteration 0 onward. See
# eco_evolutionary_investment/RESULTS.md and predpreygrass/evolutionary/RESULTS.md.
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_investment/run_fixed_genome_sweep.sh
# Logs go to /tmp/investment_sweep_<value>.log so you can tail them.

set -e
cd "$(dirname "$0")/../../.."  # repo root

VALUES=(0.15 0.25 0.35 0.55 0.70)
MAX_ITERS=100
RAY_RESULTS_DIR="$HOME/ray_results"

# Copies a finished run's /tmp console log into its own ray_results experiment
# directory, same pattern as run_replication_seeds.sh -- /tmp is not permanent.
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

for value in "${VALUES[@]}"; do
  tag=$(echo "$value" | tr -d '.')
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting fixed-genome run, investment_fraction=$value ==="
  LOG="/tmp/investment_sweep_${tag}.log"
  python predpreygrass/evolutionary/eco_evolutionary_investment/tune_ppo_investment.py \
    --fixed-investment-fraction "$value" --max-iters "$MAX_ITERS" \
    2>&1 | tee "$LOG"
  archive_console_log "PPO_ECO_EVOLUTION_INVESTMENT_FIXED${tag}_" "$LOG"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished fixed-genome run, investment_fraction=$value ==="
done

echo "=== All ${#VALUES[@]} fixed-genome sweep runs complete ==="
