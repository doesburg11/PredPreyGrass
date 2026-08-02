#!/bin/bash
# Fixed-genome fitness sweep (staged rollout step 2, see README.md): freeze
# male_donation_rate at several fixed values (genome_enabled=False, no
# inheritance/mutation/per-agent variation) and compare fitness/sustainability
# outcomes across them. Tests whether the obligate-gate mechanism actually
# behaves as designed -- rate=0.0 should be close to lethal for predator_female
# (and therefore predator_male, and eventually prey), not merely suboptimal --
# before spending compute on a full neutral-control selection replication.
#
# CPU-only and a shorter (40-iteration) pilot scale by design: this sweep was
# launched while eco_evolutionary_metabolic_code's kickback module was still
# running on the machine's only GPU (see predpreygrass/evolutionary/RESULTS.md
# for cross-module trial context) -- --cpu forces the CPU PPO config so this
# doesn't queue behind kickback's GPU reservation. The obligate gate is expected
# to produce a stark, fast signal (near-lethal at rate=0), unlike the smooth
# traits (metabolic_rate, offspring_investment_fraction, cooperation_rate) that
# needed hundreds of iterations to show a subtle drift -- so 40 iterations
# should be enough for a first look. Escalate iteration count / switch to GPU
# once the machine's GPU is free, if the first look is promising but unclear.
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/run_fixed_genome_sweep.sh
# Logs go to /tmp/nuptial_gift_sweep_<value>.log so you can tail them.

set -e
cd "$(dirname "$0")/../../.."  # repo root

VALUES=(0.0 0.25 0.5 0.75 1.0)
MAX_ITERS=40
RAY_RESULTS_DIR="$HOME/ray_results"

# Copies a finished run's /tmp console log into its own ray_results experiment
# directory, same pattern as eco_evolutionary_investment's sweep script -- /tmp
# is not permanent.
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
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting fixed-genome run, male_donation_rate=$value ==="
  LOG="/tmp/nuptial_gift_sweep_${tag}.log"
  python predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/tune_ppo_nuptial_gift.py \
    --fixed-donation-rate "$value" --max-iters "$MAX_ITERS" --cpu \
    2>&1 | tee "$LOG"
  archive_console_log "PPO_ECO_EVOLUTION_NUPTIAL_GIFT_FIXED${value}_" "$LOG"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished fixed-genome run, male_donation_rate=$value ==="
done

echo "=== All ${#VALUES[@]} fixed-genome sweep runs complete ==="
