#!/bin/bash
# Iteration 5 multi-seed replication: runs all 6 seeded runs (3 real + 3 control)
# sequentially, one at a time, since they share one GPU. Safe to Ctrl-C between
# runs -- each is a separate process and already-finished runs are unaffected.
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_metabolic_rate/run_replication_seeds.sh
# Logs go to /tmp/replication_seed_<condition>_<seed>.log so you can tail them.

set -e
cd "$(dirname "$0")/../../.."  # repo root

SEEDS=(42 43 44)
MAX_ITERS=1000

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting REAL run, seed $seed ==="
  python predpreygrass/evolutionary/eco_evolutionary_metabolic_rate/tune_ppo_metabolic_rate.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "/tmp/replication_real_seed${seed}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished REAL run, seed $seed ==="
done

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting CONTROL run, seed $seed ==="
  python predpreygrass/evolutionary/eco_evolutionary_metabolic_rate/tune_ppo_metabolic_rate_neutral_control.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "/tmp/replication_control_seed${seed}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished CONTROL run, seed $seed ==="
done

echo "=== All 6 replication runs complete ==="
echo "Run: python predpreygrass/evolutionary/eco_evolutionary_metabolic_rate/analyze_replication_seeds.py"
