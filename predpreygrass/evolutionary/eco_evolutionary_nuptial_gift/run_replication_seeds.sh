#!/bin/bash
# Neutral-control replication (staged rollout step 3, see README.md): 3 real +
# 3 neutral-control seeds, 1000 iterations each, sequential (not concurrent --
# two independent GPU-using Ray clusters sharing one physical GPU is a real
# OOM risk, see predpreygrass/non_evolutionary/project_reward_shaping/README.md's
# "Concurrent vs. sequential training" precedent). Uses the retuned config
# (initial_energy_predator_female=8.0, cooperation_range=4) validated by the
# fixed-genome sweep: rate=1.0 produced 34.5 total reproduction events over
# 60 iterations vs. 0 at rate=0.0 -- a real fitness landscape exists to test
# selection against.
#
# Logs go to /tmp/nuptial_gift_replication_<group>_<seed>.log so you can tail
# them. Analyze with analyze_replication_seeds.py once enough seeds finish
# (it reports "not enough runs yet" gracefully for missing seeds).
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/run_replication_seeds.sh

set -e
cd "$(dirname "$0")/../../.."  # repo root

SEEDS=(42 43 44)
MAX_ITERS=1000
NUM_ENV_RUNNERS=28

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting REAL run, seed=$seed ==="
  python predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/tune_ppo_nuptial_gift.py \
    --seed "$seed" --max-iters "$MAX_ITERS" --num-env-runners "$NUM_ENV_RUNNERS" \
    2>&1 | tee "/tmp/nuptial_gift_replication_real_${seed}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished REAL run, seed=$seed ==="
done

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting CONTROL run, seed=$seed ==="
  python predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/tune_ppo_nuptial_gift_neutral_control.py \
    --seed "$seed" --max-iters "$MAX_ITERS" --num-env-runners "$NUM_ENV_RUNNERS" \
    2>&1 | tee "/tmp/nuptial_gift_replication_control_${seed}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished CONTROL run, seed=$seed ==="
done

echo "=== All 6 replication runs complete ==="
echo "Run: python predpreygrass/evolutionary/eco_evolutionary_nuptial_gift/analyze_replication_seeds.py"
