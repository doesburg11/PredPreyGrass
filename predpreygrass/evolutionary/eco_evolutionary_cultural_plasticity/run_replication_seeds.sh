#!/bin/bash
# Neutral-control replication for eco_evolutionary_cultural_plasticity (Trial 8):
# 3 real + 3 neutral-control seeds, 1000 iterations each, sequential (not
# concurrent -- two GPU-using Ray clusters sharing one physical GPU is a
# documented OOM risk, see predpreygrass/non_evolutionary/project_reward_shaping/
# README.md's "Concurrent vs. sequential training"). Follows the same
# pilot-first-then-replicate discipline as every prior module in this family;
# the 300-iteration single-seed pilot (seed=1) already confirmed sustainability
# and a working cultural-learning mechanism (dialect_match_rate up to 0.96,
# far above chance) before this was launched.
#
# Compare real vs. control male_donation_rate-equivalent (here: plasticity)
# drift via analyze_replication_seeds.py once enough seeds finish.
#
# Usage: bash predpreygrass/evolutionary/eco_evolutionary_cultural_plasticity/run_replication_seeds.sh

set -e
cd "$(dirname "$0")/../../.."  # repo root

SEEDS=(42 43 44)
MAX_ITERS=1000

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting REAL run, seed=$seed ==="
  python predpreygrass/evolutionary/eco_evolutionary_cultural_plasticity/tune_ppo_cultural_plasticity.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "/tmp/cultural_plasticity_replication_real_${seed}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished REAL run, seed=$seed ==="
done

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting CONTROL run, seed=$seed ==="
  python predpreygrass/evolutionary/eco_evolutionary_cultural_plasticity/tune_ppo_cultural_plasticity_neutral_control.py \
    --seed "$seed" --max-iters "$MAX_ITERS" \
    2>&1 | tee "/tmp/cultural_plasticity_replication_control_${seed}.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished CONTROL run, seed=$seed ==="
done

echo "=== All 6 replication runs complete ==="
echo "Run: python predpreygrass/evolutionary/eco_evolutionary_cultural_plasticity/analyze_replication_seeds.py"
