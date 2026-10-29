#!/usr/bin/env bash
# Test 2's six branch runs (launchers/coevolution.py), two at a time (the GPU
# fits two learners), in the order that gives the predator contrast first:
# seed 1 both + prey_frozen, seed 2 both + prey_frozen, then both
# predator_frozen. Refuses to start if any run's folder or log exists; if a
# run fails, stops its partner and the queue.
# Run from the repository root, e.g. under systemd-run --user.
set -u
PYTHON=/home/doesburg/Projects/PredPreyGrass/.conda/bin/python
LAUNCHER=predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.launchers.coevolution
RUNS=predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_fov/runs
PAIRS=(
  "tradeoff_m120:both tradeoff_m120:prey_frozen"
  "tradeoff_m120_seed2:both tradeoff_m120_seed2:prey_frozen"
  "tradeoff_m120:predator_frozen tradeoff_m120_seed2:predator_frozen"
)
for pair in "${PAIRS[@]}"; do
  for job in $pair; do
    name="coevolution_${job%%:*}_${job##*:}"
    if [ -e "$RUNS/$name" ] || [ -e "$RUNS/$name.log" ]; then
      echo "refusing: $RUNS/$name or its log exists"
      exit 1
    fi
  done
done
for pair in "${PAIRS[@]}"; do
  pids=()
  for job in $pair; do
    source=${job%%:*} arm=${job##*:}
    echo "$(date '+%F %T') start $source $arm"
    "$PYTHON" -m "$LAUNCHER" "$source" "$arm" > "$RUNS/coevolution_${source}_${arm}.log" 2>&1 &
    pids+=($!)
  done
  running=("${pids[@]}")
  while [ ${#running[@]} -gt 0 ]; do
    if ! wait -n -p done_pid "${running[@]}"; then
      echo "$(date '+%F %T') a run of the pair failed: $pair; stopping its partner and the queue"
      kill "${pids[@]}" 2>/dev/null
      wait
      exit 1
    fi
    left=()
    for pid in "${running[@]}"; do
      [ "$pid" = "$done_pid" ] || left+=("$pid")
    done
    running=("${left[@]}")
  done
  echo "$(date '+%F %T') pair done: $pair"
done
echo "$(date '+%F %T') all pairs done"
