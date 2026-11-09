#!/usr/bin/env bash
# RESULTS.md section 3: logistic_1 and logistic_2 side by side, then
# logistic_3 (touch rule, logistic grass, 500 iterations); then evaluations
# at iterations 250 and 500 (final): evaluate.py 50 episodes, coop_fair.py 20,
# coop_by_energy.py 40; then both verdicts. Status in runs/logistic.status.
# Run from anywhere, e.g.
#   systemd-run --user --unit deer-logistic bash .../launchers/run_logistic.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
MOD=predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$MOD/runs"; RES="$MOD/results"
STATUS="$RUNS/logistic.status"
ITERATIONS=500
mkdir -p "$RUNS" "$RES" || exit 1
# On failure stop everything this script started: its whole process group
# (under systemd-run the unit's own; stopping the unit also kills its cgroup).
fail() { echo "FAILED $1 $(date)" > "$STATUS"; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 1; }
trap 'fail interrupted' INT TERM
train() { "$PY" "$MOD/launchers/replicate.py" "$1" "$ITERATIONS" --catch-rule touch > "$RUNS/$1.log" 2>&1; }

"$PY" "$MOD/launchers/check_logistic_world.py" > "$RUNS/logistic_world.log" 2>&1 || fail "world check before training"
echo "training logistic_1 logistic_2 $(date)" > "$STATUS"
train logistic_1 & P1=$!
sleep 90
train logistic_2 & P2=$!
wait $P1 || fail logistic_1
wait $P2 || fail logistic_2
echo "training logistic_3 $(date)" > "$STATUS"
train logistic_3 || fail logistic_3

for run in logistic_1 logistic_2 logistic_3; do
  # the saved settings must be section 3's world
  "$PY" "$MOD/launchers/check_logistic_world.py" "$RUNS/$run/checkpoint" >> "$RUNS/logistic_world.log" 2>&1 || fail "$run world"
  for mark in iter250 final; do
    case $mark in
      iter250) ckpt="$RUNS/$run/checkpoint/iter_000250"; name="${run}_iter250" ;;
      final)   ckpt="$RUNS/$run/checkpoint"; name="$run" ;;
    esac
    echo "evaluating $name $(date)" > "$STATUS"
    CUDA_VISIBLE_DEVICES= "$PY" -m predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.evaluate "$ckpt" 50 "$RES/${name}_eval.json" --workers 16 > "$RUNS/${name}_eval.log" 2>&1 || fail "$name eval"
    CUDA_VISIBLE_DEVICES= "$PY" "$MOD/launchers/coop_fair.py" "$ckpt" 20 10 "$RES/${name}_coop.json" > "$RUNS/${name}_coop.log" 2>&1 || fail "$name coop"
    CUDA_VISIBLE_DEVICES= "$PY" "$MOD/launchers/coop_by_energy.py" "$ckpt" 40 16 "$RES/${name}_energy.json" > "$RUNS/${name}_energy.log" 2>&1 || fail "$name energy"
  done
done
"$PY" "$MOD/launchers/verdict.py" logistic_1 logistic_2 logistic_3 > "$RES/logistic_verdict.txt" 2>&1 || fail verdict
"$PY" "$MOD/launchers/energy_verdict.py" logistic_1 logistic_2 logistic_3 --compare touch_1_iter500 touch_2_iter500 touch_3_iter500 > "$RES/logistic_energy_verdict.txt" 2>&1 || fail energy_verdict
echo "DONE $(date)" > "$STATUS"
