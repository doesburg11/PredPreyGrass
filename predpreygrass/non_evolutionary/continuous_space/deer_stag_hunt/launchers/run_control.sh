#!/usr/bin/env bash
# RESULTS.md section 5: control_1 and control_2 side by side, then
# control_3 (touch rule, young deer caught alone for certain, 500 iterations);
# then evaluations at iterations 250 and 500 (final): evaluate.py 50 episodes,
# coop_fair.py 20, coop_by_energy.py 40; then section 1's tests, section 2.1's
# tests and section 5's verdict. Status in runs/control.status.
# Run from anywhere, e.g.
#   systemd-run --user --unit deer-control bash .../launchers/run_control.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
MOD=predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$MOD/runs"; RES="$MOD/results"
STATUS="$RUNS/control.status"
ITERATIONS=500
mkdir -p "$RUNS" "$RES" || exit 1
# On failure stop everything this script started: its whole process group
# (under systemd-run the unit's own; stopping the unit also kills its cgroup).
fail() { echo "FAILED $1 $(date)" > "$STATUS"; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 1; }
trap 'fail interrupted' INT TERM
train() { "$PY" "$MOD/launchers/replicate.py" "$1" "$ITERATIONS" --catch-rule touch --set deer_young_solo='"certain"' > "$RUNS/$1.log" 2>&1; }

"$PY" "$MOD/launchers/check_control_world.py" > "$RUNS/control_world.log" 2>&1 || fail "world check before training"
echo "training control_1 control_2 $(date)" > "$STATUS"
train control_1 & P1=$!
sleep 90
train control_2 & P2=$!
wait $P1 || fail control_1
wait $P2 || fail control_2
echo "training control_3 $(date)" > "$STATUS"
train control_3 || fail control_3

for run in control_1 control_2 control_3; do
  # the saved settings must be section 5's world
  "$PY" "$MOD/launchers/check_control_world.py" "$RUNS/$run/checkpoint" >> "$RUNS/control_world.log" 2>&1 || fail "$run world"
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
# Section 1's tests and section 2.1's tests on all three runs: recorded, not
# section 5's verdict (which counts valid runs only: control_verdict.py).
NOTE="Recorded only (RESULTS.md section 5): the verdict lines below count all three runs, valid or not; section 5's verdict is control_verdict.txt."
{ echo "$NOTE"; "$PY" "$MOD/launchers/verdict.py" control_1 control_2 control_3; } > "$RES/control_section1_tests.txt" 2>&1 || fail "section 1 tests"
{ echo "$NOTE"; "$PY" "$MOD/launchers/energy_verdict.py" control_1 control_2 control_3 --compare touch_1_iter500 touch_2_iter500 touch_3_iter500; } > "$RES/control_energy_tests.txt" 2>&1 || fail "section 2.1 tests"
"$PY" "$MOD/launchers/control_verdict.py" control_1 control_2 control_3 > "$RES/control_verdict.txt" 2>&1 || fail control_verdict
echo "DONE $(date)" > "$STATUS"
