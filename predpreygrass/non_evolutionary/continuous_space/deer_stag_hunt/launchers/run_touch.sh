#!/usr/bin/env bash
# RESULTS.md section 2: touch_1 and touch_2 side by side, then touch_3, then
# every evaluation (iterations 250, 500 and final; evaluate.py 50 episodes,
# coop_fair.py 20). Status in runs/touch.status. Run from anywhere, e.g.
#   systemd-run --user --unit deer-touch bash .../launchers/run_touch.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
MOD=predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$MOD/runs"; RES="$MOD/results"
STATUS="$RUNS/touch.status"
mkdir -p "$RUNS" "$RES" || exit 1
CHILDREN=""
cleanup() { for pid in $CHILDREN; do kill "$pid" 2>/dev/null; done; wait 2>/dev/null; }
fail() { echo "FAILED $1 $(date)" > "$STATUS"; cleanup; exit 1; }
trap 'fail interrupted' INT TERM
train() { "$PY" "$MOD/launchers/replicate.py" "$1" --catch-rule touch > "$RUNS/$1.log" 2>&1; }

echo "training touch_1 touch_2 $(date)" > "$STATUS"
train touch_1 & P1=$!
CHILDREN="$P1"
sleep 90
train touch_2 & P2=$!
CHILDREN="$P1 $P2"
wait $P1 || fail touch_1
wait $P2 || fail touch_2
CHILDREN=""
echo "training touch_3 $(date)" > "$STATUS"
train touch_3 || fail touch_3

for run in touch_1 touch_2 touch_3; do
  for mark in iter250 iter500 final; do
    case $mark in
      iter250) ckpt="$RUNS/$run/checkpoint/iter_000250"; name="${run}_iter250" ;;
      iter500) ckpt="$RUNS/$run/checkpoint/iter_000500"; name="${run}_iter500" ;;
      final)   ckpt="$RUNS/$run/checkpoint"; name="$run" ;;
    esac
    echo "evaluating $name $(date)" > "$STATUS"
    "$PY" -m predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.evaluate "$ckpt" 50 "$RES/${name}_eval.json" --workers 16 > "$RUNS/${name}_eval.log" 2>&1 || fail "$name eval"
    "$PY" "$MOD/launchers/coop_fair.py" "$ckpt" 20 10 "$RES/${name}_coop.json" > "$RUNS/${name}_coop.log" 2>&1 || fail "$name coop"
  done
done
"$PY" "$MOD/launchers/verdict.py" touch_1 touch_2 touch_3 > "$RES/touch_verdict.txt" 2>&1 || fail verdict
echo "DONE $(date)" > "$STATUS"
