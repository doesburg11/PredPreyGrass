#!/usr/bin/env bash
# RESULTS.md section 1: share_1 and share_2 side by side, then share_3, then
# every evaluation (iterations 250, 500 and final; evaluate.py 50 episodes,
# coop_fair.py 20) and the verdict (results/sharing_verdict.txt). Status in
# runs/sharing.status. Before starting, waits while
# eco_evolutionary_deer_stag_hunt's section 3 runs are training (their
# status file starts with "training"; that launcher never trains again once
# it evaluates), so the GPU holds at most two learners.
#
# Restartable (RESULTS.md section 1.4): a run that finished is marked
# runs/<run>.done and skipped; a run directory without it (a crash) is moved
# to runs/<run>.crash1 and trained again from scratch, once; a second crash
# stops everything. Evaluations and the verdict are always redone (they are
# seeded). Run from anywhere, e.g.
#   systemd-run --user --unit deer-sharing bash .../launchers/run_sharing.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
MOD=predpreygrass/non_evolutionary/continuous_space/deer_food_sharing
PY="$REPO/.conda/bin/python"
RUNS="$MOD/runs"; RES="$MOD/results"
STATUS="$RUNS/sharing.status"
ECO_STATUS=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/runs/main3.status
mkdir -p "$RUNS" "$RES" || exit 1
CHILDREN=""
cleanup() { for pid in $CHILDREN; do kill "$pid" 2>/dev/null; done; wait 2>/dev/null; }
fail() { echo "FAILED $1 $(date)" > "$STATUS"; cleanup; exit 1; }
trap 'fail interrupted' INT TERM
[ -x "$PY" ] || { echo "FAILED no python at $PY $(date)" > "$STATUS"; exit 1; }
train() {
  local run=$1
  [ -f "$RUNS/$run.done" ] && return 0
  if [ -e "$RUNS/$run" ]; then
    [ -e "$RUNS/$run.crash1" ] && return 1  # crashed twice
    mv "$RUNS/$run" "$RUNS/$run.crash1" && mv "$RUNS/$run.log" "$RUNS/$run.crash1.log" 2>/dev/null
    echo "$run: unfinished earlier attempt moved to $run.crash1, rerun from scratch $(date)" >> "$RUNS/sharing.crashes"
  fi
  "$PY" "$MOD/launchers/replicate.py" "$run" > "$RUNS/$run.log" 2>&1 && [ -d "$RUNS/$run/checkpoint/learner_group" ] && touch "$RUNS/$run.done"
}
attempt() { train "$1" || train "$1"; }

while grep -q "^training" "$ECO_STATUS" 2>/dev/null; do
  echo "waiting for the GPU (eco section 3 training) $(date)" > "$STATUS"
  sleep 300
done

echo "training share_1 share_2 $(date)" > "$STATUS"
attempt share_1 & P1=$!
CHILDREN="$P1"
sleep 90
attempt share_2 & P2=$!
CHILDREN="$P1 $P2"
wait $P1 || fail share_1
wait $P2 || fail share_2
CHILDREN=""
echo "training share_3 $(date)" > "$STATUS"
attempt share_3 || fail share_3

for run in share_1 share_2 share_3; do
  for mark in iter250 iter500 final; do
    case $mark in
      iter250) ckpt="$RUNS/$run/checkpoint/iter_000250"; name="${run}_iter250" ;;
      iter500) ckpt="$RUNS/$run/checkpoint/iter_000500"; name="${run}_iter500" ;;
      final)   ckpt="$RUNS/$run/checkpoint"; name="$run" ;;
    esac
    echo "evaluating $name $(date)" > "$STATUS"
    "$PY" -m predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.evaluate "$ckpt" 50 "$RES/${name}_eval.json" --workers 16 > "$RUNS/${name}_eval.log" 2>&1 || fail "$name eval"
    "$PY" "$MOD/launchers/coop_fair.py" "$ckpt" 20 10 "$RES/${name}_coop.json" > "$RUNS/${name}_coop.log" 2>&1 || fail "$name coop"
  done
done
"$PY" "$MOD/launchers/verdict_sharing.py" > "$RES/sharing_verdict.txt" 2>&1 || fail verdict
echo "DONE $(grep '^verdict' "$RES/sharing_verdict.txt") $(date)" > "$STATUS"
