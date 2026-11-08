#!/bin/bash
# RESULTS.md section 2.1: waits for run_touch.sh to finish (runs/touch.status
# DONE; stops on FAILED), then runs launchers/coop_by_energy.py (40 episodes,
# seeds 0-39) on touch_1-3's final checkpoints (the tests), section 1's
# run_1-3 final checkpoints and touch_1-3 at iteration 500 (comparison, no
# test), and launchers/energy_verdict.py. Status in runs/energy.status.
# Run from anywhere, e.g. systemd-run --user --unit=deer-energy bash .../run_energy.sh
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
MOD=predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$MOD/runs"; RES="$MOD/results"; STATUS="$RUNS/energy.status"
export CUDA_VISIBLE_DEVICES=
fail() { echo "FAILED $1 $(date)" > "$STATUS"; exit 1; }
trap 'fail "interrupted"' INT TERM

echo "waiting for touch runs $(date)" > "$STATUS"
until grep -q '^DONE' "$RUNS/touch.status" 2>/dev/null; do
  grep -q '^FAILED' "$RUNS/touch.status" 2>/dev/null && fail "touch runs failed"
  sleep 120
done

energy() {  # checkpoint, name
  echo "analysing $2 $(date)" > "$STATUS"
  "$PY" "$MOD/launchers/coop_by_energy.py" "$1" 40 16 "$RES/$2_energy.json" > "$RUNS/$2_energy.log" 2>&1 || fail "$2"
}
for run in touch_1 touch_2 touch_3; do energy "$RUNS/$run/checkpoint" "$run"; done
for run in run_1 run_2 run_3; do energy "$RUNS/$run/checkpoint" "$run"; done
for run in touch_1 touch_2 touch_3; do energy "$RUNS/$run/checkpoint/iter_000500" "${run}_iter500"; done
"$PY" "$MOD/launchers/energy_verdict.py" touch_1 touch_2 touch_3 --compare run_1 run_2 run_3 touch_1_iter500 touch_2_iter500 touch_3_iter500 > "$RES/energy_verdict.txt" 2>&1 || fail verdict
echo "DONE $(date)" > "$STATUS"
