#!/usr/bin/env bash
# RESULTS.md section 3.2, the pilot: pilot3_L1 and pilot3_B1 side by side
# (seed 1, deer learning alongside, 500 iterations); then their own-world
# evaluations at iterations 250 and 500 and the pilot check
# (results/pilot3_check.txt). Status in runs/pilot3.status. Any failure
# stops everything this script started. Run from anywhere, e.g.
#   systemd-run --user --unit eco-deer-pilot3 bash .../launchers/run_pilot_3.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
M=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$M/runs"; RES="$M/results"
STATUS="$RUNS/pilot3.status"
mkdir -p "$RUNS" "$RES" || exit 1
fail() { echo "FAILED $1 $(date)" > "$STATUS"; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 1; }
trap 'fail interrupted' INT TERM
train() { "$PY" "$M/launchers/replicate_3.py" "$1" "$2" "$3" 500 > "$RUNS/$1.log" 2>&1; }

echo "training pilot3_L1 pilot3_B1 $(date)" > "$STATUS"
train pilot3_L1 L 1 & P1=$!
sleep 90  # let the first run claim its GPU memory before the second starts
train pilot3_B1 B 1 & P2=$!
wait $P1 || fail pilot3_L1
wait $P2 || fail pilot3_B1

for run in pilot3_L1 pilot3_B1; do
  echo "evaluating $run $(date)" > "$STATUS"
  CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/evaluate_run.py" "$run" 250 500 > "$RUNS/${run}_eval.log" 2>&1 || fail "$run eval"
done
"$PY" "$M/launchers/pilot_check_3.py" > "$RES/pilot3_check.txt" 2>&1 || fail "pilot check (error)"
if grep -q "^pilot: PASS" "$RES/pilot3_check.txt"; then
  echo "PILOT PASSED $(date)" > "$STATUS"
else
  echo "PILOT FAILED (the bar, RESULTS.md section 3.2) $(date)" > "$STATUS"
fi
