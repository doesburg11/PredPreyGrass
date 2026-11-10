#!/usr/bin/env bash
# RESULTS.md section 2.2, the pilot: pilot_L1 and pilot_B1 side by side
# (run k = 1, touch_1's deer), then pilot_L2 and pilot_L3 side by side with
# pilot_E1 (matched to pilot_L1 and pilot_B1) on the CPU alongside, all 500
# iterations; then the evaluations at iterations 250 and 500 and the pilot
# check (results/pilot2_check.txt). Status in runs/pilot2.status. Any
# failure stops everything this script started. Run from anywhere, e.g.
#   systemd-run --user --unit eco-deer-pilot2 bash .../launchers/run_pilot_2.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
M=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$M/runs"; RES="$M/results"
STATUS="$RUNS/pilot2.status"
mkdir -p "$RUNS" "$RES" || exit 1
fail() { echo "FAILED $1 $(date)" > "$STATUS"; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 1; }
trap 'fail interrupted' INT TERM
train() { "$PY" "$M/launchers/replicate.py" "$1" "$2" "$3" 500 > "$RUNS/$1.log" 2>&1; }

echo "training pilot_L1 pilot_B1 $(date)" > "$STATUS"
train pilot_L1 L 1 & P1=$!
sleep 90  # let the first run claim its GPU memory before the second starts
train pilot_B1 B 1 & P2=$!
wait $P1 || fail pilot_L1
wait $P2 || fail pilot_B1

echo "training pilot_L2 pilot_L3, evolving pilot_E1 $(date)" > "$STATUS"
"$PY" -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve pilot_E1 1 \
    --match "$RUNS/pilot_L1" "$RUNS/pilot_B1" --marks 250 500 > "$RUNS/pilot_E1.log" 2>&1 & PE=$!
train pilot_L2 L 2 & P3=$!
sleep 90
train pilot_L3 L 3 & P4=$!
wait $P3 || fail pilot_L2
wait $P4 || fail pilot_L3
wait $PE || fail pilot_E1

for run in pilot_L1 pilot_B1 pilot_L2 pilot_L3 pilot_E1; do
  echo "evaluating $run $(date)" > "$STATUS"
  CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/evaluate_run.py" "$run" 250 500 > "$RUNS/${run}_eval.log" 2>&1 || fail "$run eval"
done
"$PY" "$M/launchers/pilot_check_2.py" > "$RES/pilot2_check.txt" 2>&1 || fail "pilot check (error)"
if grep -q "^pilot: PASS" "$RES/pilot2_check.txt"; then
  echo "PILOT PASSED $(date)" > "$STATUS"
else
  echo "PILOT FAILED (the bar, RESULTS.md section 2.2) $(date)" > "$STATUS"
fi
