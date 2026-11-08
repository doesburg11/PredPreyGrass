#!/usr/bin/env bash
# The pilot (RESULTS.md section 1.2): arms L and B (run k = 1, 500
# iterations) side by side, then arm E matched to them, then the
# evaluations at iterations 250 and 500. Status in runs/pilot.status.
# Run from the repository root under systemd-run --user.
set -euo pipefail
PY=.conda/bin/python
M=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt
RUNS=$M/runs
mkdir -p "$RUNS"
status() { echo "$1 $(date)" > "$RUNS/pilot.status"; }
status TRAINING
$PY $M/launchers/replicate.py pilot_L L 1 500 > "$RUNS/pilot_L.log" 2>&1 &
L=$!
sleep 90  # let the first run claim its GPU memory before the second starts
$PY $M/launchers/replicate.py pilot_B B 1 500 > "$RUNS/pilot_B.log" 2>&1 &
B=$!
wait $L || { status "FAILED pilot_L"; exit 1; }
wait $B || { status "FAILED pilot_B"; exit 1; }
status EVOLVING
$PY -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve pilot_E 1 \
    --match "$RUNS/pilot_L" "$RUNS/pilot_B" --marks 250 500 > "$RUNS/pilot_E.log" 2>&1 || { status "FAILED pilot_E"; exit 1; }
status EVALUATING
for run in pilot_L pilot_B pilot_E; do
    $PY $M/launchers/evaluate_run.py $run 250 500 > "$RUNS/${run}_eval.log" 2>&1 || { status "FAILED eval $run"; exit 1; }
done
status DONE
