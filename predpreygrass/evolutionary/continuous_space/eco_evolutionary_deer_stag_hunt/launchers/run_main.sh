#!/usr/bin/env bash
# The main runs (RESULTS.md section 1.3): runs k = 1..5 of arms L and B
# (1000 iterations; L_k and B_k side by side), then E_1..E_5 matched to all
# ten (two at a time), then every run's evaluations at 250, 500, 750 and
# 1000. Run k uses seed k and deer_stag_hunt run ((k-1) mod 3) + 1's deer.
# Status in runs/main.status. Run from the repository root under
# systemd-run --user.
set -euo pipefail
PY=.conda/bin/python
M=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt
RUNS=$M/runs
status() { echo "$1 $(date)" > "$RUNS/main.status"; }
for k in 1 2 3 4 5; do
    status "TRAINING pair $k"
    $PY $M/launchers/replicate.py L_$k L $k > "$RUNS/L_$k.log" 2>&1 &
    L=$!
    sleep 90  # let the first run claim its GPU memory before the second starts
    $PY $M/launchers/replicate.py B_$k B $k > "$RUNS/B_$k.log" 2>&1 &
    B=$!
    wait $L || { status "FAILED L_$k"; exit 1; }
    wait $B || { status "FAILED B_$k"; exit 1; }
done
MATCH=$(for k in 1 2 3 4 5; do echo -n "$RUNS/L_$k $RUNS/B_$k "; done)
for pair in "1 2" "3 4" "5"; do
    status "EVOLVING $pair"
    pids=()
    for k in $pair; do
        $PY -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve E_$k $k \
            --match $MATCH > "$RUNS/E_$k.log" 2>&1 & pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || { status "FAILED E $pair"; exit 1; }; done
done
for k in 1 2 3 4 5; do
    for arm in L B E; do
        status "EVALUATING ${arm}_$k"
        $PY $M/launchers/evaluate_run.py ${arm}_$k > "$RUNS/${arm}_${k}_eval.log" 2>&1 || { status "FAILED eval ${arm}_$k"; exit 1; }
    done
done
status DONE
