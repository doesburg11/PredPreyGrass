#!/usr/bin/env bash
# RESULTS.md section 2.3, the main runs, only after the pilot passed: L_k
# and B_k side by side for k = 1..6 (1000 iterations, deer of
# touch_((k - 1) mod 3) + 1); then E_1..E_6 two at a time, E_k matched to
# the four L and B runs replaying the same deer; then every run's
# evaluations at 250, 500, 750 and 1000 (L and B also the fixed-deer panel)
# and the verdict (results/verdict_2.txt). Status in runs/main2.status. A
# crashed run is rerun once from scratch with the same seed (RESULTS.md
# section 2.3), the crash kept as runs/<run>.crash<n>.log. Run e.g.
#   systemd-run --user --unit eco-deer-main2 bash .../launchers/run_main_2.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
M=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$M/runs"; RES="$M/results"
STATUS="$RUNS/main2.status"
mkdir -p "$RUNS" "$RES" || exit 1
fail() { echo "FAILED $1 $(date)" > "$STATUS"; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 1; }
trap 'fail interrupted' INT TERM
grep -q "^pilot: PASS" "$RES/pilot2_check.txt" 2>/dev/null || fail "the pilot has not passed"

# One attempt, then (after a crash) one rerun from scratch with the same seed.
attempt() {  # NAME COMMAND...
  local name=$1; shift
  "$@" > "$RUNS/$name.log" 2>&1 && return 0
  mv "$RUNS/$name.log" "$RUNS/$name.crash1.log"
  mv "$RUNS/$name" "$RUNS/$name.crash1" 2>/dev/null
  "$@" > "$RUNS/$name.log" 2>&1
}
train() { attempt "$1" "$PY" "$M/launchers/replicate.py" "$1" "$2" "$3"; }
evolve() {  # E_k matched to L and B runs k and k +- 3 (same deer)
  local k=$1 j=$(( ($1 - 1) % 3 + 1 ))
  attempt "E_$k" "$PY" -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve "E_$k" "$k" \
      --match "$RUNS/L_$j" "$RUNS/B_$j" "$RUNS/L_$((j + 3))" "$RUNS/B_$((j + 3))"
}

for k in 1 2 3 4 5 6; do
  echo "training L_$k B_$k $(date)" > "$STATUS"
  train "L_$k" L "$k" & P1=$!
  sleep 90
  train "B_$k" B "$k" & P2=$!
  wait $P1 || fail "L_$k"
  wait $P2 || fail "B_$k"
done
for pair in "1 4" "2 5" "3 6"; do
  set -- $pair
  echo "evolving E_$1 E_$2 $(date)" > "$STATUS"
  evolve "$1" & P1=$!
  evolve "$2" & P2=$!
  wait $P1 || fail "E_$1"
  wait $P2 || fail "E_$2"
done
for arm in L B E; do
  for k in 1 2 3 4 5 6; do
    echo "evaluating ${arm}_$k $(date)" > "$STATUS"
    panel=""; [ "$arm" != E ] && panel="--panel"
    CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/evaluate_run.py" "${arm}_$k" $panel > "$RUNS/${arm}_${k}_eval.log" 2>&1 || fail "${arm}_$k eval"
  done
done
"$PY" "$M/launchers/verdict_2.py" > "$RES/verdict_2.txt" 2>&1 || fail verdict
echo "DONE $(tail -1 "$RES/verdict_2.txt") $(date)" > "$STATUS"
