#!/usr/bin/env bash
# RESULTS.md section 3.3, the main runs, only after the pilot passed:
# main3_L<k> and main3_B<k> side by side for k = 1..6 (1000 iterations,
# seed k, deer learning alongside); then every run's own-world evaluations
# at 250, 500, 750 and 1000, the cross-play panel at 500, 750 and 1000, and
# the verdict (results/verdict_3.txt). Status in runs/main3.status. A
# crashed run is rerun once from scratch with the same seed (section 3.3),
# the crash kept as runs/<run>.crash1.log and runs/<run>.crash1/. Run e.g.
#   systemd-run --user --unit eco-deer-main3 bash .../launchers/run_main_3.sh
set -u
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
M=predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt
PY="$REPO/.conda/bin/python"
RUNS="$M/runs"; RES="$M/results"
STATUS="$RUNS/main3.status"
mkdir -p "$RUNS" "$RES" || exit 1
fail() { echo "FAILED $1 $(date)" > "$STATUS"; trap - INT TERM; kill -TERM 0 2>/dev/null; exit 1; }
trap 'fail interrupted' INT TERM
grep -q "^pilot: PASS" "$RES/pilot3_check.txt" 2>/dev/null || fail "the pilot has not passed"

attempt() {  # NAME COMMAND...: one attempt, then one rerun from scratch after a crash
  local name=$1; shift
  "$@" > "$RUNS/$name.log" 2>&1 && return 0
  mv "$RUNS/$name.log" "$RUNS/$name.crash1.log"
  mv "$RUNS/$name" "$RUNS/$name.crash1" 2>/dev/null
  "$@" > "$RUNS/$name.log" 2>&1
}
train() { attempt "$1" "$PY" "$M/launchers/replicate_3.py" "$1" "$2" "$3"; }

ALL=""
for k in 1 2 3 4 5 6; do
  echo "training main3_L$k main3_B$k $(date)" > "$STATUS"
  train "main3_L$k" L "$k" & P1=$!
  sleep 90
  train "main3_B$k" B "$k" & P2=$!
  wait $P1 || fail "main3_L$k"
  wait $P2 || fail "main3_B$k"
  ALL="$ALL main3_L$k main3_B$k"
done
for run in $ALL; do
  echo "evaluating $run $(date)" > "$STATUS"
  CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/evaluate_run.py" "$run" > "$RUNS/${run}_eval.log" 2>&1 || fail "$run eval"
done
for mark in 500 750 1000; do
  echo "cross-play panel at $mark $(date)" > "$STATUS"
  CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/cross_play.py" "$mark" $ALL >> "$RUNS/cross_play.log" 2>&1 || fail "cross-play $mark"
done
"$PY" "$M/launchers/verdict_3.py" > "$RES/verdict_3.txt" 2>&1 || fail verdict
echo "DONE $(grep '^verdict' "$RES/verdict_3.txt") $(date)" > "$STATUS"
