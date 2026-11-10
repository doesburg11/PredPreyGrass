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
# Code frozen at the pilot (RESULTS.md section 3.2): both pilot runs started
# from one commit without uncommitted code changes, and the module's code is
# unchanged since (documents and results aside).
C=""
for run in pilot3_L1 pilot3_B1; do
  [ -e "$RUNS/$run/source_code/uncommitted.patch" ] && fail "$run started with uncommitted changes"
  c=$(sed -n 's/^commit: //p' "$RUNS/$run/source_code/git.txt")
  [ -n "$c" ] || fail "$run has no recorded commit"
  [ -z "$C" ] || [ "$C" = "$c" ] || fail "the pilot runs started from different commits"
  C=$c
done
CODE=("$M/*.py" "$M/config" "$M/launchers" "$M/tests")
git diff --quiet "$C" -- "${CODE[@]}" || fail "the code changed since the pilot ($C)"

# A run finished once (<run>.done) is never redone. A run that did not
# finish, whether it crashed or the launcher was interrupted, gets one rerun
# from scratch with the same seed (section 3.3): the failed attempt is kept
# as <run>.crash1.log and <run>.crash1/. A second failure stops everything.
attempt() {  # NAME COMMAND...
  local name=$1; shift
  [ -f "$RUNS/$name.done" ] && return 0
  if [ -e "$RUNS/$name" ] || [ -e "$RUNS/$name.log" ]; then
    [ -e "$RUNS/$name.crash1.log" ] && return 1
    mv "$RUNS/$name.log" "$RUNS/$name.crash1.log" 2>/dev/null || touch "$RUNS/$name.crash1.log"
    [ -e "$RUNS/$name" ] && { mv "$RUNS/$name" "$RUNS/$name.crash1" || return 1; }
  fi
  if "$@" > "$RUNS/$name.log" 2>&1; then touch "$RUNS/$name.done"; return 0; fi
  [ -e "$RUNS/$name.crash1.log" ] && return 1
  mv "$RUNS/$name.log" "$RUNS/$name.crash1.log"
  [ -e "$RUNS/$name" ] && { mv "$RUNS/$name" "$RUNS/$name.crash1" || return 1; }
  "$@" > "$RUNS/$name.log" 2>&1 && touch "$RUNS/$name.done"
}
train() { attempt "$1" "$PY" "$M/launchers/replicate_3.py" "$1" "$2" "$3"; }
retry() {  # LOG COMMAND...: a technically failed evaluation is rerun once
  local log=$1; shift
  "$@" >> "$log" 2>&1 && return 0
  echo "retrying after a failure $(date)" >> "$log"
  "$@" >> "$log" 2>&1
}

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
  retry "$RUNS/${run}_eval.log" env CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/evaluate_run.py" "$run" || fail "$run eval"
done
for mark in 500 750 1000; do
  echo "cross-play panel at $mark $(date)" > "$STATUS"
  retry "$RUNS/cross_play.log" env CUDA_VISIBLE_DEVICES= "$PY" "$M/launchers/cross_play.py" "$mark" $ALL || fail "cross-play $mark"
done
"$PY" "$M/launchers/verdict_3.py" > "$RES/verdict_3.txt" 2>&1 || fail verdict
echo "DONE $(grep '^verdict' "$RES/verdict_3.txt") $(date)" > "$STATUS"
