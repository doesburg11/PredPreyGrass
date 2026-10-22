#!/usr/bin/env bash
# Runs the Baldwin-branch replication (launchers/baldwin_replication.py):
# RLlib arms on up to 3 slots while the frozen lane is busy (4 after), the
# evolve_frozen arms one after another in their own lane. Arms are
# interleaved by seed so an early stop still leaves complete replicates.
# Start from the repository root:
#   systemd-run --user --unit=blrep_queue --working-directory=$PWD \
#       bash predpreygrass/evolutionary/eco_evolutionary_continuous_traits/launchers/queue_baldwin_replication.sh
set -u
PY=.conda/bin/python
M=predpreygrass.evolutionary.eco_evolutionary_continuous_traits.launchers.baldwin_replication
LOGS=predpreygrass/evolutionary/eco_evolutionary_continuous_traits/runs/logs
mkdir -p "$LOGS"
QLOG=$LOGS/blrep_queue.log
RL_JOBS=()
FROZEN_JOBS=()
for seed in 1 2 3; do
  for arm in both_learn pred_frozen prey_frozen; do RL_JOBS+=("$arm $seed"); done
  for arm in both_frozen neutral_frozen; do FROZEN_JOBS+=("$arm $seed"); done
done

declare -A RL_PIDS=()
FROZEN_PID=""
FAILED=0
note() { echo "$(date '+%F %T') $*" >> "$QLOG"; }
cleanup() {
  note "queue stopped: terminating running jobs"
  for pid in "${!RL_PIDS[@]}" $FROZEN_PID; do kill -TERM "$pid" 2>/dev/null; done
  wait
  exit 143
}
trap cleanup INT TERM

start() {  # "arm seed" -> sets STARTED to the pid (no subshell, so wait can reap it)
  set -- $1
  local log="$LOGS/blrep_${1}_s${2}.log"
  OMP_NUM_THREADS=1 $PY -m $M "$1" "$2" > "$log" 2>&1 &
  STARTED=$!
}

reap() {
  for pid in "${!RL_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; status=$?; note "done (exit $status): ${RL_PIDS[$pid]}"; unset "RL_PIDS[$pid]"
      [ $status -ne 0 ] && FAILED=$((FAILED + 1))
    fi
  done
  if [ -n "$FROZEN_PID" ] && ! kill -0 "$FROZEN_PID" 2>/dev/null; then
    wait "$FROZEN_PID"; status=$?; note "done (exit $status): $FROZEN_JOB"; FROZEN_PID=""
    [ $status -ne 0 ] && FAILED=$((FAILED + 1))
  fi
}

note "queue start: ${#RL_JOBS[@]} RLlib jobs, ${#FROZEN_JOBS[@]} frozen jobs"
r=0; f=0
while [ $r -lt ${#RL_JOBS[@]} ] || [ $f -lt ${#FROZEN_JOBS[@]} ] || [ ${#RL_PIDS[@]} -gt 0 ] || [ -n "$FROZEN_PID" ]; do
  reap
  if [ -z "$FROZEN_PID" ] && [ $f -lt ${#FROZEN_JOBS[@]} ]; then
    FROZEN_JOB="${FROZEN_JOBS[$f]}"; start "$FROZEN_JOB"; FROZEN_PID=$STARTED; f=$((f + 1)); note "start: $FROZEN_JOB"
  fi
  slots=3; [ -z "$FROZEN_PID" ] && slots=4
  if [ ${#RL_PIDS[@]} -lt $slots ] && [ $r -lt ${#RL_JOBS[@]} ]; then
    start "${RL_JOBS[$r]}"; pid=$STARTED; RL_PIDS[$pid]="${RL_JOBS[$r]}"; note "start: ${RL_JOBS[$r]}"; r=$((r + 1))
    sleep 30  # stagger Ray cluster start-ups
    continue
  fi
  sleep 60
done
note "queue finished, $FAILED failed job(s)"
[ $FAILED -eq 0 ]
