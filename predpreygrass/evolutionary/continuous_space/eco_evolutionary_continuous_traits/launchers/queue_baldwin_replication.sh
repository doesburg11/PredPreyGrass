#!/usr/bin/env bash
# Runs the Baldwin-branch replication (launchers/baldwin_replication.py):
# RLlib arms on 3 slots (each learner sits on the one 16 GB GPU; three use
# up to about 15 GB, a fourth ran out of memory on 2026-10-22), the
# evolve_frozen arms one after another in their own lane. Arms are
# interleaved by seed so an early stop still leaves complete replicates.
# A job that fails within 5 minutes stops further launches (no cascade).
# Optional arguments replace the job list, e.g. "both_learn 3" "pred_frozen 2".
# Start from the repository root:
#   systemd-run --user --unit=blrep_queue --working-directory=$PWD \
#       bash predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/launchers/queue_baldwin_replication.sh
set -u
PY=.conda/bin/python
M=predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.launchers.baldwin_replication
LOGS=predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/runs/logs
mkdir -p "$LOGS"
QLOG=$LOGS/blrep_queue.log
RL_JOBS=()
FROZEN_JOBS=()
if [ $# -gt 0 ]; then
  for job in "$@"; do
    case "$job" in
      both_frozen*|neutral_frozen*) FROZEN_JOBS+=("$job") ;;
      *) RL_JOBS+=("$job") ;;
    esac
  done
else
  for seed in 1 2 3; do
    for arm in both_learn pred_frozen prey_frozen; do RL_JOBS+=("$arm $seed"); done
    for arm in both_frozen neutral_frozen; do FROZEN_JOBS+=("$arm $seed"); done
  done
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A RL_PIDS=()
declare -A STARTED_AT=()
HALT=0
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
  STARTED_AT[$STARTED]=$(date +%s)
}

failed() {  # pid status job
  FAILED=$((FAILED + 1))
  if [ $(( $(date +%s) - ${STARTED_AT[$1]} )) -lt 300 ]; then
    HALT=1; note "fast failure of $3: no further launches"
  fi
}

reap() {
  for pid in "${!RL_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; status=$?; job_name="${RL_PIDS[$pid]}"; note "done (exit $status): $job_name"; unset "RL_PIDS[$pid]"
      [ $status -ne 0 ] && failed "$pid" $status "${job_name:-}"
    fi
  done
  if [ -n "$FROZEN_PID" ] && ! kill -0 "$FROZEN_PID" 2>/dev/null; then
    wait "$FROZEN_PID"; status=$?; note "done (exit $status): $FROZEN_JOB"
    [ $status -ne 0 ] && failed "$FROZEN_PID" $status "$FROZEN_JOB"
    FROZEN_PID=""
  fi
}

note "queue start: ${#RL_JOBS[@]} RLlib jobs, ${#FROZEN_JOBS[@]} frozen jobs"
r=0; f=0
while { [ $HALT -eq 0 ] && { [ $r -lt ${#RL_JOBS[@]} ] || [ $f -lt ${#FROZEN_JOBS[@]} ]; }; } || [ ${#RL_PIDS[@]} -gt 0 ] || [ -n "$FROZEN_PID" ]; do
  reap
  if [ $HALT -ne 0 ]; then sleep 60; continue; fi
  if [ -z "$FROZEN_PID" ] && [ $f -lt ${#FROZEN_JOBS[@]} ]; then
    FROZEN_JOB="${FROZEN_JOBS[$f]}"; start "$FROZEN_JOB"; FROZEN_PID=$STARTED; f=$((f + 1)); note "start: $FROZEN_JOB"
  fi
  slots=3
  if [ ${#RL_PIDS[@]} -lt $slots ] && [ $r -lt ${#RL_JOBS[@]} ]; then
    start "${RL_JOBS[$r]}"; pid=$STARTED; RL_PIDS[$pid]="${RL_JOBS[$r]}"; note "start: ${RL_JOBS[$r]}"; r=$((r + 1))
    sleep 30  # stagger Ray cluster start-ups
    continue
  fi
  sleep 60
done
note "queue finished, $FAILED failed job(s), $((${#RL_JOBS[@]} - r + ${#FROZEN_JOBS[@]} - f)) not started"
[ $FAILED -eq 0 ]
