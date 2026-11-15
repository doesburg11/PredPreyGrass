#!/usr/bin/env bash
# Iteration 20: compound movement + gift-bit actions, pre-registered in RESULTS.md.
# Wave 1: GIVE42, INERT42, GIVE43. At checkpoint 14 the halfway kill rule compares the gift-bit
# selectivity of GIVE42/GIVE43 with INERT42 before wave 2 (INERT43, GIVE44, INERT44) starts.
# Launch with: systemd-run --user --unit=psr-compound-gift bash <this file>
set -uo pipefail

ROOT=/home/doesburg/Projects/PredPreyGrass
MOD=predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning
RUNS="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/runs"
OUT="$RUNS/analysis"
PY="$ROOT/.conda/bin/python"
LOG="$OUT/run_all.log"
mkdir -p "$OUT"
cd "$ROOT"
export OMP_NUM_THREADS=1

log() { echo "$(date +%F_%T) $*" >> "$LOG"; }

# Killing a backgrounded `train` subshell would leave its python/Ray children running, so a stop
# terminates the training processes and then the whole systemd unit (cgroup), this script included.
UNIT=${UNIT:-psr-compound-gift}
stop_all() {
  log "stopping all training processes ($1)"
  pkill -TERM -f "$MOD.tune_ppo_compound_provisioning" 2>/dev/null || true
  sleep 60
  pkill -KILL -f "$MOD.tune_ppo_compound_provisioning" 2>/dev/null || true
  log "stopped; exiting via systemctl stop $UNIT"
  systemctl --user stop "$UNIT" 2>/dev/null
  exit "$2"
}

run_name() { echo "COMPOUND_${1}_K05_MB1024_SEED${2}"; }

train() {
  local arm=$1 seed=$2 mode name status
  mode=on
  [ "$arm" = INERT ] && mode=inert
  name=$(run_name "$arm" "$seed")
  log "start $name"
  "$PY" -m "$MOD.tune_ppo_compound_provisioning" --compound-gift "$mode" \
    --seed "$seed" --max-iters 300 \
    --reward-catch-prey 0 --reward-gather-fruit 0 --reward-per-energy 0.5 --penalty-combat-death 0.2 \
    --minibatch-size 1024 --num-epochs 30 --mate-obs-channel mate \
    --learned-gift-amount 2.0 --male-gift-donation-rate 0 \
    --name "$name" --gpu-fraction 0.33 --num-env-runners 4 --num-cpus-per-env-runner 2 --num-cpus-main 2 \
    > "$OUT/train_${arm}_s${seed}.log" 2>&1
  status=$?
  log "end $name exit $status"
  return "$status"
}

ckpt_path() {
  find "$RUNS/$(run_name "$1" "$2")" -maxdepth 2 -type d -name 'checkpoint_000014' -print -quit 2>/dev/null
}

analyse_halfway() {
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_compound_gift" --checkpoint 14 --episodes 10 --seed 7000 \
    --run GIVE_42="$RUNS/$(run_name GIVE 42)" --run GIVE_43="$RUNS/$(run_name GIVE 43)" \
    --run INERT_42="$RUNS/$(run_name INERT 42)" \
    --out "$OUT/halfway.json" > "$OUT/halfway.log" 2>&1 || return 2
  # Kill rule: continue if, in at least one of GIVE 42/43, at least one sex has a gift-bit gap
  # (eligible minus other) at least 0.02 above the INERT 42 gap of that sex.
  "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
best = []
for g in ("GIVE_42", "GIVE_43"):
    for sex in ("predator_male", "predator_female"):
        a, b = d[g][f"{sex}_gap"], d["INERT_42"][f"{sex}_gap"]
        if a is not None and b is not None:
            best.append((a[0] - b[0], g, sex))
print("halfway GIVE-minus-INERT42 gaps", best)
sys.exit(0 if any(x >= 0.02 for x, _, _ in best) else 1)
' "$OUT/halfway.json"
}

analyse_final() {
  local specs=()
  for arm in GIVE INERT; do
    for seed in 42 43 44; do
      specs+=(--run "${arm}_${seed}=$RUNS/$(run_name "$arm" "$seed")")
    done
  done
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_compound_gift" --checkpoint 29 --episodes 20 --seed 7000 \
    "${specs[@]}" --out "$OUT/final.json" > "$OUT/final.log" 2>&1
  local s=$?
  log "final analysis exit $s"
  return "$s"
}

log "queue start"
train GIVE 42 & p1=$!
train INERT 42 & p2=$!
train GIVE 43 & p3=$!

while [ -z "$(ckpt_path GIVE 42)" ] || [ -z "$(ckpt_path GIVE 43)" ] || [ -z "$(ckpt_path INERT 42)" ]; do
  for pair in "GIVE 42 $p1" "INERT 42 $p2" "GIVE 43 $p3"; do
    set -- $pair
    if [ -z "$(ckpt_path "$1" "$2")" ] && ! kill -0 "$3" 2>/dev/null; then
      log "ABORT: wave-1 run $1 $2 ended before checkpoint 14"
      stop_all "abort" 1
    fi
  done
  sleep 60
done

log "checkpoint 14 exists in wave 1; running halfway kill check"
analyse_halfway >> "$LOG" 2>&1
status=$?
if [ "$status" -eq 1 ]; then
  log "KILL: no GIVE seed/sex gap >= 0.02 above INERT 42; stopping wave 1, wave 2 not launched"
  stop_all "kill rule" 0
elif [ "$status" -ne 0 ]; then
  log "ABORT: halfway analysis failed (exit $status); training left running, wave 2 not launched"
  wait "$p1" "$p2" "$p3" 2>/dev/null || true
  exit 1
fi
log "PASS halfway kill check"
if ! wait "$p1" || ! wait "$p2" || ! wait "$p3"; then
  log "ABORT: wave-1 training failure"
  exit 1
fi

train INERT 43 & p4=$!
train GIVE 44 & p5=$!
train INERT 44 & p6=$!
if ! wait "$p4" || ! wait "$p5" || ! wait "$p6"; then
  log "ABORT: wave-2 training failure"
  exit 1
fi
log "training done; starting final analysis"
analyse_final
log "ALL DONE"
