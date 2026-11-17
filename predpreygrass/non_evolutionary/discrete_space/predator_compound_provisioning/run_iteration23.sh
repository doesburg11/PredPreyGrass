#!/usr/bin/env bash
# Iteration 23: same-batch replication, GIVE vs INERT, new seeds 45/46/47, 300 iterations. Pre-registered in RESULTS.md.
# Wave 1: GIVE45, INERT45, GIVE46. Lenient halfway kill rule at checkpoint 14. Wave 2: INERT46, GIVE47, INERT47.
# Launch with: systemd-run --user --unit=psr-replicate23 bash <this file>
set -uo pipefail

ROOT=/home/doesburg/Projects/PredPreyGrass
MOD=predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning
RUNS="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/runs"
OUT="$RUNS/analysis23"
PY="$ROOT/.conda/bin/python"
LOG="$OUT/run_all.log"
mkdir -p "$OUT"
cd "$ROOT"
export OMP_NUM_THREADS=1

log() { echo "$(date +%F_%T) $*" >> "$LOG"; }

stop_all() {
  log "stopping all training processes ($1)"
  pkill -TERM -f "$MOD.tune_ppo_compound_provisioning" 2>/dev/null || true
  sleep 60
  pkill -KILL -f "$MOD.tune_ppo_compound_provisioning" 2>/dev/null || true
  log "stopped; exit $2"
  exit "$2"
}

run_name() { echo "REP23_${1}_K05_MB1024_SEED${2}"; }

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

ckpt_ready() {
  [ -n "$(find "$RUNS/$(run_name "$1" "$2")" -maxdepth 3 -path '*checkpoint_000014/rllib_checkpoint.json' -print -quit 2>/dev/null)" ]
}

wait_all() {
  local rc=0 p
  for p in "$@"; do wait "$p" || rc=1; done
  return "$rc"
}

log "queue start"
train GIVE 45 & p1=$!
train INERT 45 & p2=$!
train GIVE 46 & p3=$!

until ckpt_ready GIVE 45 && ckpt_ready INERT 45 && ckpt_ready GIVE 46; do
  for spec in "GIVE 45 $p1" "INERT 45 $p2" "GIVE 46 $p3"; do
    set -- $spec
    if ! ckpt_ready "$1" "$2" && ! kill -0 "$3" 2>/dev/null; then
      log "ABORT: wave-1 run $1 $2 ended before checkpoint 14"
      stop_all "abort" 1
    fi
  done
  sleep 60
done
sleep 120

log "checkpoint 14 exists in wave 1; running halfway kill check"
if ! CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_compound_gift" --checkpoint 14 --episodes 10 --seed 7000 \
    --run GIVE_45="$RUNS/$(run_name GIVE 45)" --run GIVE_46="$RUNS/$(run_name GIVE 46)" \
    --run INERT_45="$RUNS/$(run_name INERT 45)" --out "$OUT/halfway.json" > "$OUT/halfway.log" 2>&1; then
  log "ABORT: halfway analysis failed"
  stop_all "halfway analysis failure" 1
fi
# Lenient kill rule: continue if GIVE 45 or GIVE 46 has a male gap above INERT 45's (difference > 0).
"$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
diffs = [(d[g]["predator_male_gap"][0] - d["INERT_45"]["predator_male_gap"][0], g) for g in ("GIVE_45", "GIVE_46")]
print("halfway GIVE-minus-INERT45 male gaps", [(round(x, 4), g) for x, g in diffs])
sys.exit(0 if any(x > 0 for x, _ in diffs) else 1)
' "$OUT/halfway.json" >> "$LOG" 2>&1
status=$?
if [ "$status" -eq 1 ]; then
  log "KILL: no GIVE male gap above INERT 45; wave 2 not launched"
  stop_all "kill rule" 0
elif [ "$status" -ne 0 ]; then
  log "ABORT: kill-rule check failed (exit $status)"
  stop_all "kill-rule failure" 1
fi
log "PASS halfway kill check"
if ! wait_all "$p1" "$p2" "$p3"; then
  log "ABORT: wave-1 training failure"
  stop_all "wave-1 failure" 1
fi

train INERT 46 & p4=$!
train GIVE 47 & p5=$!
train INERT 47 & p6=$!
if ! wait_all "$p4" "$p5" "$p6"; then
  log "ABORT: wave-2 training failure"
  stop_all "wave-2 failure" 1
fi

log "training done; starting final analysis"
specs=()
for arm in GIVE INERT; do
  for seed in 45 46 47; do
    specs+=(--run "${arm}_${seed}=$RUNS/$(run_name "$arm" "$seed")")
  done
done
if CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_compound_gift" --checkpoint 29 --episodes 20 --seed 7000 \
    "${specs[@]}" --out "$OUT/final.json" > "$OUT/final.log" 2>&1; then
  log "ALL DONE"
else
  log "FINAL ANALYSIS FAILED"
  exit 1
fi
