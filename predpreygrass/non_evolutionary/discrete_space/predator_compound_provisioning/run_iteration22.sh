#!/usr/bin/env bash
# Iteration 22: INERT control to 300 iterations, compared with the w = 0 GIVE runs of Iteration 21
# (predator_mate_regard MATEREGARD_W0_*, same code version a1854b09). Pre-registered in RESULTS.md.
# Launch with: systemd-run --user --unit=psr-inert300 bash <this file>
set -uo pipefail

ROOT=/home/doesburg/Projects/PredPreyGrass
DS=predpreygrass.non_evolutionary.discrete_space
MOD=$DS.predator_compound_provisioning
RUNS="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/runs"
W0OUT="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_mate_regard/runs/analysis"
OUT="$RUNS/analysis22"
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

run_name() { echo "COMPOUND_INERT300_K05_MB1024_SEED${1}"; }

train() {
  local seed=$1 name status
  name=$(run_name "$seed")
  log "start $name"
  "$PY" -m "$MOD.tune_ppo_compound_provisioning" --compound-gift inert \
    --seed "$seed" --max-iters 300 \
    --reward-catch-prey 0 --reward-gather-fruit 0 --reward-per-energy 0.5 --penalty-combat-death 0.2 \
    --minibatch-size 1024 --num-epochs 30 --mate-obs-channel mate \
    --learned-gift-amount 2.0 --male-gift-donation-rate 0 \
    --name "$name" --gpu-fraction 0.33 --num-env-runners 4 --num-cpus-per-env-runner 2 --num-cpus-main 2 \
    > "$OUT/train_INERT_s${seed}.log" 2>&1
  status=$?
  log "end $name exit $status"
  return "$status"
}

ckpt_ready() {
  [ -n "$(find "$RUNS/$(run_name "$1")" -maxdepth 3 -path "*checkpoint_0000$2/rllib_checkpoint.json" -print -quit 2>/dev/null)" ]
}

analyse() {  # $1 checkpoint, $2 episodes, $3 out stem
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_compound_gift" --checkpoint "$1" --episodes "$2" --seed 7000 \
    --run INERT_42="$RUNS/$(run_name 42)" --run INERT_43="$RUNS/$(run_name 43)" --run INERT_44="$RUNS/$(run_name 44)" \
    --out "$OUT/$3.json" > "$OUT/$3.log" 2>&1
}

log "queue start"
train 42 & p1=$!
train 43 & p2=$!
train 44 & p3=$!

until ckpt_ready 42 14 && ckpt_ready 43 14 && ckpt_ready 44 14; do
  for pair in "42 $p1" "43 $p2" "44 $p3"; do
    set -- $pair
    if ! ckpt_ready "$1" 14 && ! kill -0 "$2" 2>/dev/null; then
      log "ABORT: INERT $1 ended before checkpoint 14"
      stop_all "abort" 1
    fi
  done
  sleep 60
done
sleep 120

log "checkpoint 14 exists; running halfway kill check"
if ! analyse 14 10 halfway; then
  log "ABORT: halfway analysis failed"
  stop_all "halfway analysis failure" 1
fi
# Kill rule: continue if, in at least one seed, the w = 0 GIVE male gap at checkpoint 14
# (exploratory_w0fresh_ckpt14.json, same settings) exceeds the INERT male gap by >= 0.02.
"$PY" -c '
import json, sys
g = json.load(open(sys.argv[1])); i = json.load(open(sys.argv[2]))
diffs = [(g[f"W0fresh_{s}"]["predator_male_gap"][0] - i[f"INERT_{s}"]["predator_male_gap"][0], s) for s in ("42", "43", "44")]
print("halfway GIVE(w=0)-minus-INERT male gaps", [(round(d, 4), s) for d, s in diffs])
sys.exit(0 if any(d >= 0.02 for d, _ in diffs) else 1)
' "$W0OUT/exploratory_w0fresh_ckpt14.json" "$OUT/halfway.json" >> "$LOG" 2>&1
status=$?
if [ "$status" -eq 1 ]; then
  log "KILL: no seed with GIVE(w=0) male gap >= 0.02 above INERT"
  stop_all "kill rule" 0
elif [ "$status" -ne 0 ]; then
  log "ABORT: kill-rule check failed (exit $status)"
  stop_all "kill-rule failure" 1
fi
log "PASS halfway kill check"
rc=0
for p in "$p1" "$p2" "$p3"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  log "ABORT: training failure"
  stop_all "training failure" 1
fi
log "training done; starting final analysis"
if analyse 29 20 final; then
  log "ALL DONE"
else
  log "FINAL ANALYSIS FAILED"
  exit 1
fi
