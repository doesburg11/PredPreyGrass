#!/usr/bin/env bash
# Iteration 21: mate-regard reward weight w = 0.5 vs w = 0 on the 18-action compound gift, pre-registered in RESULTS.md.
# Wave 1: W05 seeds 42/43/44. At checkpoint 14 the halfway kill rule compares their gift-bit selectivity with
# Iteration 20's GIVE 42/43 checkpoint 14 (= w 0, same configuration). Wave 2 (W0 42/43/44, fresh) starts only if it
# passes and wave 1 finishes cleanly.
# Launch with: systemd-run --user --unit=psr-mate-regard bash <this file>
set -uo pipefail

ROOT=/home/doesburg/Projects/PredPreyGrass
DS=predpreygrass.non_evolutionary.discrete_space
MOD=$DS.predator_mate_regard
RUNS="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_mate_regard/runs"
IT20="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/runs"
OUT="$RUNS/analysis"
PY="$ROOT/.conda/bin/python"
LOG="$OUT/run_all.log"
mkdir -p "$OUT"
cd "$ROOT"
export OMP_NUM_THREADS=1

log() { echo "$(date +%F_%T) $*" >> "$LOG"; }

# Stop every training process of this module, then exit. When this main process exits, systemd ends the unit
# and kills anything left in its cgroup (default KillMode=control-group), so no explicit systemctl stop is needed
# and this script's exit code is what the unit records.
stop_all() {
  log "stopping all training processes ($1)"
  pkill -TERM -f "$MOD.tune_ppo_mate_regard" 2>/dev/null || true
  sleep 60
  pkill -KILL -f "$MOD.tune_ppo_mate_regard" 2>/dev/null || true
  log "stopped; exit $2"
  exit "$2"
}

run_name() { echo "MATEREGARD_${1}_K05_MB1024_SEED${2}"; }

train() {
  local arm=$1 seed=$2 w name status
  w=0.5
  [ "$arm" = W0 ] && w=0.0
  name=$(run_name "$arm" "$seed")
  log "start $name (w=$w)"
  "$PY" -m "$MOD.tune_ppo_mate_regard" --mate-regard-weight "$w" \
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

# A checkpoint counts as written once its rllib_checkpoint.json exists; the caller then waits 2 more minutes.
ckpt_ready() {
  [ -n "$(find "$RUNS/$(run_name "$1" "$2")" -maxdepth 3 -path '*checkpoint_000014/rllib_checkpoint.json' -print -quit 2>/dev/null)" ]
}

# Wait for every PID, return nonzero if any failed (never short-circuits).
wait_all() {
  local rc=0 p
  for p in "$@"; do
    wait "$p" || rc=1
  done
  return "$rc"
}

analyse_halfway() {
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$DS.predator_compound_provisioning.analyze_compound_gift" \
    --checkpoint 14 --episodes 10 --seed 7000 \
    --run W05_42="$RUNS/$(run_name W05 42)" --run W05_43="$RUNS/$(run_name W05 43)" \
    --run W05_44="$RUNS/$(run_name W05 44)" \
    --run W0_42="$IT20/COMPOUND_GIVE_K05_MB1024_SEED42" --run W0_43="$IT20/COMPOUND_GIVE_K05_MB1024_SEED43" \
    --out "$OUT/halfway.json" > "$OUT/halfway.log" 2>&1 || return 2
  # Kill rule: continue if, for seed 42 or 43, at least one sex has a W05 gift-bit gap at least 0.02 above the
  # matched-seed W0 gap of that sex.
  "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
best = []
for seed in ("42", "43"):
    for sex in ("predator_male", "predator_female"):
        a, b = d[f"W05_{seed}"][f"{sex}_gap"], d[f"W0_{seed}"][f"{sex}_gap"]
        if a is not None and b is not None:
            best.append((a[0] - b[0], seed, sex))
print("halfway W05-minus-W0 gaps", [(round(x, 4), s, x_sex) for x, s, x_sex in best])
sys.exit(0 if any(x >= 0.02 for x, _, _ in best) else 1)
' "$OUT/halfway.json"
}

analyse_final() {
  local specs=() arm seed
  for arm in W05 W0; do
    for seed in 42 43 44; do
      specs+=(--run "${arm}_${seed}=$RUNS/$(run_name "$arm" "$seed")")
    done
  done
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$DS.predator_compound_provisioning.analyze_compound_gift" \
    --checkpoint 29 --episodes 20 --seed 7000 "${specs[@]}" --out "$OUT/final.json" > "$OUT/final.log" 2>&1
}

log "queue start"
train W05 42 & p1=$!
train W05 43 & p2=$!
train W05 44 & p3=$!

until ckpt_ready W05 42 && ckpt_ready W05 43 && ckpt_ready W05 44; do
  for pair in "42 $p1" "43 $p2" "44 $p3"; do
    set -- $pair
    if ! ckpt_ready W05 "$1" && ! kill -0 "$2" 2>/dev/null; then
      log "ABORT: wave-1 run W05 $1 ended before checkpoint 14"
      stop_all "abort" 1
    fi
  done
  sleep 60
done
sleep 120

log "checkpoint 14 exists in wave 1; running halfway kill check"
analyse_halfway >> "$LOG" 2>&1
status=$?
if [ "$status" -eq 1 ]; then
  log "KILL: no seed/sex W05 gap >= 0.02 above W0; stopping wave 1, wave 2 not launched"
  stop_all "kill rule" 0
elif [ "$status" -ne 0 ]; then
  log "ABORT: halfway analysis failed (exit $status); wave 2 not launched"
  stop_all "halfway analysis failure" 1
fi
log "PASS halfway kill check"
if ! wait_all "$p1" "$p2" "$p3"; then
  log "ABORT: wave-1 training failure"
  stop_all "wave-1 failure" 1
fi

train W0 42 & p4=$!
train W0 43 & p5=$!
train W0 44 & p6=$!
if ! wait_all "$p4" "$p5" "$p6"; then
  log "ABORT: wave-2 training failure"
  stop_all "wave-2 failure" 1
fi
log "training done; starting final analysis"
if analyse_final; then
  log "ALL DONE"
else
  log "FINAL ANALYSIS FAILED (exit $?)"
  exit 1
fi
