#!/usr/bin/env bash
# Iteration 19: learned optional mate provisioning, pre-registered in RESULTS.md.
# First wave: GIVE42, INERT42, GIVE43.  At checkpoint 14 the hard kill rule
# checks the two GIVE runs before the remaining three are started.
set -uo pipefail

ROOT=/home/doesburg/Projects/PredPreyGrass
MOD=predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction
RUNS="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_sexual_reproduction/runs"
OUT="$RUNS/analysis/learned_gift"
PY="$ROOT/.conda/bin/python"
LOG="$OUT/run_all.log"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1

log() { echo "$(date +%F_%T) $*" >> "$LOG"; }

train() {
  local arm=$1 seed=$2 mode name status
  mode=on
  [ "$arm" = INERT ] && mode=inert
  name="PPO_PREDATOR_SEXUAL_REPRODUCTION_${arm}_K05_MB1024_SEED${seed}"
  log "start $name"
  "$PY" -m "$MOD.tune_ppo_predator_sexual_reproduction" \
    --seed "$seed" --max-iters 300 \
    --reward-catch-prey 0 --reward-gather-fruit 0 --reward-per-energy 0.5 --penalty-combat-death 0.2 \
    --minibatch-size 1024 --num-epochs 30 --mate-obs-channel mate --learned-gift "$mode" \
    --learned-gift-amount 2.0 --male-gift-donation-rate 0 \
    --name "$name" --gpu-fraction 0.33 --num-env-runners 4 --num-cpus-per-env-runner 2 --num-cpus-main 2 \
    > "$OUT/train_${arm}_s${seed}.log" 2>&1
  status=$?
  log "end $name exit $status"
  return "$status"
}

ckpt_path() {
  local arm=$1 seed=$2
  find "$RUNS/PPO_PREDATOR_SEXUAL_REPRODUCTION_${arm}_K05_MB1024_SEED${seed}" \
    -maxdepth 2 -type d -name 'checkpoint_000014' -print -quit 2>/dev/null
}

analyse_halfway() {
  local give42="$RUNS/PPO_PREDATOR_SEXUAL_REPRODUCTION_GIVE_K05_MB1024_SEED42"
  local give43="$RUNS/PPO_PREDATOR_SEXUAL_REPRODUCTION_GIVE_K05_MB1024_SEED43"
  "$PY" -m "$MOD.analyze_learned_gift" --checkpoint 14 --episodes 10 --seed 7000 \
    --run GIVE_42="$give42" --run GIVE_43="$give43" \
    --out "$OUT/halfway_gift.json" > "$OUT/halfway_gift.log" 2>&1
  "$PY" -c '
import json, sys
d=json.load(open(sys.argv[1])); rates=[d[k]["gift_events_per1000_pred_steps"][0] for k in ("GIVE_42", "GIVE_43")]
print("halfway GIVE rates", rates)
sys.exit(1 if all(x < 1.0 for x in rates) else 0)
' "$OUT/halfway_gift.json"
}

analyse_final() {
  local specs=()
  for arm in GIVE INERT; do
    for seed in 42 43 44; do
      local p="$RUNS/PPO_PREDATOR_SEXUAL_REPRODUCTION_${arm}_K05_MB1024_SEED${seed}"
      specs+=(--run "${arm}_s${seed}=$p")
    done
  done
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_learned_gift" --checkpoint 29 --episodes 20 --seed 7000 \
    "${specs[@]}" --out "$OUT/learned_gift_final.json" > "$OUT/learned_gift_final.log" 2>&1 &
  local p1=$!
  CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_mate_tag" --checkpoint 29 --episodes 20 --part B --seed 7000 \
    "${specs[@]}" --out "$OUT/pair_outcomes_final.json" > "$OUT/pair_outcomes_final.log" 2>&1 &
  local p2=$!
  wait "$p1"; local s1=$?; log "final learned-gift analysis exit $s1"
  wait "$p2"; local s2=$?; log "final pair-outcome analysis exit $s2"
  [ "$s1" -eq 0 ] && [ "$s2" -eq 0 ]
}

log "queue start"
train GIVE 42 & p1=$!
train INERT 42 & p2=$!
train GIVE 43 & p3=$!

while [ -z "$(ckpt_path GIVE 42)" ] || [ -z "$(ckpt_path GIVE 43)" ]; do
  if { [ -z "$(ckpt_path GIVE 42)" ] && ! kill -0 "$p1" 2>/dev/null; } || \
     { [ -z "$(ckpt_path GIVE 43)" ] && ! kill -0 "$p3" 2>/dev/null; }; then
    log "ABORT: first-wave GIVE run ended before checkpoint 14"
    kill "$p1" "$p2" "$p3" 2>/dev/null || true
    wait "$p1" "$p2" "$p3" 2>/dev/null || true
    exit 1
  fi
  sleep 60
done

log "checkpoint 14 exists; running halfway kill check"
if ! analyse_halfway >> "$LOG" 2>&1; then
  log "KILL: both GIVE rates below 1 per 1000 predator steps; stopping first wave and not launching second wave"
  kill "$p1" "$p2" "$p3" 2>/dev/null || true
  wait "$p1" "$p2" "$p3" 2>/dev/null || true
  exit 0
fi
log "PASS halfway kill check"
if ! wait "$p1" || ! wait "$p2" || ! wait "$p3"; then
  log "ABORT: first-wave training failure"
  exit 1
fi

train INERT 43 & p4=$!
train GIVE 44 & p5=$!
train INERT 44 & p6=$!
if ! wait "$p4" || ! wait "$p5" || ! wait "$p6"; then
  log "ABORT: second-wave training failure"
  exit 1
fi
log "training done; starting final analyses"
analyse_final
log "ALL DONE"
