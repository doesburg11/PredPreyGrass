#!/usr/bin/env bash
# Iteration 23 wave-2 futility check, added 2026-11-18 while wave 2 was running and before any wave-2 checkpoint 14
# existed (see RESULTS.md, Iteration 23 amendment). Runs alongside run_iteration23.sh without modifying it.
# At checkpoint 14 of INERT 46, GIVE 47 and INERT 47, compare matched pairs (GIVE 46 uses its existing checkpoint 14).
# H1 needs all three seeds, so if either wave-2 pair shows GIVE male gap <= INERT male gap, stop the training; the
# launcher then logs ABORT and Iteration 23 is scored as H1 not met.
# Launch with: systemd-run --user --unit=psr-futility23 bash <this file>
set -uo pipefail

ROOT=/home/doesburg/Projects/PredPreyGrass
MOD=predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning
RUNS="$ROOT/predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/runs"
OUT="$RUNS/analysis23"
PY="$ROOT/.conda/bin/python"
LOG="$OUT/run_all.log"
cd "$ROOT"
export OMP_NUM_THREADS=1

log() { echo "$(date +%F_%T) [futility] $*" >> "$LOG"; }
run_name() { echo "REP23_${1}_K05_MB1024_SEED${2}"; }
ckpt_ready() {
  [ -n "$(find "$RUNS/$(run_name "$1" "$2")" -maxdepth 3 -path '*checkpoint_000014/rllib_checkpoint.json' -print -quit 2>/dev/null)" ]
}

log "waiting for wave-2 checkpoint 14"
until ckpt_ready INERT 46 && ckpt_ready GIVE 47 && ckpt_ready INERT 47; do
  if ! pgrep -f "$MOD.tune_ppo_compound_provisioning" > /dev/null; then
    log "no training running before checkpoint 14; nothing to check"
    exit 0
  fi
  sleep 60
done
sleep 120

log "wave-2 checkpoint 14 exists; running futility check"
if ! CUDA_VISIBLE_DEVICES="" "$PY" -m "$MOD.analyze_compound_gift" --checkpoint 14 --episodes 10 --seed 7000 \
    --run GIVE_46="$RUNS/$(run_name GIVE 46)" --run INERT_46="$RUNS/$(run_name INERT 46)" \
    --run GIVE_47="$RUNS/$(run_name GIVE 47)" --run INERT_47="$RUNS/$(run_name INERT 47)" \
    --out "$OUT/futility_wave2.json" > "$OUT/futility_wave2.log" 2>&1; then
  log "futility analysis failed; training left running"
  exit 1
fi

"$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
diffs = []
for s in ("46", "47"):
    a, b = d[f"GIVE_{s}"]["predator_male_gap"], d[f"INERT_{s}"]["predator_male_gap"]
    diffs.append((s, None if a is None or b is None else a[0] - b[0]))
print("wave-2 halfway GIVE-minus-INERT male gaps", [(s, None if x is None else round(x, 4)) for s, x in diffs])
sys.exit(0 if all(x is not None and x > 0 for _, x in diffs) else 1)
' "$OUT/futility_wave2.json" >> "$LOG" 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  log "PASS wave-2 futility check; training continues"
  exit 0
fi
log "STOP: a wave-2 pair has GIVE male gap <= INERT (or missing); H1 cannot be met; stopping training"
pkill -TERM -f "$MOD.tune_ppo_compound_provisioning" 2>/dev/null || true
sleep 60
pkill -KILL -f "$MOD.tune_ppo_compound_provisioning" 2>/dev/null || true
log "stopped"
