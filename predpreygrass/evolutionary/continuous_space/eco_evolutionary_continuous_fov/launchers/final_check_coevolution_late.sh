#!/usr/bin/env bash
# Coexistence checks of the test-2 follow-up (design note addendum of
# 2026-10-30): final_check.py, 50 episodes, at iterations 700, 800, 900 and
# 1,000 of each coevolution_late run, on the CPU, six at a time. Refuses to
# overwrite an existing summary. Run from the repository root, e.g. under
# systemd-run --user, after launchers/queue_coevolution.sh late.
set -u
PYTHON=/home/doesburg/Projects/PredPreyGrass/.conda/bin/python
MODULE=predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_fov
CHECK=predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.final_check
jobs=()
for source in tradeoff_m120 tradeoff_m120_seed2; do
  for arm in both prey_frozen predator_frozen; do
    for iteration in 0700 0800 0900 1000; do
      run=$MODULE/runs/coevolution_late_${source}_${arm}
      checkpoint=$run/checkpoint
      [ "$iteration" = 1000 ] || checkpoint=$checkpoint/iter_00$iteration
      out=$MODULE/results/final_check_coevolution_late_${source}_${arm}_iter${iteration}_summary.json
      if [ -e "$out" ]; then echo "refusing: $out exists"; exit 1; fi
      if [ ! -f "$checkpoint/gene_pools.json" ]; then echo "missing: $checkpoint"; exit 1; fi
      jobs+=("$checkpoint $out")
    done
  done
done
failed=0
pids=()
for i in "${!jobs[@]}"; do
  set -- ${jobs[$i]}
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PYTHON" -m "$CHECK" "$1" 50 "$2" --workers 5 > "${2%_summary.json}.log" 2>&1 &
  pids+=($!)
  if (( ${#pids[@]} == 6 )) || (( i + 1 == ${#jobs[@]} )); then
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    pids=()
  fi
done
for job in "${jobs[@]}"; do
  set -- $job
  [ -f "$2" ] || { echo "no summary: $2"; failed=1; }
done
echo "$(date '+%F %T') checks done, failed=$failed"
exit $failed
