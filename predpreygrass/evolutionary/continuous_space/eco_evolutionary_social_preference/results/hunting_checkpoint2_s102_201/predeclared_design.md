# Hunting captures, feeding energy and movement costs

## Design and definitions

Replay the same mixed-population episodes on seeds **102–201**, checkpoint
`blrep2500_both_learn_s2/checkpoint`, strength 2 and assignment seed 20261027.
Preserve frozen PPO, rewards, physical dynamics, reproduction rules and the
non-heritable preference assignments. Append hunting measurements, then
compare every prior saved episode field (including fitness) with its baseline.
This is a diagnostic extension of the same trajectories, not fresh replication.

The observer wraps the existing environment instance's `_energy` and `_gain`
methods, calling each original exactly once and preserving arguments, return
values and animal ordering. The environment source is unchanged. Measurements
use the existing catch list and actual post-movement velocity, not inferred
captures or moving-action frequency.

- **Captures:** credit each caught prey to its original catcher in the existing
  simulation. This measures individual capture credit, not cooperative effort,
  attack attempts, prey encounters or all participants in a group capture.
- **Offered hunting energy:** catch efficiency times prey energy at capture,
  when the original rule allows the predator to feed.
- **Retained hunting energy:** actual before/after change in the predator's
  energy for each original `_gain` call. Feeding at the energy cap may retain
  less than the offered amount. Their difference is storage-cap loss.
- **Movement energy charged:** the existing predator speed-cost coefficient
  times its velocity magnitude after movement, once per physics tick.
- **Resting energy charged:** the resting-cost coefficient times the animal's
  physical-trait cost multiplier. Reproductive expenditure is tracked separately.

The observer checks each individual's predator-tick exposure against its
fitness record, then checks conservation: final minus initial energy equals
retained hunting energy minus movement, resting and offspring energy costs.
Death-interval costs and negative terminal energy are included, following the
existing environment. A final-tick newborn has zero subsequent exposure.

Compare rates per **1,000 observed predator-ticks** by preference type. All
animals are the main descriptive cohort; founder-only rates and offered energy,
resting cost, net feeding-minus-metabolism and feeding/metabolic-cost ratios
are secondary diagnostics. Rates in the comparison table average episode-level
rates, not pooled animal-time ratios. Raw exposure and energy totals are also
saved. Age, physical-trait and survivor composition may affect these comparisons.

Use 10,000 paired episode bootstrap resamples (seed 20261026) for descriptive
95% intervals. The episode seed is the replicate, not an animal or a catch.
Intervals are exploratory and not adjusted for multiple comparisons. Retain
all 100 seeds, including the early ending at seed 133; do not impute hunting
after episode end. Do not tune or analyse interim outcomes.

Four low-priority workers run the same 25-seed chunks as the baseline. Each
completed episode must match every prior saved field before its new record
is written. After the 100 replays, seed 102 is replayed serially and must also
match the new hunting ledger. Modules, checkpoint metadata, source, runtime
and baseline hashes are checked. Use a new output directory; no experiment
is overwritten.

## Reproduce

From the repository root:

```bash
nice -n 10 .conda/bin/python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.hunting_replay \
  predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/fitness_confirmation_checkpoint2_s102_201 \
  --output-dir /tmp/ppg_hunting_new_run
```
