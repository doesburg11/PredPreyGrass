# Fresh-seed confirmation of mixed-preference founder outcomes

## Predeclared design

Run exactly 100 new mixed-population episodes on seeds **102–201**, with
checkpoint `blrep2500_both_learn_s2/checkpoint`, strength 2 and assignment seed
20261027. Six founders give two per preference (-1, 0, +1). Preserve the pilot's
measurement definitions, frozen PPO, rewards and non-heritable preferences.
The pilot's 20 episodes are not pooled into these estimates.

The two primary outcomes are founder observed alive time and offspring per
founder. Each episode contributes the attraction-minus-neutral difference
between its two-founder means. The independent replicate is the episode seed.

Use 10,000 paired episode bootstrap resamples with RNG seed 20261028 and
97.5% marginal percentile intervals for each of the two primary outcomes.
This Bonferroni-style adjustment targets approximate 5% familywise error;
bootstrap coverage is approximate. Joint positive confirmation requires both
lower bounds above zero and no founder censored by an early episode ending.
A failure to confirm does not establish equivalence or absence of an effect.

Keep all seeds, including early episode endings, and report right-censoring
separately from deaths. Founder time is restricted by the observed episode
horizon. If founders are censored before 8,000 ticks, do not interpret their
observed times as complete 8,000-tick restricted survival. Offspring counts
are limited to the observed episode. Born cohorts have unequal follow-up.

Avoidance contrasts, stored energy, reproduction per 1,000 predator-ticks,
and energy in 1,000-tick windows are secondary, descriptive measurements.
Their ordinary 95% bootstrap intervals are not adjusted for multiple comparisons.
No interim outcome analysis, tuning, seed removal or early stopping is planned.
This tests fixed preferences in interacting mixed populations at one checkpoint;
it is not a test of inherited social traits or evolutionary selection.

Four low-priority processes each run 25 consecutive seeds with one Torch
thread. Record the plan, runtime, source/pilot/checkpoint hashes (including all checkpoint metadata files) before any
fresh outcomes. Validate trained modules and weights before and after each
worker and the complete run. A pilot episode (82) must match every saved field,
including fitness. After all 100 episodes finish, a serial fresh-seed replay
(102) must match every field from its parallel run. This is an empirical
fidelity check for those seeds, not a universal equivalence proof.

## Reproduce

From the repository root, choose a new output directory:

```bash
nice -n 10 .conda/bin/python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.fitness_confirmation \
  predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/fitness_mixed_checkpoint2_s82_101/fitness.json \
  --output-dir /tmp/ppg_fitness_confirmation_new
```

The runner fixes seeds 102–201 and refuses to overwrite an existing directory.
Detailed episode records remain in four JSONL journals; the final compact
manifest hashes these journals, and the summary contains aggregate estimates.
