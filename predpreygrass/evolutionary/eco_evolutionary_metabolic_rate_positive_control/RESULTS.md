# Training Analysis — eco_evolutionary_metabolic_rate_positive_control

**Status: Pilot 1 complete (weak partial signal, not a clean pass). Pilot 2 in progress,
testing whether mutation rate rather than effect size was the bottleneck.** See `README.md`
for what this module is and why it exists — a positive control, not a new trait design.

---

## 1. Why this module exists

Three trait designs in this project have now come back flat or null:

- Trial 3 (`eco_evolutionary_metabolic_rate`, sub-linear `metabolic_rate`, α ∈ {0.7, 0.4}) —
  null even after sharpening the fitness gradient; individual-level `mr_repro_spearman`
  also flat.
- Trial 8 (`eco_evolutionary_cultural_plasticity`) — stopped early, `plasticity` flat across
  3 real seeds.
- Trial 9 (`eco_evolutionary_cultural_plasticity_seasonal`) — seed 42's full 1000-iteration
  run also shows `plasticity_mean` and `plasticity_repro_spearman` bouncing with no
  consistent direction throughout (not yet formally analyzed, but the shape matches Trial 8).

None of this rules out that the whole pipeline — genome inheritance, mutation, differential
reproduction propagating into population-level drift — is simply unable to produce a
detectable signal at the population sizes / reproduction-event counts / episode lengths this
project runs at, independent of any trait's design. That question has never been directly
tested, because every trait tried so far has had a *plausible, biologically-motivated, but
comparatively subtle* fitness effect.

This module removes the subtlety. `metabolic_rate_alpha = 3.0` (vs. the parent module's
0.4-0.7) makes net efficiency scale as `metabolic_rate ** 2`: at the trait's upper bound
(2.0) vs. lower bound (0.5), that's a 16x difference in net energy efficiency — an
overwhelming, monotonic advantage for high `metabolic_rate`, not dependent on policy quality
or any interior-optimum balancing act.

**If this comes back flat too:** the problem is very likely structural (not enough
reproduction events for selection to outrun drift, or the shared-per-species-PPO-policy
architecture itself), and the project should stop trying additional trait designs and
address that instead.

**If this shows a clean, fast drift toward the upper trait bound:** the pipeline can detect
selection given a strong enough gradient, which means prior null results are about
insufficient fitness leverage in those specific trait designs, not a broken pipeline — and
increasing effect size / reproduction-event count becomes the productive lever for future
trait designs, rather than continuing to guess at mechanism variants.

## 2. What's been done so far

- **Implementation.** Cloned from `eco_evolutionary_metabolic_rate` (Trial 3); the only
  substantive change is `metabolic_rate_alpha: 0.4 → 3.0` in
  `config/config_env_eco_evolutionary.py`. Genome bounds, mutation rate/std, founder
  distribution, population caps, satiation throttle, and all other sustainability mechanics
  were unchanged from the already-validated Trial 3 setup for Pilot 1.
- **Unit tests.** 24/27 passing — the 3 failures (`test_episode_return_callback_logs_*`) are
  pre-existing in the parent module too (an RLlib test-harness API mismatch unrelated to this
  change), not something this clone introduced.
- **CSV-logging fragility fixed.** `{species}_mr_repro_spearman` and `_mr_repro_rate_q1-q4`
  are now always emitted (NaN when not yet computable) instead of being omitted when fewer
  than 4 qualifying individuals exist. Ray Tune's CSV logger fixes its column set from the
  first reported iteration and silently drops any key that only starts appearing later; with
  this module's small starting population, that had permanently excluded these columns from
  `progress.csv` (values were still recoverable from the TensorBoard event file, which is how
  Pilot 1's results below were read).

### Pilot 1 — alpha=3.0, mutation std=0.04 (parent module's default) — complete, weak partial signal

200 iterations, single seed (config default, no `--seed` override). Not a clean pass:

- `{species}_metabolic_rate_mean` climbed from the founder value (1.0) to only ~1.05-1.06 for
  both species, then plateaued — far short of the 2.0 upper bound despite the deliberate 16x
  efficiency gradient.
- `{species}_metabolic_rate_std` narrowed early (population converging) then re-widened in
  the back half of the run (0.018 → 0.056 for predator around iteration 50-92) — consistent
  with mutation re-randomizing the population faster than selection could lock in an
  advantage.
- `predator_mr_repro_spearman` (the direct individual-level test: does higher MR actually
  predict reproduction?) stayed **consistently positive** through the second half of the run
  — last-quarter mean +0.046, final 8 readings 0.053, 0.056, 0.064, 0.059, 0.063, -0.003,
  0.051, 0.148. Small, but a real, non-noise signal — this is the strongest evidence yet in
  this project that the pipeline *can* detect selection, given a strong enough gradient.
- `prey_mr_repro_spearman` showed no consistent direction (last-quarter mean -0.024, final 8
  readings -0.012, -0.007, -0.036, -0.059, -0.025, -0.221, -0.023, 0.088) — no signal.

**Interpretation:** rules out "the pipeline can't detect selection at all" (predator's
correlation is real and directionally stable, not scattered noise). But also rules out
"effect size alone was the bottleneck" — a 16x gradient should produce a much larger, faster
shift than a 5% mean move that plateaus, if effect size were the only constraint. Most likely
explanation: population size (~20-30 reproducing individuals) and/or the mutation rate
(re-randomizing a meaningful fraction of a small population every generation) are capping how
much of a real fitness advantage can compound into visible drift, independent of how strong
the advantage is. The predator/prey split is itself informative, not just noise: predator
reproduction is tightly bottlenecked on scarce, effortful catches (satiation-throttled),
where a per-catch efficiency edge matters a lot; prey reproduction depends on locally-abundant
grass, where the same edge barely moves the needle on whether any given individual finds
enough food. Traits tied to a scarce, contested resource may be inherently more detectable
than traits tied to an abundant one, independent of raw effect size.

### Pilot 2 — alpha=3.0 (unchanged), mutation std=0.04 → 0.01 — in progress

Isolates the mutation-rate hypothesis: same overwhelming 16x gradient as Pilot 1, but less
mutation noise fighting it. If the predator signal strengthens and the population mean
actually climbs toward the bound instead of plateauing, mutation rate (relative to population
size) is the real bottleneck — a cheap fix (lower mutation, and/or larger population) rather
than requiring the bigger architectural change (per-agent/genome-conditioned policies).

## 3. Not yet done

- Pilot 2 results (in progress).
- Isolating population size specifically (Pilot 2 only changes mutation; a further run
  holding mutation at Pilot 1's value but relaxing whatever caps active population at ~20-30
  would isolate that variable separately).
- A second seed of whichever pilot config looks most promising, before trusting the direction
  as real rather than one-seed luck.
- No neutral-control comparison run yet (needed to confirm any drift seen is selection, not
  just mutation + finite-population sampling noise, same discipline as every prior module).

## 4. Next step

Let Pilot 2 finish and compare its `mr_repro_spearman` and `metabolic_rate_mean` trajectories
against Pilot 1's numbers above.
