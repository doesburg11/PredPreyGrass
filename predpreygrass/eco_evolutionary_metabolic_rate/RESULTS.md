# Eco-Evolutionary Metabolic Rate — Training Results

Results from PPO training runs on the `eco_evolutionary_metabolic_rate` environment. Analysis
focuses on whether the Darwin/Baldwin loop — genetic evolution and within-lifetime RL learning
feeding back into each other — is actually operating, whether it's sustainable, whether the
population-regulation mechanism is biologically realistic, and whether any observed genome
drift is statistically distinguishable from neutral noise. See `README.md` for the trait design
rationale (why `metabolic_rate` rather than `offspring_investment_fraction`).

This is a running research log, not just a final write-up — it records the trial-and-error
trail (what was tried, why, what was rejected and why) so the search can be re-evaluated later
without reconstructing it from conversation history.

---

## Iteration log

Each entry: the config that was run, the result, the diagnosed cause of whatever wasn't working,
and the adjustment made in response — which becomes the config for the next entry. This is the
literal trial-and-error trail; read top to bottom to re-derive the reasoning without needing
conversation history. Full data tables for each iteration are in the matching detail section
further below.

### Iteration 0 — Baseline (original run, no reproduction throttle)

**Config:** no reproduction throttle at all (`predator_reproduction_max_ratio`,
`predator_satiation_cooldown`, `max_energy_gain_per_prey` didn't exist yet). Run:
`PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-06-28_15-14-26`, 411 iterations.

**Result:** 46% of training iterations produced valid eco-evolution data (rest crashed
mid-episode); prey showed a clean Darwin/Baldwin cycle (live MR 0.990 → 1.031 peak → 1.003),
predators showed none (flat at founder-mean equilibrium the entire run, MR ≈0.994-1.000). Full
analysis in the "Baseline" detail section below.

**Possible cause:** predator reproduction was unconditional — the instant a predator crossed
`predator_creation_energy_threshold` it reproduced, with no limit on catch frequency or catch
size. A lucky run of catches could push predator numbers up faster than prey scarcity could
push back: the classic timing-lag instability behind boom-bust collapse in simple
Lotka-Volterra systems with no saturating functional response. Predators being flat is
consistent with them starting already near whatever equilibrium exists, with no perturbation to
reveal a cycle.

**Adjustment → Iteration 1:** this run's own "next steps" list (see the detail section below)
prioritized a freeze-genome controlled experiment first and crash-rate reduction last (#4).
Reordered around the actual goal — a *sustainable* loop, not just a scientifically rigorous
demonstration — which pointed at #4 first: add a hard throttle on predator reproduction. Chose
a population-ratio cap as the simplest, most direct lever: `predator_reproduction_max_ratio: 1.5`.

### Iteration 1 — Population-level ratio cap

**Config:** `predator_reproduction_max_ratio: 1.5` (block predator reproduction outright once
`active_predators ≥ 1.5 × active_prey`), everything else unchanged.

**Result:** sustainability solved — 99.7% valid iterations, episode length climbed to ~98%+ of
full length by the run's end. But predator genome drift stayed pinned in a narrow 0.97-0.99
band — muted, not fixed.

**Possible cause:** measured directly, not guessed — only 11-16% of energy-eligible predator
reproduction attempts actually succeeded, the rest blocked purely by population timing
regardless of which predator was trying or how fit it was. Reproductive success is supposed to
track genome quality; blocking 84-89% of attempts indiscriminately dilutes that signal to a
small residual. Separately, the mechanism itself is not biologically motivated — no individual
predator could ever sense a population census — which was flagged as disqualifying regardless
of the sustainability win.

**Adjustment → Iteration 2:** reject the ratio cap; replace with individual-level throttles
that regulate population through each predator's own energy/hunting history instead of a
population-wide rule: `predator_reproduction_max_ratio: None` (disabled),
`predator_satiation_cooldown: 8` (steps before the same predator can catch again),
`max_energy_gain_per_prey: 8.0` (per-catch energy ceiling).

### Iteration 2 — Satiation throttle, run 1 (1000 iterations)

**Config:** `predator_satiation_cooldown: 8`, `max_energy_gain_per_prey: 8.0`,
`predator_reproduction_max_ratio: None`.

**Result:** real predator genome cycle for the first time in this whole search — live MR
1.003 → 0.954 (trough) → 0.963, a ~4.6% decline with partial recovery. But episode-completion
rate plateaued at 45.3% by the run's end, well short of Iteration 1's ~98%.

**Possible cause:** the completion-rate trend was still climbing at iteration 1000, not
plateaued — the open question was whether this was a "needs more runway" problem or a real
ceiling of the mechanism. Separately, an apparent puzzle: the selection-gap and
`predator_mr_repro_spearman` metrics were both positive (implying selection favors *higher* MR)
while the population mean *fell* — investigated later and resolved as a metric-definition
mismatch (`eco_evolution_MR` is an unweighted per-individual-ever-born average,
`live_genome_MR` is lifespan-weighted; they don't decompose into "why did the mean move" the
way originally assumed). Not a real contradiction, but also not an explanation for the decline.

**Adjustment → Iteration 3:** rerun the identical config for longer (2000 iterations, new seed)
to test two things at once: does completion rate keep closing the gap with Iteration 1 given
more training time, and does the predator genome cycle reproduce on an independent seed or was
it a one-off trajectory.

### Iteration 3 — Satiation throttle, run 2 (2000 iterations, replication)

**Config:** identical to Iteration 2, `max_iters: 2000`, new seed.

**Result:** completion rate climbed further (12.0% → 62.1%) but decelerated in the second half
rather than continuing toward Iteration 1's ~98% — looks like a real ceiling in the 60-70%
range for this configuration, not just a runway problem. Predator genome MR declined again
(1.018 → 0.992, ~2.6%) — same *direction* as run 1, but smaller and noisier (a mid-training
checkpoint at iteration 1648 even looked like pure oscillation with no net direction before the
final quintile revealed the decline).

**Possible cause:** two seeds now show the same directional tendency, which is better evidence
than one run — but a repeating-but-variable ~2.6-4.6% shift in population-mean MR is exactly
the magnitude that neutral genetic drift (pure mutation + finite-population sampling noise,
with zero fitness difference between genome values) could plausibly produce on its own, given
how many generations turn over within a single long episode (~140+ unique predators born per
episode). Two runs both showing a modest, noisy decline doesn't by itself rule this out.

**Adjustment → Iteration 4:** rather than run a third real trial (more of the same kind of
ambiguous evidence), implement a neutral-drift control — same config, but genome inheritance
decoupled from reproductive success — to establish what pure drift looks like at this
population scale, and compare directly against the ~2.6-4.6% observed in runs 2 and 3. A
displaced-founder-mean test (start predator MR at 0.7 or 1.3, check for return-to-equilibrium)
was considered as an alternative next step but set aside: it would have tested for a real
attractor, but wouldn't have addressed whether the *magnitude* of any observed movement is
distinguishable from noise, which is the more fundamental open question at this point.

### Iteration 4 — Neutral-drift control (2000 iterations) — in progress

**Config:** identical to Iteration 3, plus `genome_neutral_drift_control: True` — offspring
genome template is a uniformly random currently-alive same-species agent instead of the actual
parent; reproduction eligibility, timing, and all energy dynamics are otherwise unchanged.
Started 2026-07-08 (`PPO_ECO_EVOLUTION_METABOLIC_RATE_NEUTRAL_CONTROL_2026-07-08_03-41-51`).

**Result so far (iteration 637/2000, not final):** population dynamics track the real runs
closely (episode length, peak predator/prey counts all in the same range as run 2 at a
comparable stage) — confirms the control preserves ecology while only severing genome
inheritance. Predator MR is already declining under zero selection pressure: 0.989 → 0.985 →
0.970 (~1.9% over the first third of the run).

**Possible cause (preliminary, not conclusive):** if this continues at a similar pace, the
neutral control could plausibly reach a decline in the same 2.6-4.6% range as the real runs by
iteration 2000 — which would mean the observed predator genome cycle is largely indistinguishable
from noise rather than real selection. Too early to conclude; the real runs' trajectories
weren't monotonic either.

**Adjustment → Iteration 5 (pending):** to be determined once this run completes and the final
comparison is in. If the neutral control's drift is comparable to the real runs', the next step
is likely to either (a) look for a selection mechanism strong enough to exceed the drift
baseline (e.g. coupling satiation cooldown to metabolic rate, adding a second genome-linked
fitness channel), or (b) accept that MR isn't under meaningful selection pressure at these
settings and treat the sustainability work as the deliverable rather than the Darwin/Baldwin
signal. If the real runs' drift is substantially larger than the control's, that's the
confirmation needed to call the loop established.

---

## Baseline — original run analysis (Iteration 0 detail)

Run: `PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-06-28_15-14-26`, 411 iterations, 422K steps.
Analysis snapshots at iter 217 (225K steps) and iter 410 (422K steps).

### Environment recap

Agents carry a **metabolic rate genome** (MR, founder mean 1.0 ± 0.10, mutation rate 5%, std
0.04, bounds [0.5, 2.0]).

The tradeoff is designed to create a policy-dependent optimum:

| Effect | Formula | Direction |
|---|---|---|
| Energy cost | `basal_cost × MR` | Higher MR → costlier to exist |
| Energy gain (eating) | `food_energy × MR^α` (α = 0.70) | Higher MR → more energy per bite |

With α < 1 (sub-linear gain, linear cost) there is an interior optimum whose location depends
on *how often the agent eats* — i.e., on policy quality. This is what links Darwin (genome
selection) to Baldwin (RL learning).

Rewards: only **reproduction** yields signal (`+10`). Catching prey (predators) and eating
grass (prey) give `0`. This is intentionally sparse.

### Why predator_episode_return_p50 = 0 throughout

`predator_episode_return_p50` is 0.000 in every iteration. This is **not evidence that
predators fail to learn or fail to reproduce**.

The callback measures the individual-lifetime return of *every* predator agent ID that ever
lived in the episode, including the large majority of short-lived offspring who die before
reproducing. With ~170 unique predator IDs per episode and fewer than 50% reproducing before
death, the median is 0 by construction regardless of how good the policy is. The real learning
signal is in:

- `predator_spawned_total`: grew from 0 (iter 1) → ~127 (iter 21) → ~170–195 (iter 41+). That
  jump is the policy learning to hunt.
- `predator_offspring_count_mean ≈ 0.97`: survivors at episode end each produced ~1 offspring on
  average. Nearly every surviving predator reproduced.

Predators do reproduce. The p50 = 0 reflects population structure (deep reproduction chain of
~1 offspring per parent), not learning failure.

### Ecology

At the stable phase (iter 40 onward, sustained to iter 410):
- Prey alive (live_genome): ~15–22 per step
- Predators alive (live_genome): ~16–20 per step
- Ratio ~1:1 — ecologically tight, consistent with a persistent predator-prey arms race

Predator reproduction never hit the `n_possible_predators = 500` capacity ceiling
(reproduction_blocked = 0 throughout). The ecosystem is under pressure but not collapsing.

About 46% of iterations report valid eco-evolution metrics (223 of 411); the rest are NaN due
to within-episode population crashes.

### RL learning trajectory

Quintiles over the full 411-iteration run (valid rows only):

| Quintile | Iters | Prey spawned | Prey ret p75 | Pred spawned |
|---|---|---|---|---|
| Q1 | 1–111 | 440 | 5.85 | 145 |
| Q2 | 117–191 | 534 | 6.01 | 168 |
| Q3 | 192–273 | 511 | 5.55 | 160 |
| Q4 | 275–340 | 490 | 6.30 | 154 |
| Q5 | 343–411 | 524 | 5.19 | 160 |

Prey policy improved in Q1–Q2 (387 → 534 spawns), then plateaued with noise at 490–534. Prey
return p75 is flat to slowly rising overall (5.2–6.3), with no clear further gains after iter
200. Predator spawned_total stable at 145–168, consistent with the predator:prey ceiling.

### Genome evolution — the Darwin signal

#### Prey live_genome MR trajectory

| Quintile | Iters | Live_genome MR | Std |
|---|---|---|---|
| Q1 | 1–111 | 0.990 | ±0.024 |
| Q2 | 117–191 | **1.031** | ±0.011 |
| Q3 | 192–273 | 1.013 | ±0.013 |
| Q4 | 275–340 | 1.015 | ±0.020 |
| Q5 | 343–411 | 1.003 | ±0.022 |

The prey genome completed **one full cycle**: rising from 0.990 to a peak of 1.031 in Q2, then
returning to ~1.003 in Q5. Not random drift — variance collapsed from ±0.024 to ±0.011 at the
Q2 peak (the population converged on the new optimum) and widened again as the genome settled
back. The Q5 endpoint (~1.003) is slightly above the Q1 start (0.990), suggesting the
equilibrium shifted a small but persistent amount upward from the founder mean of 1.0.

#### Predator live_genome MR trajectory

| Quintile | Iters | Live_genome MR | Std |
|---|---|---|---|
| Q1 | 1–111 | 0.999 | ±0.017 |
| Q2 | 117–191 | 0.994 | ±0.017 |
| Q3 | 192–273 | 0.996 | ±0.010 |
| Q4 | 275–340 | 1.000 | ±0.011 |
| Q5 | 343–411 | 0.997 | ±0.008 |

Predators show no directional genome drift across 411 iterations. Variance steadily decreased
(±0.017 → ±0.008), meaning the predator population is converging tightly on MR ≈ 1.0. The
genome stabilised at the founder mean.

### The Baldwin signal — selection gap reversal

The **selection gap** = `eco_evolution MR − live_genome MR` (see the metric-definition
correction in the Fix 2 section below — this framing was later found to be less clean than it
looks here, since these two metrics are different weighting schemes of the same population, not
survivors vs. non-survivors).

#### Prey selection gap

| Quintile | Iters | Gap | Interpretation |
|---|---|---|---|
| Q1 | 1–111 | −0.001 | Cost-side dominant: survive longer by burning less |
| Q2 | 117–191 | **+0.006** | Gain-side takes over: eat enough that higher MR pays off |
| Q3 | 192–273 | +0.008 | Gain-side still dominant |
| Q4 | 275–340 | −0.004 | Oscillates back to cost-side |
| Q5 | 343–411 | +0.006 | Gain-side again |

The sign flipped at Q2 and oscillated since. The initial reversal (Q1→Q2) coincides with the
Baldwin mechanism: the prey policy improved enough (spawned_total 440 → 534) for eating
frequency to tip the energy tradeoff. The genome peak in Q2 is the direct consequence.

#### Predator selection gap

| Quintile | Iters | Gap | Interpretation |
|---|---|---|---|
| Q1 | 1–111 | +0.003 | Gain-side mildly dominant from the start |
| Q2 | 117–191 | +0.009 | Strengthening |
| Q3 | 192–273 | +0.004 | Still positive |
| Q4 | 275–340 | +0.008 | Persistent |
| Q5 | 343–411 | +0.009 | Sustained |

Predators show a consistently positive gap throughout, yet the genome doesn't move — consistent
with the population already being near its equilibrium for the current policy quality, with
mutation balancing any gain-side pull back toward the founder mean.

### Darwin/Baldwin loop verdict (at the time)

**Yes, the loop ran one complete cycle — clearly for prey, stably at equilibrium for
predators.** The loop requires three things:

1. **RL improvement changes which genome is adaptive.** ✓ The prey selection gap reversed sign
   as the policy improved in Q1→Q2.
2. **The genome responded to that new selection pressure.** ✓ Prey MR rose to a peak of 1.031
   with sharply reduced variance, then returned toward ~1.003 as the system settled.
3. **The genome shift feeds back into RL.** Plausible but not directly measurable from these
   logs — this became the motivation for the `predator_mr_repro_spearman` /
   `prey_mr_repro_spearman` reverse-leg metric used in later iterations.

For predators: the selection gap was consistently positive, but the genome showed no
directional shift — the population appeared to already be at equilibrium from the start, so
there was no displacement to observe a cycle from.

**Caveats at the time:**
- The prey genome cycle amplitude was modest (+0.041 peak, settling to +0.013 above baseline).
- 46% of episode iterations reported valid eco-evolution metrics — meaningful selection only
  operates in episodes that don't crash mid-episode, limiting effective selection pressure.
  This is exactly the problem Iteration 1 onward addresses.
- The reverse leg (genome → RL) was unconfirmed and would need either a controlled experiment
  or (as implemented later) a within-run MR-vs-reproduction correlation metric.

### Original next-steps list (context for why Iteration 1 happened)

This is the prioritized list from the original analysis, reordered as described in Iteration 0's
"Adjustment" above:

| Priority (original) | Action | What it establishes |
|---|---|---|
| 1 | Freeze-genome controlled experiment | Genome → RL direction (missing leg) |
| 2 | Raise α or grass density, run longer | Second cycle; loop is repeatable |
| 3 | Start predator genome at MR 0.80 | Observable predator Darwin cycle |
| 4 | Reduce ecosystem crash rate | Stronger selection signal per step |

Item 4 was pursued first because the stated goal was specifically a *sustainable* loop, not
just a scientifically rigorous demonstration of one. Items 1-3 remain open; item 1 was
substantially addressed by the `mr_repro_spearman` metric introduced in later iterations
(a within-run correlation rather than a separate frozen-genome experiment).

---

## Fix 1 — Population-level ratio cap (Iteration 1 detail)

**Run: `PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-07-05_21-16-40`, 1000 iterations.**

`predator_reproduction_max_ratio: 1.5` — a predator eligible to reproduce is blocked outright
if `active_predators ≥ 1.5 × active_prey`, regardless of which predator it is or how it got
there. This is a population census rule: it has no connection to any individual agent's own
energy or hunting history.

### Result: crash rate solved, genome drift muted

| Quintile | Ep. len (of 1000) | Peak pred | Peak prey | Predator live MR | Prey live MR | Repro success rate* |
|---|---|---|---|---|---|---|
| Q1 (4-202) | 776 | 21.7 | 43.7 | 0.993 | 0.978 | 10.8% |
| Q2 (203-401) | 941 | 25.6 | 41.0 | 0.986 | 0.982 | 13.9% |
| Q3 (402-600) | 944 | 25.4 | 40.7 | 0.980 | 1.005 | 12.8% |
| Q4 (601-799) | 966 | 25.0 | 40.6 | 0.970 | 0.984 | 15.0% |
| Q5 (800-1000) | 984 | 25.0 | 40.4 | 0.981 | 0.989 | 16.4% |

\* share of energy-eligible predator reproduction attempts that weren't blocked by the ratio cap.

- **997/1000 iterations (99.7%) valid** — a huge jump from the baseline run's 46%.
- Episode length climbed to ~980-1000 of 1000 steps by the end — episodes routinely running
  to completion instead of crashing.
- **Only 11-16% of energy-eligible predator reproduction attempts actually succeeded** — the
  rest were blocked purely by population timing, with no regard for which predator was trying
  or how fit it was.
- **Predator live-genome MR stayed pinned in a narrow 0.97-0.99 band the entire run** — no
  directional cycle at all, versus the baseline run's flat predator (which at least had an
  excuse: it started at equilibrium). Selection gap stayed small (+0.000 to +0.010) and
  `predator_mr_repro_spearman` flipped the expected sign (−0.088 → +0.01 to +0.03) but the
  magnitude never grew enough to move the population mean noticeably.

### Why the cap muted the drift

Reproduction success is supposed to track genome quality (a predator that hunts well should
out-reproduce one that doesn't). The ratio cap blocks reproduction indiscriminately once the
population ratio is exceeded — it doesn't ask which predator is trying. With 84-89% of
attempts blocked regardless of fitness, the realized variance in reproductive success
attributable to genome quality is diluted to a small residual (the correctly-signed but
weak spearman correlation above), not eliminated but heavily attenuated.

**Verdict:** solves sustainability, but the mechanism is not biologically motivated — no
individual predator could ever sense or respond to a population census — and it comes at a
real cost to the very selection signal the experiment is trying to observe.

---

## Fix 2 — Individual-level satiation throttle (Iteration 2 detail)

**Run: `PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-07-06_12-11-02`, 1000 iterations.**

The ratio cap was disabled (`predator_reproduction_max_ratio: None`) and replaced with two
throttles that operate purely on each predator's own hunting history:

- `predator_satiation_cooldown: 8` — a predator that just caught prey cannot catch again for
  8 steps (digestion/rest). A prey encountered during this window survives.
- `max_energy_gain_per_prey: 8.0` — a satiation ceiling; a single kill cannot provide more
  energy than this regardless of how much the prey itself had accumulated.

Both are per-individual: whether a specific predator gets to hunt again depends only on *its
own* recent catches, never on how many other predators exist. Functionally this is a Holling
Type II "handling time" term — the standard mechanism ecologists use to make a predator's
catch rate saturate rather than scale unboundedly with prey density, which is the actual
missing ingredient behind the original instability.

### Result: real genome drift, weaker (but improving) sustainability

| Quintile | Ep. len (of 1000) | % full-length | Peak pred | Peak prey | Predator live MR | Prey live MR | Catches blocked by satiation |
|---|---|---|---|---|---|---|---|
| Q1 (4-202) | 684 | 18.6% | 21.3 | 54.6 | 1.003 | 0.952 | 107.2 |
| Q2 (203-401) | 786 | 24.6% | 26.9 | 46.1 | 0.984 | 0.989 | 147.6 |
| Q3 (402-600) | 900 | 40.2% | 26.3 | 48.6 | 0.968 | 0.995 | 168.9 |
| Q4 (601-799) | 893 | 41.7% | 26.1 | 48.5 | **0.954** | 0.991 | 165.4 |
| Q5 (800-1000) | 873 | 45.3% | 25.8 | 46.0 | 0.963 | 0.984 | 159.7 |

Overall: 997/1000 iterations valid (99.7%, same as Fix 1); 34.1% of episodes reached the full
1000-step length overall, climbing steadily quintile over quintile (18.6% → 45.3%) but never
reaching Fix 1's ~98%+ plateau within this run's 1000 iterations.

- **Predator live-genome MR moved substantially: 1.003 → 0.984 → 0.968 → 0.954 (trough) →
  0.963** — a real, mostly-monotonic ~4.6% decline followed by a partial recovery. This is the
  predator-side genome cycle that neither the baseline run nor the ratio-cap run produced.
  Individual-level dispersion (`live_genome_predator_metabolic_rate_std`) also grew over the
  run (0.041 → 0.054 → 0.062 → 0.062 → 0.060) rather than shrinking — the population stayed
  genetically diverse instead of converging tightly, likely because mutation keeps injecting
  variance at a higher absolute rate now that far more reproduction events occur per episode.
- Peak populations stabilized cleanly from Q2 onward (~26 predators, ~46-49 prey) with no
  divergence — the coexistence itself is stable even though episode-completion rate is lower
  than Fix 1's.
- Prey live-genome MR shows a smaller, familiar-shaped bump (0.952 → 0.995 peak in Q3 → 0.984),
  similar in character to the baseline run's prey cycle.

### An open puzzle — resolved as a metric-definition mismatch, not a real contradiction

The selection gap for predators was *positive* in Q1-Q4 (+0.005 to +0.010) and
`predator_mr_repro_spearman` was also weakly *positive* from Q2 onward (+0.007 to +0.031) — both
signals superficially pointing toward selection favoring *higher* MR, while the population mean
MR fell substantially over the same stretch. Reading the actual metric-computation code
resolved this:

- `live_genome/predator_metabolic_rate_mean` is built from `self.agents` (the currently-alive
  population) snapshotted every step and reduced over the episode — a **lifespan-weighted**
  average (an agent that lives 500 steps contributes 500× more than one that lives 1 step).
- `eco_evolution/predator_metabolic_rate_mean` is built from `_iter_all_agent_records()`, which
  chains `agent_stats_live` and `agent_stats_completed` — literally every agent ever born in the
  episode, each counted **exactly once** regardless of how long it lived. This is *not*
  "survivors at episode end" as originally assumed when this section was first written — it's an
  unweighted per-individual headcount average.
- `predator_mr_repro_spearman` correlates genome MR against a **binary** "did this agent ever
  reproduce" flag, not reproduction rate or lifespan.

None of these three decompose cleanly into "why did the population mean shift." The "gap" was
comparing two different weighting schemes of the same population, not survivors against
non-survivors — so a positive gap never actually implied the population should trend up. The
apparent contradiction dissolves once the mismatch is recognized; it does not, however, explain
*why* predator MR declined — that remains a real, unresolved empirical fact, addressed by the
neutral-control experiment below.

**Verdict:** meaningfully more biologically grounded (population regulation is now emergent
from individual starvation risk, not a census rule) and it delivered the missing predator
genome cycle — but sustainability, while much improved over baseline, hasn't caught up to the
ratio cap's episode-completion rate within the same 1000 iterations, and whether the observed
drift is real selection (vs. neutral genetic drift) was still untested at this point.

---

## Fix 2, replication — 2000-iteration run, second seed (Iteration 3 detail)

**Run: `PPO_ECO_EVOLUTION_METABOLIC_RATE_2026-07-07_09-54-25`, 2000 iterations, same config as
Fix 2 (satiation cooldown 8, per-catch cap 8.0, ratio cap disabled).** `max_iters` was doubled
to test two things: whether episode-completion rate keeps closing the gap with Fix 1 given more
training time, and whether the predator genome cycle from the first Fix-2 run reproduces on an
independent seed or was a one-off trajectory.

| Quintile | Ep. len (of 2000) | % full-length | Peak pred | Peak prey | Predator live MR | Prey live MR |
|---|---|---|---|---|---|---|
| Q1 (4-402) | 601 | 12.0% | 25.2 | 41.8 | 1.018 | 0.991 |
| Q2 (403-801) | 917 | 51.4% | 25.3 | 41.2 | 1.016 | 0.959 |
| Q3 (802-1200) | 945 | 50.6% | 25.1 | 43.0 | 1.005 | 0.969 |
| Q4 (1201-1599) | 993 | 58.9% | 24.7 | 44.8 | 1.005 | 0.963 |
| Q5 (1600-2000) | 992 | **62.1%** | 24.2 | 47.6 | **0.992** | 0.958 |

Overall: 1997/2000 iterations valid (99.85%).

**Sustainability: real improvement from the extra training, but plateauing rather than closing
the gap with Fix 1.** Full-length fraction climbed from 12.0% to 62.1% — clearly ahead of the
first Fix-2 run's final 45.3% — and episode length hit ~992-993 by Q4-Q5, close to saturating.
But the rate of improvement decelerates in the second half (Q3→Q4→Q5: 50.6%→58.9%→62.1%), and
doubling the training budget did not double the completion rate. This suggests satiation+cap
alone likely has a real ceiling somewhere in the 60-70% range for this configuration, rather
than eventually converging on Fix 1's ~98%+ given unlimited time.

**Genome drift: reproduces directionally on a second seed, but smaller and noisier than the
first run.** An earlier read of this run at iteration 1648 (mid-training) looked like the
predator genome was "just oscillating" in a 1.00-1.03 band with no net direction — a real
concern, since it would have meant the first run's cycle was a one-off. The completed run
changes that read: **Q5 shows a genuine decline to 0.992**, meaning there's a real net
downward trend across the full run (1.018 → 0.992, ≈2.6% down from its own early peak), just
substantially smaller and noisier than the first Fix-2 run's clean drop to 0.954 (≈4.6%
decline). Two independent seeds now show the same *direction* — predator MR trending down over
training — even though magnitude and path differ. That is better evidence than a single run,
but it is not a clean, tightly-reproducing cycle either.

**Verdict:** this replication upgrades "predator genome drift" from "one observed trajectory"
to "a direction that recurs across two independent seeds, with inconsistent magnitude" — real
progress, but not yet strong enough to call the Darwin/Baldwin loop statistically established.
The recurring-but-variable nature of both the drift and the sustainability ceiling is exactly
what motivates the neutral-control experiment below: with two seeds both showing a modest,
noisy decline, the open question is whether a 2.6-4.6% shift is distinguishable from pure
demographic/mutational noise at this population scale, or whether it's coincidental drift that
happens to recur.

---

## Neutral-drift control — isolating selection from pure genetic drift (Iteration 4 detail)

Even a real, reproducible-looking directional shift in population-mean MR is not, on its own,
evidence of selection. In a finite population with many generations turning over within a
single long episode (~140+ unique predators born per episode), pure chance — which specific
individual's genome happens to propagate, compounded through many generations — can produce a
shift indistinguishable in shape from the ones observed above, entirely without any fitness
difference between genome values. This is neutral genetic drift, and it is the standard
alternative hypothesis that has to be ruled out before trusting an observed trend as real
selection.

**Design.** A `genome_neutral_drift_control` config flag severs the link between reproductive
success and genome inheritance while leaving population dynamics untouched:

- The reproducing agent is still whichever predator/prey actually crosses the energy
  threshold — same energy economy, same reproduction rate, same population sizes, same
  ecology as the real run.
- But the **genome template** used to produce the offspring (before mutation) is *not* the
  reproducing agent's own genome — it's sampled uniformly at random from the currently-alive
  same-species population instead.

This keeps every population-level statistic (episode length, peak counts, spawn totals)
comparable to a real run with the same seed and config, while making genome propagation
completely blind to which individual actually reproduced. Any drift observed under this control
is attributable purely to mutation plus finite-population sampling noise — not selection. If
the real run's drift magnitude (2.6-4.6% across the two seeds above) is comparable to what the
neutral control produces, the observed "cycles" are likely coincidental drift, not evidence of
a working Darwin/Baldwin loop. If the real run's drift is substantially larger, that's genuine
evidence of selection.

Implementation: `predpreygrass_rllib_env.py`'s `_inherit_genome`, config key added to
`config_env_eco_evolutionary.py`, dedicated config/tune-script pair
(`config_env_eco_evolutionary_neutral_control.py` /
`tune_ppo_metabolic_rate_neutral_control.py`) so the comparison run only differs from the real
Fix-2 run in this one flag.

**Status:** running. `PPO_ECO_EVOLUTION_METABOLIC_RATE_NEUTRAL_CONTROL_2026-07-08_03-41-51`,
started 2026-07-08, same duration (2000 iterations) and PPO/env config as Fix-2 run 2, only
`genome_neutral_drift_control: True` differs.

### Interim data (iteration 637 of 2000 — not final, do not conclude from this alone)

| Group | Ep. len | % full-length | Peak pred | Peak prey | Predator live MR | Prey live MR |
|---|---|---|---|---|---|---|
| G1 (4-214) | 646 | 14.2% | 22.4 | 49.4 | 0.989 | 0.950 |
| G2 (215-425) | 585 | 9.0% | 26.3 | 42.3 | 0.985 | 0.952 |
| G3 (426-637) | 810 | 34.0% | 25.9 | 43.8 | **0.970** | 0.961 |

**Population dynamics track the real runs closely** (episode length, peak predator/prey counts
all in the same range as Fix-2 run 2 at a comparable early stage) — confirms the control is
doing what it's designed to do: leave population/energy dynamics untouched while only severing
genome inheritance from reproductive success.

**The concerning early signal**: predator MR is already declining here too — 0.989 → 0.985 →
0.970, a ~1.9% drop over the first 637 iterations, under a mechanism with **zero** selection
pressure on genome by construction. This is not yet conclusive (only ~32% of the way through,
and the real runs' trajectories weren't monotonic either), but it is not an encouraging sign for
the "real selection" hypothesis. If the neutral control keeps producing declines of comparable
magnitude to the real runs' 2.6-4.6% by the time it also reaches 2000 iterations, that would mean
the observed predator genome cycle is largely indistinguishable from noise. This section will be
updated with the completed run and the final side-by-side comparison against runs 2 and 3.

---

## Comparison at a glance

| | Baseline (no throttle) | Ratio cap | Satiation throttle (run 1, 1000 iters) | Satiation throttle (run 2, 2000 iters) |
|---|---|---|---|---|
| Valid iterations | 46% | 99.7% | 99.7% | 99.85% |
| Episodes reaching full length | — (not tracked this way) | ~98%+ by Q5 | 45.3% by Q5, still climbing | 62.1% by Q5, decelerating |
| Peak predators / prey | ~16-20 / ~15-22 | ~25-29 / ~40-45 | ~21-27 / ~46-55 | ~24-25 / ~42-48 |
| Predator genome trend | none (flat at equilibrium) | none (muted, 0.97-0.99 band) | real: 1.00 → 0.95 → 0.96 | real but smaller: 1.02 → 0.99 |
| Prey genome trend | clear: 0.99 → 1.03 → 1.00 | muted: 0.98 → 1.00 → 0.99 | similar: 0.95 → 1.00 → 0.98 | similar: 0.99 → 0.96 |
| Mechanism realism | n/a (uncapped, unstable) | population census (not diegetic) | individual hunting/energy history | individual hunting/energy history |

---

## Next steps

1. **Run the neutral-drift control** (running, see Iteration 4) and compare its drift
   magnitude directly against the two real satiation-throttle runs above. This is the priority
   — it determines whether anything above constitutes real evidence of selection.
2. **Track per-capita reproduction rate directly**, binned by MR quartile, separate from the
   existing (binary-outcome) spearman correlation, as a cleaner selection-differential metric
   than the mismatched eco_evolution/live_genome comparison this document originally relied on.
3. **Try coupling satiation cooldown to metabolic rate** (e.g. `cooldown = base / metabolic_rate`,
   faster metabolism → faster digestion) — deliberately deferred across both satiation runs to
   keep each a single-variable change relative to the last. Worth revisiting once the
   neutral-control result is in, since it would add a second genome-linked fitness channel and
   may also help the episode-completion ceiling.
4. **Investigate the episode-completion ceiling directly** (both satiation runs decelerate
   rather than approach Fix 1's ~98%). Options: tune `predator_satiation_cooldown` /
   `max_energy_gain_per_prey`, or accept the lower completion rate as the price of a more
   realistic mechanism.
