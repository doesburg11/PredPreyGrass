# Training Analysis — eco_evolutionary_erl_baldwin

**Status (2026-08-10, latest): World AL rebuild (§6) plus a full 500-run comparative study
(§7) plus a two-stage parameter retune (§8) after that study's null result was root-caused
to an agent-population boom-bust, not the ERL mechanism. A second full 500-run study is in
progress on the retuned config.** See `README.md` for exactly what's matched vs. adapted
from the paper. §1-5 below describe the superseded, simpler-ecology version (pre-2026-08-09)
and should not be read as describing the current codebase.

---

## 6. World AL rebuild (2026-08-09)

**Why:** the comparative study in §5 below (run on the old, simpler ecology) reproduced the
paper's headline result (ERL beats luck) but not its internal ranking (E vs. L), and scaling
up the sample size and step budget didn't resolve it within the time tried. Before sinking
more compute into scale alone, the world was rebuilt to remove world/mechanics as a
confound -- see README.md's "World AL rebuild" section for the full list of what's now
matched (including every exact number the paper actually publishes: grid size, sense
ranges, carnivore spawn interval, min_plants) vs. still necessarily chosen by me (the paper
never publishes damage/threshold/growth-rate constants).

**Structural correction, not just cosmetic:** the old world had two adaptive species
(predator + prey, both genome+learning). The paper has exactly one adaptive species
("agents") plus a separate, permanently non-adaptive species ("carnivores", hard-coded FSA,
never affected by `strategy`). This is now correctly reflected -- `Carnivore` has no genome
field at all (see `test_carnivores_have_no_genome_or_learning`).

**Status:**
- 18/18 unit tests passing (up from 12; new tests cover the Agent/Carnivore split and
  updated strategy-comparison mechanics against the new reproduction method names).
- One full-scale smoke run (`--seed 1 --steps 20000`, 100×100 grid, default population):
  ran 365 steps before agent-population extinction (carnivores overwhelmed a population that
  had itself boomed rapidly -- final state 0 agents, 351 carnivores). Confirms the mechanics
  run without crashing; this specific early extinction is not evidence of anything beyond
  "my chosen, unpublished-by-the-paper constants produce a fast boom-bust here," same
  caveat as every other first-pass parameterization in this project.
- **Performance dropped substantially**: ~30 steps/sec on this machine (100×100 grid,
  carnivores, trees, walls, corpses), down from ~240 steps/sec on the old, simpler ecology.
  Reaching the paper's 1,000,000-step comparative-study ceiling is now estimated at **~9
  hours per seed** that survives that long -- worth confirming scope again before launching
  another 500-run study, since the earlier ~90-min-per-seed estimate no longer applies.

## 7. First full-scale comparative study on the rebuilt World AL (2026-08-09/10) — clean null, root-caused

Full paper-scale study: 5 conditions × 100 seeds × 1,000,000-step ceiling, 24-way
parallel. All 500 runs completed.

| strategy | median | mean | max | reached 1M |
|---|---|---|---|---|
| ERL | 371 | 408 | 2,004 | 0 |
| E | 380 | 513 | 12,285 | 0 |
| L | 384 | 1,344 | 95,065 | 0 |
| F | 391 | 419 | 1,915 | 0 |
| B | 393 | 857 | 17,172 | 0 |

Every median sits in a 371-393 band. **No pairwise comparison was significant** (Mann-
Whitney, n=100 each) -- not even ERL vs. B (p=0.056), the one result that had survived
every prior version of this study. Not a single one of 500 runs reached even 10% of the
step ceiling.

**Diagnosis:** direct population trace (seed 1, ERL) showed the founder population of 60
agents exploding to 1,916 by step 120 -- fed by abundant plants and an easy reproduction
threshold -- which then fed an equally explosive carnivore boom (6→739 by step 280),
collapsing the agent population entirely by ~step 400. Every strategy dies at the same
rate because death arrives via this boom-bust cycle before behavioral differences have any
time window to matter, not because learning/evolution don't work. A structural/parameter
problem in my own necessarily-guessed constants (see config.py's limitation notice), not a
finding about the ERL mechanism.

## 8. Two-stage retune (2026-08-10)

**Attempt 1 (didn't work):** raised `carnivore_reproduction_energy_threshold` (14→18) and
`carnivore_reproduction_energy_cost` (7→10), reasoning that carnivore population growth was
the runaway variable. Validation (5 seeds, 50k-step budget) showed no improvement --
extinction still at 369-534 steps, carnivore populations still exploding to 300+.

**Root cause, found by tracing the actual population dynamics:** the trigger wasn't
carnivore reproduction, it was agent reproduction being too easy. At
`reproduction_energy_threshold_agent=10.0` (only ~2 plant bites above the starting energy
of 5.0) against abundant plants (up to ~2,700 on this grid), the agent population had no
real ceiling before it overshot the environment's actual carrying capacity -- and that
overshoot is what fed the carnivore boom, not carnivore-side parameters.

**Attempt 2 (fixed it):** raised `max_energy_agent` (15→30) and
`reproduction_energy_threshold_agent` (10→22) and `reproduction_energy_cost_agent` (5→10)
-- directly targeting the trigger. Re-traced the same seed: agent population now oscillates
in a sustained 100-700 range (rather than overshooting to ~1,900) and was still healthy at
step 3,000, roughly 7.5x longer than the previous full collapse point. Quick multi-seed
check (6 seeds) confirmed the same pattern -- all 6 outlasted a 90-second screening cutoff
that previously killed every seed within it.

**A second full 500-run study was launched on the retuned config** (same 5×100×1M design as
§7) once this validation held up. Not yet analyzed -- see top of this file for current
status.

## 5. Sections below (§1-5): results from the SUPERSEDED simpler-ecology world

Everything from here down describes the version of this module before the 2026-08-09
rebuild above -- kept for the record, not because it describes current behavior. In
particular, the 5-strategy comparative study below (ERL/E/L/F/B) was run on the old
predator-prey-grass world, not the rebuilt World AL, and would need to be re-run on the
current codebase to say anything about the rebuilt version.

## 1. What's been done so far

- **Implementation.** `genome.py` (per-agent genome, mutation, crossover), `networks.py`
  (evaluation/action network forward pass, local REINFORCE-style reinforcement update),
  `world.py` (self-contained predator-prey-grass simulator, no RLlib/PPO/Ray),
  `metrics.py` (functional-constraint tracker + CSV logger), `run_erl_simulation.py`
  (CLI entry point).
- **Unit tests.** 12/12 passing, including the critical correctness property flagged during
  design: `test_offspring_genome_does_not_inherit_parents_learned_weights` directly asserts
  that an agent's learned, post-lifetime action-network weights are never copied into
  offspring -- only the untouched genome record is (Darwinian, not Lamarckian; see README
  for why this matters).
- **Smoke run** (2026-08-09, seed=41, `--steps 20000 --log-every 1000
  --constraint-window 2000`): ran 9,830 steps at ~241 steps/sec (40.7s wall time) before
  predator extinction. No crashes; population dynamics show real predator-prey oscillation
  (predator count: 75→86→42→87→15→34→46→19→61 across the run; prey: 202→28→44→161→26→28→17→76→9
  at the same checkpoints) rather than a degenerate/frozen population. Extinction this early
  is expected and consistent with Ackley & Littman's own observation that "most initial agent
  populations die out quite quickly" -- not evidence of a bug.
- Genome-level stats (`{species}_eval_weight_absmean`, `_action_weight_absmean`) stayed
  roughly flat over the run (predator eval ~0.40-0.43, action ~0.38-0.40; prey eval
  ~0.34-0.38, action ~0.39-0.40) -- expected at this timescale; the paper's own clearest
  genetic-assimilation results took ~3 million steps to appear.
- Functional-constraint rates (`{species}_eval_site_change_rate` /
  `_action_site_change_rate`) were logged successfully and are in a sane range
  (predator: eval ~0.004-0.007, action ~0.005-0.007, no separation yet; prey: eval rate
  dropped from ~0.05 to ~0.007 over the run, action stayed ~0.007-0.009) -- too early and
  too few reproduction events (population maxed around 87) to read a genetic-assimilation
  signature into this; needs a much longer run.

## 2. Performance fixes (found while screening seeds)

The smoke run's ~240 steps/sec turned out not to hold once population grew past ~100
agents. Profiling a slow seed found two O(agents²)-per-step hotspots, both fixed
2026-08-09:
- `_observe` rebuilt the full prey/predator position sets from scratch for *every
  individual agent's* observation each step. Fixed: built once per step instead (a
  documented simplification -- agents now sense positions as of the start of the step,
  not a live view updated by earlier-acting agents within the same step; eating/death
  still use fully live, current positions regardless).
- `_try_eat` linearly scanned the *entire* agent list for every predator, every step, to
  check for a co-located prey. Fixed: an O(1) per-step `row,col -> prey` dict, kept in
  sync as prey move or get eaten during the step.
- Profiling also found `rng.choice`'s generic validation overhead dominated
  `sample_action` at this (tiny, 5-action) scale. Replaced with a direct
  cumulative-probability draw.
- Net effect: ~2.3x speedup on a representative seed (232 vs. ~100 steps/sec at a
  ~230-agent population size). All 12 unit tests still pass unchanged.

## 3. Survival screen (15 seeds, up to 100k steps each, 2026-08-09)

Following the paper's own comparative-study spirit (run many random initial
populations, most die quickly, a minority survive far longer) rather than hand-tuning
parameters to force artificial stability:

| seed | extinction step | which species died |
|---|---|---|
| 3, 9 | 104, 91 | predator (near-instant) |
| 7, 11 | 722, 389 | prey / predator |
| 4, 5, 6, 13 | 2134, 1750, 2369, 1659 | mixed |
| 8, 12, 14, 15 | 2466, 4711, 3462, 3633 | mixed |
| 2, 10 | 12877, 12682 | predator (prey overran) |
| **1** | **not yet extinct** | still running |

Median extinction ~2,400 steps; two seeds (2, 10) reached ~12,800; **seed 1 is a clear
outlier**, surviving past the 90s screening cutoff twice. This distribution (most die
young, a minority survive far longer) is qualitatively the same shape the paper itself
reports (only ~18% of *their* 100 random populations reached even 10,000 steps) --
read as consistent with the mechanism working as expected, not as evidence of a bug
worth re-tuning away.

## 4. Long run (seed 1) — completed, not the survivor it looked like

Seed 1's extended run (target 1,000,000 steps) terminated by prey extinction at step
45,107 -- far short of the target, and short of any timescale where a genetic-assimilation
signature would be expected to appear. Population trace (published as an artifact during
this work) showed real oscillation for ~40,000 steps, then a sharp, fast collapse in the
final ~2,000 -- diagnosed at the time as *not* a growing-amplitude instability (checked
against the full trace, no clean amplitude trend), more consistent with ordinary
demographic stochasticity hitting the zero boundary while population counts happened to be
lower than typical. No clean, actionable fix identified; treated as expected rarity of
long-term survival, consistent with the paper's own low success rate.

## 4b. Five-strategy comparative study (ERL vs. E vs. L vs. F vs. B)

The paper's actual headline result is this comparison, not any single population's
survival -- run three times, each fixing a real methodological problem found in the
previous pass.

**Pass 1** (15 seeds/condition, 20,000-step cap): ERL significantly beat L, F, and B
(Mann-Whitney p ≤ 0.002 each) but *not* E (p=0.47). Internal ranking was the reverse of the
paper's: E beat L significantly (p=0.0005), and L was statistically indistinguishable from
pure luck (p=0.51) -- opposite of the paper's finding that learning-alone was their
second-best strategy.

**Root cause found and fixed:** strategy L's "no evolution" was implemented as fully
independent random genome resampling at every birth (zero heritability) -- stricter than
the paper's own description ("L can never move beyond the randomly generated evaluation
functions found in the *initial* populations", implying cloning without mutation, i.e.
inheritance still happens). Fixed: L/F now clone the parent's genome exactly (no mutation,
no crossover) instead of resampling.

**Pass 2** (L/F re-run, same 15 seeds, corrected cloning): median survival rose sharply for
both (L: 148→1,014; F: 293→551). New ranking: ERL (2,369) > E (1,538) > L (1,014) > F (551)
> B (160) -- monotonic, intuitive, and much closer to the paper's picture. ERL vs. L became
not significant (p=0.33, matching the paper's own finding that ERL and L track closely for
a long stretch); E vs. L became a statistical tie (p=0.73, softer than the paper's
significant L-beats-E finding but no longer a reversal). One clean remaining discrepancy:
F significantly beat B (p=0.008) where the paper found F doing *worse* than B -- plausibly
because F's cloned-but-frozen action network lets a luckily-decent founder policy persist
and spread via ordinary reproduction, which Brownian (re-randomized every single step) has
no way to do.

**Pass 3, attempted then abandoned for cause:** scaled to 20 seeds/condition (ERL/E/L only)
at a 300,000-step budget, to reach closer to the paper's own timescale (they report the
ERL-vs-L separation only appears past ~500,000 steps) with more statistical power (paper
used 100 seeds/condition, not 15-20). Killed partway through at the user's request in favor
of rebuilding the world itself (§6 above) rather than continuing to scale up a world never
calibrated against the paper's actual mechanics -- see §6 for why.

**A full-scale attempt (5 conditions × 100 seeds × 1,000,000-step ceiling, matching the
paper's actual sample size and ceiling) was launched and then explicitly killed** when the
user asked for the world rebuild instead, on the reasoning that scale alone couldn't resolve
whether the remaining discrepancies (E vs. L, F vs. B) were a power problem or a
world/mechanics problem. That question is what §6 exists to answer.

## 5. Still not done (on the rebuilt World AL)

- **No comparative study run yet on the rebuilt world at all.** Everything in §4/4b above is
  from the superseded simpler ecology.
- Reading the functional-constraint signature on a real long run of the rebuilt world: does
  `action_site_change_rate` drop below `eval_site_change_rate` over generations?
- Tuning the rebuilt world's chosen-by-me constants (damage, thresholds, growth rates) --
  the one smoke run went extinct in 365 steps, which says more about first-pass parameter
  balance than about the mechanism.
- Given the ~9-hour-per-long-seed cost estimate (§6), a full 500-run study at the paper's
  exact scale needs an explicit go-ahead on scope/time before launching again.
