# Results — eco_evolutionary_erl_coevolution

Consolidated findings, 2026-09-28 to 2026-09-30. The dated, step-by-step log with
every run, including the dead ends, is in [README.md](README.md). All runs use
ERL prey (evolution + lifetime learning) unless stated. Results live under
`~/simulation_results/erl_results/coevo_*`.

## Summary

Starting from erl_baldwin's §9 result (ERL beats E/L/F/B, p<0.00001), this module
moves step by step toward predator–prey coevolution.

1. **Faithful baseline.** Step 0 reproduces §9 run-for-run (100/100). erl_baldwin's
   own HEAD does not, because later genes shifted its RNG stream (noted in its
   RESULTS.md).
2. **Prey-regulated carnivores coexist robustly, but only in a large enough world.**
   On 150×150 (step-0 densities), 60/63 established populations keep their
   carnivores after immigration stops. On 100×100 only 13/45 do (8/17 for ERL), most
   lost at cycle lows when few carnivores remain.
3. **The ERL advantage holds against prey-regulated carnivores, with one change.**
   ERL beats E, F and B. Against learning-alone (L), prey survival was not detectably
   different in the larger world (5/60 vs. 8/60, p=0.56), but L systems lost their
   carnivores 7/52 times vs. ERL's 0/55 (p=0.005). ERL's observed edge there is lower
   carnivore loss, consistent with greater ecosystem stability. This replicated at
   n=100 in a second setting: after the switch, carnivores died out in 0/170 ERL runs
   vs. 11/166 L runs (p=0.0004), against both evolving and fixed carnivores (§7).
4. **Carnivore evolution: purifying selection, but no better reactive behavior found.**
   Evolving carnivores coexist and recover to the seed rule's level, no further, and
   their seeded traits show no consistent directional change. Competition tests found
   selection purging a worse variant (an exploratory whole-trajectory analysis) and no
   better reactive behavior among the variants tested, even with richer senses. That is
   consistent with the seed rule lying on or near a local fitness peak.
5. **A memory-based strategy has headroom, and a plateau appears to block it.**
   Persistent search beats the seed carnivore (p=1.6e-5, pre-registered). Starting near
   zero with small variation, evolution did not produce a detectable rise, and an
   exploratory lineage analysis estimated a near-zero selection gradient. With wide
   founder variation, the mean rose in 18/20 seeds (p=0.0002), meeting two of three
   pre-registered conditions.
6. **Paper-style lifetime learning did not bridge that plateau in exploratory smokes.**
   ERL carnivores learned persistence slightly *down* in 18/18 smoke seed-runs across
   one-step, baseline, trace and pure-energy variants. That makes the planned Baldwin
   test unpromising as designed. The signal scale and credit horizon suggest learning
   may be too weak and too short-horizon for a search strategy whose payoff comes many
   steps later. The full pre-registered comparison was not run.

Against the project's three criteria: **sustainability** met (2); **Darwin/Baldwin loop**
met for prey (3); **coevolution** not yet. Predators can evolve, and headroom exists
(5), but no arms race has been produced.

## 1. Step 0 — faithful baseline

`config_step0` = erl_baldwin at commit 47534a0 (the §9 study), with the §9-era genome.
It matches §9's extinction steps on 100/100 runs (20 seeds × 5 strategies, 50k-step
budget). At that sample size ERL beats L (p=0.008), F (p=2e-6) and B (p=0.005); ERL vs. E
is p=0.1 (§9 needed n=100 and 1M steps).

## 2. Step 1 — carnivores regulated by prey

In §9, carnivore numbers are set by immigration (one per 200 steps). Step 1 keeps that
for a 20k-step warm-up, then stops it; afterwards carnivores live or die by prey through
their own energy budgets.

- Stopping immigration from the start fails in ~90% of runs. Naive prey die on walls,
  the corpses fuel a carnivore boom, and carnivores then starve. Faster carnivore
  breeding looked better only because newborns received more energy than the parent
  paid (fixed by an energy-conserving birth option; with it, ~1/10).
- **Warm-up + 150×150 world (all densities scaled by area) is the step-1 preset.**
  Carnivore minima after the switch are 18–54, vs. 2–14 on 100×100.

| 150×150, 20 seeds × 60k | prey extinct | alive at switch | coexist to 60k | carnivores die out |
|---|---|---|---|---|
| ERL | 1 | 19 | 19 | 0 |
| E | 10 | 10 | 10 | 0 |
| L | 3 | 17 | 15 | 2 |
| F | 12 | 8 | 7 | 1 |
| B | 11 | 9 | 9 | 0 |

ERL vs. E/F/B on prey extinction: Fisher p=0.003 / 0.0004 / 0.001.

**ERL vs. L at n=60:** prey extinct 5 vs. 8 (p=0.56). Carnivores die out after the
switch 0/55 vs. 7/52 (p=0.005). A founder-count check (60 vs. 135 founders) showed
most of L's improvement goes with the larger world rather than founder count: with the
same 60 founders, L goes from 13/20 prey extinct on 100×100 to 5/20 on 150×150. Which
feature of the larger world matters is unresolved. Every prey
extinction on 150×150 happens in the first ~11k steps, so once established, both prey
types almost always survive the budget. This early-extinction pattern may explain why
prey survival no longer separates ERL from L. The mechanism behind L's lost carnivores is not established.

Predation is only ~13% (100×100) to ~20% (150×150) of prey deaths. Most deaths are
wounds, largely from walking into walls.

## 3. Step 2 — carnivores with a genome (evolution only)

Carnivores act with a single-layer network over 10 inputs (prey signal and blocked, per
direction; energy; health). Founders are seeded to approximate the hand-coded rule
("chase the strongest prey signal, avoid obstacles") plus N(0, 1) noise, with sexual
reproduction like the prey. The neutral control is a neutral marker: identical
inheritance, but the genome isn't expressed.

- **Coexistence holds** (20/20).
- **No directional genome change** in the seeded traits, and no single weight changes
  consistently. Performance recovers from the founder-noise handicap to about the seed
  rule's level (kill rate 14.7 → 16.7 per 1,000 carnivore-steps; neutral 16.4 → 17.1),
  never beyond.
- **Per-capita kill rate is not a valid selection measure here.** It tracks ecological
  state (numerical response), not hunting skill. A matched non-heritable control
  diverged from step 2 within a few hundred steps despite identical founders. That
  control also lost its prey in the opening 8/20 vs. 0/20; the cause is not known, and no
  bug was found.

**Headroom probes.** In monomorphic populations (every carnivore using one fixed behavior),
per-capita carnivore fitness is nearly flat across quite different behaviors. The
differences appear at the population level instead: carnivore numbers 87–149, prey
385–737. Competition tests, where one variant is mixed 50/50 with seed-network
residents, measure within-population selection:

| mutant vs. seed resident | result |
|---|---|
| no obstacle avoidance | purged: 0.5 → 0.09 by the switch, 0 by 40k (p=0.0006 vs. identical) |
| sharper / softer pursuit | neutral |
| richer senses: live-prey or carcass preference, ignoring sheltered prey | neutral or slightly worse |
| wounded / sated scavenger (state-dependent) | worse / tends worse |
| **persistent search** (repeat the last move when nothing is visible) | **better** |

Confirmatory test for persistent search, with a 50/50 split after the warm-up and no
immigration afterwards: net growth rate difference +0.44 vs. +0.04 for the identical
control (p=1.6e-5). It took over in 13/13 completed runs. In 2 of the persistent-search
runs the carnivores later died out (identical: 0); possibly over-exploitation, but too
few cases to say.

## 4. Can evolution find persistence?

With the previous move added as an input (22 inputs), a single-layer network can express
persistence (weight ≈ +3.3 matches the hand-coded rule).

| founders' persistence weights | persist mean, early → late | rises in | selection gradient |
|---|---|---|---|
| seed 0, noise N(0, 1) (step 2b) | +0.04 → +0.14 | 10/20 (neutral 9/18) | ≈ 0 (median −0.001) |
| **N(0, 3)** (step 2c) | **+0.45 → +2.03** | **18/20, p=0.0002** | **positive in 17/20** |

In step 2c, two of the three pre-registered conditions were met. The third, a per-seed
upper-tail test, failed; seeds that converged just below +2 made it a poor indicator.
Together these runs are consistent with a **fitness plateau near zero**: persistence
pays, but small amounts carry no detectable advantage, so small mutations are unlikely
to start the climb. This
is the Hinton & Nowlan (1987) setting in which learning is predicted to help evolution.

## 5. Step 3 — can lifetime learning bridge the plateau?

ERL carnivores: a live network learns each step by the prey's one-step REINFORCE rule on
the change of an evolvable innate evaluation (founders seeded "more energy = good").
Offspring inherit the genome only.

In exploratory smoke runs (6 seeds × 25k each), learned persistence (live − genome) is negative in
18/18 seed-runs across all variants: one-step; + reward baseline; + eligibility trace;
+ both; pure-energy goal. Magnitudes range from −0.003 to −0.075. Scale arithmetic
suggests why (a hypothesis, not tested directly):
- Reinforcement is ~0.03 per step. With a learning rate of 0.05 over ~280-step
  lifetimes, a weight can move at most ~0.1–0.4, far from +3.3.
- Persistence pays off over a whole search (tens to hundreds of steps), which
  one-step or ~10-step credit cannot see. What it does see is that walking costs
  energy.

**With the paper's learning rule, lifetime learning did not increase persistence in these
smokes, so a Baldwin test on this trait is unpromising as designed.** The full step-3
comparison was not run.

## 6. ERL vs. L prey against evolving vs. fixed carnivores (2026-09-30): pilot, inconclusive

2×2, 20 seeds per cell, 60k steps, 150×150, rich_memory carnivores with wide founder
persistence variation. The ERL cells reuse step 2c (replay-verified); the L cells are new
(commit b126e0d). Analysis `analyze_prey_vs_predators.py`, fixed before the run. Results
in `~/simulation_results/erl_results/coevo_prey_vs_pred` and `coevo_step2c`.

| prey × carnivores | prey extinct | carnivores extinct | any collapse |
|---|---|---|---|
| ERL × evolving | 0 | 0 | 0/20 |
| L × evolving | 2 | 1 | 3/20 |
| ERL × fixed | 3 | 0 | 3/20 |
| L × fixed | 3 | 1 | 4/20 |

- **Primary (vs. evolving carnivores): not significant.** Any collapse is 0/20 vs.
  3/20 (Fisher p=0.23); prey extinction 0/20 vs. 2/20 (p=0.49). Against fixed carnivores
  the two are nearly equal (3/20 vs. 4/20). The interaction points the predicted way
  (+0.10) but is far from significant (permutation p=0.74). n=20 is underpowered.
- Most collapses happen in the opening (steps 1,077–3,946), before carnivores have
  evolved much. Only 3 are late (> 19k steps): L × evolving 2, L × fixed 1, ERL 0. Too few
  to test.
- Exploratory: carnivores evolved somewhat less persistence against L prey (late median
  +1.58 vs. +2.04 against ERL prey, p=0.18). n.s.
- Reading: consistent with the prediction that L falls behind against a moving target,
  but not evidence for it. Late collapses are rare (~0–10% per cell), so separating the
  cells would take on the order of 100 seeds per cell.

## 7. ERL vs. L prey against evolving vs. fixed carnivores at n=100 (2026-10-01)

Seeds 21–100 added to all four cells under the same tags and pre-registered analysis
(job finished 2026-10-01 01:55; log `~/simulation_results/erl_results/coevo_prey_vs_pred_n100.log`).

| prey × carnivores | prey extinct | carnivores extinct | any collapse |
|---|---|---|---|
| ERL × evolving | 3 | 0 | **3/100** |
| L × evolving | 10 | 7 | **17/100** |
| ERL × fixed | 27 | 0 | 27/100 |
| L × fixed | 24 | 4 | 28/100 |

**Pre-registered:** primary PASS. Any collapse against evolving carnivores is ERL 3/100
vs. L 17/100 (Fisher p=0.0015). Prey extinction alone: 3 vs. 10 (p=0.08). Against fixed
carnivores: 27 vs. 28 (p=1). Interaction: +0.13, permutation p=0.12, not significant.

**When collapses happen** (exploratory breakdown):

| | before the switch (all prey extinctions) | after the switch (all carnivore extinctions) |
|---|---|---|
| ERL × evolving | 3 | 0 / 97 |
| L × evolving | 10 | 7 / 90 (vs. ERL p=0.005) |
| ERL × fixed | 27 | 0 / 73 |
| L × fixed | 24 | 4 / 76 (vs. ERL p=0.12) |

Reading, hedged:
- The primary pass combines two effects, and neither is specific to *evolving*
  predators. (a) Opening prey extinctions: ERL 3 vs. L 10, p=0.08 n.s. (b)
  Post-switch carnivore loss, which happens only in L systems. It occurs against
  fixed carnivores too (4/76), and pooled it is ERL 0/170 vs. L 11/166, p=0.0004.
  That replicates step 1's finding (L 7/52 vs. ERL 0/55) in a second setting.
- The key prediction, that L falls *further* behind against a moving target, is
  **not supported**. The interaction is n.s. (p=0.12), and L's post-switch carnivore
  loss appears against fixed carnivores as well.
- **Design confound:** "fixed" carnivores are the neutral marker. They all act with the
  noise-free seed network, so from the start they are more competent hunters than the
  noisy evolving founders. That likely explains their many opening prey extinctions
  (24–27 vs. 3–10). Fixed vs. evolving therefore differs in starting competence as
  well as in evolution, which weakens the interaction test.
- So the robust result is the general one: with ERL prey, the predator population
  never died out after the switch in 170 runs, while with L prey it did in 11 of 166.
  Why is still not established.

## 8. Moving-target retest, competence-matched fixed control (2026-10-01): prediction not supported

`step2_nonheritable` control, ERL and L prey × 100 seeds (commit 816db53; job finished
12:31; log `~/simulation_results/erl_results/coevo_matched_fixed.log`). The evolving cells
are reused (n=100).

| prey × carnivores | before the switch (prey extinct) | after the switch (carnivores extinct) | any collapse |
|---|---|---|---|
| ERL × evolving | 3 | 0/97 | 3/100 |
| L × evolving | 10 | 7/90 | 17/100 |
| ERL × matched-fixed | 40 | 0/60 | 40/100 |
| L × matched-fixed | 37 | 5/63 | 42/100 |

- **Primary (pre-registered): FAIL.** Interaction on any collapse: +0.12, permutation
  p=0.18. Post-switch only: +0.02, p=0.77. With two different fixed-control designs, L
  does not fall detectably further behind against evolving carnivores. **The
  moving-target prediction is not supported.**
- **Robust in every setting tested:** after the switch, carnivores die out only with L
  prey. Pooled over the three carnivore settings run at n=100 (evolving, neutral-marker
  fixed, matched-fixed), that is ERL 0/230 vs. L 16/229 (Fisher p≈1e-5). Step 1
  (hand-coded-rule carnivores) adds 0/55 vs. 7/52. Mechanism unknown.
- **New, strong and unexplained: heritable carnivores do far less damage in the opening.**
  Opening prey extinction is 3/100 vs. 40/100 (ERL prey) and 10/100 vs. 37/100 (L prey)
  for evolving vs. matched non-heritable carnivores. Both are p < 1e-6 (exploratory). The
  founders are identical (verified), so the difference arises from inheritance during
  the first ~1–4k steps. This replicates step 2's unexplained 0/20 vs. 8/20 at n=100.
  Candidate explanations, not tested: selection removing over-exploiting carnivore
  lineages early, which would be ecologically interesting; reduced phenotypic diversity
  as a few lineages take over; or both. This also means the "matched" control is matched
  in starting competence but not in opening dynamics, which loads the any-collapse
  interaction with opening noise. The post-switch interaction avoids that and is also null.

## Corrections made along the way

Each was caught before it became a conclusion; details are in README.md.

- "6/6 coexist" after the warm-up came from runs with 10 initial carnivores instead of 5.
- Fast carnivore breeding "worked" only because births created energy.
- The first neutral control leaked viability selection. It drew donor genomes from
  living carnivores; Codex review caught this.
- The step-2 kill-rate "pass" compared against a control that started ahead.
- The competition test's pre-registered window (after the switch) missed selection
  that happened during the warm-up. It also has low power, because of the opening
  bottleneck and immigration reset. Both were fixed by the delayed-split design.

## Open questions

- **A stronger learner** (long eligibility trace or full-return REINFORCE, larger
  learning rate): can a capable learner discover persistence and produce a Baldwin
  effect? This departs from Ackley & Littman.
- **The ERL-vs-L gap against evolving predators** (§7). Tested at n=100 with carnivores
  evolving persistence from wide founder variation. ERL's advantage is there (collapse
  3/100 vs. 17/100), but it is not detectably larger than against fixed carnivores,
  with either control design (interaction p=0.12 neutral marker; p=0.18, post-switch
  p=0.77 matched non-heritable, §8). Not supported.
- **Why heritable carnivores are so much milder in the opening** (§8: 3/100 vs. 40/100
  prey extinction). Over-exploiting lineages being selected out early? Testable from
  carnivore lineage data (kill rates of lineages that die early vs. survive).
- Why L systems lose their carnivores, and why the non-heritable control lost its prey
  in the opening.
- An arms race (step 4) needs both sides to keep adapting; not reached.
