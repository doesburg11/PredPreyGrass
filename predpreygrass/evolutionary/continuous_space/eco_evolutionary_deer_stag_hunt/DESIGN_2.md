# Design note 2: nature, nurture and both, with hunters and deer starting together (2026-11-10)

Status: **draft 2 approved 2026-11-10 with every recommended option (decisions 1-5
below); pre-registered in RESULTS.md section 2.** Draft 1 was revised after a Codex review ("Revisions after review", end).
This is a **new stage**, prompted by a new user decision after section 1's stop: it
gets its own pre-registration (RESULTS.md section 2) and does not continue or rerun
section 1. Everything not named here stays as in DESIGN.md (approved 2026-11-08).

## Why a redesign

The first design's pilot failed (RESULTS.md section 1.5): hunters died out in 50 of
50 evaluated episodes in every arm, E, L and B alike. Their reward, births, stayed
almost zero for all 500 iterations, so PPO had nothing to learn from and genes had
almost no selection.

The likely cause is the deer, not the genes. The first design froze deer_stag_hunt's
*final* deer from the first episode. In deer_stag_hunt, hunters and deer started
naive together: early deer were easy, early hunters caught and bred, and that reward
let PPO learn. Here naive hunters met trained deer and starved. Arm L, with the genes
switched off and a world equal to deer_stag_hunt's, failed the same way, which points
at the deer, though no run has isolated it.

So the deer must start as weak as the hunters and get better as the hunters do,
while staying the same across arms, so that a difference between arms is the
hunters'.

## The change: deer replayed from a deer_stag_hunt run

The deer in run k of every arm follow **the deer of one deer_stag_hunt run, in the
order they were trained**. They never learn here; only the schedule moves on.

**The schedule, exactly.** Training iteration t (t = 1 ... 1000) collects its samples
with the deer of checkpoint D(t):

- D(t) = iter_000010 for t = 1 ... 19;
- D(t) = iter_(10 · floor(t / 10)) for t = 20 ... 999;
- D(1000) = the run's final checkpoint (iteration 1000).

So the deer change before sampling iterations 20, 30, ..., 990 and 1000, in the
learner and every env runner at once (PPO samples synchronously, so one iteration's
batch never mixes deer). An episode running across a switch goes on with the new deer,
as it goes on with updated hunter weights.

**Evaluation:** a checkpoint saved after iteration t is evaluated against D(t), the
deer its last batch was collected with, using the same episode seeds (0-49) in every
arm.

**What this keeps:** the opening that let deer_stag_hunt's hunters learn, easy deer
first; deer matched across arms; no deer adapting differently to L and B hunters.

**What it gives up:** the deer do not respond to these hunters. Each deer policy was
trained against deer_stag_hunt's hunters at that point, so its difficulty is
inherited from that history, not calibrated, and need not rise steadily. If hunters
here learn faster, the deer may be too weak and be eaten out; if slower, too strong,
as in the failed pilot. This is a replayed schedule of opponents, not coevolution. A
B-vs-L result is conditional on these schedules.

**Alternatives considered:**

- **Deer learning in every arm, from scratch.** Real coevolution, but deer would adapt
  to each arm's hunters, so a B-vs-L difference could come from the deer as much as
  from the hunters. Suits a later stage.
- **Hunters warm-started from deer_stag_hunt's trained hunters.** Rejected: learning
  would have little left to do, and that network lacks the gene inputs.

## The world: the touch rule

The touch rule (deer_stag_hunt RESULTS.md section 2) is now the better-understood
world: 50 of 50 coexisting in all three runs, adult approaches with a partner near
above chance, and pairing on large middle deer that a control world ties to the
payoff (sections 2.1 and 5). Grass stays instant.

## Runs: 6 per arm, 2 per deer schedule

Only three touch runs exist, so there are three deer schedules. Runs 1-6 of each arm
use touch_1, touch_2, touch_3, touch_1, touch_2, touch_3 (two runs per schedule, with
seeds 1-6). Every arm's run k has the same schedule and seed, so runs are **paired
across arms**.

## Arm E: reported, not tested

Arm E switches deer by physics steps, not by PPO iterations, so it cannot meet the
same deer for the same stretch of its training as L and B: E has no PPO iterations,
and its episodes and generations run at another pace. A B-vs-E test would compare two
different curricula as much as nature against both. E therefore stays in, as in
DESIGN.md, but **its results are reported, not tested**:

- each of E's 14 environments switches to D(t) when its own physics steps pass 1/14
  of the median total physics steps of the L and B runs of the same schedule at
  iteration t, read from their saved budget records. E runs after the L and B runs of
  its schedule, and the threshold table is saved with it;
- E's checkpoints are taken at the same matched totals as in DESIGN.md (250, 500, 750,
  1000) and evaluated against D(t).

## Measures and tests

**Primary estimand: performance against the schedule**, not general hunter quality.
Each run value is the mean over checkpoints 500, 750 and 1000, each evaluated against
its own deer D(t) (DESIGN.md's measures A, S and C otherwise unchanged).

**Tests** (set before any main run):

1. **B beats L on A and on S**, paired: per run k, the difference B_k − L_k; one-sided
   sign-flip permutation test on the mean difference over the 6 pairs (64
   arrangements), p < 0.05, for A and for S. Both must hold. (p < 0.05 allows at most
   one small negative difference.)
2. **B sustains coexistence:** in at least 5 of 6 B runs, hunters and deer are both
   alive at the end of at least 45 of 50 episodes, averaged over the three
   checkpoints.

**Verdict:** if both are met: "in this world, against a replayed schedule of deer, an
innate attraction bias that evolves plus shared PPO learning catches more adults, and
more per attempt, than learning alone." Inference is conditional on the three deer
schedules.

**Also reported:**

- **Against fixed deer:** every L and B final checkpoint (iteration 1000) against each
  of the three final deer (touch_1-3, iteration 1000), 50 episodes, seeds 0-49: a
  common panel independent of the schedule, as a check on general hunter quality.
- **Per schedule:** B − L for the two runs of each schedule.
- E's values on every measure, beside L's and B's.
- Everything else DESIGN.md reports: per-run values, bootstrap intervals, C, gene
  means, selection differentials against L's unexpressed genes, checkpoints 250 and
  500.

## Pilot, and the hard stop

- **Pilot runs:** L on all three schedules and B on touch_1 (runs pilot_L1, pilot_L2,
  pilot_L3, pilot_B1), 500 iterations, two at a time; E on touch_1 on the CPU
  alongside. Pilot runs enter no test.
- **Pilot passes if,** in every one of the four L and B pilot runs:
  1. nothing crashes;
  2. at iterations 250 and 500, hunters and deer are both alive at the end of at least
     40 of 50 evaluated episodes;
  3. at iteration 500, hunter births per evaluated episode are at least 5
     (deer_stag_hunt touch_1-3 at iteration 500: 18-22).
- **E in the pilot:** reported only. If E's hunters die out or barely breed, E is
  still run and reported, with that stated.
- **Reference, no test:** deer_stag_hunt touch_1-3 at iterations 250 and 500 coexisted
  in 50 of 50 episodes.
- **If the pilot fails:** this stage stops. Nothing is tuned or re-run: no change to
  β, the schedule, the schedule-to-run allocation, D(t), the thresholds, or any world
  setting, and no run is excluded. A new design would be a new user decision.
- **If it passes:** the 12 main L and B runs and 6 E runs, then the stop, whatever the
  outcome. A run that crashes is rerun from scratch with the same seed, and the crash
  recorded; no other rerun.

## Implementation sketch (after approval)

- `environment.py`: this module's world was copied from deer_stag_hunt before the
  touch rule existed, so the rule is ported (`deer_catch_rule: "touch"`). The
  equivalence test against deer_stag_hunt is extended to the touch rule (same seed
  and actions, identical episodes, genes unexpressed).
- **Schedule copy and checks:** before training, the deer module of every scheduled
  checkpoint is copied into the run with a manifest (source path, iteration, SHA-256,
  source run's settings). Each source is checked: touch rule, instant grass, the
  module's world settings otherwise equal, deer observation and action shapes.
- `train.py`: D(t) loaded into learner and every env runner before sampling iteration
  t; a check after each switch that all copies equal the manifest's weights, and that
  only deer weights changed (hunter weights and optimizer state untouched). The
  active deer is recorded per iteration and in every saved checkpoint. No resume.
- `evolve.py` (E): per-environment switching from the saved threshold table; several
  thresholds crossed in one step are handled; the active deer recorded per episode.
- `launchers/evaluate_run.py`: evaluation of a checkpoint against D(t), and the fixed
  panel; both deer recorded in the results.
- **Tests:** D(t) at t = 1, 9, 10, 19, 20, 21, 990, 999, 1000; weights equal after each
  switch and unchanged between switches; hunter weights unchanged by a switch; E's
  mapping, including several thresholds in one step; manifest checks refuse a
  wrong-world checkpoint; evaluation uses the intended deer.
- Codex reviews the code before any run.

## Cost

- **Pilot:** 4 GPU runs of 500 iterations, two at a time, about 4 h, plus E and
  evaluations: about half a day.
- **Main runs:** 12 GPU runs of 1000 iterations, two at a time, about 25-30 h, plus E
  on the CPU and the evaluations.

## Decisions for the user

1. **Deer replayed from deer_stag_hunt's touch runs** (recommended), or deer learning
   in every arm (coevolution, harder to attribute).
2. **The touch world** (recommended), or the original party world.
3. **6 runs per arm, 2 per schedule, paired test** (recommended), or 5 runs with
   DESIGN.md's unpaired test (schedules then unbalanced, runs not independent).
4. **Arm E reported, not tested** (recommended), or kept as test 2 of DESIGN.md
   despite the curriculum mismatch.
5. **Pilot bar:** coexistence at least 40 of 50 at 250 and 500 and at least 5 hunter
   births per episode, in L on all three schedules and B on one (recommended).

## Revisions after review (Codex, 2026-11-10)

- **Runs not independent:** draft 1 had 5 runs over 3 schedules, tested as 5
  independent runs. Now 6 runs, 2 per schedule, and a paired test across arms;
  inference stated as conditional on the schedules.
- **E not matched on deer:** B-vs-E is no longer a test; E is reported. Its switching
  is defined per environment from a saved table.
- **Estimand:** now named as performance against the schedule, with a fixed-deer panel
  reported beside it.
- **Pilot:** raised from 25 to 40 of 50, checked at 250 and 500, with a births floor,
  and run on all three schedules (L) instead of one.
- **Switch timing, provenance, checks:** D(t) defined exactly; manifest with hashes;
  tests at the boundaries.
- **Verdict wording:** "against a replayed schedule of deer", not "fixed deer".
- **Hard stop:** what cannot change is listed; this stage is labelled new, not a
  continuation of section 1.
