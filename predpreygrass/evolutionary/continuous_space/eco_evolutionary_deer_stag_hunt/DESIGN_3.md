# Design note 3: nature and nurture with deer learning alongside (2026-11-10)

Status: **draft 2, awaiting the user's decisions (end). Nothing built.** Draft 1 was
revised after a Codex review ("Revisions after review", end).
This is a **new stage**, prompted by a user decision after section 2's stop. It gets
its own pre-registration (RESULTS.md section 3) and does not continue or rerun
sections 1 or 2. Everything not named here stays as in DESIGN.md (approved
2026-11-08): the four genes, β₀ ≈ 3.44, the bias inside the RLModule, the touch
world of DESIGN_2.md, measures A, S and C.

## Why another design

Both earlier pilots failed for the same reason, in opposite directions: the deer
could not respond to these hunters.

- **Section 1 (RESULTS.md §1.5):** deer frozen at deer_stag_hunt's final policy.
  Naive hunters starved; hunters extinct in 50 of 50 episodes in every arm.
- **Section 2 (RESULTS.md §2.5):** deer replayed in their training order. Hunters
  bred, but ate the deer out (coexisting 15-38 of 50, bar 40), catching 3-6 times as
  many adults as deer_stag_hunt's hunters against the same deer checkpoints.

In deer_stag_hunt, where hunters and deer learned together from scratch, all three
touch runs coexisted in 50 of 50 episodes. The likely reading (not tested) is that
the deer adapting to the hunters keeps that balance. So this design goes back to
deer_stag_hunt as it is and adds only the genes.

## The change: deer learn in every run

- **Deer:** a fresh deer policy in every run, trained by PPO alongside the hunters,
  exactly as in deer_stag_hunt (same network, same settings, both policies in
  `policies_to_train`). No deer are copied from anywhere.
- **World:** the touch rule, instant grass, `deer_young_solo` "graded": deer_stag_hunt's
  touch runs. Shown equal to deer_stag_hunt for the same actions with the genes
  unexpressed (tests/test_equivalence.py).
- **Hunters:** as before. L: genes inherited and mutated, visible to the network, not
  expressed (β = 0). B: genes expressed through the bias (β = β₀).
- **So arm L is close to deer_stag_hunt's touch runs**, apart from the four gene
  inputs and the seed. That is what gives this design a better chance of coexisting.

**What this gives up:** deer adapt to each run's hunters, so a B-vs-L difference in
a run's own world mixes hunter quality with deer quality. B's deer could become
weaker, or stronger, because of B's hunters. This is coevolution, and the result in a
run's own world is a whole-system result. The cross-play panel below separates the
hunters' share.

## Arm E: dropped

E (genes alone, no learning, fixed cruise rule) would face deer that learn against
it, while its own behaviour can change only by selection across generations. Its
hunters would most likely be outlearned, and E's result would say little about the
B-vs-L question. Section 2's E (against replayed deer) coexisted but caught few
adults (1.4 per episode at iteration 500). E is not run.

## Runs

- **6 runs per arm, 12 in all,** 1000 iterations, seeds 1-6. Seed k is used for run k
  of both arms, for bookkeeping. Both policies learn and the runs diverge from the
  first iteration, so the shared seed is at most a weak block. The tests treat the 12
  runs as exchangeable units under the null and permute arm labels over all of them
  (924 arrangements), which also gives finer p-values than a 6-pair sign flip (64).
- **Two at a time** on the GPU, as in deer_stag_hunt.
- **Checkpoints** every 10 iterations; evaluated at 250, 500, 750 and 1000.

## Measures

DESIGN.md's measures, unchanged in definition:

- **A:** adults caught per episode.
- **S:** adults caught ÷ (adults caught + failed adult attempts), pooled.
- **C (supporting):** adult approaches with a partner near ÷ chance (coop_fair.py,
  20 episodes).
- **Coexisting:** both species alive at the episode's last physics step (an episode
  ends early only when a species dies out; evaluate.py `outcome`).

What is evaluated is a **hunter checkpoint system**: the shared policy at that
checkpoint plus the founder gene pool saved with it (DESIGN.md). Not "hunter quality"
in general.

Two ways of measuring:

1. **Own world:** each checkpoint against its own run's deer at that checkpoint, 50
   episodes, seeds 0-49. What a run became, the whole system.
2. **Cross-play panel:** at checkpoints 500, 750 and 1000, every run's hunters against
   the deer of every run at the same checkpoint, 50 episodes each, seeds 0-49: a 12 ×
   12 matrix per checkpoint. The own-run cells (the diagonal) are the own-world
   evaluations; the **primary panel value leaves them out**, because a hunter's own
   deer adapted to it and the others did not. A hunter run's panel value is its mean
   over the 11 other runs' deer and the 3 checkpoints (A as a mean of cell values;
   S pooled over all those episodes).

**What the panel removes and what it doesn't.** Within a checkpoint, every hunter
faces nearly the same set of deer, so a B-vs-L gap is not simply "B's own deer were
weaker". But every panel deer was itself produced by a B or an L world, and each
hunter system was shaped by its own coevolving deer, its gene pool and its PPO
history. The panel measures performance against a **common mixture of coevolved
deer**; it does not isolate an innate-bias effect from those histories.

**Deer side, reported:** a deer run's mean adults lost against the 11 other runs'
hunters, B's deer against L's.

## Tests, set before the main runs

**Statistic and test, for every test below:** the difference of arm means of the run
values, B − L; one-sided exact permutation test over all 924 ways of labelling 6 of
the 12 runs B, p = share of labellings with a difference at least the observed one.
In the panel, permuting labels permutes whole runs, hunters and deer together, over
the fixed 12 × 12 matrices, so the panel's shared deer and episode seeds are held
fixed under the null. Ties need no rule. No power analysis was done: with 6 runs per
arm only a large, consistent difference can pass.

1. **Primary: B's hunter systems beat L's on the panel,** on A and on S, p < 0.05
   each. Both must hold (one conjunctive claim, no correction needed). A met without
   S, or S without A, is "not met", with both reported.
2. **Condition: B sustains coexistence.** In at least 5 of 6 B runs, the own-world
   coexisting counts at checkpoints 500, 750 and 1000 sum to at least 135 of 150.
3. **Secondary: genes under selection beyond drift.** Per run and gene, the mean of
   the per-episode selection differential cov(relative offspring, gene)
   (environment.py `gene_stats`, over all founders and newborns of the episode) over
   the own-world evaluation episodes at checkpoints 500, 750 and 1000, each episode
   weighted equally, episodes without a defined value (fewer than two hunters or no
   offspring) left out and counted. B against L, whose genes are unexpressed (the
   drift reference), one-sided upward, for `a_hunter` and for `a_adult`, p < 0.025
   each (Bonferroni over this secondary family of two). Each gene is its own claim.

**Verdicts, written in advance:**

- **Tests 1 and 2 met:** "in this world, with deer learning alongside, the hunter
  systems of arm B (shared PPO learning plus an evolving innate attraction bias)
  catch more adults, and more per attempt, than those of arm L (learning alone),
  against a common panel of coevolved deer." This does not say the genes' evolution
  caused it: the bias's mere presence, or another gene, could.
- **Test 3 met for a gene:** "in B, the gene `<gene>` had a larger positive selection
  differential than in L." That is evidence of selection on it, not by itself proof
  of a lasting evolutionary response; gene means over time are reported beside it.
  It stands on its own, whatever tests 1 and 2 say.
- **B ≈ L with no selection beyond drift:** learning may make the genes irrelevant to
  fitness, the opposite of a Baldwin effect. A useful negative.
- **B < L:** the bias may get in the way of learning.

**Also reported, not tested:** own-world A, S and C per run and arm; the panel's
diagonal; C in the panel; deer side per arm; B − L with bootstrap 95% intervals over
runs; gene means over time; checkpoints 250 and 500 on their own; hunter births;
`a_young` and `neutral` selection differentials; how many episodes had no defined
differential.

## Pilot, and the hard stop

- **Code frozen first:** the code, launchers and verdict scripts are committed and
  Codex-reviewed before the pilot; each run saves its source and the commit hash.
- **Pilot runs:** pilot_L1 and pilot_B1, 500 iterations, seed 1, two at a time. They
  enter no test, and nothing in this note (tests, wording, analysis) changes after
  seeing them.
- **Pilot passes if, in both runs:**
  1. nothing crashes;
  2. at iteration 250 and, separately, at iteration 500, at least 40 of 50 own-world
     episodes are coexisting;
  3. at iteration 500, hunter births per own-world episode are at least 5
     (deer_stag_hunt touch_1-3 at iteration 500: 18-22).
- **If the pilot fails:** this stage stops. Nothing is tuned or re-run: no change to
  β, the genes, any PPO or world setting, and no run is excluded. The eco-evolutionary
  line in this world then ends; a further design would be a new user decision.
- **If it passes:** the 12 main runs, the evaluations and the panel, then the stop,
  whatever the outcome.
- **Failures that aren't results:**
  - a run that crashes, or a hardware failure, is rerun from scratch with the same
    seed, the event recorded; no other rerun;
  - a bug found in code (not a setting) is fixed, recorded with the reason, and every
    run it affects is rerun from scratch, the pilot included if it was affected,
    with no pre-registered setting changed;
  - an evaluation that fails technically (crash, missing or corrupt checkpoint
    file) is rerun; a checkpoint that cannot be loaded at all is reported missing,
    and the stage reports what can be computed without it.

## Implementation sketch (after approval)

- `train.py`, a third deer mode (`deer: "learn"` beside the replay and frozen modes;
  exactly one set, validated):
  - the deer module built fresh from deer_stag_hunt's spec, not loaded from disk;
  - both policies in `policies_to_train`;
  - none of the frozen or switch checks in this mode; instead a check that deer
    weights change across iterations and that hunters still train;
  - the settings record the mode; `SOURCE_FILES` gains DESIGN_3.md.
- `checkpoints.py`: in learn mode a checkpoint's deer are the deer module inside that
  RLlib checkpoint (learner_group/learner/rl_module/deer_policy), with its hash
  recorded, not `<run>/deer_policy`.
- `launchers/cross_play.py` (new): for a mark, every hunter run × every deer run,
  calling evaluate.py's Python API with the other run's deer module; refuses unless
  both runs' world settings and deer observation and action shapes are equal;
  results record both runs, both checkpoints and both modules' hashes. Runs
  evaluations one at a time so the CPU isn't oversubscribed.
- `launchers/verdict_3.py`, `pilot_check_3.py`, `run_pilot_3.sh`, `run_main_3.sh`,
  modelled on section 2's; the main launcher refuses unless the pilot passed.
- **Tests:** learn mode trains deer weights and hunter weights, and a checkpoint
  restores both; checkpoint_deer finds a learn-mode checkpoint's own deer; the
  cross-play loader puts run j's deer with run i's hunters and refuses a mismatched
  world; the permutation test and the diagonal exclusion on known inputs;
  equivalence test unchanged.
- Codex reviews the code before any run.

## Cost

Wall-clock estimates, to be replaced by the pilot's measured throughput (learning
deer add learner work, so learn mode may be slower than section 2's pilot):

- **Pilot:** 2 runs of 500 iterations together, about 2-3 h (deer_stag_hunt's
  control runs: 3 runs of 500 iterations in 3 h 51 min, two at a time), plus
  evaluations, about 10 min.
- **Main runs:** 12 runs of 1000 iterations, two at a time, about 18-30 h.
- **Evaluations:** own world 12 × 4 checkpoints plus coop_fair, under 1 h; panel 3 ×
  12 × 11 = 396 off-diagonal evaluations of 50 episodes, at about 30 s each when run
  one at a time (section 2's pilot evaluations), about 3.5-6 h of wall-clock on the
  CPU.

## Limits, set in advance

- **Coevolution:** the deer in B and L differ. The panel compares hunter systems
  against a common mixture of coevolved deer; those deer came from the B and L worlds
  themselves, and each hunter system was shaped by its own deer.
- **Shared learning, not individual:** one PPO policy for all hunters; the genes are
  individual. Not a classic Baldwin effect; the shared policy is not culture.
- **One mechanism:** an additive attraction bias with four genes.
- **C is company, not partner seeking** (deer_stag_hunt RESULTS.md §1.1).
- **Six runs per arm:** a test across runs, not a population estimate; low power.

## Decisions for the user

1. **Deer learning in every run, arms B and L** (recommended), or also run E against
   learning deer (likely outlearned, little to say about B vs L).
2. **Primary hunter test on the cross-play panel, other runs' deer only**
   (recommended), or on each run's own world (whole-system, mixes in the deer).
3. **Gene-selection test 3 included as its own claim** (recommended), or reported only.
4. **6 runs per arm, permutation test over the 12 runs** (recommended), or 5 per arm
   (cheaper, 252 arrangements, less power).
5. **Pilot bar:** 40 of 50 coexisting at 250 and 500 and at least 5 births, in both
   pilot runs (recommended, same as section 2), with the line ending if it fails.

## Revisions after review (Codex, 2026-11-10)

- **Panel claim weakened:** the panel deer are products of the B and L worlds, so the
  panel measures hunter systems (policy plus gene pool) against a common mixture of
  coevolved deer, not hunter quality cleaned of history; the verdict says so and no
  longer implies the genes' evolution caused a gap.
- **Own-run cells left out** of the primary panel value (a hunter's own deer adapted
  to it).
- **Dependence in the panel:** the Mann-Whitney test on row means is replaced by a
  permutation test that relabels whole runs over the fixed matrices; the same test
  is used everywhere. The shared seed is explained as bookkeeping, not a block.
- **Multiplicity and partial results:** test 1 is one conjunctive claim; test 3 is a
  separate secondary family with Bonferroni over its two genes; A without S is "not
  met".
- **Selection statistic defined** from the existing per-episode `gene_stats`, its
  aggregation and missing values fixed; test 3's wording is now "a larger positive
  selection differential", with gene means reported beside it.
- **Coexistence rule exact:** summed counts ≥ 135 of 150; pilot at each checkpoint
  separately.
- **Hard stop completed:** code frozen before the pilot, nothing changes after it,
  bug and crash policy, failed evaluations.
- **Implementation:** learn mode, checkpoint deer lookup, cross-play via the Python
  API with world checks, DESIGN_3.md in the saved source.
- **Cost:** wall-clock ranges widened, to be replaced by the pilot's throughput.
