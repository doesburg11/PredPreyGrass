# Design note 4: selection between groups (draft 2, 2026-11-14)

Status: **draft 2, not approved.** Draft 1 was revised after a Codex review
("Revisions after review", end). Nothing here is pre-registered yet; once the
decisions below are made it goes into RESULTS.md as section 4, before any code.
This is a new stage, prompted by a user decision after section 3 closed. It does
not continue or rerun sections 1-3. Everything not named here stays as in
DESIGN_3.md (approved 2026-11-10): the touch world, deer learning alongside, the
four genes, β₀ ≈ 3.44, the bias inside the RLModule, measures A, S and C.

## Why

Section 3 (RESULTS.md §3.8, §3.9) ended on the classic stag hunt problem. With
genes selected only through individual hunters, in all six B runs they moved
toward young deer (caught alone, safe) and away from adults (need two, risky),
and B's hunters partnered on adults below chance. Switching the bias off raised
partnering above chance but not adult catches. Sharing did not help either, and
a theory sweep said why: sharing pays only if big game pays 1.2-1.75 times the
alternative in expectation.

The standard theoretical answer to a stag hunt is **selection between groups**
(Boyd & Richerson 1990; Bowles 2006). A stag hunt has two stable outcomes; within a
group the safe one tends to win, but if groups that settle on the joint hunt do
better as groups, they can spread by replacing others. This is equilibrium
selection, not altruism, so it may need less than sharing did. It still needs one
thing: **groups whose genes lead them to hunt adults together must grow faster.**
Whether they do in the deer world is open; section 3 hints they may not (adults are
a small part of the diet, per-attempt success S ≈ 0.02).

## Step 0: theory screen (before any training code)

A screening heuristic, not a calibrated threshold: the theory model has pure
strategies, synchronous generations, fixed group sizes and no learning, so passing
shows only that the mechanism is plausible at these payoffs, and failing is a
reason to stop.

- **Model:** EvolvedCooperation `ecological_models/stag_hunt_sharing`, sharing off,
  the grid split into **14 groups** (as the training stage), **no migration**
  between groups beyond the replacement itself, and, every **T = 10** generations,
  the lowest-fitness group replaced by a copy of the highest (ties: random among
  the tied). Group fitness = mean individual fitness in the group over the window.
- **Fixed settings, no search:** diminishing fitness; q ∈ {0.3, 0.45, 0.6};
  ratio ∈ {1, 1.1, 1.2, 1.3, 1.5, 1.75}; start 5% and 50% stag hunters; **20
  seeds** (0-19) per cell, with and without group selection. Ratio is the expected
  payoff ratio of the existing model (stag payoff = ratio × hare / q), not the deer
  world's energy ratio.
- **Outcome per seed:** final stag share = mean over the last 100 generations.
  Takeover = final share > 0.5.
- **Pre-registered rule:** continue to training only if, at ratio 1.2 (the lowest
  ratio sharing needed anywhere), for **every q and both starts**, group selection
  gives takeover in at least 15 of 20 seeds **and** at least 10 more seeds than
  without group selection. Otherwise this line stops at step 0, as sharing did.
  The full sweep is reported either way.

## The training stage (only if step 0 passes)

### Groups and world identity

The 14 env-runner worlds of a run are the groups. Each already keeps its own gene
pool across episodes (environment.py: founders are the hunters alive at the end of
the last episode) and never exchanges genomes. The PPO policy stays shared by all
worlds: **only the genes can differ between groups**, so group selection can act
only through the innate bias.

Each world gets a durable identity (env-runner index plus a UUID created at
construction, saved with its pool). A selection event proceeds only if exactly 14
distinct world identities answer; a missing world, a duplicate or a changed UUID
(a restarted runner, which would have lost its pool) **aborts the run**, which is
then rerun from scratch and recorded, as section 3.7's power cut was. Env-runner
restarts stay disabled.

### Group fitness (exposure-normalised)

Per world, since its last selection event, over **completed episodes only**:
**hunter births per hunter-step** (births divided by the summed number of living
hunters over the physics steps of those episodes). This removes the size and
episode-length confounds of raw births. A world needs at least 2 completed
episodes since the last event to take part; worlds below that are neither source
nor sink in that event.

### The replacement rule (new code)

Every T = 10 iterations, after training results: read every world's fitness and
pool (all reads before any write); sink = lowest fitness, source = highest; ties
random (seeded). The source's pool is copied by value (genomes and generations) and
**queued** in the sink: it becomes the founder pool at the sink's next reset, and
the sink's end-of-episode pool update is skipped for that episode, so the hunters
alive at the event cannot overwrite it. No new mutation (mutation stays at births).
Deer, grass and positions are untouched. Every event is logged: source and sink
identity, fitness and exposure of every world, pool hashes before and after, and an
acknowledgement from the sink when the queued pool is used.

Selection events and checkpoints never coincide in the same callback without a fixed
order: the event runs first, the checkpoint (with every world's pool, queue and
fitness accumulator) after.

### Arms

- **G:** arm B of section 3 (genes expressed, β = β₀) plus the replacement rule.
- **B:** the same code with the rule off (fitness still computed and logged, never
  acted on), on RLlib 2.60.0.

Six runs per arm (decision 3), 1000 iterations.

### Evaluation panel (external, balanced)

Every G and B hunter checkpoint at 500, 750 and 1000 plays against **the same
external panel**: the deer of section 3's 12 main runs (main3_L1-6, main3_B1-6) at
the same iteration, evaluate.py 50 episodes and coop_fair.py 20 per cell. No hunter
meets its own run's deer, and both arms face exactly the same deer, so differences in
deer evolution between G and B cannot tilt the panel.

### Pilot

One G and one B run, 500 iterations. Bars:

- coexistence as section 3 (at least 40 of 50 at 250 and at 500, own world);
- every scheduled event completed with 14 identities, no abort;
- every queued pool used at the sink's next reset (acknowledgements match events);
- **a reliable signal between worlds:** in the B pilot (rule off), the split-half
  correlation of world fitness ranks (odd vs even windows) is positive (Spearman
  > 0), so worlds differ by more than noise.

Before the pilot, unit tests with fixture worlds: exact copy of genomes and
generations, persistence through reset, no overwrite by the old episode, abort on a
missing or restarted world. Hard stop: if the pilot fails, the stage stops; no
recalibration round.

### Measures and tests (to be fixed before the main runs)

- **Manipulation check:** G − B in group fitness (births per hunter-step), own
  world, mean of 500/750/1000. Reported first; if G is not above B, the test results
  below cannot be read as group selection working.
- **Test 1 (primary):** G − B, panel A (adults caught per episode), one-sided
  924-labelling permutation test, p < 0.05.
- **Test 2 (condition):** at least 5 of 6 G runs coexist in at least 135 of 150
  own-world episodes.
- **Test 3 (secondary, each on its own):** G − B on panel C, `a_adult` and
  `a_hunter` at 1000.
- **Reported:** events per run, which worlds won, between-world gene variance
  (Q_ST), births, S, young caught solo and with a partner.

**Power, stated now:** from §3.8's between-run spread of panel A (pooled SD about
4.5), a simulated one-sided 924-labelling test with 6 runs per arm has about 40%
power for an effect of +4 adults per episode, 66% for +6 and 86% for +8. Only a
large effect is likely to be detected; a null is weak evidence of no effect. Runs and
worlds are not treated as interchangeable: the run is the unit.

## Decisions for the user

1. **Group fitness:** hunter births per hunter-step *(recommended; removes size and
   duration confounds)*, or population growth rate. Adults caught is excluded: it
   would reward the target behaviour directly.
2. **Replacement:** best replaces worst every 10 iterations *(recommended: the
   Moran rule, strongest test of the mechanism)*, or a gentler rule: source drawn
   with probability proportional to fitness rank (less noise-driven, weaker).
3. **Runs per arm:** 6 *(recommended: as section 3; one G/B pair took about 7-10
   hours there, so about 2 days, plus about 10 hours for the panel's 432
   evaluations)*, or 10 (power for +6 adults rises to about 85%; about 3.5 days plus
   about 17 hours of panel).
4. **Step 0 first** *(recommended)*, or go straight to training.
5. **Hard stop:** one pilot, then main runs or stop; no calibration rounds.

## Known limits, stated now

- **Only genes are group-level.** The shared policy learns from all worlds at once,
  so behaviour learned in winning groups spreads to all groups anyway.
- **14 groups is few,** and they share PPO and selection events: not 14 independent
  replicates. Q_ST is reported to show whether groups differ enough to select on.
- **Copying a pool is not group fission.** It is the simplest stand-in (the Moran
  model's rule); wording will say so.
- **Group fitness may favour young deer.** If young deer feed groups better, group
  selection will push the genes the same way as individual selection; that would be
  a real negative answer, not a failure of the design.

## Revisions after review (Codex, 2026-11-14)

- Group fitness: raw births per episode replaced by births per hunter-step over
  completed episodes, with a minimum exposure (size and episode-length confounds).
- Replacement queued for the sink's next reset, with the old episode's pool update
  skipped (otherwise living hunters overwrite the imported pool).
- Durable world identity, all-14 requirement, abort on restart, logged events,
  read-before-write, fixed event/checkpoint order.
- Step 0: fixed to the training setting (14 groups, no migration, T = 10), seeds,
  outcome window, ties and a rule over every q and both starts, instead of passing
  on any of 96 settings; framed as a screen.
- Pilot: the replacement bar (which could not fail) replaced by integrity bars and a
  between-world reliability check; fixture unit tests before the pilot.
- Panel: an external, balanced panel (section 3's 12 runs' deer) instead of the
  arms' own deer.
- Primary test: adults caught alone, with a manipulation check on the selected
  fitness first; `a_adult` moved to secondary.
- Power simulation added.
