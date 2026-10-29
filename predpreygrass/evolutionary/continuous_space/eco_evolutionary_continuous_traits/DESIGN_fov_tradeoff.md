# Design note: a field-of-view trade-off (2026-10-26)

Status: **approved 2026-10-26; built in eco_evolutionary_continuous_fov; strike catching added 2026-10-27 (addendum); pilot 2026-10-28: test 1 failed (predators stayed at the fully binocular corner instead of narrowing), see that module's RESULTS.md section 3; pilot 2 (monocular field 120°, re-registered) 2026-10-29: test 1 passed in both seeds (predators narrowed, prey panoramic), test 3 passed, RESULTS.md section 5.** Decisions taken: no
energy cost for field of view, a new module, noisy monocular distance, and
the parameters proposed below.

## Why

This module's learning-to-body effect is replicated (RESULTS.md sections 5,
8), but the coevolution half of the project's goal is missing: both species
evolve the same way (shorter view distance, wider field of view; section
2), and the prey's gains come from behaviour, not bodies. There is nothing
for one species' body to respond to in the other's.

The reason is in the model. Inside its view cone an animal sees everything
perfectly: each observation slot holds the exact offset, distance and
velocity of the nearest animals in view (`environment.py`, `_slots`). A
wider field of view therefore only costs energy (`trait_cost_fov`) and never
loses anything. In animals it does lose something: forward-facing eyes give
a large binocular overlap and depth perception, which predators use to
judge distance when catching prey; eyes on the sides give panoramic
coverage, which prey use to spot predators, at the cost of overlap (Heesy
2004; Banks et al. 2015; README, "Vision and energy").

## Proposal: binocular overlap

Give each animal two eyes, each covering a monocular field of `M` degrees.
The field-of-view gene sets how far apart the eyes point: the total field is
the expressed field of view `F`, and the binocular overlap in front is

    overlap = clip(2 M - F, 0, F)

so a narrow field is mostly binocular and a wide one mostly monocular.

- **Inside the overlap** (|bearing| ≤ overlap / 2): observations as now,
  exact distance.
- **Outside it** (monocular): the direction is exact but the distance is
  noisy: the slot's offset (dx, dy) and distance are multiplied by one
  lognormal factor exp(σ ε), ε ~ N(0, 1), drawn per observer, target and
  step. The seen flag and the target's velocity are unchanged.
- The same rule applies to the grass observation.

Predicted effect: a predator closing on prey needs accurate distance in
front, so selection should favour a narrower field (more overlap); prey need
to detect predators from any side, so they should keep or widen theirs. If
that happens, the two species' field-of-view genes diverge in sign, which is
the precondition for bodies to respond to each other.

Why this mechanism: it follows the biology the README already cites, it is
a single change in one function (`_slots`, plus the grass observation), it
keeps every other rule as it is, and with σ = 0 it reproduces the current
environment exactly (testable bit for bit, like `trait_expression`).

**Alternatives considered:**

- *Acuity spread over the field* (detection range falls as the field
  widens, for a fixed number of receptors): couples view distance and field
  of view through one parameter, but makes a wider field directly shorten
  vision, which overlaps with the vision gene's own cost.
- *More observation slots in front*: simple, but not tied to anything in
  animals.

## Parameters

| Parameter | Proposal | Reasoning |
| --- | --- | --- |
| Monocular field `M` | 150° for both species | Predator base field 150° is then fully binocular, prey base 240° has 60° overlap; the gene range (×0.5 to ×2) spans fully binocular to 300° with no overlap. |
| Distance noise σ | 0.3 (about ±30%) | Large enough to hurt interception; calibrate in a short test (below). |
| `trait_cost_fov` | 0 (decided) | Geometry is the only cost of a wide field, the cleanest test of the trade-off; CT-05 had 0.0625. |
| Costs of speed and vision | as CT-05 (0.25, 0.125) | The setting where the learning-to-body effect replicated. |
| Policy input | unchanged; the policy already sees its own genes | It can learn how far to trust distances. |

## What counts as success (pre-specified before any run)

1. **Divergence (primary, pilot).** In runs with the trade-off, the change
   in the mean field-of-view gene from start to end (mean of the last 300
   iterations) is negative for predators and positive (or at least
   non-negative) for prey, in every run; in a matched run without it (σ = 0)
   both rise, as in CT-04 and CT-05.
2. **Reciprocal response (coevolution, after replication).** Branch a
   trade-off run into arms where the prey's gene pool evolves or is held
   fixed (policies keep learning in both). Coevolution in bodies means the
   predator's trait trajectory depends on whether the prey's bodies evolve:
   pre-specified contrast, predator field-of-view change with evolving prey
   minus with fixed prey; and the mirror test for prey. This needs a
   per-species `freeze_gene_pool`, which does not exist yet.
3. **Sustainability (guard).** Both species coexist in at least 45 of 50
   final-check episodes, as CT-04 and CT-05 did (50 of 50).

A result of 1 without 2 would be divergence, not yet coevolution.

## Plan and cost

| Step | What | Compute |
| --- | --- | --- |
| 0 | Implement in a copy of this module (`eco_evolutionary_continuous_fov`), tests incl. σ = 0 bit-for-bit equivalence; Codex review | none |
| 1 | Calibrate σ: catches per predator against fixed prey with σ = 0, 0.15, 0.3, 0.5, frozen CT-05 policies, prey in front vs to the side | about 1 h CPU |
| 2 | Pilot: one trade-off run and one σ = 0 control, CT-05 settings, 3,000 iterations, side by side | about 24 h, 2 GPU slots |
| 3 | If divergence: 3 seeds per condition | about 24–36 h |
| 4 | Coevolution branches (test 2), after implementing per-species gene-pool freezing | about 30 h |

Stop rules: if the pilot shows no divergence, analyse why (selection
differentials, the calibration's catch rates) before any replication; if the
ecosystem collapses (guard 3), lower σ once, then stop.

## Decisions (2026-10-26)

- Energy cost of field of view: dropped (`trait_cost_fov = 0`).
- A new module, `eco_evolutionary_continuous_fov`, so that this module's
  results stay reproducible.
- Monocular vision keeps the direction and gets a noisy distance (not
  direction only, which would make the monocular field nearly useless for
  hunting and grazing).
- Parameters as proposed: `M` = 150°, σ = 0.3 (to be checked in the
  calibration step), CT-05's costs of speed and vision.

## Addendum (2026-10-27): strike catching

**Why.** The calibration (eco_evolutionary_continuous_fov, RESULTS.md
section 1) found that noisy monocular distances barely hurt the predators
(at most 3–4% fewer catches per opportunity at σ = 0.5, about equally in
front and to the side), while they clearly hurt the prey. A catch was a
contact, which needs the direction to the prey but no judged distance, so
the premise for predators ("a predator closing on prey needs accurate
distance in front") did not hold in the model. Decision by the user: add a
strike.

**Rule** (`strike_catching`). Each physics step, a predator that is not
cooling down strikes at the nearest prey in its view when it perceives that
prey within reach (sum of radii plus `strike_reach`, 0). The perceived
distance is exact inside the binocular overlap and noisy outside it, with
the same σ as the observation but its own random draws. A strike at prey
truly within reach catches it; a premature strike misses, and the predator
cannot strike for `strike_cooldown` physics steps (16, two decisions).
Without noise a strike catches exactly the prey in view that contact would
catch; prey touching a predator from outside its view are no longer caught.

**Why a reflex, not an action.** The strike is triggered by perception, not
chosen by the policy, so the action and observation spaces stay as they are
and the CT-05 policies can calibrate it at once; a strike action would need
new policies before any calibration. The policy can still adapt how it
approaches (from the front, more slowly). A strike action, or observing the
cooldown, remains an option for later.

**Calibration** (the same design as section 1, `calibrate_noise.py
--strike`), with a contact-catching σ = 0 condition for the change of
ecology from strikes alone. Pilot only if predators then clearly lose
catches to noise, more to the side than in front.

## Addendum (2026-10-28): pilot 2, monocular field 120°, re-registered

**Why.** Pilot 1 failed test 1 (eco_evolutionary_continuous_fov
RESULTS.md section 3): with the trade-off the predators' field-of-view
gene stayed near its start (+0.05) instead of falling, while without it it
rose to +0.57. With M = 150° the predators' starting field, 150°, is
already fully binocular, so narrowing it gains no overlap: the prediction
"falls" did not fit the geometry. Written before pilot 2 is started.

**Change.** M = 120° (`fov_monocular_field`), nothing else (σ 0.3, strikes,
CT-05 settings). The predators then start at 150° with 90° overlap (60%
binocular); narrowing to 120° (gene −0.22) makes them fully binocular, so
narrowing now gains depth vision. The prey start at 240° with no overlap.
The σ = 0 control does not depend on M (without noise M is never used),
so pilot 1's control is pilot 2's control.

**Runs.** `launchers/pilot.py tradeoff_m120` (seed 1) and
`tradeoff_m120_seed2` (seed 2), 3,000 iterations, 10 workers each, side by
side; no new control.

**Test 1, re-registered.** Change in the mean field-of-view gene from the
founders of every worker's first episode to the mean of the gene-pool
snapshots over the last 300 iterations (`analyze_pilot.py`):
negative for predators and non-negative for prey, in both runs. The
control's part (both rise) is already met by pilot 1's control. Secondary,
also stated now: the predators' end gene is below pilot 1's trade-off run
(+0.07) in both runs. Test 3 (coexistence in at least 45 of 50 final-check
episodes) as before. Not calibrated again: M changes only where the noise
applies; a collapse of either population would show in the first hours
and stop the run.

## Addendum (2026-10-29): test 2, coevolution branches, pre-registered

**Why now.** Pilot 2 passed test 1 in both seeds (eco_evolutionary_continuous_fov
RESULTS.md section 5). The plan put replication (step 3) before test 2; the
user chose test 2 first. Written before any branch run; the branch and
freeze code (`trait_initial_gene_pool`, `trait_frozen_species`,
`init_checkpoint`) exists, was reviewed on 2026-10-27, and a 3-iteration
smoke test confirmed that a frozen species keeps exactly the branch point's
pooled genomes in every worker.

**Design.** [`launchers/coevolution.py`](../eco_evolutionary_continuous_fov/launchers/coevolution.py).
Branch points: iteration 10 of each pilot-2 run (seed 1 and seed 2), where
both species' genes are still near the founders' (predator fov gene +0.03
and −0.05, prey +0.00 and +0.19), so that both arms of a contrast see the
whole move; this point was chosen after seeing the pilot trajectories.
From each branch point three new runs, initialised from the iteration-10
policy weights (fresh PPO optimizer state; both policies keep learning)
and the iteration-10 gene pool pooled over the 10 workers (fresh episodes;
every worker draws from the same pool, not the original islands), with
the source run's seed and settings (σ 0.3, M 120°, strike catching; the
launcher aborts if any other setting differs from the source run's saved
settings): `both` (both gene pools carried over between episodes),
`prey_frozen`, `predator_frozen`. Freezing stops inheritance between
episodes only: the frozen species' founders are drawn from the branch
point's pool every episode, but its offspring still mutate and are
selected within an episode. So the contrast is "with against without
between-episode inheritance of the other species' genes", not "with
against without any evolution of it"; within-episode change may shrink a
contrast. Manipulation check: the frozen species' living genes at each
snapshot, reported. 1,000
iterations each (pilot 2's predators settled within 30 iterations, prey
reached 360° within 190); 6 runs, two at a time (the GPU fits two), about
34 h. Founders' start is the same for all arms of a branch point, so a
contrast is a difference of end values.

**Measures** (from the gene-pool snapshots, end = mean of the last 300
iterations, as in test 1):

- Predator: mean field-of-view gene.
- Prey: mean expressed field of view in degrees, min(240° × exp(gene),
  360°) averaged over genomes, because above gene 0.405 the prey's gene is
  capped and not under selection.
- Prey, secondary (uncapped, for the ceiling): the mean expressed field
  over iterations 0 to 300 (how fast prey widen), and the end gene.

**Test 2, pre-specified.**

1. Predators respond to prey evolution: C_pred = predator end gene in
   `both` minus in `prey_frozen`. Supported if |C_pred| ≥ 0.05 with the
   same sign at both branch points. (0.05 is the difference between pilot
   2's two seeds' predator end genes, −0.20 and −0.15: a between-history
   scale, fixed now.)
2. Prey respond to predator evolution (mirror): C_prey = prey end expressed
   field in `both` minus in `predator_frozen`. Supported if |C_prey| ≥ 10°
   with the same sign at both branch points. (10° is fixed now without a
   noise reference; pilot 2's prey ended at 360° in both seeds.)
   Secondary, interpretive only: the same contrast on the prey's
   0-to-300 mean expressed field and on the end gene, same thresholds
   (10°; 0.05 for the gene).
3. Coevolution in bodies (reciprocal) if 1 and 2 are both supported; a
   one-sided response if only one is; otherwise none detected.
4. Guard (test 3) for every arm: coexistence in at least 45 of 50
   final-check episodes; an arm that fails is reported but its contrast is
   flagged.

Raw contrasts are reported with per-worker values (workers share the
policy and are not independent replicates), the snapshot SD over the end
window, and which nearby thresholds would change a verdict. The thresholds
are pre-committed smallest effects, not significance tests; with two
histories a result is scoped to them.

**Expectation, stated before the runs.** A null is likely and would be
informative. Predators narrowed within 30 iterations in both pilot-2 runs,
fast enough to be driven by their own geometry; the prey reached the 360°
cap in all four pilot runs whatever the predators did, and a capped trait
can hardly respond (a ceiling on test 2.2: a null there says little,
the secondary prey measures are the check). The coupling through which prey
eyes could change the predators' best field is behavioural (prey that see
predators earlier flee earlier), which §4 found real but did not link to
predator eyes. The confusion effect (README TODO) is the stronger coupling
and stays open.

## Related work (2026-10-27, from abstracts and summaries)

- **Coevolved vision in predator and prey simulations.** Buason and Ziemke
  (2003) co-evolved the controllers of simulated predator and prey robots
  together with each robot's camera view angle and range, and found that
  the two species evolved different, task-dependent visual morphologies:
  the nearest precedent for test 1, there with evolved network weights
  instead of a learned policy and without a binocular trade-off. There
  (one predator, one prey robot; angle and range coupled) the prey evolved
  to circle the arena whatever the predator did (Smith 2004): prey that
  ignore predators cannot evolve eyes in response to them, so check that
  our prey's behaviour depends on seeing a predator. Coupling angle and
  range is the fallback trade-off ("acuity spread over the field" above). Olson et
  al. (2016) found a coevolutionary cycle between the predator's visual
  system and prey behaviour (a focused predator view selected for
  dispersing prey, a broad one for swarming), and suggested that predators
  would gain from a hybrid of narrow, high-acuity tracking vision and broad
  vision for finding prey, close to this design's binocular centre and
  monocular periphery; a cycle, not a steady state, is a possible outcome
  of test 2. Georgiev et al. (2019) co-evolved predator behaviour with an
  asymmetric sensor offset (predators only).
- **Learning and evolution of eyes.** Tiwary et al. (2025) evolved eye
  shape, optics and neural processing while agents learned tasks by deep
  reinforcement learning: navigation tasks gave compound-like eyes,
  object discrimination high-acuity camera-like eyes. No predators and
  prey, no coevolution.
- **Biology.** Cartmill's visual predation hypothesis links forward-facing
  eyes and binocular overlap to visually guided predation; comparative
  tests are mixed (some find no association between orbit convergence and
  predation across mammals; the arboreal hypothesis competes), while
  side-facing eyes for panoramic vision are better supported (Heesy 2004).
  This design builds the predation link in, so a positive result shows that
  the mechanism can produce divergence, not that it did in animals.
- **Background.** Pursuit and evasion as a coevolutionary arms race (Miller
  and Cliff 1994; Cliff and Miller 1996); evolution of visual sensor
  morphology with the controller (Husbands et al. 1997); Aquarium (2024),
  the reinforcement-learning predator-prey framework this world reproduces,
  does not evolve sensors.

Not found: a study in which learned behaviour and heritable bodies
interact while predator and prey eyes face a depth-against-coverage
trade-off. Full papers not yet read; the bibliographic details of the
added references are from abstract pages and still to be checked.

## References

- Banks, M. S., Sprague, W. W., Schmoll, J., Parnell, J. A. Q., and Love,
  G. D. (2015). Why do animal eyes have pupils of different shapes?
  *Science Advances*, 1(7), e1500391.
- Buason, G., and Ziemke, T. (2003). Co-evolving task-dependent visual
  morphologies in predator-prey experiments. *GECCO 2003*, LNCS 2723,
  458–469.
- Cliff, D., and Miller, G. F. (1996). Co-evolution of pursuit and evasion
  II: simulation methods and results. *From Animals to Animats 4*.
- Georgiev, M., Tanev, I., and Shimohara, K. (2019). Coevolution of the
  asymmetric morphology and the behaviour of simple predator agents in
  predator-prey pursuit problem. *Information*, 10(2), 72.
- Heesy, C. P. (2004). On the relationship between orbit orientation and
  binocular visual field overlap in mammals. *The Anatomical Record Part A*,
  281A(1), 1104–1110.
- Husbands, P., Harvey, I., Cliff, D., and Miller, G. (1997). Artificial
  evolution: a new path for artificial intelligence? *Brain and Cognition*,
  34, 130–159.
- Miller, G. F., and Cliff, D. (1994). Co-evolution of pursuit and evasion
  I: biological and game-theoretic foundations. CSRP 311, University of
  Sussex.
- Olson, R. S., Hintze, A., Dyer, F. C., Moore, J. H., and Adami, C. (2016).
  Exploring the coevolution of predator and prey morphology and behavior.
  *Proceedings of ALIFE 2016*; arXiv:1602.08802.
- Smith, N. W. (2004). Evolving a vision-based predator-and-prey system for
  two robots with a learning classifier system. GECCO 2004 workshop paper.
- Tiwary, K., et al. (2025). What if Eye...? Computationally recreating
  vision evolution. *Science Advances*, 11(51); arXiv:2501.15001.
