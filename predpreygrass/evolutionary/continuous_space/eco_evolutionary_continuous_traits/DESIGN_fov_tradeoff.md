# Design note: a field-of-view trade-off (2026-10-26)

Status: **approved 2026-10-26; built in eco_evolutionary_continuous_fov; strike catching added 2026-10-27 (addendum).** Decisions taken: no
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

## References

- Banks, M. S., Sprague, W. W., Schmoll, J., Parnell, J. A. Q., and Love,
  G. D. (2015). Why do animal eyes have pupils of different shapes?
  *Science Advances*, 1(7), e1500391.
- Heesy, C. P. (2004). On the relationship between orbit orientation and
  binocular visual field overlap in mammals. *The Anatomical Record Part A*,
  281A(1), 1104–1110.
