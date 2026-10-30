# eco_evolutionary_continuous_fov

**Status (2026-10-30): test 2, no coevolution detected; prey-frozen arms
collapsed.** Pilot 2 (monocular field 120°, re-registered, RESULTS.md
section 5) passed: predators narrowed to mostly binocular fields (gene
−0.22 and −0.13), prey went panoramic (360°), all runs sustainable.
Test 2 (coevolution branches, section 6): no coevolution detected. The
prey's field did not respond to predator evolution (+0.6°/+0.9°,
threshold 10°). The predator contrast (−0.010/−0.028, threshold 0.05) is
flagged and not interpreted: the guard failed in both `prey_frozen` arms
(12 and 0 of 50, prey extinct). Exploratory, not pre-specified: with prey
frozen, predators evolved faster (speed gene +0.33/+0.26 against
−0.32/−0.11) and the prey collapsed; consistent with prey having to keep
evolving to persist, but confounded (the frozen prey pool is an early,
costly one, and all three genes are frozen together). Next: freeze the
prey at a late, adapted pool (pre-register first); confusion effect;
replication seeds.

Can predators and prey evolve different eyes? A copy of
[`eco_evolutionary_continuous_traits`](../eco_evolutionary_continuous_traits/)
(Trial 15) with one change: a wide field of view now costs depth vision
instead of energy. In that module both species evolved the same way,
shorter view distance and a wider field of view, partly because a wider
field only ever cost energy and never lost anything. Here a narrow,
forward field gives accurate distances in front, as binocular vision does
in animals, so predators might evolve "hunting eyes" and prey "panoramic
eyes", the precondition for the two species' bodies to respond to each
other. Design, predictions and the pre-specified tests:
[DESIGN_fov_tradeoff.md](../eco_evolutionary_continuous_traits/DESIGN_fov_tradeoff.md).

Closest related work: simulated predator and prey robots co-evolving their
camera view angle and range with their controllers evolved different eyes
(Buason and Ziemke 2003), and a predator's visual system and prey swarming
coevolved in a cycle (Olson et al. 2016); what is new here is learned
behaviour (shared PPO policies) together with heritable bodies and a
depth-against-coverage trade-off. Details and references: the design
note's "Related work".

## The change

- **Binocular overlap.** Each eye covers `fov_monocular_field` degrees
  (150). For an animal with expressed field of view F the overlap in front
  is clip(2 × 150 − F, 0, F): the predator's base field (150°) is fully
  binocular, the prey's (240°) has 60° of overlap.
- **Monocular distances are noisy.** An animal or grass patch seen outside
  the overlap is observed in the right direction but with its offset and
  distance multiplied by exp(0.3 × N(0, 1)) (`fov_monocular_distance_noise`),
  capped at the observation scale. The noise has its own random generator,
  so it changes observations only, never the world.
- **Strike catching** (`strike_catching`, design note addendum): a
  predator strikes at the nearest prey in view when it perceives it within
  reach (sum of radii), with the same noisy distance outside the overlap;
  a premature strike misses and the predator cannot strike for 16 physics
  steps. Without noise it catches the in-view prey contact would (at most
  one per step).
- **No energy cost for field of view** (`trait_cost_fov` 0); speed and
  vision cost as in CT-05 (0.25 and 0.125).
- **Per-species gene-pool freeze** (`trait_frozen_species`): a listed
  species' founders are drawn from its initial gene pool every episode
  (without `trait_initial_gene_pool`, from the first episode's founders),
  for the coevolution branches (the design note's test 2). Checkpoints
  mark the pool as frozen, so a run resumed from one starts from the
  frozen pool, not from the mutated offspring alive at the time.

As for the other random generators, the noise generator's state is not
saved in checkpoints: a resumed run reseeds it at the next episode, so it
continues the experiment but not the exact noise draws.

With the noise at 0 the environment is that of
`eco_evolutionary_continuous_traits`, bit for bit (`tests/test_fov.py`).
Everything else, rules, logging and checkpoints, is unchanged; see that
module's README and [SPEC.md](SPEC.md) (sections 1 to 11; the change above
is not in SPEC.md).

## Files

As in `eco_evolutionary_continuous_traits`: [`environment.py`](environment.py)
(the change is in `_monocular_factor`, `_slots`, `_observe` and
`_update_gene_pool`), [`train.py`](train.py), [`callbacks.py`](callbacks.py),
[`checkpoints.py`](checkpoints.py), [`evolve_frozen.py`](evolve_frozen.py),
[`tournament.py`](tournament.py), [`eval.py`](eval.py), [`config/`](config/)
(the experiment's settings in `config_env.py`), [`tests/`](tests/)
(`test_fov.py` for the trade-off). New here:
[`calibrate_noise.py`](calibrate_noise.py) (the calibration),
[`launchers/pilot.py`](launchers/pilot.py) (the pilot),
[`analyze_pilot.py`](analyze_pilot.py) (test 1 from the gene-pool
snapshots), [`final_check.py`](final_check.py) (test 3),
[`prey_behaviour.py`](prey_behaviour.py) (prey reaction and grouping) and
[`RESULTS.md`](RESULTS.md). Not copied: that module's analyses, launchers and
results.

## Usage

From the repository root:

```bash
python predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_fov/train.py
python -m pytest predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_fov/tests
```

Runs are saved in the module's `runs/` folder (gitignored).

## TODO

- [x] Codex review of the implementation (2026-10-27): the trade-off
  itself correct; two bugs in the gene-pool freeze fixed (without an initial
  pool it resampled every episode; a checkpoint restored the living, not
  the frozen pool), tests added.
- [x] Calibrate the noise (design note, step 1; RESULTS.md sections 1
  and 2): contact catching gave predators almost no cost of noise; with
  strike catching σ = 0.3 costs them 41% of side catches, 6% in front.
- [x] Pilot (step 2, 2026-10-28; RESULTS.md section 3): test 1 failed on
  the predators (+0.05 instead of < 0) but the arms differ widely
  (predators +0.05 against +0.57); test 3 passed.
- [x] Decided (2026-10-28): pilot 2 with the monocular field at 120°, so
  that the predators' 150° start is no longer fully binocular; test 1
  re-registered before launch (design note, addendum of 2026-10-28).
- [x] Pilot 2 (2026-10-28/29; RESULTS.md section 5): `tradeoff_m120` and
  `tradeoff_m120_seed2`, pilot 1's control as control (without noise M has
  no effect). Re-registered test 1 passed in both seeds (predators −0.22
  and −0.13, prey +0.59), secondary passed, test 3 passed (50 and 50).
- [x] Test 2, coevolution branches (pre-registered 2026-10-29, design
  note addendum): `launchers/coevolution.py`, queue
  `launchers/queue_coevolution.sh`, `analyze_coevolution.py`; six
  1,000-iteration runs from iteration 10 of both pilot-2 runs, 2026-10-29
  16:38 to 2026-10-30 09:55. Prey contrast null; predator contrast
  flagged (guard failed in both `prey_frozen` arms, prey extinct); exploratory speed and
  vision differences (RESULTS.md section 6).
- [ ] Follow-up to test 2: freeze the prey at a late, adapted pool (e.g.
  `both`'s end) and ask whether evolving predators then overtake them;
  separates "prey must keep evolving" from "the iteration-10 prey pool is
  maladapted". Pre-register first.
- [ ] Replication (plan step 3): more seeds per condition, M 120°, with a
  σ 0 control per seed.
- [x] Descriptive analyses after Olson et al. (2016) and Buason and Ziemke
  (2003) on pilot 1 (RESULTS.md section 4, `prey_behaviour.py`): prey do
  flee predators they see (both arms); no cycle of predator field of view
  and prey grouping within 3,000 iterations (r about 0).
- [ ] Before test 2: consider a confusion effect in the strike (hit
  probability 1 / number of prey near the target in the predator's view),
  so that prey grouping protects and the predators' best field of view
  depends on prey behaviour, the coupling behind Olson et al.'s cycle; the
  geometric trade-off alone may give divergence but no reciprocal response.
  Calibrate as in RESULTS.md section 2 first; decide after the pilot.
