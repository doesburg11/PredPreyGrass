# foraging_swarms

**Question.** Can predators and prey develop collective movement out of
individual self-interest, with no reward for cooperation and no built-in
Boids rules? And do they use what they see of their own species to find
food or prey that is out of their own view?

The starting point is Hahn et al. (2020). In that study, 25 agents follow a moving target that
each agent sees only when it is among its 8 nearest neighbours. The agents
learned to keep their distance and to follow neighbours when the target
was out of sight. Agents trained without information about their neighbours did worse. This module asks the same
questions in the Predator-Prey-Grass world, where the only reward is +10
per birth and coexistence of the two species comes before reward.

**Result (Stage 2, 2026-10-20).** Prey that cannot see other prey (C)
group and align *more*, not less, than prey that can (A, B), in all three
runs and from early in training. So the prey grouping in this world is not
driven by social information. What prey seem to do with the sight of
neighbours is keep their distance. Seeing neighbours' movement (A against
B) makes no measurable difference, and neither seeing positions nor
seeing movement helps prey find grass that is out of view. C runs were the
slowest to reach coexistence, and one never did. Details and caveats:
"Stage 2" below.

**Environment.** A copy of
[`base_environment_continuous`](../base_environment_continuous/) (torus,
steering physics, view cones, energy, reproduction, moving grass clusters;
[SPEC.md](SPEC.md)) with one addition: four settings that control what an
animal sees of its own species. With the defaults it is
`base_environment_continuous` exactly (same observations and dynamics;
`tests/test_social_observation.py`). The world used here is the milestone
world with every quirk fixed, as in base_environment_continuous runs BE-04
and BE-05 ([config/config_env.py](config/config_env.py)). The quirks would
distort the measurements: a stopped animal's heading reset, prey bumps that
replace their acceleration, and sequential movement.

## Conditions

Each animal's observation has 1 predator slot and 3 prey slots,
`[seen, dx, dy, distance, vx, vy]`, for the nearest animals of that
species in its view cone. **Neighbour velocity is already observed** in
`base_environment_continuous`, so the comparison removes social
information rather than adding it:

| Condition | An animal sees of its own species | Setting |
|---|---|---|
| A | positions and velocities (as base_environment_continuous) | defaults |
| B | positions only (velocities 0) | `<species>_observe_neighbor_movement: False` |
| C | nothing (own-species slots empty; the animals are still there and still bump) | `<species>_observe_neighbors: False` |

The other species' slots are never changed: a prey keeps seeing a
predator's velocity, which is information about a threat, not social
information. The observation length is the same in every condition. A
blanked input is constant, so it carries no information.
The policies are memoryless MLPs, so in B and C an animal cannot recover
its neighbours' movement from successive frames.

The prey are manipulated first (B, C), with predators keeping full
information: prey have 3 own-species slots, predators only 1. BP and CP
are the predator mirrors, for later.

## What the environment imposes (not emergent)

| Rule | Looks like | Control in the measurements |
|---|---|---|
| Newborns, and every animal at reset, start with the same velocity (prey (-1, 0), predators (0, -1)) | perfect alignment | animals younger than `MIN_AGE` = 10 decisions are left out (ages count from reset) |
| Offspring are placed 2 radii (32) from the parent | kin clumps | the same age filter; parent-offspring pairs are excluded from pair persistence. R over all ages is reported alongside to show the size of the effect |
| The bump rule pushes touching animals apart | separation / minimum spacing | spacing near 32 is physics; only clustering against a null counts |
| Grass grows in 20 clusters | co-location without coordination | alignment split by whether the animal's target is in view |
| About 45% of predators stand still | no heading | alignment over moving animals only (speed > 0.5) |
| Population sizes change | nearest-neighbour distance changes with density | Clark-Evans R and group shares against random placement of the same number of animals |

The shuffled-heading null for alignment keeps positions and the
population's headings, so a population-wide drift does not count as local
alignment. It does **not** remove alignment caused by neighbours heading for the
same visible food or prey; the split by target visibility addresses that.

## Measurements

[`measure_swarms.py`](measure_swarms.py) plays seeded episodes of a
checkpoint with sampled actions (as in training; seeds 0-49, the same as
`base_environment_continuous`'s final checks), in parallel. The metric
functions are in [`swarm_metrics.py`](swarm_metrics.py) and are tested on
hand-made configurations (`tests/test_swarm_metrics.py`). For each species:

- **Spacing:** nearest-neighbour distance (NN) and Clark-Evans
  R = mean NN / mean NN of random placement at the same count. R < 1 means
  clustered. Real animals cannot overlap, which pushes R up, so R < 1 is
  conservative.
- **Alignment:** the cosine between an animal's heading and the mean
  heading of moving neighbours within 96 units (3 body diameters), against
  shuffled headings. Global polarization is also reported.
- **Groups:** share of animals in groups of 3 or more (single linkage at
  64 and 96 units) against random placement; how long pairs stay within 96
  units (pair spells, in decisions); group lifetimes (a group continues while
  it shares at least half of its members, by the Jaccard index). Durations
  are Kaplan-Meier estimates: a spell cut short by a death or the end of the
  episode counts as lasting at least what was seen.
- **Does grouping pay?** Prey caught, and grass eaten, per prey-decision,
  in a group or alone. This is descriptive only: being in a group and being
  near grass are confounded.
- **Target out of view:** how often an animal sees no target (grass for
  prey, prey for predators), and how often it then sees a conspecific.
- **Losing the target** (Hahn et al. 2020, Fig. 4): distance to the
  nearest target over the 20 decisions after it leaves the view cone, split
  by whether a conspecific was in view at that moment. Every loss is
  followed for the full window, also when the animal loses the target
  again within it (the windows overlap).
- **Counterfactual probe:** take observations with the target out of view
  and a moving conspecific in view, rotate the conspecifics' velocity
  vectors (45° to 315°), and measure the change in the policy's distribution
  over the 16 directions (total-variation distance, TV). Also measure the
  turn of its mean direction: a policy that follows its neighbours turns by
  the rotation angle; one that ignores them has TV 0. For scale, the probe
  also reports the TV when the conspecific velocities are zeroed, and when
  the animal's own velocity is turned by 90° instead. The turn is
  measured only on observations whose direction distribution has a clear
  mean direction (mean unit vector of length 0.5 or more) before and after
  the rotation; "clear" gives their share.
- **Intervals:** 95% bootstrap intervals over episodes (whole episodes
  resampled, since decisions within an episode are not independent) for
  R, alignment minus shuffled, group share minus random, and catches in a
  group minus alone. They describe these policies, not the variation
  between training runs.
- **Catch risk** is catches per prey per 1,000 physics steps, with prey
  exposure estimated per decision from the counts before and after it, as
  in base_environment_continuous's `tournament.py` and `final_check.py`
  (approximate when births and deaths fall inside a decision; kept the same
  so the numbers compare with BE-01 to BE-05).

```bash
python -m predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_swarms \
    <checkpoint> <label> [--episodes 50] [--override '{"prey_observe_neighbor_movement": false}']
```

`--override` changes the environment at test time, not the policy (a
test-time ablation). It writes `analysis/results/<label>.json` and `.txt`.

## Stage 0: the existing policies (2026-10-18)

BE-04 and BE-05 were trained in this world, in condition A, on code
identical to this environment at its defaults (diffed against their saved
`source_code/`), so they serve as condition A without new training. Final
checkpoints (iteration 2,990), 50 episodes each. The coexistence counts
reproduce their final checks (47 and 46 of 50).

### As trained (condition A)

| | BE-04 | BE-05 |
|---|---:|---:|
| Coexisting (of 50) | 47 | 46 |
| Catch risk (per prey per 1,000 steps) | 1.93 | 1.69 |
| **Prey** | | |
| Clark-Evans R (all ages) | 0.93 (0.88) | 0.93 (0.88) |
| Alignment vs shuffled | 0.17 vs 0.01 | 0.18 vs 0.01 |
| Alignment minus shuffled, 95% interval | 0.16 (0.15–0.17) | 0.16 (0.15–0.18) |
| Alignment, grass in view / out of view | 0.20 / 0.13 | 0.22 / 0.14 |
| Polarization | 0.23 | 0.24 |
| In groups of 3 or more at 96, vs random | 0.43 vs 0.35 | 0.45 vs 0.37 |
| Group share minus random, 95% interval | 0.077 (0.072–0.081) | 0.075 (0.070–0.079) |
| Pair spells (Kaplan-Meier): median, share lasting 10 or more decisions | 3, 5.5% | 3, 5.9% |
| Grass out of view (share of prey-decisions) | 43% | 49% |
| Conspecific in view when grass is out of view | 99.7% | 99.8% |
| Distance to grass after losing it, decision 0 → 20 | 228 → 205 | 232 → 213 |
| Caught per decision, in a group / alone | 0.0047 / 0.0185 | 0.0035 / 0.0192 |
| Grass per decision, in a group / alone | 0.032 / 0.064 | 0.029 / 0.067 |
| Probe TV at 90° (zeroed / own velocity 90° for scale) | 0.69 (0.61 / 0.61) | 0.67 (0.59 / 0.63) |
| Probe mean turn at 90° (resultant length; share with a clear direction) | 0° (0.64; 95%) | -1° (0.69; 95%) |
| **Predators** | | |
| Clark-Evans R | 1.03 | 1.01 |
| In groups of 3 or more at 96, vs random | 0.14 vs 0.11 | 0.15 vs 0.11 |
| Alignment vs shuffled | 0.19 vs 0.02 | 0.20 vs 0.02 |
| Alignment, prey in view / out of view | 0.25 / -0.04 | 0.26 / -0.00 |
| Probe TV at 90° (zeroed / own velocity 90°) | 0.68 (0.61 / 0.31) | 0.66 (0.59 / 0.29) |

- **Prey aggregate and align somewhat more than the nulls, in transient
  groups.** R is about 0.93, and group shares are about 8 percentage points
  above random placement. Alignment is 0.17 against a shuffled 0.01, and
  remains (0.13) when the prey itself sees no grass. The intervals over
  episodes exclude zero for all of these (effect sizes for these policies,
  not a test across training runs). Groups do not last: the Kaplan-Meier
  median pair spell is 3 decisions, and about 6% of pairs stay together for
  10 or more. This is loose, short-lived grouping, not a cohesive swarm.
- **Predators hardly group.** R is 1.01-1.03, slightly more evenly spread
  than random placement. Their share in groups of 3 or more is 3-5 points
  above random. Their alignment exists only while they see prey (0.25); with
  no prey in view it is slightly below the shuffled null. This is
  consistent with several predators chasing the same prey, not with social
  coordination.
- **The policies are highly sensitive to neighbour velocity but do not
  follow it.** Rotating the conspecifics' velocities changes the prey's
  direction distribution about as much as turning its own velocity by 90°
  (TV 0.67-0.69 against 0.61-0.63; nudging its energy by 0.05 gives 0.13 in
  a smaller check). However, the mean direction does not turn with the
  neighbours (turn about 0°, among the 95% of observations with a clear
  direction). A synthetic policy that does follow its neighbours turns by
  the rotation angle in the same probe (`tests/test_measure_swarms.py`), so
  the probe can detect following. The probe changes the standing
  probability by 0.004 at most. Predators react to the other predator's
  velocity more than to a 90° turn of their own (0.68 against 0.31).
  Caveat: a rotated velocity with an unchanged position can be a state the
  policy never met in play (e.g. a neighbour moving straight through the
  observer). A high TV shows sensitivity to this intervention, and the
  turn of about 0° shows no simple following. Neither shows how the
  information is used in normal play.
- **Social foraging information is available.** Prey see no grass in
  43-49% of their decisions and almost always see another prey then. This
  makes the Hahn-style foraging question worth asking here: grass is often
  out of sight, so neighbours could guide prey to it. Within condition A,
  "lost grass without a conspecific in view" is too rare (about 530 events
  against 100,000) for a comparison. That comparison needs conditions B and
  C.
- **Grouping coincides with safety and with eating less.** Per decision,
  prey in groups are caught about 4 times less often and eat about half as
  much grass. This is descriptive only (dilution or selfish herd, group
  location, and grass access are confounded).

### Test-time ablations (same policies, information removed only at evaluation)

| | BE-04 A | BE-04 Bt | BE-04 Ct | BE-05 A | BE-05 Bt | BE-05 Ct |
|---|---:|---:|---:|---:|---:|---:|
| Coexisting (of 50) | 47 | 43 | **0** | 46 | 43 | **0** |
| Predators extinct | 2 | 6 | 50 | 3 | 4 | 50 |
| Catch risk | 1.93 | 1.24 | 0.19 | 1.69 | 1.31 | 0.17 |
| Prey alive on average | 27 | 37 | 51 | 29 | 35 | 52 |
| Prey R | 0.93 | 0.92 | 0.85 | 0.93 | 0.93 | 0.83 |
| Prey alignment vs shuffled | 0.17 vs 0.01 | 0.24 vs 0.03 | 0.36 vs 0.11 | 0.18 vs 0.01 | 0.23 vs 0.03 | 0.41 vs 0.18 |
| Prey in groups ≥3 at 96 vs random | 0.43 vs 0.35 | 0.59 vs 0.50 | 0.86 vs 0.75 | 0.45 vs 0.37 | 0.55 vs 0.47 | 0.87 vs 0.76 |

Bt: prey see other prey's positions only. Ct: prey see no other prey.

- Zeroing neighbour velocity (Bt) **does not break prey grouping or
  alignment**: both rise slightly. Catch risk falls, and predators go
  extinct a little more often. Prey alignment does not depend on seeing
  neighbours' velocity.
- Hiding neighbours entirely (Ct) makes prey nearly uncatchable (catch risk
  divided by 10), and predators die out in all 100 episodes. **This is an
  out-of-distribution response**: the prey policy has almost never seen
  empty own-species slots (a conspecific is in view 99.8% of the time), so
  what it does then says nothing about what prey would learn without
  social information. One possible reading, not tested: the co-adapted
  predators rely on regularities in prey behaviour that depend on the prey
  seeing each other. Prey clustering under Ct (R 0.83-0.85) cannot come
  from following neighbours the prey cannot see. That is a reminder that
  aggregation alone is not evidence of social behaviour.

### Over training (BE-04, 20 episodes per checkpoint)

| Iteration | Coexisting (of 20) | Prey R (all ages) | Prey alignment vs shuffled | Prey groups ≥3 at 96 vs random | Probe TV at 90° |
|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 0.96 (0.78) | 0.39 vs 0.05 | 0.17 vs 0.09 | 0.24 |
| 300 | 2 | 0.94 (0.80) | 0.37 vs 0.06 | 0.17 vs 0.09 | 0.41 |
| 1,000 | 19 | 0.92 (0.87) | 0.20 vs 0.01 | 0.45 vs 0.35 | 0.59 |
| 2,000 | 19 | 0.93 (0.87) | 0.20 vs 0.01 | 0.39 vs 0.31 | 0.66 |
| 2,990 (50 ep.) | 47 of 50 | 0.93 (0.88) | 0.17 vs 0.01 | 0.43 vs 0.35 | 0.69 |

Sensitivity to neighbour velocity grows throughout training, while prey
alignment falls as the populations stabilize. At iterations 100-300, R
over all ages (0.78-0.80) is far below R for settled animals
(0.94-0.96). Most of the apparent clustering early in training is
newborns next to their parents, which is imposed by the birth rule.

### Stage 0 conclusion

The decision rule was: if the policies ignored neighbour velocity, the
training comparison would likely be null and could be skipped. They do
not ignore it, and grass is out of view in nearly half of the prey's
decisions with another prey in sight. **The training comparison (Stage 2)
is warranted.** The test-time ablations cannot replace it (Ct is far out of
distribution). All of this is two training runs of one condition; it
describes these policies, not the variation between training runs.

## Stage 2: training comparison (2026-10-18 to 2026-10-20)

| Condition | Runs | |
|---|---|---|
| A | BE-04, BE-05, at their iteration-2,000 checkpoints | the code is identical to this environment at its defaults (diffed) |
| B | seeds 1, 2, 3 | prey see other prey's positions only |
| C | seeds 1, 2, 3 | prey see no other prey |

Each run trains for **2,000 iterations** rather than BE-04/05's 3,000, and
A is measured at BE-04/05's iteration-2,000 checkpoints, so all
conditions are compared at the same training length. The cut saves about
a third of the time. In BE-04, coexistence and the swarm measurements were
already stable by iteration 1,000 (Stage 0, "Over training"). The risk is
a B or C run that settles late: BE-05 settled only at iteration 1,366. A
new A run was planned, as a check that this module trains like BE-04/05,
but was dropped to save about 10 hours; the diff of the environment code
stands in for it.

[`launchers/launch.py`](launchers/launch.py) `<A|B|C|BP|CP> <seed>`
trains one run with 14 simulation workers, into
`runs/foraging_swarms_<time>_<condition>_seed<seed>/`.
[`launchers/queue_stage2.sh`](launchers/queue_stage2.sh) runs several one
after another. Two queues run side by side on 32 cores, as BE-04 and BE-05
did:

```bash
systemd-run --user --unit=sw-lane1 --working-directory=$PWD \
    bash predpreygrass/non_evolutionary/continuous_space/foraging_swarms/launchers/queue_stage2.sh "B 1" "C 2" "B 3"
systemd-run --user --unit=sw-lane2 --working-directory=$PWD \
    bash predpreygrass/non_evolutionary/continuous_space/foraging_swarms/launchers/queue_stage2.sh "C 1" "B 2" "C 3"
```

Progress is in `runs/logs/queue.log` and `runs/logs/train_*.log`. A run
takes about 9.5 hours (BE-04 needed 9.5 hours for its first 2,000
iterations: about 10 s per iteration while populations are small, 18-21 s
once both species coexist). Three rounds take about 28 hours. A first
start with 3,000 iterations was stopped after about 300 iterations and
restarted with 2,000; its partial runs are in `runs/aborted_3000iter/` and
are not used.

The `seed` setting (`config_ppo`) is recorded with the run, but training
is **not** bit-reproducible with remote simulation workers: two
1-iteration runs with the same seed collected different episodes.

**Pre-specified outcomes**, measured with `measure_swarms.py` at iteration
2,000 for every condition (and every 300 iterations for the course of
training):

- **Primary:** coexistence (of 50 evaluation episodes, seeds 0-49), prey
  catch risk, prey grass intake.
- **Secondary:** the swarm measurements above.
- **Social information:** distance to grass after losing it, and time
  until it is seen again, in A against B and C. The velocity probe applies
  to A only: B and C policies never see a conspecific's velocity, so there
  is nothing to rotate (`measure_swarms` collects no probe observations
  for them).

Between-run variation is large: BE-04 settled at iteration 371 and BE-05
at 1,366. With two or three runs per condition, report directions across
runs and avoid causal claims beyond the manipulation itself.

### Results

All runs at iteration 2,000, 50 episodes each (seeds 0-49). Outputs:
`analysis/results/stage2_*`; course of training (20 episodes every 300
iterations): `analysis/results/course_*`.

| | A (BE-04) | A (BE-05) | B1 | B2 | B3 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Coexisting (of 50)** | **49** | **32** | **49** | **47** | **48** | **49** | **40** | **1** |
| Prey extinct / predators extinct | 0 / 1 | 16 / 2 | 1 / 0 | 3 / 0 | 0 / 2 | 0 / 1 | 8 / 2 | 48 / 1 |
| Catch risk | 2.06 | 2.15 | 1.68 | 2.19 | 1.81 | 1.78 | 1.85 | 3.01 |
| Predators / prey alive | 11 / 26 | 11 / 24 | 10 / 30 | 11 / 24 | 10 / 29 | 11 / 29 | 11 / 27 | 11 / 20 |
| **Prey R** | 0.92 | 0.93 | 0.93 | 0.92 | 0.93 | **0.81** | **0.82** | **0.81** |
| **Prey alignment minus shuffled** | 0.18 | 0.19 | 0.21 | 0.20 | 0.21 | **0.29** | **0.29** | **0.32** |
| Same, grass out of view | 0.14 | 0.15 | 0.18 | 0.17 | 0.18 | 0.21 | 0.21 | 0.23 |
| **Prey in groups ≥3 minus random** | 0.082 | 0.072 | 0.082 | 0.090 | 0.078 | **0.200** | **0.192** | **0.187** |
| Prey pairs lasting ≥10 decisions | 6.4% | 6.7% | 7.0% | 7.2% | 7.0% | 12.3% | 12.4% | 14.2% |
| Grass per prey-decision (in group / alone) | 0.032 / 0.066 | 0.029 / 0.074 | 0.032 / 0.057 | 0.032 / 0.070 | 0.031 / 0.062 | 0.031 / 0.071 | 0.031 / 0.074 | 0.032 / 0.090 |
| After losing grass: distance 0 → 20 | 227 → 203 | 230 → 211 | 231 → 209 | 229 → 206 | 230 → 207 | 235 → 206 | 236 → 204 | 225 → 196 |
| After losing grass: back in view ≤20 | 92% | 87% | 92% | 91% | 90% | 92% | 90% | 83% |
| Predator alignment minus shuffled | 0.20 | 0.21 | 0.15 | 0.18 | 0.17 | 0.17 | 0.20 | 0.25 |
| Predators in groups ≥3 minus random | 0.029 | 0.036 | 0.058 | 0.053 | 0.039 | 0.066 | 0.049 | 0.044 |

"After losing grass" uses the events with a conspecific in view in A and B
and those without one in C (the only kind there).

**Course of training** (prey; 20 episodes per checkpoint):

| Iteration | Prey R: A | B | C | Groups ≥3 minus random: A | B | C | Coexisting (of 20): A | B | C |
|---:|---|---|---|---|---|---|---|---|---|
| 300 | 0.94, 0.90 | 0.89, 0.91, 0.95 | 0.80, 0.80, 0.77 | 0.09, 0.09 | 0.11, 0.07, 0.06 | 0.24, 0.19, 0.22 | 2, 0 | 4, 0, 1 | 0, 0, 0 |
| 600 | 0.94, 0.90 | 0.94, 0.92, 0.96 | 0.82, 0.84, 0.82 | 0.07, 0.12 | 0.07, 0.08, 0.05 | 0.18, 0.16, 0.18 | 14, 0 | 20, 0, 18 | 0, 4, 0 |
| 900 | 0.92, 0.92 | 0.94, 0.96, 0.96 | 0.82, 0.83, 0.81 | 0.09, 0.09 | 0.07, 0.05, 0.05 | 0.17, 0.17, 0.18 | 19, 1 | 20, 9, 20 | 5, 4, 0 |
| 1,200 | 0.93, 0.94 | 0.93, 0.95, 0.93 | 0.84, 0.82, 0.80 | 0.08, 0.07 | 0.08, 0.06, 0.08 | 0.16, 0.18, 0.21 | 18, 3 | 19, 18, 19 | 8, 9, 0 |
| 1,500 | 0.93, 0.93 | 0.94, 0.94, 0.93 | 0.82, 0.82, 0.80 | 0.08, 0.07 | 0.08, 0.06, 0.08 | 0.19, 0.19, 0.20 | 17, 7 | 20, 19, 20 | 14, 16, 0 |
| 1,800 | 0.93, 0.94 | 0.94, 0.94, 0.93 | 0.81, 0.81, 0.80 | 0.08, 0.07 | 0.07, 0.07, 0.08 | 0.20, 0.20, 0.19 | 20, 16 | 18, 18, 20 | 16, 19, 1 |

**Findings.** Runs are the unit of replication: 2 A, 3 B, 3 C. "Complete
separation" means every run of one condition lies beyond every run of the
other. For 3 against 2 runs, the one-sided Mann-Whitney p of complete
separation is 0.1. For 3 against 3 it is 0.05. These are the smallest p
values these numbers of runs allow, however large the effect.

1. **Prey that cannot see each other group more (C > A, B).** On R,
   alignment, group share and pair persistence, all three C runs lie
   beyond all five A and B runs (C against B: p = 0.05; C against A:
   p = 0.1). The difference is large: the excess of prey in groups over
   random placement is about 0.19-0.20 in C against 0.07-0.09 in A and B.
   The three C runs are nearly identical to each other. It holds from
   iteration 300 to 2,000, before any C run coexisted. So it is not a
   late-learned strategy. Grouping in this world therefore arises without
   social information, from shared responses to grass and predators. What
   A and B prey do with the sight of other prey reduces grouping: they
   appear to keep apart (Boids' separation rather than cohesion). That
   reading fits the numbers, but the mechanism was not tested directly.
   For example, A prey avoiding the patch another prey is heading for
   would produce it.
2. **Seeing neighbours' movement adds nothing measurable (B ≈ A).** On
   every measure the B runs fall within or next to the A runs. This agrees
   with Stage 0: the A policies are sensitive to neighbour velocity, but
   do not follow it.
3. **No social foraging (Hahn et al.'s main effect is absent).** After
   losing sight of grass, prey in all conditions get back to it about
   equally well (83-92% within 20 decisions, the same distance curves).
   Prey with no view of other prey (C) do not do worse. Grass intake per
   decision is also similar.
4. **Coexistence: C was slowest to settle.** First checkpoint with at
   least 15 of 20 episodes coexisting: B at iterations 600, 600, 1,200; A at
   900, 1,800; C at 1,500, 1,800 and never (C3: 1 of 50 at iteration 2,000,
   prey extinct in 48). B against C is completely separated (p = 0.05); A
   overlaps both. At iteration 2,000, B coexists in 47-49 of 50 episodes, A
   in 32-49, C in 1-49. One possible reading: prey that cannot see each
   other crowd together, which makes them easier for predators to exploit
   while the two species are still co-adapting. Untested, and based on
   three runs. C3 might settle with more training, as BE-05 did between
   iterations 2,000 and 2,990 (32 → 46 of 50).
5. **Predators are unaffected**, as intended (only the prey's information
   changed): their alignment is driven by visible prey, and they hardly
   group.

**Answer to the question.** In this world, prey and predators do not
develop swarming out of self-interest the way Hahn et al.'s foragers did.
Prey grouping is real but weak, short-lived and not social: it gets
stronger when social information is removed. The social information prey
do use seems to serve spacing, and neighbours' movement plays no visible
role. The difference from Hahn et al. is plausible given the setup. Their
agents were rewarded every step for tracking a target that most of them
could not see, so following neighbours was the only way to track it. Here
grass is in view about half the time, the only reward is a birth, and
predators add a cost to crowding.

**What could change this** (not tested): make following neighbours pay,
for example less visible food (shorter prey view distance or scarcer
grass), or predators that are confused by groups, as in Hahn et al. (2019).
More runs per condition, including new A runs, would make the coexistence
result testable (5 against 5: smallest possible p 0.004).

## Predator cooperation: restraint and group hunting (2026-10-20)

**Question.** Prey grouping turned out not to be social (Stage 2). Do the
predators cooperate instead? Ritz et al. (2021) found two forms among
selfish PPO predators with no cooperation reward: restraint (not hunting
the last prey, "herding") and group hunting. Both are measured here on the
trained policies, without new training.

[`measure_predators.py`](measure_predators.py) (18 checkpoints: BE-04 and
BE-05 at iterations 100-2,000 and final, B1-3, C1-3; 20-50 episodes each;
predators at least 10 decisions old):

- **No consistent restraint.** With 6-15 prey alive instead of 16-30,
  low-energy predators go for a prey 50-100 units away at most about 1
  point less hard (BE-04 -1.2 points, interval -1.8 to -0.6; BE-05 +0.3,
  -0.5 to +1.0), and the share of encounters ending in a catch does not
  change. Predators do stop more often when prey are scarce, but the
  hungriest stop most (42% against 26% when well fed): sit-and-wait energy
  saving (75% of the energy cost is movement), not restraint. Unlike Ritz
  et al.'s predators, these cannot see how many prey are left.
- **An apparent group-hunting signal.** At encounters (a prey within 64
  units) where another predator's chosen action heads for the same prey,
  the prey is caught by anyone in 0.33 of cases against 0.19 alone, and by
  the focal predator in 0.21 against 0.18, in all 8 trained runs; early in
  training (iterations 100-300) the focal predator's rate was lower when
  joined. This is an association: both predators may pick the same easy
  prey.

[`counterfactual_hunting.py`](counterfactual_hunting.py) removes that
selection. At a joint encounter it copies the world and replays the same
decision from the same state with the helper's action replaced (stop, or
run away from the prey), following each branch for 3 decisions with the
same random seed. The factual branch reproduced every real outcome
(0 mismatches in about 19,000 replays). Placebo: stopping a nearby
predator that was *not* heading for the prey.

| Checkpoint | Prey caught by anyone | … by the helper | … by the focal predator | Placebo (focal) |
|---|---:|---:|---:|---:|
| BE-04 final | +0.063 (+0.052, +0.074) | +0.072 | **-0.009** (-0.016, -0.003) | +0.006 |
| BE-05 final | +0.061 (+0.048, +0.075) | +0.077 | **-0.016** (-0.024, -0.008) | -0.002 |
| BE-04 iter 600 | +0.064 (+0.051, +0.077) | +0.078 | **-0.014** (-0.024, -0.004) | +0.004 |
| B1 | +0.047 (+0.037, +0.058) | +0.065 | **-0.018** (-0.026, -0.010) | -0.004 |
| C1 | +0.050 (+0.039, +0.061) | +0.061 | **-0.011** (-0.017, -0.005) | +0.000 |

Effect of the helper being there (factual minus helper stopped), within 3
decisions, with 95% intervals over episodes; outputs `analysis/results/cf_*`.

**Answer.** The helper does not help: it slightly lowers the focal
predator's catch chance, and the extra catches are only its own (no
synergy: the gain for "anyone" is the helper's catches minus what it takes
from the focal predator). Joint pursuit here is scramble competition, and
the earlier association was selection. This fits the ecology: the first
predator to touch a prey gets all its energy, so nothing makes hunting
together pay. With prey grouping (environmental), predator restraint
(absent) and group hunting (competition) all tested, the current PPG
world shows no emergent cooperation. Next: change the ecology so that
cooperation can pay, as Ritz et al. did (shared carcass, harder solo
catches), without rewarding it.

## Shared carcass: can cooperation pay? (2026-10-20 to 2026-10-21)

**Question.** In the current world the first predator to touch a prey
gets all its energy, so hunting together cannot pay, and it does not
emerge (counterfactual test above). Does predator cooperation emerge when
the ecology lets it pay, still with no reward for it? After Ritz et al.
(2021), who found group hunting when the catch was shared among predators
nearby and solo catches were made hard.

**Change** (`energy_catch_share_radius`, default 0 = unchanged; SPEC.md
section 5): a catch's energy (0.5 × the prey's energy) is split equally
among the catcher and every living predator within the radius of the
catch spot. This is an ecological rule (shared kills), not a reward: the
only reward is still +10 per birth. It also creates a free-rider dilemma:
a predator can be fed by staying near a hunt without chasing.

| Condition | Change | Runs |
|---|---|---|
| Baseline | none: BE-04 and BE-05 at iteration 2,000 | 2 (existing) |
| S1 | shared carcass, radius 96 | seeds 1, 2 |
| S2 | shared carcass, radius 96, and slower predators (top speed 4.5 instead of 5; prey 6.0): solo catches harder | seeds 1, 2 |

2,000 iterations each (as Stage 2), launched with `launchers/launch.py
<S1|S2> <seed>` in two queues side by side. Risk: in S2, slower predators
that also share food may starve (Ritz et al. found that too much pressure
stopped learning altogether); baseline against S1 stands on its own.

**Measures** (pre-specified): the counterfactual test (`counterfactual_hunting.py`)
on the final checkpoints, with the new outcome "focal predator fed from the
prey" (catches by others count when the focal predator is within the
radius); group hunting means the helper raises the focal predator's chance
to be fed. Free-riding: predators near catches that were not heading for
the prey. Coexistence and catch risk (as in Stage 2), since better hunting
could also drive prey extinct.

### Results

All at iteration 2,000, 50 episodes each (seeds 0-49). Counterfactual: the
helper's effect (factual minus helper stopped) on the focal predator within
3 decisions, 95% intervals over episodes. "Fed" means the focal predator
received energy from the prey (catcher, or within 96 units of a catch
when the carcass is shared). Outputs: `analysis/results/cf_*`,
`pred_*`, `stage3_*`.

| Run | Coexisting (of 50) | Catch risk | Predators / prey alive | Helper effect: focal fed | Helper effect: focal's own catches | Placebo | Joint pursuit share | Predators in groups ≥3 minus random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (BE-04) | 49 | 2.06 | 11 / 26 | -0.012 (-0.022, -0.003) | -0.012 (-0.022, -0.002) | +0.003 | 0.19 | 0.029 |
| A (BE-05) | 32 | 2.15 | 11 / 24 | -0.001 (-0.008, +0.007) | -0.001 (-0.008, +0.008) | -0.001 | 0.21 | 0.036 |
| S1 seed 1 | 46 | 1.61 | 10 / 32 | +0.062 (+0.049, +0.075) | -0.011 (-0.019, -0.003) | +0.003 | 0.15 | 0.034 |
| S1 seed 2 | 42 | 1.93 | 11 / 28 | +0.049 (+0.037, +0.062) | -0.013 (-0.020, -0.005) | -0.005 | 0.18 | 0.026 |
| S2 seed 1 | 50 | 1.72 | 11 / 30 | +0.049 (+0.038, +0.060) | -0.010 (-0.017, -0.003) | +0.007 | 0.15 | 0.015 |
| S2 seed 2 | 43 | 1.21 | 10 / 38 | +0.058 (+0.046, +0.069) | -0.004 (-0.011, +0.003) | +0.003 | 0.15 | 0.025 |

**Control: the rule alone.** The baseline policies (BE-04 and BE-05 at
iteration 2,000), which never trained with sharing, run in a world with a
shared carcass at test time: helper effect on "focal fed" +0.054
(+0.042, +0.067) and +0.055 (+0.043, +0.067); on the focal predator's own
catches -0.008 and -0.012. All replays reproduced the real outcome
(0 mismatches).

**Findings.**

1. **No learned cooperation.** With a shared carcass the helper raises
   the focal predator's chance of being fed by about 5-6 points in all four
   runs, but so does a helper running the baseline policies, which never
   trained with sharing. The gain is the sharing rule (a catch nearby feeds
   the focal predator), not behaviour learned to exploit it. The helper
   still slightly lowers the focal predator's own catch chance (no
   synergy), predators do not group more (excess over random 0.015-0.034
   against 0.029-0.036), and joint pursuit is, if anything, less common
   (0.15-0.18 of encounters against 0.19-0.21; all four S runs below both
   baseline runs, one-sided p = 0.07 for 4 against 2 runs).
2. **The ecology shifts: fewer catches, more prey.** Catch risk 1.21-1.93
   against 2.06-2.15 and prey alive 28-38 against 24-26: in both, all four
   S runs lie beyond both baseline runs (p = 0.07). One reading, not
   tested: sharing takes energy from the catcher, which slows predator
   reproduction and lets prey numbers rise. Coexistence overlaps (S 42-50,
   baseline 32-49 of 50).
3. **Slower predators (S2) did not starve and did not change the
   picture**: lone predators still catch about 19% of encounters. Solo
   catches stayed easy, so hunting together was never needed.

**Answer.** Sharing the carcass makes being *near* a catch pay, not
*helping* with it, and lone predators still succeed often. Neither creates
a reason to coordinate, and none emerged. Following Ritz et al., group
hunting needs lone predators to rarely succeed. A next step would be prey
that only two predators together can catch (an imposed capture rule, as in
the discrete `mammoths` modules, but coordination would still have to be
learned without a reward for it).

## Large prey: when lone hunting fails (2026-10-21 to 2026-10-22)

**Question.** Shared carcass and solo-catchable prey gave no reason to
coordinate. Does cooperation emerge when lone predators *cannot* catch
some prey, still with no reward for it?

**Rule** (`capture_assist_radius`, `capture_assist_threshold`; default
off; SPEC.md): a prey with at least the threshold energy is caught only
if, besides the predator touching it, another predator is within the
radius of it; the catch's energy is then shared with those helpers.
Smaller prey are caught as before. An imposed capture rule (as in the
discrete `mammoths` modules), but how to coordinate (finding a partner,
waiting, choosing large or small prey) has to be learned. Also available,
unused here: `observe_energy_in_slots` (each slot carries the observed
animal's energy).

**Why warm-started and mild.** From scratch, random predators catch so
little under the rule that they never reproduce (0 births per episode
against 1.5 in the current world), and the only reward is a birth, so
there would be no learning signal. Even predators trained in the current
world (BE-04 at iteration 2,000) die out when the rule is switched on,
unless it is mild (12 episodes each):

| Rule (warm-started predators, untrained for it) | Predator births / episode | Coexisting (of 12) |
|---|---:|---:|
| none | 34 | 12 |
| energy ≥ 5, helper within 64 or 96 | 0 | 0 |
| energy ≥ 6, helper within 96 | 3 | 1 |
| **energy ≥ 6, helper within 128** | **12** | **6** |
| every prey, helper within 96 or 128 | 0 | 0 |

A threshold of 5 fails because escaped prey keep grazing and an adult
prey drops back to energy 5 after each birth: nearly every adult becomes
large. The mild rule (energy ≥ 6, about 16% of living prey; helper within
128) is used.

| Condition | World | Start | Runs |
|---|---|---|---|
| K | large prey (energy ≥ 6) need a helper within 128 | BE-04 (seed 1) / BE-05 (seed 2) at iteration 2,000 (weights only) | 2 |
| W | unchanged (control) | the same | 2 |

1,000 iterations each (`launchers/launch.py <K|W> <seed>`). W separates
the rule from the effect of the warm start itself (fresh optimizer).

**Learned cooperation would show as**, in K against W:
(1) predators heading for *large* prey together more often than for
small prey (`measure_predators.py`, joint pursuit by prey size), (2) the
share of large-prey encounters ending in a catch rising over training,
(3) K predators recovering from the crash (6 of 12 coexisting untrained).

### Results

All after 1,000 iterations from the warm start (K, W), 50 episodes each;
"untrained in K" = the starting policies (BE-04 / BE-05 at iteration
2,000) run in the large-prey world without training for it. Large prey:
energy ≥ 6. Joint pursuit: another predator's chosen action heads for the
same prey at an encounter. Outputs: `analysis/results/pred_K*`,
`pred_W*`, `pred_untrained*`, `cf_K*`, `cf_W*`, `cf_untrainedK*`,
`stage4_*`.

| | Coexisting (of 50) | Joint pursuit: small / large prey | Large minus small (95%) | Large prey caught by anyone: alone / joint | Helper effect on focal fed (counterfactual) |
|---|---:|---:|---:|---:|---:|
| Untrained in K, from BE-04 | 18 | 0.120 / 0.127 | +0.007 (-0.008, +0.024) | 0.073 / 0.290 | +0.002 (-0.012, +0.019) |
| Untrained in K, from BE-05 | 17 | 0.141 / 0.147 | +0.006 (-0.011, +0.022) | 0.073 / 0.275 | +0.000 (-0.013, +0.014) |
| K seed 1 | 45 | 0.187 / 0.200 | +0.013 (+0.002, +0.025) | 0.110 / 0.295 | +0.007 (-0.005, +0.019) |
| K seed 2 | 34 | 0.161 / 0.162 | +0.001 (-0.010, +0.012) | 0.101 / 0.331 | +0.011 (+0.001, +0.021) |
| W seed 1 | 42 | 0.162 / 0.198 | +0.037 (+0.026, +0.047) | 0.171 / 0.309 | -0.015 (-0.023, -0.008) |
| W seed 2 | 45 | 0.180 / 0.225 | +0.044 (+0.035, +0.054) | 0.186 / 0.318 | -0.021 (-0.030, -0.013) |

Over training (K, 20 episodes per checkpoint), joint pursuit of large
minus small prey: seed 1 +0.004, +0.026, +0.009, +0.013 at iterations
50, 200, 500, 1,000; seed 2 +0.020, +0.018, +0.014, +0.001. Coexistence
stayed at 15-18 of 20 throughout. Predators in groups of 3 or more minus
random: K 0.025 / 0.033, W 0.034 / 0.047. All replays reproduced the real
outcome (0 mismatches); placebos about 0.

**Findings.**

1. **The predators adapted to the rule.** Untrained, they coexist in 17-18
   of 50 episodes in the large-prey world; after 1,000 iterations in it, 34
   and 45, close to W (42, 45). A lone predator's chance with a large prey
   rose from 0.073 to 0.10-0.11 (W, where large prey are no harder: 0.17-0.19).
2. **But not by hunting together.** Predators do not head for large prey
   together more than for small ones: the large-minus-small difference in
   joint pursuit is +0.001 and +0.013 in K, about the same as untrained
   (+0.006, +0.007) and *smaller* than in W (+0.037, +0.044), where large
   prey draw more joint pursuit without any rule. They do not group more
   either. Over training the difference does not grow.
3. **A helper is no longer a competitor, which is the rule.** In W a
   helper lowers the focal predator's chance of being fed (-0.015, -0.021);
   in K it is about neutral (+0.007, +0.011), as it already is untrained
   (+0.002, +0.000): helpers make large-prey catches possible, which offsets
   the competition. Training moved this by at most about 0.01.

**Answer.** Even when lone predators fail at large prey, the predators
adapted without learning to hunt together: the recovery came from somewhere
else (one possible reading, not tested: hunting small prey more and taking
large prey only when another predator happens to be near). As with the shared
carcass, the benefit of a second predator is what the rule makes it, not
learned coordination. Caveat: predators cannot see a prey's energy here
(the warm start needed the old observation shape), so they cannot tell
which prey need a partner; `observe_energy_in_slots` exists for a version
trained from scratch, which in turn needs a way past the zero-birth start.

## Files

| File | |
|---|---|
| `environment.py`, `grass.py`, `geometry.py` | the environment (copied; social-information settings added in `environment.py`) |
| `config/` | environment (quirks-fixed world), PPO (with `seed`), evaluation settings |
| `train.py`, `eval.py`, `tournament.py`, `render.py`, `checkpoints.py` | as in base_environment_continuous |
| `swarm_metrics.py` | spacing, alignment, groups, pair and group persistence |
| `measure_swarms.py` | Stage 0 / evaluation measurements and the counterfactual probe |
| `launchers/launch.py` | Stage 2 training runs |
| `measure_predators.py` | predator restraint and group-hunting measurements |
| `counterfactual_hunting.py` | counterfactual replay test of group hunting |
| `analysis/results/` | Stage 0 outputs (`stage0_<run>_<A|Bt|Ct>`, `_iter<n>`) |
| `tests/` | the copied environment tests, plus the social-information settings and the metrics |

## References

- Hahn, C., Ritz, F., Wikidal, P., Phan, T., Gabor, T., & Linnhoff-Popien, C. (2020). Foraging swarms using multi-agent reinforcement learning. *ALIFE 2020: The 2020 Conference on Artificial Life*, 333–340.
- Hahn, C., Phan, T., Gabor, T., Belzner, L., & Linnhoff-Popien, C. (2019). Emergent escape-based flocking behavior using multi-agent reinforcement learning. *ALIFE 2019*.
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *SIGGRAPH '87*, 25–34.
- Ritz, F., Ratke, D., Phan, T., Belzner, L., & Linnhoff-Popien, C. (2021). A sustainable ecosystem through emergent cooperation in multi-agent reinforcement learning. *ALIFE 2021*, 74.
- Hahn, C., Phan, T., Feld, S., Roch, C., Ritz, F., Sedlmeier, A., Gabor, T., & Linnhoff-Popien, C. (2020). Nash equilibria in multi-agent swarms. *ICAART 2020*, 234–241.
- Clark, P. J., & Evans, F. C. (1954). Distance to nearest neighbor as a measure of spatial relationships in populations. *Ecology*, 35(4), 445–453.
