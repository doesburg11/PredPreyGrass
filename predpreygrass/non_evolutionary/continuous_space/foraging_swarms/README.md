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

## Stage 2: training comparison (prepared, not started)

| Condition | Runs | |
|---|---|---|
| A | BE-04, BE-05 + 1 new | the new A run checks that this module at HEAD trains like BE-04/05 |
| B | 3 | prey see other prey's positions only |
| C | 3 | prey see no other prey |

```bash
systemd-run --user --unit=sw-B-1 --working-directory=$PWD \
    .conda/bin/python predpreygrass/non_evolutionary/continuous_space/foraging_swarms/launchers/launch.py B 1
```

[`launchers/launch.py`](launchers/launch.py) `<A|B|C|BP|CP> <seed>`: 3,000
iterations, 14 simulation workers (two runs side by side on 32 cores, as
BE-04 and BE-05), into `runs/foraging_swarms_<time>_<condition>_seed<seed>/`.
BE-04 took 14.2 hours, so 7 runs two at a time take about 50 hours. Four at
a time may be faster; measure that on the first iterations before relying
on it. The `seed` setting (`config_ppo`) is recorded with the run, but
training is **not** bit-reproducible with remote simulation workers: two
1-iteration runs with the same seed collected different episodes.

**Pre-specified outcomes:**

- **Primary:** coexistence (of 50 evaluation episodes, seeds 0-49), prey
  catch risk, prey grass intake.
- **Secondary:** the swarm measurements above at the final checkpoint and
  every 300 iterations.
- **Social information:** distance to grass after losing it, and time
  until it is seen again, in A against B and C. The velocity probe applies
  to A only: B and C policies never see a conspecific's velocity, so there
  is nothing to rotate (`measure_swarms` collects no probe observations
  for them).

Between-run variation is large: BE-04 settled at iteration 371 and BE-05
at 1,366. With three runs per condition, report directions across runs
and avoid causal claims beyond the manipulation itself.

## Files

| File | |
|---|---|
| `environment.py`, `grass.py`, `geometry.py` | the environment (copied; social-information settings added in `environment.py`) |
| `config/` | environment (quirks-fixed world), PPO (with `seed`), evaluation settings |
| `train.py`, `eval.py`, `tournament.py`, `render.py`, `checkpoints.py` | as in base_environment_continuous |
| `swarm_metrics.py` | spacing, alignment, groups, pair and group persistence |
| `measure_swarms.py` | Stage 0 / evaluation measurements and the counterfactual probe |
| `launchers/launch.py` | Stage 2 training runs |
| `analysis/results/` | Stage 0 outputs (`stage0_<run>_<A|Bt|Ct>`, `_iter<n>`) |
| `tests/` | the copied environment tests, plus the social-information settings and the metrics |

## References

- Hahn, C., Ritz, F., Wikidal, P., Phan, T., Gabor, T., & Linnhoff-Popien, C. (2020). Foraging swarms using multi-agent reinforcement learning. *ALIFE 2020: The 2020 Conference on Artificial Life*, 333–340.
- Hahn, C., Phan, T., Gabor, T., Belzner, L., & Linnhoff-Popien, C. (2019). Emergent escape-based flocking behavior using multi-agent reinforcement learning. *ALIFE 2019*.
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *SIGGRAPH '87*, 25–34.
- Clark, P. J., & Evans, F. C. (1954). Distance to nearest neighbor as a measure of spatial relationships in populations. *Ecology*, 35(4), 445–453.
