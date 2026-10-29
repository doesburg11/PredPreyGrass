# eco_evolutionary_continuous_fov: results

## 1. Calibration of the distance noise (2026-10-27)

**Question** (design note, plan step 1). How much does monocular distance
noise σ hurt hunting and escaping, before any policy has learned with it?

**Design.** [`calibrate_noise.py`](calibrate_noise.py): the CT-05 branch
point (eco_evolutionary_continuous_traits at iteration 1,500), its predator
and prey policies (trained without noise) and its gene pools held fixed; σ
0, 0.15, 0.3 and 0.5, applied to the predators only, the prey only, or
both; 400 episodes per condition on common seeds and common noise draws.
Rates are catches per 1,000 predator decisions, and per 1,000 decisions in
which the predator's nearest prey in view was in front (inside its
binocular overlap) or to the side; ratios to σ 0 with 95% bootstrap
intervals over seeds, paired across conditions
([`results/calibrate_noise_summary.json`](results/calibrate_noise_summary.json)).

| Noise on | σ | Catches per predator decision (vs σ 0) | Per front opportunity | Per side opportunity | Prey survive the episode |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 1.00 | 1.00 | 1.00 | 99% |
| predators | 0.15 | 1.01 (1.00–1.02) | 0.99 (0.98–1.01) | 1.00 (0.98–1.02) | 97% |
|  | 0.3 | 1.01 (1.00–1.02) | 0.98 (0.96–1.00) | 0.97 (0.95–1.00) | 98% |
|  | 0.5 | 1.02 (1.01–1.03) | 0.97 (0.95–0.99) | 0.96 (0.94–0.98) | 100% |
| prey | 0.15 | 0.98 (0.97–0.99) | 1.02 (1.00–1.04) | 1.02 (1.00–1.05) | 96% |
|  | 0.3 | 0.94 (0.93–0.95) | 1.11 (1.09–1.13) | 1.11 (1.09–1.14) | 89% |
|  | 0.5 | 0.92 (0.91–0.93) | 1.27 (1.25–1.29) | 1.28 (1.25–1.30) | 69% |
| both | 0.15 | 0.98 (0.97–0.99) | 1.02 (1.00–1.04) | 1.02 (1.00–1.04) | 96% |
|  | 0.3 | 0.95 (0.94–0.96) | 1.11 (1.09–1.13) | 1.10 (1.08–1.13) | 92% |
|  | 0.5 | 0.93 (0.92–0.94) | 1.23 (1.21–1.25) | 1.25 (1.23–1.28) | 76% |

**Result.** Noise barely hurts the predators: with noise on predators only,
catches per opportunity fall by at most 3–4% at σ = 0.5, about equally in
front and to the side, and catches per predator decision even rise
slightly. Noise hurts the prey: with noise on prey only, catches per
opportunity rise by 11% at σ = 0.3 and 28% at σ = 0.5, and prey survive
fewer episodes (89% and 69%, against 99%); catches per predator decision
fall because fewer prey are left (prey per predator decision 2.4 at σ 0,
1.5 at σ 0.3, 1.1 at σ 0.5).

**Reading.** The design's premise for predators does not hold for these
policies: they hunt about as well with noisy distances, presumably because
pursuit needs the direction to the prey, which stays exact, more than its
distance (a hypothesis, not tested here). A catch is a contact, with no
strike that needs a judged distance. So the trade-off as built puts little
selection on the predators' field of view, and test 1's predicted predator
narrowing is unlikely; the prey bear the cost, which could select for a
narrower prey field (more overlap) as well as a wider one (more coverage).
Caveats: policies trained without noise (trained ones would cope better,
making the predator cost smaller still), one branch point, end-to-end rates
rather than an interception assay.

## 2. Calibration with strike catching (2026-10-27)

**Question.** With catches by strikes (design note, addendum), does
distance noise now cost the predators, and more to the side than in front?

**Design.** As section 1 (`calibrate_noise.py --strike`, 400 episodes per
condition, 2026-10-27 10:57 to 11:44), every condition with
`strike_catching` (reach 0, cooldown 16 physics steps), plus contact
catching at σ 0 for the change of ecology from strikes alone
([`results/calibrate_noise_strike_summary.json`](results/calibrate_noise_strike_summary.json)).
Ratios to strikes at σ 0, 95% bootstrap intervals over seeds.

| Noise on | σ | Strikes that miss | Per front opportunity | Per side opportunity | Predator births per episode | Prey survive |
| --- | --- | --- | --- | --- | --- | --- |
| none, contact catching | 0 | – | 0.99 (0.97–1.01) | 1.07 (1.05–1.10) | 36.8 | 99% |
| none | 0 | 0% | 1.00 | 1.00 | 36.9 | 98% |
| predators | 0.15 | 22% | 0.98 (0.96–0.99) | 0.71 (0.69–0.73) | 32.3 | 100% |
|  | 0.3 | 36% | 0.94 (0.92–0.96) | 0.59 (0.57–0.61) | 29.5 | 100% |
|  | 0.5 | 50% | 0.91 (0.89–0.93) | 0.54 (0.53–0.56) | 26.1 | 100% |
| prey | 0.3 | 0% | 1.11 (1.09–1.13) | 1.09 (1.06–1.11) | 39.0 | 92% |
|  | 0.5 | 0% | 1.26 (1.24–1.29) | 1.25 (1.23–1.28) | – | 70% |
| both | 0.3 | 34% | 1.04 (1.02–1.06) | 0.67 (0.65–0.68) | 36.4 | 98% |
|  | 0.5 | 48% | 1.15 (1.13–1.18) | 0.70 (0.68–0.72) | – | 94% |

**Result.** Strikes alone hardly change the ecology: at σ 0 the catch rate
per opportunity is within 1% of contact catching in front and 7% lower to
the side (prey touching from outside the view are no longer caught), with
the same predator births. With noise on the predators, catches of prey to
the side fall by 29%, 41% and 46% at σ 0.15, 0.3 and 0.5, against 2%, 6%
and 9% in front, and predator births per episode fall from 37 to 30 at
σ 0.3; prey accumulate (prey per predator decision 2.6 at σ 0, 5.7 at
σ 0.3), which is why catches per predator decision rise slightly. The
prey's cost of noise is as in section 1. With noise on both at σ 0.3 both
species survive 98–99% of episodes.

**Reading.** With strikes the design's premise holds for these policies:
for a predator, prey to the side of its binocular overlap are much harder
to catch, so a narrower field of view (more overlap) should pay, while the
prey keep their cost of noise. σ = 0.3 gives a large asymmetry for
predators (front −6%, side −41%) and a viable ecosystem with noise on both;
it stays the pilot's setting. Caveats as in section 1: policies trained
without noise or strikes, one branch point, end-to-end rates.

## 3. Pilot: does the trade-off make the species' eyes diverge? (2026-10-28)

**Question** (design note, plan step 2, pre-specified test 1). With the
trade-off (σ 0.3), does the predators' field-of-view gene fall from start
to end while the prey's does not, when without it (σ 0) both rise?

**Design.** [`launchers/pilot.py`](launchers/pilot.py): one run per arm, σ
0.3 (trade-off) and σ 0 (control), both with strike catching, otherwise the
CT-05 settings (`config/config_env.py`); 10 workers, seed 1, 3,000
iterations, run side by side (2026-10-27 12:41 to 2026-10-28 about 05:20).
Analysis: [`analyze_pilot.py`](analyze_pilot.py), from the gene-pool
snapshots saved every 10 iterations (not the EMA-smoothed TensorBoard
means). Start: the founders of every worker's first episode, rebuilt
exactly from the run's seeds (identical in both arms); end: mean of the
snapshots over the last 300 iterations (2,710 to 3,000). Sustainability
guard (test 3): [`final_check.py`](final_check.py), 50 episodes from each
final checkpoint, gene pools held fixed. Summary:
`results/pilot_summary.json`; figure: `assets/pilot_traits.png`.

| Arm | Species | fov gene, start | end | change | Workers with change > 0 | Field of view at end | Binocular overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trade-off (σ 0.3) | predator | +0.02 | +0.07 | **+0.05** | 10 of 10 (+0.01 to +0.09) | 161° | 139° |
|  | prey | −0.04 | +0.55 | +0.59 | 10 of 10 | 360° (capped) | 0° |
| control (σ 0) | predator | +0.02 | +0.59 | +0.57 | 10 of 10 | 272° | 28° |
|  | prey | −0.04 | +0.56 | +0.60 | 10 of 10 | 360° (capped) | 0° |

| Arm | Coexisting (of 50) | Catch risk | Mean predators | Mean prey |
| --- | --- | --- | --- | --- |
| trade-off | 50 | 1.08 | 14.7 | 44.0 |
| control | 48 (2 prey extinct) | 1.98 | 13.6 | 26.6 |

(Catch risk: prey caught per prey per 1,000 physics steps.)

**Result.** Test 1 fails as pre-specified: the predators' field-of-view
gene did not fall, it rose slightly (+0.05; in every worker). The other two
parts hold: the prey's rose (+0.59), and in the control both rose (+0.57,
+0.60), as in CT-04 and CT-05. The guard (test 3) passes in both arms (50
and 48 of 50, threshold 45). Measured from the first snapshot (iteration
10) instead of the founders, the predators' change is −0.02; the
pre-specified baseline is the start, so this does not change the verdict.

**What happened instead.** The trade-off made a large difference to the
predators, not in the predicted direction but in size: their gene stayed
near 0 (field 161°, 86% of it binocular) for the whole run, while in the
control it rose to 0.59 within 30 iterations (272°, 11% binocular). The
prey reached the full 360° in both arms. So in the trade-off run the two
species ended with different eyes, a narrow mostly binocular predator field
and a panoramic prey field, while in the control both ended wide.

**Reading (post hoc, one seed per arm).** The pre-specified direction did
not fit the geometry. With the monocular field M at 150°, the predators'
starting field (150°, gene 0) is already fully binocular: narrowing it
adds no binocular overlap and only loses field, so the trade-off can hold
the predators' gene at 0 but has no reason to push it below. The predators
stayed just above that corner, which is what a fitness optimum at the
corner would look like; it is not a test of it. Likewise, above gene 0.405
the prey's field is capped at 360° and the gene costs nothing, so the
prey's level above that is not under selection. Prey in the trade-off run
first narrowed their field (gene −0.56 around iteration 80, about 137°,
fully binocular) and switched to panoramic only from about iteration 640;
the control's prey did so by iteration 100. Why is open. Other
differences between the arms: predator speed gene −0.13 against +0.38,
and fewer catches (catch risk 1.08 against 1.98) with more prey (44 against
27). Caveats: one seed per arm, workers share policies (not independent
replicates), and the size of the difference from the control was not
the pre-specified criterion.

## 4. Pilot 1: do prey react to predators they see, and do eyes and grouping cycle? (2026-10-28)

**Questions** (descriptive, not pre-specified; design note, related work).
(a) Buason and Ziemke's (2003) prey evolved to circle whatever the
predator did; a prey that ignores predators cannot coevolve its eyes with
them. Do our prey react to a predator in view? (b) Olson et al. (2016)
found predators' view angle and prey swarming cycling together. Do they
here?

**Design.** [`prey_behaviour.py`](prey_behaviour.py): pilot 1's checkpoints
every 100 iterations, both arms, 8 episodes each (seeds 0 to 7, gene pools
held fixed). (a) At every decision of a prey with a predator in its
observation, the prey policy's action distribution from the observation as
it is and with the predator slot blanked (as when none is in view): the
total-variation distance between them, and the expected velocity away from
that predator (true offset, in units of top speed) under each. Placebo:
blanking another prey instead. (b) The prey's Clark-Evans ratio R (mean
nearest-neighbour distance over its value for the same number of prey
placed at random on the same torus; R < 1: grouped) at every decision,
correlated over checkpoints with the predators' mean field-of-view gene
(section 3), as levels and as changes. Summary:
`results/prey_behaviour_summary.json`; figure: `assets/prey_behaviour.png`.

| Arm | Iteration | Velocity away, predator seen | predator blanked | Gain | Action change (TV), predator blanked | other prey blanked | Prey R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trade-off | 100 | −0.20 | −0.42 | +0.22 | 0.34 | 0.26 | 0.75 |
|  | 700 | +0.24 | −0.22 | +0.45 | 0.68 | 0.43 | 0.81 |
|  | 3,000 | +0.31 | −0.16 | +0.48 | 0.81 | 0.52 | 0.89 |
| control | 100 | +0.34 | −0.16 | +0.50 | 0.68 | 0.25 | 0.86 |
|  | 3,000 | +0.24 | −0.13 | +0.37 | 0.83 | 0.54 | 0.88 |

**Result.** (a) Yes. In both arms, from early on, a predator in view moves
the prey's actions away from it: at the end the expected velocity away
from the predator is 0.31 (trade-off) and 0.24 (control) of top speed, and
−0.16 and −0.13 with the predator blanked; blanking the predator changes
the action distribution more than blanking another prey (TV 0.81 against
0.52, 0.83 against 0.54). (Blanked, prey move slightly towards predators
they would see, probably because what is in view tends to lie ahead of a
moving prey.) The trade-off run's prey learned this later: at iteration
100 to 400 they still moved towards seen predators (−0.20 to −0.06) and
saw few (3,000 to 5,000 decisions with a predator in view against 60,000
to 70,000 in the control); flight appeared between iterations 400 and
700, about when their field of view widened from about 137° to 360°
(section 3, from about iteration 640). Which came first, and whether
either caused the other, these data cannot say. (b) No cycle within 3,000
iterations: the predators' gene changed little after the first 30 to 100
iterations in either arm, prey were mildly grouped throughout (R 0.75 to
0.89, drifting up), and the two were uncorrelated (r 0.04 and −0.05 for
levels, −0.15 and −0.11 for changes).

**Reading.** The prey in this world do use vision against predators, so
the failure mode of Buason and Ziemke's prey does not apply. Olson et
al.'s cycle took thousands of generations in 30 replicates; one run per
arm of 3,000 iterations shows none, which does not rule one out. Without
a confusion effect (README TODO) prey grouping gives no protection the
predators' eyes would respond to, so no coupling is expected yet.

## 5. Pilot 2: monocular field 120°, re-registered test 1 (2026-10-29)

**Question** (design note, addendum of 2026-10-28, written before the
runs). With the monocular field M lowered from 150° to 120°, the
predators' starting field (about 150°) is only about 60% binocular, so
narrowing it now gains depth vision. Does the predators' field-of-view
gene fall from the founders to the end, and the prey's not fall, in both
of two runs? Secondary: the predators' end gene is below pilot 1's
trade-off run (+0.07) in both. Test 3 as before.

**Design.** [`launchers/pilot.py`](launchers/pilot.py), arms
`tradeoff_m120` (seed 1) and `tradeoff_m120_seed2` (seed 2): σ 0.3, M
120°, strike catching, otherwise as pilot 1; 10 workers, 3,000
iterations, run side by side (2026-10-28 about 10:20 to 2026-10-29 12:48
and 15:24). Seed 1 has the same seeds, and so the same founders, as both
pilot-1 arms: it differs from pilot 1's trade-off run only in M. The σ 0
control is pilot 1's (without noise M has no effect). Analysis as in §3
([`analyze_pilot.py`](analyze_pilot.py), now with all four runs);
summary: `results/pilot_summary.json`; figure: `assets/pilot_traits.png`;
test 3: `results/final_check_tradeoff_m120[_seed2]_summary.json`.

| Run | Species | fov gene, start | end | change | Workers with change < 0 | Field of view at end | Binocular overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed 1 | predator | +0.02 | −0.20 | **−0.22** | 10 of 10 (−0.26 to −0.16) | 122° | 118° (96%) |
|  | prey | −0.04 | +0.54 | +0.59 | 0 of 10 | 360° (capped) | 0° |
| seed 2 | predator | −0.03 | −0.15 | **−0.13** | 10 of 10 (−0.17 to −0.08) | 129° | 111° (86%) |
|  | prey | −0.03 | +0.55 | +0.59 | 0 of 10 | 360° (capped) | 0° |
| *pilot 1, M 150° (§3)* | predator | +0.02 | +0.07 | +0.05 | 0 of 10 | 161° | 139° |
| *control, σ 0 (§3)* | predator | +0.02 | +0.59 | +0.57 | 0 of 10 | 272° | 28° |

(Founders' field: 153° with 87° binocular, 57%, for seed 1; 146° with
94°, 64%, for seed 2.)

| Run | Coexisting (of 50) | Catch risk | Mean predators | Mean prey |
| --- | --- | --- | --- | --- |
| seed 1 | 50 | 1.06 | 14.4 | 47.9 |
| seed 2 | 50 | 1.16 | 15.2 | 43.9 |
| *pilot 1 trade-off* | 50 | 1.08 | 14.7 | 44.0 |
| *control* | 48 | 1.98 | 13.6 | 26.6 |

**Result.** Test 1, re-registered, passes in both runs: the predators'
gene fell (−0.22 and −0.13, in every worker of both), the prey's rose
(+0.59 in both). The secondary criterion holds (end −0.20 and −0.15,
below +0.07). Test 3 passes (50 and 50 of 50). Measured from the first
snapshot (iteration 10) the predators' change is −0.23 and −0.11.

**Course.** The predators' gene fell below −0.1 within 30 iterations in
both runs and then stayed in a narrow band (iterations 300 to 3,000: mean
−0.17 and −0.16, SD 0.03 in both), close to but on average above the
corner at −0.22 (field 120°, fully binocular); single snapshots reached
−0.27 and −0.26. The prey's field passed 360° (gene 0.405) by iteration
190 and 120, without the detour through a narrow field that pilot 1's
trade-off prey took (until about iteration 640, §3).

**Reading.** For seed 1, whose founders and seeds match pilot 1's and
which differs from pilot 1's trade-off run only in M, the M 150° run
stayed near 0 while the M 120° run ended near the new corner, and the σ 0
control rose to +0.59. Seed 2, with other founders and no matched M 150°
run, reproduced the narrowing at M 120°. This satisfies the re-registered
directional prediction: predators narrowed, to mostly binocular fields,
while prey became panoramic and monocular. It is consistent with selection
towards a nearly fully binocular predator field, but two seeds and two
values of M do not establish that the predators follow the corner wherever
it is. Nor do these data tell an optimum at the corner itself from an
interior optimum that trades a little overlap for field; one possible
explanation for a mean slightly above the corner, untested, is that a
mutating population around a kinked optimum spreads mostly to the side
where fitness falls more slowly. Caveats: two seeds, the control from one
seed (pilot 1's), workers share the policy and are not independent
replicates. Other differences, not pre-specified: seed 1's predators also
ended slower (speed gene −0.24, against −0.13 in pilot 1's trade-off run
and +0.38 in the control; seed 2: −0.10); vision fell to about −0.55 to
−0.6 in both species in all four runs, the control included, so it is not
an effect of the trade-off. This is divergence, not coevolution: test 2
(does one species' field evolve differently when the other's gene pool is
held fixed) is still to come.
