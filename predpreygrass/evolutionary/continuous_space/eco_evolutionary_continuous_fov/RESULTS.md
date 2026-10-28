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
