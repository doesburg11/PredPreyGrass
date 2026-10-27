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
