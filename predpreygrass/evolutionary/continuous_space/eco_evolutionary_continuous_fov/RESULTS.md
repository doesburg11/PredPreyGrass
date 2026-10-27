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
