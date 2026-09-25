# predator_complementary_diet

A clone of [`predator_sexual_reproduction`](../predator_sexual_reproduction/) (sexed predators, two-parent
reproduction, sex-specific hunting odds, mate gift, parental care) with a **complementary diet** added. Nothing in
the original module was changed, so its results and runs stay valid. Non-evolutionary: learning is PPO only.

## STATUS (2026-09-25): under calibration, not yet viable

The diet requirement has been built and tested (81 tests) but two pilot runs did not produce a viable population:

| Pilot | Meat share of costs | Result |
|---|---|---|
| 1 (37 of 100 iterations, stopped) | 0.25 | Females extinct in every episode; ~7.5 female deaths per episode from meat deficiency |
| 2 (60 iterations) | 0.10 | Females still extinct in ~90-100% of episodes; deaths flipped to fruit deficiency (~10 per episode) plus combat deaths (females hunt ~98 times per episode at 20% success, 10% death per attempt) |

The male side is fine (89% hunting success, 4-6 alive at the end). Moving the meat share only moves the failure from
one store to the other. Both pilots are short (the base module needed ~300 iterations for its sex split to settle), so
they are calibration evidence, not a verdict. A three-value sweep (0.10 / 0.15 / 0.20, 150 iterations, scripted prey)
is the next step. Run names: `PPO_PREDATOR_COMPLEMENTARY_DIET_*` under `~/simulation_results/ray_results/`.

## What is new

- **Two stores.** Every predator keeps total energy `E` (as before) plus a **fruit store** `F`; the meat store is
  `E - F`. Fruit adds to `F`, prey only to `E`.
- **Fixed-share costs.** Running (homeostatic, movement) and birth costs are drawn `diet_meat_cost_share` from meat and the
  rest from fruit, so neither food substitutes for the other.
- **Death and breeding gated on both stores** (`diet_required`): a predator dies when either store reaches zero, and can
  reproduce only with both stores above a floor that also covers its share of the birth cost.
- **Reciprocal exchange.** The existing male meat gift to the recorded mate has a fruit counterpart: a female passes
  `female_gift_donation_rate` of the fruit she eats to her recorded mate (within `predator_gift_range`). Parental care keeps
  the food type. Gifts and care never rescue a recipient already doomed by a store (turn-order safe).
- **Observation.** A sixth channel holds every predator's fruit store in the window; channel 1 still holds total energy.
  Sex and mate identity are still not observable.
- **Scripted prey (`scripted_prey`, default off).** Prey are moved by a fixed rule (flee a predator within
  `prey_flee_radius`, else walk to the nearest grass) and are not learning agents. They were ~79% of sampled agent
  steps; with them scripted, an iteration takes 3.4 s instead of 17.6 s (5.1x) at the same predator data. It changes the
  ecology (fleeing prey survive better: ~50 alive versus ~45) and is not directly comparable with learned-prey runs.
- **Controls via flags:** `--diet-required 0` (no requirement), `--female-gift-rate 0 --male-gift-rate 0` (diet without
  exchange), `--diet-meat-cost-share`, `--initial-num-fruit`, `--scripted-prey 1`.
- **Viewer:** prey are drawn as a teal rabbit icon (the mammoth icon is kept for planned big game).

## Running

```
python -m predpreygrass.non_evolutionary.predator_complementary_diet.tune_ppo_predator_complementary_diet \
    --seed 42 --max-iters 150 --reward-per-energy 0.5 --penalty-combat-death 0.2 --minibatch-size 1024 \
    --diet-meat-cost-share 0.15 --scripted-prey 1
python -m predpreygrass.non_evolutionary.predator_complementary_diet.random_policy      # random-policy viewer
pytest predpreygrass/non_evolutionary/predator_complementary_diet/tests/ -v
```

## Ideas discussed, not built

Band formation (initial bands of families and singles with within-band sharing, marriage between bands, kin exclusion)
and big game ("mammoths" needing several hunters, with the kill split by band strength) are design ideas only; they
would go in a further module (`predator_bands`).
