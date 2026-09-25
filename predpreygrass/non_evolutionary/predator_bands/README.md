# predator_bands

A clone of [`predator_complementary_diet`](../predator_complementary_diet/) (sexed predators, two-parent reproduction,
fruit/meat stores with a complementary-diet requirement, scripted prey) with **band living** added. Earlier modules are
unchanged. Non-evolutionary: PPO learning only. The design and its open questions are in
[`../predator_complementary_diet/BANDS_DESIGN.md`](../predator_complementary_diet/BANDS_DESIGN.md).

## STATUS (2026-09-25): built, under calibration

Built and tested (113 tests; Codex review found 4 issues, all fixed). First calibration runs (seed 42, 100 iterations,
5 s per iteration, 30 learning predators):

| Run | Meat share | Band sharing | Result |
|---|---|---|---|
| Pilot (stopped at 82 iterations) | 0.25 | 0.3 | Females extinct in every episode from iteration 25; hunters supplied only ~120 energy of meat per episode against ~250 needed |
| CALIB M010_SHARE030 | 0.10 | 0.3 | Females extinct in 92% of episodes early, 67% late; episode length 362 to 737; ~7 female births and ~7 marriages per episode |
| CALIB M010_NOSHARE (control) | 0.10 | 0 | Females extinct in ~100% of episodes; episode length ~390; ~3 births per episode |
| CALIB M005_SHARE030 | 0.05 | 0.3 | (was running when this was written) early: fruit-deficiency deaths dominate |

Band sharing clearly helps (control loses all females), but females are not yet sustained, and 100 iterations is short.
Fruit supply (100 patches, 0.04 regrowth per step) was sized for 6-10 predators and looks under-scaled for 30. Next planned
runs: meat share 0.10 with more fruit regrowth, sharing on and off, 200 iterations.

## What is new (on top of the diet module)

- **Initial bands:** `num_bands` (5) bands of a founding couple (recorded mates), `band_children_per_couple` (2) dependent
  children (alternating male, female), and one unpaired male and female each; members start on the free cells nearest a band
  centre (`band_spawn_radius`, enforced), centres spread by farthest-point sampling. 30 predators by default.
- **Band sharing:** `band_share_rate` (0.3) of any forage, prey or fruit, is split equally among the forager's living band
  members within `band_share_range` (5); meat stays meat and fruit stays fruit; doomed members are not rescued. Mechanical, no
  reward for grouping. The old mate gifts are set to 0 here; parental care stays at 0.2.
- **Kin exclusion** (`kin_exclusion`): no parent-child or sibling mating.
- **Marriage** (`marriage_rule`): a cross-band pairing moves one partner (default the female) and their dependent children
  into the other's band; newborns join the father's band.
- **Observation:** channels 6 and 7 mark same-band and other-band predators (8 channels in total). Sex is still not visible.
- `num_bands = 0` turns all band features off (the diet-module behaviour).
- New metrics: `band_share_events`, `band_share_meat_total`, `band_share_fruit_total`, `marriages`, `within_band_pairings`,
  `kin_blocked_checks`, `bands_alive`.

## Running

```
python -m predpreygrass.non_evolutionary.predator_bands.tune_ppo_predator_bands \
    --seed 42 --max-iters 100 --reward-per-energy 0.5 --penalty-combat-death 0.2 --minibatch-size 1024 \
    --diet-meat-cost-share 0.10 --band-share-rate 0.3
python -m predpreygrass.non_evolutionary.predator_bands.random_policy
pytest predpreygrass/non_evolutionary/predator_bands/tests/ -v
```
Flags: `--num-bands`, `--band-share-rate`, `--band-share-range`, `--kin-exclusion 0/1`, `--marriage-rule`,
`--diet-meat-cost-share`, `--scripted-prey 0/1`, `--initial-num-fruit`.

## Not built

Big game ("mammoths" needing several hunters, kill split by band strength): see the design draft.
