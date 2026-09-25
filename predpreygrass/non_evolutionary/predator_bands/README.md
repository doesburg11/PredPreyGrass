# predator_bands

A clone of [`predator_complementary_diet`](../predator_complementary_diet/) (sexed predators, two-parent reproduction,
fruit/meat stores with a complementary-diet requirement, scripted prey) with **band living** added. Earlier modules are
unchanged. Non-evolutionary: PPO learning only. The design and its open questions are in
[`../predator_complementary_diet/BANDS_DESIGN.md`](../predator_complementary_diet/BANDS_DESIGN.md).

## STATUS (2026-09-25): built, under calibration

Built and tested (113 tests; Codex review found 4 issues, all fixed). First calibration runs (seed 42, 100 iterations,
5 s per iteration, 30 learning predators):

| Run (seed 42) | Meat share | Fruit regrowth | Band sharing | Result |
|---|---|---|---|---|
| Pilot (stopped at 82 iterations) | 0.25 | 0.04 | 0.3 | Females extinct in every episode from iteration 25; hunters supplied only ~120 energy of meat per episode against ~250 needed |
| CALIB M010_SHARE030 (100 it.) | 0.10 | 0.04 | 0.3 | Female extinction 92% early, 67% late; episode length 362 to 737; ~7 female births and ~7 marriages per episode |
| CALIB M010_NOSHARE (100 it., control) | 0.10 | 0.04 | 0 | Females extinct in ~100% of episodes; episode length ~390; ~3 births per episode |
| CALIB M005_SHARE030 (100 it.) | 0.05 | 0.04 | 0.3 | Female extinction 80% early, 13% late; episodes near the 1000-step cap; ~5 females and ~11 males alive; deaths now mostly fruit deficiency |
| CALIB F008_SHARE030 (200 it.) | 0.10 | 0.08 | 0.3 | Female extinction 100% at iterations 25-75, then 77-85% (flat); episode length ~750; ~23 female meat-deficiency deaths per episode, ~1-3 from fruit; ~14 marriages per episode |
| CALIB F008_NOSHARE (200 it., control) | 0.10 | 0.08 | 0 | Female extinction 96-100% throughout; episode length ~410; ~15 female meat-deficiency deaths per episode |
| CALIB F008_MEAT060 (200 it., seed 42) | 0.10 | 0.08 | meat 0.6 / fruit 0.3 | **Viable.** Female extinction 61% (iterations 25-50), then 2-12% and stable; episodes ~970-990 steps (cap 1000); 6-9 females and ~28 males alive; ~35-40 births and ~35 marriages per episode; ~470 energy of meat shared per episode. High turnover: ~20 female meat- and ~14 fruit-deficiency deaths per episode. One seed |
| CALIB F008_MEAT060 seed 43 (200 it.) | 0.10 | 0.08 | meat 0.6 / fruit 0.3 | **Replicates seed 42.** Female extinction 85% (iterations 0-25), 36%, 11%, then 2-7% and stable; episodes ~1000 steps; 6-9 females and ~26-29 males alive; ~36 births and ~37-40 marriages per episode; ~520-550 energy of meat shared per episode; ~14 female fruit- and ~19-21 meat-deficiency deaths per episode |
| CALIB F008_MEAT060_MJF (200 it., seed 42, `male_joins_female`) | 0.10 | 0.08 | meat 0.6 / fruit 0.3 | **Viable, more so for females than the default rule.** Female extinction 68% (iterations 0-25), 5%, 2%, then 0%; episodes ~1000 steps; 12-14 females and ~30 males alive (default rule: 6-9 and ~28); ~44-46 female births and ~32 marriages per episode; ~55 within-band pairings (default: ~35); ~700 energy of meat shared per episode (default: ~470); **only ~2.4 of 5 bands survive (default: ~4.5)**, i.e. the population merges into a few large bands; ~20 female fruit- and ~17 meat-deficiency deaths per episode. One seed |

Band sharing clearly helps (the controls lose all their females), but at meat share 0.10 a 30% share does not sustain them. More fruit
regrowth removed the fruit-deficiency deaths but not the meat ones, so the limit is how meat is distributed, not the fruit supply:
males catch ~135 prey (~400 energy) per episode but keep ~70% of it in an uncapped store while females starve. Hence the last run
splits the sharing rate by food type (meat shared more widely, as real bands do): `--band-meat-share-rate 0.6 --band-fruit-share-rate 0.3`.
**This worked:** with meat sharing 0.6, females are sustained at meat share 0.10 (see the last row). Seed 43 replicates it (female extinction 2-7% late). The `male_joins_female`
marriage rule is also viable, with more surviving females but far fewer bands (one seed; see the last row).

## Behaviour analysis (2026-09-25, descriptive)

`analyze_band_behavior.py` rolls out the final checkpoint (19) of the three viable runs (meat share 0.10, fruit regrowth 0.08, meat sharing 0.6 / fruit
0.3): seed 42, seed 43 and the `male_joins_female` variant (seed 42), 20 episodes each. Rollout-based and descriptive: one checkpoint per run, two
seeds plus one variant, no causal claims. A Codex review of the script led to fixes before these numbers: parental care and gifts are now counted
next to band sharing, moves onto occupied cells are treated as blocked, and the random baseline is used for the first 100 steps only.
Full logs: `~/simulation_results/band_behavior/behavior_{s42,s43,mjf}.log`.

| Per life (pooled over lives) | seed 42 | seed 43 | `male_joins_female` |
|---|---|---|---|
| Life length, male / female (steps, all lives) | 508 / 230 | 469 / 245 | 472 / 258 |
| Prey share of own foraging, male / female | 16% / 2.7% | 18% / 2.7% | 20% / 2.4% |
| Own meat per life, male / female | 23.3 / 1.2 | 23.8 / 1.3 | 24.5 / 1.2 |
| Meat that came from others, male / female | 22% / 76% | 21% / 78% | 26% / 80% |
| Single females: life, meat from others | 294 steps, 79% | 282 steps, 73% | 234 steps, 75% |
| Founding-couple females: life, meat from others | 246 steps, 75% | 295 steps, 74% | 252 steps, 74% |
| Approach bias toward prey, male / female | +0.17 / +0.04 | +0.19 / +0.03 | +0.22 / +0.02 |
| Approach bias toward fruit, male / female | +0.21 / +0.21 | +0.22 / +0.21 | +0.19 / +0.22 |
| Approach bias toward same-band / other-band members (males; females) | -0.09 / -0.08; -0.08 / -0.08 | -0.08 / -0.06; -0.06 / -0.05 | -0.07 / -0.07; -0.09 / -0.09 |

(Approach bias: positive means the policy's intended move ends closer to the target than a uniform random mover's would; every interval above excludes
zero; the 95% episode-cluster intervals are in the logs.)

What the runs show:
- **A sex differentiation in foraging emerges.** Males approach and take prey; females approach fruit like males do but barely approach prey, so
  females live almost entirely on fruit they gather themselves and on meat that reaches them through the band (75-80% of their meat, against ~22% for males).
  Band sharing moves meat from the hunters to everyone in range, and females pass on a similar amount of fruit (~12-14 energy given per life).
- **Single females are not worse off than paired ones** in these runs: the same reliance on band meat (73-79%) and similar life lengths (234-295 against
  246-295 steps for founding-couple females). Females of every kind live about half as long as males (~230-260 against ~470-510 steps), which
  matches the high female turnover seen in training.
- **No sign of active cohesion beyond the starting layout.** In the first 100 steps the share of decisions with a same-band member in range is
  0.78 (male) and 0.85 (female) for the trained policies against 0.77 for random movers (a small edge for females only). Both sexes move *away* from any
  nearby predator, same band or other (-0.05 to -0.09), i.e. they avoid crowding while staying within sharing range (about 3.5-4.6 cells from the nearest
  band-mate). There is no valid random comparison after step 100, because random-policy populations die out.
- **`male_joins_female` merges the population into fewer bands:** later in episodes the nearest same-band member is ~2.4-2.9 cells away (against ~3.5-4.6)
  and 90-96% of decisions have a band-mate in range (against 71-86%), consistent with the ~2.4 surviving bands reported above.

- **Minimum meat a female needs** (`analyze_female_meat_intake.py`, same runs): the rules put the meat drain at 0.010 (always idle) to 0.018 (always
  moving) energy per step at meat share 0.10, from a 2.5 starting store. Females that lived at least 800 steps had a net meat intake (own hunting +
  received - given) of at least 0.016-0.017 per step (median about 0.021), i.e. at the upper end of that floor, with the starting store covering the
  rest; among females that lived at least 300 steps the lowest was about 0.009 (5th percentile about 0.010). Intake is not consumption, and the
  samples of 800-step lives are small (11, 26 and 24 females).

- **Who is scattered** (`analyze_band_scatter.py`, same runs, sampled every 10 steps): the share of observations in which a predator has
  no band-mate within sharing range is about 31-37% for founders (never changed band), 5-19% for members born into a band, and
  8-12% for members that changed band less than 25 steps ago, rising to 12-25% more than 100 steps after the move. So marriage is not the
  main source of scatter (a marriage needs a mate within 3 cells, so the mover arrives next to its new band); scatter grows with time in a
  band, and the policies themselves do not hold bands together. Distance to the band centroid (about 8 cells for every group) is
  uninformative because bands are spread over several clusters. Descriptive, one checkpoint per run; age and sex effects not separated.

Limits: a state-dependent policy comparison would need the shared-state method used in `predator_sexual_reproduction`; the counts of founder roles are
small (100 lives each); lives cut off at the episode end are included (many males, few females). See the Codex-reviewed docstring of
`analyze_band_behavior.py` for the exact definitions.

## What is new (on top of the diet module)

- **Initial bands:** `num_bands` (**3 by default since 2026-09-25; all calibration and behaviour results above used 5**) bands of a founding couple (recorded mates), `band_children_per_couple` (2) dependent
  children (alternating male, female), and one unpaired male and female each; members start on the free cells nearest a band
  centre (`band_spawn_radius`, enforced), centres spread by farthest-point sampling. 18 predators by default (30 with 5 bands). The food settings (fruit regrowth 0.04 by default; the viable 5-band runs used 0.08) have not been re-calibrated for 3 bands.
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
`--band-meat-share-rate`, `--band-fruit-share-rate` (each defaults to the single rate), `--diet-meat-cost-share`, `--scripted-prey 0/1`,
`--initial-num-fruit`, `--energy-gain-per-step-fruit`.

## Not built

Big game ("mammoths" needing several hunters, kill split by band strength): see the design draft.
