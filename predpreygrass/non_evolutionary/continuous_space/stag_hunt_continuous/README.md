# stag_hunt_continuous

**Status (2026-11-04): calibrating (DESIGN.md, plan step 1).** Hunters
are viable on rabbits alone. With mammoths, no round so far gives both
coexistence and cooperation: worlds where hunters learn to hunt mammoths
together (joint arrivals up to 2.4x chance) end in a hunter boom that eats
out a prey species, and the brakes tried that keep the prey (a wait
between births, faster rabbits) also remove the hunters' interest in
mammoths. Rabbit burrows (round 10, DESIGN.md section 7) went unused by
the rabbits, and well-fed mammoths harder to kill (round 11, DESIGN.md
section 8) turned the hunters away from mammoths. Round 12, one warren
beside each grass cluster (DESIGN.md section 7, point 9), boomed with
the rabbits still not hiding; without mammoths rabbits learned to hide,
but no evaluated episode kept both species, even with hidden rabbits
unseen by hunters; burrows are set aside. Round 13 ported the discrete module's
summed-energy capture rule (DESIGN.md section 9): at ratio 1 no boom and
mammoths about half the hunters' food, with joint arrivals 1.8x chance,
but the rabbits are still eaten out (10 of 50 evaluated episodes keep all
three species); at ratio 1.5 hunters leave the mammoths. Round 14 added
faster rabbits: 47 of 50 keep all three species, without a boom, but
mammoths fall to about a fifth of the hunters' food and joint arrivals to
1.2x chance. Round 15 made rabbits a poorer food: no joint arrivals above
chance, mammoths still about a fifth of the food, stopped early. Round 16
gave the rabbits' grass seasons instead: 26 and 31 of 50 keep all three
species, mostly losing the rabbits in or just after the first lean season,
with joint arrivals 1.2-1.3x chance. Round 17 trained hunters on rabbits
first: 39 of 50 keep all three species with or without seasons, mammoths
about a fifth of the food, joint arrivals 1.1-1.3x chance, and hunters join
about as often alone as with a partner. See "Calibration log"
below and
[DESIGN.md](DESIGN.md).

Cooperative hunting in the continuous Predator-Prey-Grass world. Hunters
(the predators, standing for humans) live with two prey species:
**rabbits**, which one hunter can catch, and **mammoths**, which need a
party of hunters. Each hunter decides whether to join a hunt (and pay
its cost and risk) or stand by and take a scavenger share. The question
is whether hunters learn to hunt together, and whether cooperation,
solo hunting or free-riding pays.

It transfers the discrete
[`stag_hunt_forward_view`](../../discrete_space/project_cooperation/stag_hunt_forward_view/README.md)
(there hunters joined hunts more often next to mammoths than next to
rabbits, and hunters labelled group hunters reproduced more, a descriptive
result) to the continuous world of
[`base_environment_continuous`](../base_environment_continuous/README.md),
with a capture chance by party size from
[`predator_bands`](../../discrete_space/predator_bands/README.md), and is
designed around the reasons the continuous foraging_swarms experiments
found no learned cooperation (DESIGN.md, section 2).

## Files

- [`environment.py`](environment.py): `StagHuntContinuous`, an RLlib
  `MultiAgentEnv` (three species, the hunt rule, observations, hunt
  counters).
- [`grass.py`](grass.py), [`geometry.py`](geometry.py): copies of the base
  world's; the rabbits' grass and the mammoths' steppe are two `Grass`
  layers.
- [`config/config_env.py`](config/config_env.py),
  [`config/config_ppo.py`](config/config_ppo.py): settings for training.
- [`train.py`](train.py): PPO, one policy per species (predator, rabbit,
  mammoth); hunt counters logged to TensorBoard under `hunts/` by
  [`callbacks.py`](callbacks.py).
- [`calibrate.py`](calibrate.py): the hunt's expected-value table for the
  current settings.
- [`prey_switching.py`](prey_switching.py), [`crossplay.py`](crossplay.py),
  [`seasons.py`](seasons.py): prey switching by rabbit count, policies of two
  checkpoints mixed, and populations, food and kills by phase of the grass
  seasons.
- [`evaluate.py`](evaluate.py): plays seeded episodes of a checkpoint (or
  untrained policies): coexistence of the three species, hunt counters,
  and joint arrivals per mammoth encounter episode against a
  density-matched chance level (each predator's track shifted on the
  torus).
- [`watch.py`](watch.py) with [`render.py`](render.py): one episode in a
  pygame window or as an mp4 (orange ring: a predator whose join is on;
  the rabbit sprite, `assets/rabbit.png`, was drawn for this module).
- [`checkpoints.py`](checkpoints.py): a checkpoint's settings and
  policies, and untrained policies.
- [`launchers/calibration.py`](launchers/calibration.py): the
  calibration runs (one arm per round, settings listed in `ARMS`), to
  runs/calibration_<arm>/; evaluations in `results/`.
- [`tests/`](tests/): spaces, determinism, catching, food layers, episode
  ends and the hunt rule.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.train
    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.evaluate <checkpoint | untrained> 50 out.json
    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.watch <checkpoint | untrained> --seed 0
    python -m pytest predpreygrass/non_evolutionary/continuous_space/stag_hunt_continuous/tests

## TODO

- [x] Decide the open points (DESIGN.md, section 6; 2026-10-30).
- [x] Build: environment, three policies, hunt rule, observations,
  metrics, tests; Codex review (2026-10-30).
- [x] Evaluation tools: evaluate.py (coexistence, hunt counters, joint
  arrivals against a chance level), watch.py and render.py (2026-10-30).
- [ ] Calibrate viability from scratch (in progress, see the log).
- [ ] Pre-register and run the main test (hunt arm against a solo
  control and untrained policies, joint arrivals above chance).

## Calibration log (not confirmatory)

Runs from scratch with [`launchers/calibration.py`](launchers/calibration.py),
1,000 iterations planned, 14 env runners, one seed each; values are
per-episode training metrics (TensorBoard `hunts/`, averaged over ten
iterations) unless an evaluation is named. Up to round 7 an episode ended
early only when no rabbits and no mammoths were left (or no hunters), so
long training episodes did **not** show coexistence: round 5's hunters
kept episodes long on mammoths after eating out the rabbits. From round 8
training episodes are truncated at the first prey extinction
(`truncate_on_prey_extinction`), and coexistence is judged by evaluations
of whole episodes. Mammoth share = mammoth energy / (mammoth + rabbit
energy) gained by hunters.

| Round (arm) | Change | Outcome | Stopped |
| --- | --- | --- | --- |
| 1a `rabbits_only` | no mammoths | early crash (rabbits eaten out by iteration ~50), then recovery to full-length episodes; evaluation of the final checkpoint: hunters and rabbits coexist in 49 of 50 episodes (`results/calibration_rabbits_only_eval.json`) | finished 1,000 |
| 1b `with_mammoths` | configured world (6 hunters, 4 mammoths, carcass 32) | mammoth share ~1%; ~100 solo attempts per episode, join rate on contact 0.49 → 0.25 | iteration 410 |
| 2 `with_mammoths_r2` | 10 hunters, 8 mammoths; struggle 0.02, join cost 0.02 (failed solo attempt −0.66 instead of −1.65) | mammoth share peaked ~9–14% (iteration ~50–150), then fell to ~4%; joining fell, alone faster than with a partner (0.26 vs 0.20 at the end) | iteration 370 |
| 3 `with_mammoths_r3` | round 2 with carcass 64 (struggle 0.01, solo still −0.66) | hunter boom: mammoth share ~90%, both prey eaten out, episodes ~265 decisions from iteration ~150 with no recovery; joint arrival rate 0.43, join with partner 0.74 vs alone 0.60 | iteration 270 |
| 4 `with_mammoths_r4` | carcass 48 | same as round 3, later: mammoth share ~84%, episodes ~246 decisions by iteration ~190; joint arrival rate 0.29, join with partner 0.64 vs alone 0.56 | iteration 190 |
| 5 `with_mammoths_r5` | round 4 with the living-hunter cap 50 → 25 | stable training episodes (~1,000 decisions), mammoth share ~85%; evaluation of the final checkpoint: **coexistence in 3 of 50** (rabbits eaten out in 47, then hunters starved in 9 of those); joint arrival rate 0.197 vs chance 0.082, join with partner 0.50 vs alone 0.26 (`results/calibration_with_mammoths_r5_eval.json`) | finished 1,000 |
| 6 `with_mammoths_r6` | round 4 (cap 50) with costlier offspring: birth threshold 20 (was 12), offspring energy 10 (was 5) | boom delayed by ~100 iterations, not prevented: episodes ~450 decisions from iteration ~350, mammoth share ~80%; at iteration 920 a hunter's median wait between births was 188 physics steps (90% under 903) | iteration 925 |
| 7 `with_mammoths_r7` | round 4 with a wait of 2,000 physics steps between a hunter's births (`reproduction_predator_interval`) | no boom (full-length episodes), but mammoth share ~3%, pair kills <1 per episode, joint arrival rate 0.02, join with partner = alone | iteration 571 |
| 8a `with_mammoths_r8a` | round 7 with wait 1,000, training truncated at the first prey extinction | as round 7: mammoth share ~4%; evaluation at iteration 450: coexistence in 43 of 50 (hunters died out in 7), joint arrival rate 0.026 vs chance 0.025 (`results/calibration_with_mammoths_r8a_iter450_eval.json`) | iteration 924 |
| 8b `with_mammoths_r8b` | as 8a with wait 600 | mammoth share ~6%; evaluation at iteration 480: coexistence in 38 of 50 (hunters died out in 11, rabbits in 1), joint arrival rate 0.029 vs chance 0.026 (`results/calibration_with_mammoths_r8b_iter480_eval.json`) | iteration 978 |
| 9a `with_mammoths_r9a` | round 5 (cap 25) with rabbit top speed 7 (was 6; hunters 5), training truncated at the first prey extinction | rabbits multiplied instead of escaping (~250 births per training episode against ~90 in round 5) and became the staple; mammoth share ~10% and flat, joining fell with and without a partner; evaluation at iteration 290: **coexistence in 46 of 50**, but joint arrival rate 0.036 vs chance 0.032, mammoths ~9% of the hunters' food (`results/calibration_with_mammoths_r9a_iter290_eval.json`) | iteration 290 |
| 9b `with_mammoths_r9b` | as 9a with rabbit top speed 8 | as 9a; evaluation at iteration 270: coexistence in 41 of 50 (rabbits died out in 6, hunters and rabbits in 3), joint arrival rate 0.032 vs chance 0.029, mammoths ~7% of the hunters' food (`results/calibration_with_mammoths_r9b_iter270_eval.json`) | iteration 270 |
| 10a `with_mammoths_r10a` | round 5 (cap 25) with 10 rabbit burrows of radius 24 (DESIGN.md, 7), training truncated at the first prey extinction | rabbits did not learn to use the burrows (~3% of their time inside, the burrows' share of the area, through iteration 330); hunters lived on rabbits (mammoth share ~13%, rabbits caught = rabbits born), joint arrival rate 0.049, join with partner 0.33 vs alone 0.28; training episodes ~745 decisions (some prey extinctions); evaluation at iteration 330: **coexistence in 33 of 50** (rabbits died out, then hunters, in 15), joint arrival rate 0.042 vs chance 0.026, rabbits inside 3.4% (median rabbit never inside) (`results/calibration_with_mammoths_r10a_iter330_eval.json`) | iteration 335 |
| 10b `with_mammoths_r10b` | round 4 (cap 50) with the same burrows | rabbits did not use the burrows (~3%); round 4's boom, slower: mammoth share ~76%, training episodes ~350 decisions from iteration ~150, cap binding 25% of physics steps; strongest cooperation signals with a cap so far (joint arrival rate 0.255, join with partner 0.63 vs alone 0.40); evaluation at iteration 280: **coexistence in 2 of 50** (both prey eaten out in 39), cap binding 56% of physics steps, joint arrival rate 0.326 vs chance 0.191, join with partner 0.63 vs alone 0.33 (`results/calibration_with_mammoths_r10b_iter280_eval.json`) | iteration 289 |
| 11a `with_mammoths_r11a` | round 4 (cap 50, no burrows) with hunt success × (1 − 0.5 × mammoth condition), condition = mammoth energy / its birth threshold, at most 1 (`hunt_condition_factor`, DESIGN.md, 8); hunters see the condition of mammoths in view; training truncated at the first prey extinction | no boom (training episodes ~1,100 decisions, cap never binding) but hunters left the mammoths: mammoth share ~2%, rabbits the staple (~400 births per training episode), joint arrival rate ~0.03; killed mammoths' condition 0.67 vs hunted 0.71; evaluation of the final checkpoint: **coexistence in 35 of 50** (hunters died out in 15), joint arrival rate 0.028 vs chance 0.022 (`results/calibration_with_mammoths_r11a_eval.json`) | finished 1,000 |
| 11b `with_mammoths_r11b` | as 11a with factor 0.8 | as 11a: mammoth share ~1.5%, joint arrival rate ~0.025; killed mammoths leaner than hunted (condition 0.49 vs 0.71); evaluation of the final checkpoint: **coexistence in 28 of 50** (hunters died out in 22), joint arrival rate 0.021 vs chance 0.018 (`results/calibration_with_mammoths_r11b_eval.json`) | finished 1,000 |
| 12 `with_mammoths_r12` | round 10b (cap 50) with one warren per grass cluster (20), beside it and moving with it (`burrow_placement = "cluster"`, DESIGN.md, 7, point 9); untrained, the median edible grass patch is 31 units from a warren's edge (98 in round 10) | rabbits still hardly used the warrens: ~7% of their time inside from iteration ~150 (~6% untrained), ~18% of edible grass within reach of a rabbit inside; round 4's boom again: training episodes ~255 decisions, cap binding 29%, joint arrival rate 0.315, join with partner 0.64 vs alone 0.44 (iterations 250–300); rabbit births ~61 per episode (~100 in 10b); not evaluated | iteration 449 |
| check `rabbits_only_warrens` | round 1a (no mammoths) with round 12's warrens: can rabbits learn to hide in a world with full-length episodes? | rabbits learned to hide, late: time inside ~7% to iteration 300, 18% at 500, 35% at 1,000 and still rising; but training episodes ~455 decisions (1a: ~1,230), rabbit births ~83 per episode (1a: ~440); evaluation of the final checkpoint: **coexistence in 0 of 50** (rabbits eaten out in 34, hunters starved in 16; 1a: 49 of 50), rabbits inside 43% (quartiles 0, 0.06, 0.39), hunters within reach of a warren's edge 22% of their steps, 67,381 rabbit-steps inside a warren touching a hunter, 11% of catches within 8 steps of an exit (`results/calibration_rabbits_only_warrens_eval.json`) | finished 1,000 |
| check 2 `rabbits_only_warrens_hidden` | check 1 with rabbits inside a warren hidden from the hunters (`burrow_hides_rabbits`, DESIGN.md, 7, point 10) | rabbits learned to hide sooner and more (35% of their time inside by iteration ~275, 45% at 750); training episodes ~450 decisions; evaluation of the final checkpoint: **coexistence in 0 of 50** (rabbits eaten out in 27, hunters starved in 23), rabbits inside 50% (quartiles 0, 0.07, 0.47), hunters at a warren's edge 18% of their steps, 8% of catches exit ambushes; per episode ~102 rabbit births and ~103 catches (1a: ~439 and ~378) (`results/calibration_rabbits_only_warrens_hidden_eval.json`) | finished 1,000 |
| 13a `with_mammoths_r13a` | round 4 (cap 50, no burrows) with the discrete module's capture rule: a hunt succeeds when the contributors' summed energy exceeds the mammoth's (`hunt_rule = "energy"`, no draw; DESIGN.md, 9); hunters see mammoth and partner energies; training truncated at the first prey extinction | no boom (cap binding ~3%); mammoths ~50-60% of the hunters' energy from iteration ~200, falling to ~45% by 850 as rabbits became the staple again; training episodes grew from ~100 to ~510 decisions; joint arrival rate peaked ~0.23 (iterations 400-650); evaluation of the final checkpoint: **coexistence in 10 of 50** (rabbits eaten out in 38, median at ~1,000 physics steps, hunters starved afterwards in 30; mammoths out in 10), joint arrival rate 0.116 vs chance 0.066 (1.8x), join with partner 0.34 vs alone 0.19, cap binding 4% (`results/calibration_with_mammoths_r13a_eval.json`) | finished 1,000 |
| 13b `with_mammoths_r13b` | as 13a with summed energy above 1.5 x the mammoth's | round 9's regime: rabbits the staple (mammoths ~11% of the hunters' energy), pair kills rare (~0.2 per episode), joint arrival rate ~0.04, join with partner = alone (~0.39); training episodes ~740 decisions and rising; not evaluated | stopped at iteration 258 (user) |
| 14 `with_mammoths_r14` | round 13a with round 9's faster rabbits (top speed 7, hunters 5) | no boom (cap binding 0-3%); training episodes ~1,170 decisions; hunters drifted to rabbits: mammoths ~48% of their energy at iterations 200-250, ~41% at 300-350, ~31% at 500-550, ~22% at the end. Evaluations (50 episodes): iteration 300 **14 coexist**, joint 0.087 vs chance 0.058, partner 0.49 vs alone 0.33; iteration 500 **28**, joint 0.058 vs 0.044, partner 0.35 vs 0.25; final checkpoint **47**, joint 0.034 vs 0.028 (1.2x), partner 0.25 vs 0.19, rabbits out in 1, hunters out in 3, mammoths in 0 (`results/calibration_with_mammoths_r14_eval.json`, `_iter300_eval.json`, `_iter500_eval.json`) | finished 1,000 |
| 15a `with_mammoths_r15a` | round 14 with rabbits a poorer food: hunters gain 0.3 of a caught rabbit's energy (0.5 before) | mammoths ~29% of the hunters' energy at iterations 150-223 (round 14 at that stage ~50%), pair kills ~0.3 per episode (~1.9); evaluation at iteration 220: **13 coexist** (hunters out in 37), joint 0.014 vs chance 0.016, partner 0.44 vs alone 0.42 | stopped at iteration 223 (for round 16's slots) |
| 15b `with_mammoths_r15b` | as 15a with 0.4 | mammoths ~23% of the hunters' energy at iterations 350-400, ~20% at 590-639 (round 14: ~38%, ~28%), cap binding ~0; evaluations (50 episodes): iteration 390 **45 coexist**, joint 0.022 vs chance 0.024, partner 0.28 vs alone 0.24; iteration 630 **37** (hunters out in 13), joint 0.019 vs 0.020, partner 0.34 vs 0.28 | stopped at iteration 639: no joint arrivals above chance, and fewer coexisting late |
| 16a `with_mammoths_r16a` | round 14 with seasons in the rabbits' grass (`grass_season_period` 2000, `grass_season_amplitude` 0.5; rabbit efficiency back at 0.5) | no boom (cap binding ~1-3%); mammoths ~45% of the hunters' energy at iterations 250-300, ~27% at 850-900 (round 14: ~45%, ~23%). Evaluations (50 episodes): iteration 500 **18 coexist**, joint 0.074 vs chance 0.057, partner 0.29 vs alone 0.17; final checkpoint **31** (hunters out in 19, rabbits in 16), joint 0.054 vs 0.042 (1.3x), partner 0.22 vs 0.15, mammoths 24% of the food | seasons slowed the drift to rabbits without ending it |
| 16b `with_mammoths_r16b` | as 16a with period 1000, amplitude 1 | no boom (cap binding up to ~4%); mammoths ~50% of the hunters' energy at iterations 250-450, ~33% at 850-900; training episodes ~720 decisions at 850-900 (round 14: ~1,160). Evaluations (50 episodes): iteration 500 **7 coexist**, joint 0.110 vs chance 0.074, partner 0.41 vs alone 0.23; final checkpoint **26** (hunters out in 23, rabbits in 20), joint 0.050 vs 0.041 (1.2x), partner 0.26 vs 0.16, mammoths 26% of the food | |
| 17 stage 1 `rabbits_only_r17` | rabbits-first curriculum (user's idea): round 14 without mammoths, 400 iterations; the mammoth policy stays at its initial weights | evaluation (50 episodes): **48 keep hunters and rabbits**, rabbits out in 1, hunters out in 1, cap 0 (`results/calibration_rabbits_only_r17_eval.json`) | finished 400 |
| 17a `with_mammoths_r17a` | round 14 continued from stage 1's hunters and rabbits, mammoths added and learning from scratch | no boom (cap ~0.1%); mammoths ~32% of the hunters' energy at iterations 50-250, ~24% at 800; training joint rate 0.042-0.049 and partner below alone throughout (0.34 vs 0.37 at 800; round 14 at 850-900: 0.38 vs 0.29). Evaluations (50 episodes): iteration 500 **22 coexist**, joint 0.038 vs chance 0.028, partner 0.32 vs alone 0.34; final checkpoint **39** (hunters out in 11, rabbits in 3), joint 0.037 vs 0.028 (1.3x), partner 0.28 vs 0.29, mammoths 19% of the food, 46 of 437 kills by a group (`results/calibration_with_mammoths_r17a_eval.json`, `_iter500_eval.json`) | finished 1,000 |
| 17b `with_mammoths_r17b` | as 17a with round 16b's seasons (period 1000, amplitude 1) | no boom (cap ~0.1%); mammoths ~38% at iterations 0-50, ~25% at 750-780; training joint rate 0.044-0.058, partner about level with alone by 750 (0.39 vs 0.39). Evaluations (50 episodes): iteration 500 **14 coexist**, joint 0.037 vs chance 0.029, partner 0.40 vs alone 0.40; final checkpoint **39** (hunters out in 11, rabbits in 8), joint 0.032 vs 0.029 (1.1x), partner 0.30 vs 0.27, mammoths 18% of the food, 34 of 405 kills by a group (round 16b: 94 of 421) (`results/calibration_with_mammoths_r17b_eval.json`, `_iter500_eval.json`) | finished 1,000 |

Reading so far: hunters live on rabbits; between a carcass of 32 and 48
the mammoth world tips from hunters ignoring mammoths into a hunter boom
that removes a prey species within an episode. Round 5's cap kept the
mammoths but not the rabbits, and a fixed cap is not realistic, so the
aim is a world that works without it. The brakes on hunter births tried
since split the same way: costlier offspring keep the mammoth payoff and
the boom; a wait between births stops the boom, but then time, not
energy, limits a hunter's births, rabbits alone reach the birth threshold
within the wait, and the mammoth payoff (the reason to hunt together)
loses its value (an interpretation, not tested). A mammoth payoff large
enough to make group hunting pay may be large enough to fuel a boom; in
nature, long coexistence of hunters and large prey has been linked to
space (refuges, movement), which these worlds lack. A larger world at
equal densities was timed and found costly (1,131 x 1,131: ~40-55 s per
iteration; 1,600 x 1,600: 80-200 s) and is likely still small against a
hunter's range. Faster rabbits (round 9) kept the rabbits but did not
make them hard to live on: they bred faster (more grass reached, a guess
not checked) and the hunters turned to them, the round 7/8 regime again.
Refuges in the terrain, rabbit burrows, which protect rabbits without
making them more plentiful (DESIGN.md, section 7, reviewed by Codex), were
tried in round 10, but the rabbits did not learn to use them within
~300 iterations. The likely reason (measured, not tested): the burrows lay
far from the clustered grass (median patch 98 units from a burrow's edge,
hunters see 200), so a feeding rabbit seldom had a refuge in reach. Round
11 tried a brake on the hunters instead, a hunt less likely to succeed the
better fed the mammoth (DESIGN.md, section 8), meant to slow hunting as
mammoths become scarce; instead the hunters left the mammoths, which then
stayed well fed and hard to kill (an interpretation). Round 12 puts a
warren beside each grass cluster (DESIGN.md, section 7, point 9): the
rabbits still did not hide (a guess not tested: births as their only
reward, with training episodes cut short by the boom, give too weak a
signal). A check without mammoths showed they can learn it, late (35% of
their time inside after 1,000 iterations), but the hunters learned to wait
at and on the warrens, and that world lost a species in all 50 evaluated
episodes (49 of 50 coexisting without warrens): as built, warrens harmed
the rabbits. Hunters see a rabbit inside a warren (its slot and an
inside bit, DESIGN.md 7.5), which real predators mostly cannot. Check 2
hid them (DESIGN.md 7.10): rabbits hid more and sooner, but again no
evaluated episode kept both species. In both checks rabbits bred about a
quarter as much as without warrens (~100 births per episode against
~440), while catches fell less (an interpretation, not tested: time spent
hidden is time not grazing, and hunters still wait where rabbits gather).
Burrows as built here do not protect the rabbit population, so they are
set aside as a lever. Round 13 tried the discrete module's capture rule: hungry
hunters cannot kill a mammoth (DESIGN.md, section 9). At ratio 1 it
removed the boom (cap binding ~4% against 25-56% in rounds 4, 10b and 12)
and kept mammoths about half the hunters' food with cooperation signals,
but did not save the rabbits: they went first in 38 of 50 evaluated
episodes. An interpretation, not tested: the rule brakes the mammoth
harvest, not the rabbit harvest, and the rabbits are the prey being eaten
out. At ratio 1.5 the hunters left the mammoths, as in rounds 7 to 11.
Round 14 gave the rabbits top speed 7: the first world that passes the
coexistence target without a boom and with mammoths still ~22% of the
hunters' food (round 9: ~9%), but cooperation is near chance. Within this
run coexistence rose as cooperation fell (14, 28, 47 of 50 at iterations
300, 500, 1,000; joint arrivals 1.5x, 1.3x, 1.2x chance), and across the
evaluated rounds both prey survived in nearly every world near chance
and in few of those well above it; an association over 11 unlike runs, not
a tested cause. Apparent competition through the shared hunters (Holt 1977)
is one candidate, not tested.

Prey switching (`prey_switching.py`, 50 seeded episodes per checkpoint,
`results/prey_switching_*.json`; descriptive, rabbit numbers are not set by
the experimenter). The diet shows no detected departure from proportional
use: slope b of diet ratio on availability ratio 1.16 [0.96, 1.38] in 13a,
1.05 [0.86, 1.22] in 14, 1.03 [0.83, 1.21] in 14 at iteration 300 (b > 1
would be switching; these intervals still allow moderate switching either
way). Scarce rabbits are not safer: rabbits caught per rabbit x hunter x
physics step do not rise with rabbit numbers (14: 25.7 x 10^-5 at 1-5
rabbits, 13.9 at 41-70). What differs between 14's final checkpoint (47 of
50 coexist) and its iteration 300 (14 of 50), in the same world, is lower
hunting at every rabbit count: hunters' movement toward the nearest rabbit
in view is about half (ratio 0.44-0.53, paired resampling of seeds), and the
catch rate 0.47-0.68 of it up to 40 rabbits. Iteration 300 at rabbit speed 7
matches 13a's final checkpoint at speed 6 on both measures up to 40 rabbits.
An association between checkpoints whose rabbit and hunter policies both
changed.

Cross-play (`crossplay.py`, all 8 mixes of hunters, rabbits and mammoths
from 14's final checkpoint and iteration 300, 50 seeded episodes each,
`results/crossplay_r14_final_vs_iter300.json`; the unmixed cells repeat the
evaluations exactly). The policies never trained against these partners, and
rabbit counts, hunter condition and episode stage all shift between cells.
Final rabbits go with a lower catch rate from either kind of hunter (0.42-0.74
of that for iteration-300 rabbits at 11-40 rabbits, paired resampling of
seeds). Against iteration-300 rabbits, final hunters catch about as many as
iteration-300 hunters (0.92-1.16) with about half the pursuit (0.46-0.52), so
the lower pursuit does not by itself lower catches. Pursuit, mammoth food
share (0.15-0.19 against 0.40-0.51) and joint arrivals above chance (1.2-1.4x
against 1.5-1.7x) go mostly with the hunters' checkpoint, whichever rabbits
they meet. Iteration-300 hunters die out against final rabbits (50 and 49 of
50), which also selects what their rates measure (before the collapse); final
hunters against iteration-300 rabbits keep 28 and 40 of 50 episodes. This fits
final rabbits being harder to catch and final hunters persisting on them with
less movement and fewer mammoths, rather than hunters easing off; it does not
establish evasion (spatial avoidance or abundance could also lower the rate),
starvation as the cause of the collapse, or learned cooperation as such.

Seasons (`Grass`, `seasons.py`): grass regrowth runs on a clock moving
1 + amplitude x sin(2 pi t / period) per physics step, starting into the good
season; the steppe has its own settings, off so far. The clock's rate averages
1 over a cycle, but realised grazing need not match a world without seasons.
Round 14's final policies, which never trained with seasons, played in such
worlds (30 episodes each, first cycle left out, ranges over the 8 phases,
`results/season_pilot_r14_<period>_<amplitude>.json`):

| Period, amplitude | Coexist | Rabbits | Hunter food per 1,000 hunter steps | Grass eaten per 1,000 steps |
|---|---|---|---|---|
| no seasons | 29 | 30-35 | 12-13 | 207 |
| 2000, 0.5 | 27 | 25-46 | 7-18 | 205 |
| 1000, 1 | 20 | 26-48 | 4-21 | 200 |
| 2000, 1 | 8 | 18-74 | 3-22 | 197 |
| 4000, 0.5 | 18 | 24-69 | 7-19 | 192 |
| 4000, 1 | 2 | 4-96 | 0.4-33 | 173 |

The swings reach rabbits and hunters' food, with rabbits lagging the grass;
mammoth kills were too few to show a seasonal pattern in these policies.
Round 16's arms differ in both period and amplitude: calibration, not a test
of either.

Round 16 by season (`seasons.py`, final checkpoints, 50 episodes,
`results/seasons_r16a.json`, `seasons_r16b.json`; cycles after the first, so
only worlds still running; intervals resample episodes). Mammoths' share of the
hunters' food in the grass's lean half against its good half: 16b 26% vs 18%
(difference 95% interval +0.02 to +0.15, 38 group kills), 16a 6% vs 14% (-0.12
to -0.03, 11 group kills). So a lean-season shift towards mammoths appears only
in the shorter, stronger seasons, on few kills, and two arms cannot separate
period from amplitude. Most rabbit extinctions come early: 16 of 20 in 16b by
step ~1170 (first lean half: 500-1000), 13 of 16 in 16a by step ~1730 (1000-2000),
which may reflect the first lean season arriving before rabbit numbers have
built up from the 8 at the start.

Delaying the first lean season (`grass_season_start`, plain growth until then;
round 16's final policies replayed, not retrained, on the same 50 seeds;
`results/delayed_lean_<arm>_<start or off>.json`), episodes with all three
species:

| Arm | Start 0 | Start 1000 | Start 2000 | Start 4000 | Seasons off |
|---|---|---|---|---|---|
| 16a | 31 | 36 | 35 | 40 | 40 |
| 16b | 26 | 33 | 34 | 35 | 37 |

Change against start 0, paired by seed: start 4000 +9 in both arms (95% interval
0 to 17); the others include no change. A delay recovers at most what switching
seasons off does, and these policies still lose 10-13 worlds with no seasons at
all (rabbits out at a median step of ~1,420-1,570, hunters later), where round
14's lost 3. So the first lean season accounts for part of the die-off, not all
of it; the rest goes with the policies trained in seasonal worlds or with the
opening. These policies did not train with a delay.

More rabbits at the start (`seasons.py --rabbits`, the same replay,
`results/more_rabbits_<arm>_rabbits<n>_<on or off>.json`) did not help, episodes
with all three species:

| Arm, seasons | 8 rabbits | 30 | 60 |
|---|---|---|---|
| 16a, on | 31 | 31 | 26 |
| 16a, off | 40 | 37 | 31 |
| 16b, on | 26 | 23 | 10 |
| 16b, off | 37 | 40 | 34 |

Paired by seed, only two changes exclude none: 16b on at 60 (-16, 95% interval
-25 to -6) and 16a off at 60 (-9, -17 to -1). So too few rabbits at the start
does not look like the cause of the die-off, which points to the policies. With
60 rabbits they still die out first, which would fit an overshoot (more early
food for hunters, then a crash), not measured here.

Round 17 against the same worlds trained without the rabbit stage, final
checkpoints, 50 episodes (one training run per arm; shares here are from
evaluation, so round 14's is below the ~22% it ended on in training):

| | 14 | 17a | 16b | 17b |
|---|---|---|---|---|
| Seasons | no | no | yes | yes |
| Rabbit stage first | no | yes | no | yes |
| All three species | 47 | 39 | 26 | 39 |
| Joint arrivals vs chance | 1.2x | 1.3x | 1.2x | 1.1x |
| Partner minus alone | +0.06 | -0.01 | +0.10 | +0.03 |
| Mammoths' share of food | 17% | 19% | 26% | 18% |
| Group kills, of all mammoth kills | 43 of 360 | 46 of 437 | 94 of 421 | 34 of 405 |

The rabbit stage went with fewer worlds kept without seasons (exact test
p = 0.04) and more with them (p = 0.01), both uncorrected, and with lost or
smaller partner effects in both. Seasons did not raise the mammoth share of
hunters started on rabbits. By season (`results/seasons_r17b.json`, computed as
for round 16): 19% in the lean half against 14% in the good half (difference
95% interval +0.01 to +0.09, 27 group kills), the same direction as 16b's. So
hunters shift somewhat towards mammoths when rabbits are scarcer, but the rabbit
stage gave neither the mammoth share nor the cooperation signals asked for. One
possible reading, not tested: hunters arrive with a solo rabbit-hunting policy,
and there is little reward left for learning joint hunts. Stage 1 had no
seasons, so hunters first met scarcity after they had learned on plenty.

Accepted earlier: mammoths need
not be the hunters' only food (round 5's ~80-85% share is acceptable,
cooperation stays voluntary), and a forced-join-off test was dropped as
it adds no biological realism. The checkpoints of rounds 1 to 11 are kept
for evaluate.py.

Also found: RLlib's environment check at start-up rejects the final
observation of an agent that dies in its one random step (round 4's
first launch); `train.py` now disables that check
(`disable_env_checking`), the environment tests cover those invariants.
Added for these rounds (defaults keep the earlier behaviour, and older
checkpoints load with them, `checkpoints.ADDED_SETTINGS`):
`reproduction_<species>_interval` (physics steps between an animal's
births), `truncate_on_prey_extinction` (training only; evaluate.py
always plays whole episodes), `burrow_count` and `burrow_radius` (rabbit
burrows, DESIGN.md 7) and `hunt_condition_factor` (DESIGN.md 8). The Codex review of the latter found that
an animal dying in the step that ended an episode was flagged both
terminated and truncated; it is now terminated only.

