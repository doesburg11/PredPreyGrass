# stag_hunt_continuous: results

Calibration only, one seed per arm, no confirmatory test (see the
[README](README.md) for the module, [DESIGN.md](DESIGN.md) for its design).

## 1. Where it ended

Calibration stopped on 2026-11-05 after 20 rounds, with the target not met, and the
work continues in [`../stag_hunt_rule_prey/`](../stag_hunt_rule_prey/README.md).
Section 2 gives the reasons, section 3 the rounds in brief, section 4 the full log
and analyses.

The follow-up with rule-following prey was also stopped on 2026-11-05, and the stag
hunt line was closed on 2026-11-06 (see
[`../stag_hunt_rule_prey/RESULTS.md`](../stag_hunt_rule_prey/RESULTS.md), section 15). Across both modules, about 25 runs, one seed each, the
results lay along one trade-off: hunters cooperated strongly where they could
overrun the prey, and coexisted with it where they had little need to cooperate
(round 20, with spaced births: 50 of 50, mammoths 7%, joint arrivals at chance). No
setting gave both. One reading, not tested: as in the classic stag hunt, hunters
settle on the safe solo prey when it is plentiful, and turned to mammoths only under
a scarcity that this world could not sustain. See
[`../stag_hunt_rule_prey/RESULTS.md`](../stag_hunt_rule_prey/RESULTS.md), section 9.

## 2. Why each round, and why the trail stopped

The target was all three species in at least 45 of 50 evaluated episodes,
mammoths about half or more of hunters' food (about 80% acceptable), and
cooperation signals, without a tight cap on hunters. Section 4 gives each
round's change and outcome; this table gives the reason for the change. The
reasons were written up afterwards from working notes (those for rounds 2-4 from
the sequence of results alone), not recorded at the time of each round.

| Round | Why this change |
| --- | --- |
| 1a, 1b | Check that hunters can live on rabbits alone, then try the configured world with mammoths. |
| 2 | Hunters barely touched mammoths in 1b: more animals to meet, and a failed solo attempt made cheaper. |
| 3, 4 | Mammoths still a small share: a larger carcass (64), then 48 after 64 boomed, to look between the two regimes. |
| 5 | Stop the boom with a lower hunter cap (25). |
| 6 | A fixed cap is not realistic and the user wanted the world to work without it: costlier offspring as a brake that depends on energy. |
| 7, 8 | Costly offspring only delayed the boom: a fixed wait between births (2000, then 1000 and 600 when mammoths lost their value). |
| 9 | Round 5 lost its rabbits: faster rabbits so they survive the hunters. |
| 10 | Faster rabbits made rabbits the staple: refuges instead, since long coexistence in nature has been linked to space. |
| 11 | Burrows went unused: a brake from the prey's side instead, with well-fed mammoths harder to kill. |
| 12, checks 1-2 | Burrows may have failed because they lay far from grass: warrens beside each cluster. When that boomed, checks without mammoths asked whether rabbits can learn to hide at all, then whether hiding helps if hunters cannot see them inside. |
| 13 | Burrows set aside. The user chose the discrete module's summed-energy rule over a carcass sweep or accepting a cap, tying kills to hunters' condition. |
| 14 | 13a kept mammoths but lost rabbits, so rabbits were the bottleneck: round 9's faster rabbits. |
| 15 | Rabbits became the staple again: make them a poorer food so hunters need mammoths. |
| 16 | Constant scarcity looked like a dead end. The user's idea: seasonal lean periods, pressure from the world and not forced cooperation. |
| 17 | The user's idea: rabbits-first training, so hunters are viable before mammoths appear. |
| 18 | Round 17's rabbit stage had no seasons, so hunters may have started well fed: seasons in both stages. |
| 19a-c | Energy accounts showed group hunts paid but parties were far too weak for most mammoths, so lower ratios. A replay showed that lower ratios alone mainly added solo kills, hence a minimum party of 2. 0.75 and 0.5 went to opposite regimes, so 0.6 lies between. |
| 20 | 19b had the strongest cooperation with a boom: last attempt to brake it with a birth wait, under a test set in advance. |

**The pattern.** Worlds with strong cooperation boomed and lost prey, and the
brakes that kept the prey also lost the group hunting. Round 19c's iteration 250
was the best mix (30 of 50, mammoths 46%, joint 1.3x chance), and with more
training it moved to the stable side (39 of 50, mammoths 24%, joint 1.1x).
Round 20 repeated this at the extreme: 50 of 50 with mammoths at 7% and joint
arrivals at chance. All of this is one seed per arm and not confirmatory.

**Why it stopped.** Round 20 failed its test, set in advance, on joint arrivals
and mammoth share. The user was also troubled by the series of adjustments
leading nowhere, and each new setting made the results harder to interpret.

**Why rule-following prey.** The alternatives were to give hunters larger
brains or prey smaller ones, and the user chose simpler prey
([`../stag_hunt_rule_prey/`](../stag_hunt_rule_prey/README.md)), because:

- All three species here have the same network, though human brains are far
  more developed than a rabbit's or a mammoth's.
- Larger hunter networks add computation; simpler prey removes it, and leaves a
  simpler base to build on.
- A learning arms race between humans and their prey is not a real feature of
  the human case, since prey adapt over generations and hunters within a
  lifetime.

That module gets one calibration with settings chosen in advance, then a hard
stop and a write-up, however it comes out.

**Open, untested.** Reward here is births only, so an individual hunter gains
nothing by holding back. Sustainability may therefore be its own social dilemma,
which no adjustment of prey or carcass would remove.

## 3. The rounds in brief

Hunters are viable on rabbits alone. With mammoths, no round so far gives both
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
about as often alone as with a partner. Round 18 kept the seasons in the
rabbit stage too: 40 of 50, mammoths about a fifth of the food, hunters
joining more with a partner (0.38 vs 0.31), but joint arrivals near chance
(1.1x). Round 19 required a party of at least two: ratio 0.5 boomed (0 of 50,
joint 2.2x), 0.75 left the mammoths (46 of 50 at iteration 400, mammoths 3%),
and 0.6 ended at 39 of 50 with mammoths 24%. Round 20 added a birth wait to
ratio 0.5: 50 of 50, but mammoths 7% and joint arrivals at chance. See section 3 and [DESIGN.md](DESIGN.md).

## 4. Calibration log (not confirmatory)

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
| 18 stage 1 `rabbits_only_r18` | round 17's first stage with round 16b's seasons, 400 iterations; stage 2 only if at least 20 of 50 evaluated episodes keep hunters and rabbits | evaluation (50 episodes): **31 keep hunters and rabbits**, rabbits out in 19, 12 of them by step ~1170 (`results/calibration_rabbits_only_r18_eval.json`) | finished 400 |
| 18 `with_mammoths_r18` | 17b continued from stage 1 of round 18, so seasons in both stages | no boom (cap ~0.4%); mammoths ~40% of the hunters' energy at iterations 0-150, ~25% at 950-981, mammoth food steady while rabbit food grew; training joint rate 0.072 falling to 0.054, partner above alone from the start (0.46 vs 0.41 at 950-981), pair kills ~1.1 per episode. Evaluations (50 episodes): iteration 500 **29 coexist**, joint 0.039 vs chance 0.034, partner 0.40 vs alone 0.35, mammoths 23% of the food; final checkpoint **40** (hunters out in 10, rabbits in 7), joint 0.032 vs 0.030 (1.1x), partner 0.38 vs 0.31, mammoths 21% of the food, 50 of 413 kills by a group (`results/calibration_with_mammoths_r18_eval.json`, `_iter500_eval.json`) | finished 1,000 |
| 19a `with_mammoths_r19a` | round 14 with `hunt_min_party` 2 and energy ratio 0.75 | no boom (cap ~0.2%); mammoths ~13% of the hunters' energy at iterations 50-100, ~6% at 300-325; evaluation at iteration 400 (50 episodes): **46 coexist** (hunters out in 4), joint 0.026 vs chance 0.024 (1.1x), partner 0.25 vs alone 0.18, mammoths 3% of the food, 32 kills by a group (`results/calibration_with_mammoths_r19a_iter400_eval.json`) | stopped at iteration ~410 (for round 20's slot): hunters left the mammoths |
| 19b `with_mammoths_r19b` | as 19a with energy ratio 0.5 | boom: cap binding ~29% and training episodes ~220 decisions at iterations 150-194, mammoths ~76% of the hunters' energy. Evaluation at iteration 190 (50 episodes): **3 coexist** (both prey out in 46: rabbits at a median step of ~900, mammoths ~1,280), joint 0.248 vs chance 0.113 (2.2x), partner 0.66 vs alone 0.54, mammoths 82% of the food, 1,237 kills by a group, none solo (`results/calibration_with_mammoths_r19b_iter190_eval.json`); iteration 290 **0 coexist** (both prey out in 46, hunters out in the other 4), joint 0.337 vs 0.152 (2.2x), partner 0.64 vs 0.55, mammoths 87%, cap 49% (`_iter290_eval.json`) | stopped at iteration 294 (for 19c's slot) |
| 19c `with_mammoths_r19c` | as 19a with energy ratio 0.6 | evaluation (50 episodes) by iteration: 250 **30 coexist**, joint 1.3x chance, partner 0.42 vs alone 0.30, mammoths 46% of the food, 842 kills by a group (`results/calibration_with_mammoths_r19c_iter250_eval.json`); 500 **42 coexist**, joint 0.039 vs 0.033 (1.2x), partner 0.39 vs 0.26, mammoths 19%, 253 group kills (`_iter500_eval.json`); 1000 **39 coexist**, joint 0.036 vs 0.031 (1.1x), partner 0.36 vs 0.23, mammoths 24%, 276 group kills, hunters out in 10, rabbits out in 6 (`_eval.json`). Accounts at 1000: pair hunts +8.7 energy per hunter, 15,590 solo hunts at -0.66 each, all hunts together -0.39 per 1000 hunter steps (interval -1.9 to +1.0) against rabbits' +10.8 (`results/payoffs_r19c.json`) | completed 1000 iterations; coexistence recovers while mammoths and joint arrivals fall back, the same trade-off as earlier rounds |
| 20 `with_mammoths_r20` | 19b with a wait of 1000 physics steps between a hunter's births (`reproduction_predator_interval`). Set in advance: continue if at least 35 of 50 keep all three species, joint arrivals stay well above chance and mammoths stay about half the food or more; otherwise stop calibrating | evaluation (50 episodes) at 1000: **50 coexist**, joint 0.019 vs chance 0.020 (0.95x), partner 0.27 vs alone 0.19, mammoths 7% of the food, 82 kills by a group, cap never binding (`results/calibration_with_mammoths_r20_eval.json`); iteration 500: 38 coexist (hunters out in 12), joint 0.86x, partner 0.21 vs 0.13, mammoths 3% (`_iter500_eval.json`). Accounts at 1000: pair hunts +12.7 energy per hunter but only 98 of them against 11,572 solo hunts at -0.66 each, all hunts together -1.76 per 1000 hunter steps (interval -2.0 to -1.6) against rabbits' +13.4 (`results/payoffs_r20.json`) | completed 1000 iterations; **test failed** on joint arrivals and mammoth share, so calibration of this module stops. The birth wait gives stable worlds without group hunting, as in rounds 7-8 |

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

Round 18 tested that, seasons in both stages (one training run). Final checkpoint
against 17b, 16b and 14, in that order: all three species 40 of 50 against 39, 26
(exact test p = 0.006, uncorrected) and 47 (p = 0.07); joint arrivals 1.1x chance
against 1.1x, 1.2x and 1.2x; partner minus alone +0.08 against +0.03, +0.10 and
+0.06; mammoths' share 21% against 18%, 26% and 17%; group kills 50 of 413 against
34 of 405, 94 of 421 and 43 of 360. By season (`results/seasons_r18.json`): 21% in
the lean half against 18% in the good half (difference 95% interval -0.02 to
+0.09, 35 group kills), which includes no shift. So seasons in the first stage went
with a partner effect that 17b lacks, but hunters do not arrive together much
above chance, so this may not be coordination; the join rates could also reflect
where partners tend to be, not measured. Against round 14 there is no clear gain.

Why do hunters drift to rabbits? Their energy accounts (`payoffs.py`, 50 episodes
each, `results/payoffs_<run>.json`; no training), energy per 1000 hunter steps:

| Run | All three species | Rabbit food | Hunts, net (95% interval) | Scavenged | Hunts with 2+ hunters |
|---|---|---|---|---|---|
| 16b iter 500 | 7 | 6.3 | +2.11 (+1.27 to +2.96) | 1.78 | 6.6% |
| 13a | 10 | 7.4 | +2.20 (+1.32 to +3.28) | 1.87 | 6.4% |
| 14 iter 300 | 14 | 9.0 | +0.77 (-0.17 to +1.78) | 1.46 | 5.1% |
| 16b | 26 | 9.7 | +0.26 (-0.06 to +0.66) | 0.66 | 2.6% |
| 18 | 40 | 11.3 | -1.52 (-1.76 to -1.25) | 0.52 | 1.5% |
| 14 | 47 | 10.6 | -1.00 (-1.18 to -0.82) | 0.39 | 1.5% |

So at the end of training the hunts of rounds 14 and 18 cost the hunters more than
they bring, while group hunts themselves still pay. Per hunter per hunt, solo
hunts net -0.21 to -0.34 in all six (2-4% kill), pairs +2.82 to +4.69 (18-31% kill).
Under the energy rule a hunt's result is fixed by the energies, and hunters are
lean against mammoths: parties average about 5 energy per hunter, mammoths hunted
about 14-20. In training the hunted mammoth's energy rises with the mammoths'
learning, 11 to 19 in round 14, 11 to 16 in 16b and 11 to 18 in 18, while the
party's stays about 5 per hunter, so a typical pair cannot win. Only about 1% of a
lone hunter's contacts could win, yet hunters alone still join 15-30% of those
that cannot (46-76% of those that can, round 18 38%), so they use the energies
partly. So both may contribute: mammoths getting harder to kill, and hunters not
learning to wait or to hunt together. Across these six checkpoints the worlds where
hunts pay most keep the fewest species (an association; each differs in training
and rule), which fits round 13's earlier reading that mammoth food feeds hunters
who then eat out the rabbits, not tested here. Bout-level rates in the same files
are descriptive only: bouts are chosen by the policy and take their group from
their largest party.

Replay at a lower energy ratio (`payoffs.py --energy-ratio`, final policies, not
retrained, same seeds, `results/replay_ratio_<run>_<ratio>.json`):

| Run | Ratio | All three species | Rabbits out | Hunters out | Mammoths out | Solo kills | Kills by 2+ |
|---|---|---|---|---|---|---|---|
| 14 | 1 | 47 | 1 | 3 | 0 | 317 | 43 |
| 14 | 0.75 | 25 | 25 | 25 | 0 | 775 | 151 |
| 14 | 0.5 | 0 | 50 | 33 | 17 | 1,258 | 265 |
| 18 | 1 | 40 | 7 | 10 | 0 | 363 | 50 |
| 18 | 0.75 | 13 | 35 | 37 | 0 | 643 | 146 |
| 18 | 0.5 | 0 | 50 | 29 | 21 | 1,463 | 343 |

Rabbits go first (median step at ratio 0.5: 998 and 657), against about 1,000-1,700
at 0.75. A lower ratio mostly adds solo kills: the solo hunt turns profitable (+1.4
per hunter at 0.5), so it does not make a partner necessary. These policies and
their rabbits never trained under these ratios, so this is what the rule does
at once, not what training would reach. Round 19 at ratio 0.5 was not started.

The energy rule also lets one hunter kill: 317 of round 14's 360 kills were solo.
With a minimum party of two (`hunt_min_party`; smaller parties always fail, at the
cost of any failure; same replay, `results/replay_minparty2_<run>_<ratio>.json`):

| Run | Ratio | All three species | Rabbits out | Hunters out | Kills by 2+ | Pair, net per hunter per hunt | Rabbit food per 1000 steps |
|---|---|---|---|---|---|---|---|
| 14 | 1 | 38 | 0 | 12 | 25 | +3.57 | 12.6 |
| 14 | 0.75 | 40 | 1 | 10 | 92 | +7.79 | 11.6 |
| 14 | 0.5 | 34 | 6 | 16 | 191 | +11.31 | 10.1 |
| 14 | 0.25 | 31 | 9 | 17 | 452 | +15.49 | 7.5 |
| 18 | 1 | 41 | 0 | 9 | 21 | +3.53 | 15.1 |
| 18 | 0.75 | 42 | 0 | 8 | 34 | +5.01 | 14.7 |
| 18 | 0.5 | 40 | 3 | 10 | 92 | +9.09 | 13.3 |
| 18 | 0.25 | 25 | 20 | 18 | 765 | +15.29 | 6.2 |

Against the same ratios without the minimum (0 of 50 at 0.5), the rabbits mostly
survive, so solo kills seem to carry much of what eats them out; hunters lose more
often than with their own solo kills at ratio 1. A solo hunt now always costs 0.66,
and these policies still try: hunters alone join 16-31% of contacts. This is
behaviour never trained under the rule, with few pairs forming, so it does not show
what happens once hunters learn to hunt together, when more mammoth food could
bring the boom back.

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

## 5. One more run: lean and fat rabbits, rabbits learning (2026-11-06)

At the user's request, after the stag hunt line was closed: one more run, the
lean-and-fat world of
[`../stag_hunt_rule_prey/RESULTS.md`](../stag_hunt_rule_prey/RESULTS.md) (sections
11 and 12), but with hunters and rabbits both learning, and round 1a as the comparison.

**World (arm `lean_fat_learning`, `launchers/calibration.py`):** round 1a unchanged
(no mammoths, 6 hunters, 8 rabbits at the start, rabbit speed 6, 800 x 800, births
reward for both species, whole episodes in training), plus:

- **The lean and fat rule** of the rule-prey module (`rabbit_fat_energy` 6): a hunter
  touching a rabbit with no other hunter within 64 units catches it with probability 1
  at energy 3 or less, falling linearly to 0 at energy 6; with another hunter within 64
  units, always, shared equally. A failed attempt blocks the hunter's catching for 8
  physics steps.
- **Hunters see each rabbit's energy** in their rabbit slots, and their own catch
  cooldown (`observe_rabbit_condition`). In the rule-prey runs they could not: they saw
  rabbits' positions and velocities only, so they could not tell a fat rabbit from a
  lean one. That gap was found while building this run and is recorded there too.

From scratch, 1000 iterations, 14 env runners, one seed. With the rule off and the
observation unchanged, the code replays round 1a's evaluated episodes exactly (checked
on 6 episodes).

**Tests, set before training** (final checkpoint):

1. **Coexistence:** hunters and rabbits both alive at step 8000 in at least 45 of 50
   evaluated episodes (round 1a without the rule: 49).
2. **Cooperation:** with the fair measure (`launchers/coop_fair.py`, 20 episodes, each
   approach of a hunter to a rabbit counted once), approaches to fat rabbits have another
   hunter within 64 units more often than approaches to lean rabbits and than chance
   (hunter time with another hunter within 64 units).
3. **Learned, not imposed:** the share in test 2 for fat rabbits is also above that of
   round 1a's policies replayed with the rule switched on (trained without it, so neither
   hunters nor rabbits learned it).

Also recorded, not tests: iterations 250 and 500, and the reference replay of test 3
(coexistence and the fair measure).

**Hard stop:** one run, no changes afterwards, whatever the result.

**Caveats, set in advance:** one seed. Compared with the rule-prey lean-and-fat run, two
things change at once (learning rabbits and visible rabbit energy), as well as the world
(6 instead of 10 hunters, slower rabbits, no truncation at prey extinction in training),
so a difference from that run cannot be put down to either. Compared with round 1a, the
rule and the hunters' view of rabbit energy change together. Rabbits that learn may
avoid lone hunters while lean, or stay near others; that would be adaptation, not a
fault.

**Reference, before training** (round 1a's policies, trained without the rule, replayed
with it on; `results/replay_rabbits_only_lean_fat_*.json`): hunters died out in all 50
episodes (median step about 1,500, latest 4,354). The fair measure (20 episodes): chance
7.1%, approaches with a partner near lean 5.5% (256 approaches), middle 8.9%, fat 9.9%
(181); fat above lean in 10 of 20 episodes. These untrained-for-the-rule policies would
pass test 2 on small counts, which is why test 3 compares with them.

### 5.1 Result (2026-11-06): coexistence met, cooperation not shown

Training ran 1000 iterations without failure. Evaluated with the same scripts as the
reference (`results/calibration_lean_fat_learning{_iter250,_iter500,}_{eval,coop}.json`):

| Checkpoint | Both alive at step 8000 (of 50) | Chance | Partner near: lean | middle | fat | Fat above lean (episodes of 20) | Fat above chance (episodes of 20) |
|---|---|---|---|---|---|---|---|
| Iteration 250 | 42 (rabbits eaten out in 8) | 33.3% | 23.0% | 23.3% | 29.6% | 17 | 2 |
| Iteration 500 | **50** | 28.5% | 19.3% | 18.1% | 24.1% | 18 | 1 |
| Final (1000) | **46** (hunters died out in 4) | 24.6% | 16.8% | 15.4% | 23.6% | 18 | 6 |
| Reference: round 1a policies, rule on | 0 (hunters died out in 50) | 7.1% | 5.5% | 8.9% | 9.9% | 10 | – |

Approach counts at the final checkpoint: lean 5,542, middle 6,758, fat 4,699.
Catches over the 50 final episodes: 16,348 by a lone hunter, 7,199 by a party, and
27,456 failed lone attempts.

**Tests at the final checkpoint:**

1. **Coexistence: met.** 46 of 50 (target 45; round 1a without the rule 49).
2. **Cooperation: not met.** Fat approaches had a partner near more often than lean
   ones (23.6% against 16.8%, in 18 of 20 episodes), but not more often than chance
   (24.6%; above it in 6 of 20 episodes; fat minus chance -1.0 points, bootstrap 95%
   interval over episodes -2.2 to +0.1).
3. **Above the reference: met** (23.6% against 9.9%). With test 2 failed this does not
   show learned cooperation: the reference's chance level was 7.1%, so most of the gap
   is that the trained hunters simply spend more time near each other.

Under the pre-registered tests, the stag hunt was not shown. Per the hard stop there
are no further changes.

**Reading (exploratory, one seed, not tested):** the pattern is the same at all three
checkpoints. Lean and middle approaches have a partner near *less* often than chance,
and fat approaches about at chance. One description that fits: hunters approach fat
rabbits as often with a partner as their general closeness to other hunters gives, and
peel off alone for lean ones; that is, they learned which rabbits a lone hunter can
take, not to seek a partner for the ones it cannot. Lone attempts that fail outnumber
lone catches (27,456 against 16,348), so hunters still try fat rabbits alone. This is
the closest the line came to both targets in one world: coexistence as in round 1a,
and a fat-above-lean difference that round 1a's policies did not show (10 of 20) —
but the fat share does not exceed chance, which is what the stag hunt needed.

## 6. Follow-up: bigger fat rabbits, catches shared evenly (2026-11-06)

At the user's request, after section 5 (which had been set as a hard stop). Section
5's hunters learned to skip partners for lean rabbits but did not seek them for fat
ones. One account: the payoffs barely made the fat rabbit a stag. A hunter gains half
a caught rabbit's energy, and rabbits bred at 8 (dropping to 5), so a fat rabbit (6
to 8) shared by two paid each about 1.75, against up to 1.5 for a lean rabbit alone
(in replays, 1.4 times per head on average).

**World (arm `lean_fat_big`):** section 5's arm with rabbits breeding at energy 16
instead of 8 (`reproduction_rabbit_threshold`; a parent still pays its child's 3) and
fat from 12 instead of 6 (`rabbit_fat_energy`; lean still 3 or less). Catches are
shared evenly, as before. A fat rabbit (12 to 16) shared by two pays each 3 to 4,
against up to 1.5 for a lean rabbit alone. From scratch, 1000 iterations, 14 env
runners, one seed.

**Paper check before training** (scratch replays of section 5's final policies, 10
episodes; these policies never trained here): a party's fat rabbit paid each hunter
2.6 times a lone lean catch (section 5's world: 1.4), but rabbits were fat 90% of the
time and lean 2%, hunters' food per step fell to about their resting cost, and every
episode ended before step 2,800. The run tests whether training from scratch finds a
way to live in this world. (A no-split variant, not what the user asked for, was
started by mistake and stopped at iteration 186 unevaluated; its code was reverted.)

**Tests, set before training** (final checkpoint; the same as section 5):

1. **Coexistence:** hunters and rabbits both alive at step 8000 in at least 45 of 50
   evaluated episodes.
2. **Cooperation:** with the fair measure (`launchers/coop_fair.py`, 20 episodes),
   approaches to fat rabbits have another hunter within 64 units more often than
   approaches to lean rabbits and than chance.
3. **Learned, not imposed:** the fat share in test 2 above that of section 5's final
   policies replayed in this world (never trained in it).

Also recorded, not tests: iterations 250 and 500.

**Reference, before training** (section 5's final policies in this world;
`results/replay_lean_fat_learning_big_*.json`): both alive at step 8000 in 0 of 50
(hunters died out in 30, median step about 2,000; rabbits in 20, median about 1,200).
Fair measure: chance 20.5%, fat 17.4% (447 approaches), lean 12.5% (8), middle 0%
(45).

**Hard stop:** one run, no changes afterwards, whatever the result.

**Caveats, set in advance:** one seed. Rabbits breed half as often per unit of food
gathered, so rabbit numbers and the hunters' food supply change along with the payoff;
a difference from section 5 can come from either. Lean rabbits are mainly the young,
so lone hunting depends on catching them before they grow.

**Stopped (2026-11-06), not evaluated:** I stopped this run at iteration 752 of 1000,
reading a request to change the rabbits' energies as a request to replace it; the
user had not asked for it to be stopped, and chose to go on to section 7 rather than
resume it. Checkpoints to iteration 750 are kept. Training curves at the stop: no
rabbit was born in any training episode at any point (all 752 iterations); episodes
ended at about step 900 (at most 1,013), with hunters catching about 7 of the 8
starting rabbits. A reading, not tested: starting at 3 and breeding at 16, a rabbit
needs about 7 grass patches or more before its first birth, all that time catchable
by a lone hunter; the hunters ate them first, and with no births the rabbits, rewarded
for births only, had nothing to learn from.

## 7. Follow-up: the user's rabbit energies (2026-11-06)

The user's settings, after section 6.

**World (arm `lean_fat_r7`):** section 5's arm with rabbits born at energy 2 (the
starting rabbits too; was 3), breeding at 7 (was 8) and paying 3 per birth (new
setting `reproduction_rabbit_cost`; before, a parent paid its child's start energy),
so 1 is lost in each birth and an adult drops to 4. Lean: 2 or less (was 3); fat: 5
or more (was 6). Catches shared evenly, as in section 5. From scratch, 1000
iterations, 14 env runners, one seed.

A lone hunter catches a rabbit with probability 1 at energy 2 or less, falling
linearly to 0 at 5; a party (another hunter within 64 units) always, shared evenly.
A hunter gains half a caught rabbit's energy, so the best a lone hunter can expect
per attempt is about 1.04 (rabbit energy 2.5, chance 5/6), and two hunters sharing a
fat rabbit (5 to 7) gain 1.25 to 1.75 each.

**Paper check** (scratch replays of section 5's final policies, 10 episodes; these
policies never trained here): a party's fat rabbit paid each hunter 1.8 times a lone
lean catch; hunters starved out by step 2,223 in all 10, rabbits multiplying to about
65.

**Tests, set before training** (final checkpoint; the same as section 5):

1. **Coexistence:** hunters and rabbits both alive at step 8000 in at least 45 of 50
   evaluated episodes.
2. **Cooperation:** with the fair measure (`launchers/coop_fair.py`, 20 episodes),
   approaches to fat rabbits have another hunter within 64 units more often than
   approaches to lean rabbits and than chance.
3. **Learned, not imposed:** the fat share in test 2 above that of section 5's final
   policies replayed in this world (never trained in it).

Also recorded, not tests: iterations 250 and 500.

**Reference, before training** (section 5's final policies in this world;
`results/replay_lean_fat_learning_r7_*.json`): both alive at step 8000 in 0 of 50
(hunters died out in all 50, median step about 1,750). Fair measure: chance 7.4%,
lean 8.2% (522 approaches), middle 6.1%, fat 9.2% (519); fat above lean in 11 of 20.
Like section 5's reference, these policies would pass test 2 on small margins, which
is why test 3 compares with them.

**Caveats, set in advance:** one seed. Several energies change at once, so a
difference from section 5 cannot be put down to one of them. The per-head gain of a
pair on a fat rabbit over the best lone attempt is modest (about 1.2 to 1.7 times).

### 7.1 Result (2026-11-07): coexistence met, cooperation not shown

Training ran 1000 iterations without failure. Evaluated with the same scripts as
section 5 (`results/calibration_lean_fat_r7{_iter250,_iter500,}_{eval,coop}.json`):

| Checkpoint | Both alive at step 8000 (of 50) | Chance | Partner near: lean | middle | fat | Fat above lean (episodes of 20) | Fat above chance (episodes of 20) |
|---|---|---|---|---|---|---|---|
| Iteration 250 | **49** (hunters died out in 1) | 19.6% | 13.8% | 13.4% | 16.9% | 16 | 3 |
| Iteration 500 | **47** (hunters died out in 3) | 21.8% | 14.7% | 13.7% | 18.9% | 20 | 2 |
| Final (1000) | **46** (hunters died out in 4) | 14.6% | 9.6% | 10.2% | 13.2% | 16 | 5 |
| Reference: section 5's final policies | 0 (hunters died out in 50) | 7.4% | 8.2% | 6.1% | 9.2% | 11 | – |

Approach counts at the final checkpoint: lean 4,880, middle 7,396, fat 4,391.
Catches over the 50 final episodes: 19,373 by a lone hunter, 5,605 by a party, and
39,920 failed lone attempts.

**Tests at the final checkpoint:**

1. **Coexistence: met.** 46 of 50 (target 45).
2. **Cooperation: not met.** Fat approaches had a partner near more often than lean
   ones (13.2% against 9.6%, in 16 of 20 episodes), but less often than chance
   (14.6%; above it in 5 of 20 episodes; fat minus chance -1.4 points, bootstrap 95%
   interval over episodes -2.2 to -0.6, so below chance).
3. **Above the reference: met** (13.2% against 9.2%). With test 2 failed this does not
   show learned cooperation; as in section 5, most of the gap is that the trained
   hunters spend more time near each other (chance 14.6% against 7.4%).

**Reading (exploratory, one seed, not tested):** the same pattern as section 5 at all
three checkpoints, with the fat share a little further below chance (fat minus chance
-2.7, -2.9 and -1.4 points at iterations 250, 500 and 1000, each interval below 0).
Hunters approach lean rabbits alone and fat ones at about their general closeness to
other hunters; failed lone attempts outnumber lone catches about 2 to 1, so hunters
still try fat rabbits alone. The new energies kept coexistence but did not change the
cooperation result. Section 8 (calls and catching together) was set before these
results were known.

## 8. Follow-up: calls and catching together (2026-11-06)

At the user's request, while section 7 trains. Section 7's hunters, like section
5's, skipped partners for rabbits they could take alone but did not seek one for fat
rabbits beyond chance (early look at iteration 340: fat 18.2% against chance 20.0%,
lean 13.6%; not a test). Two missing pieces of human group hunting are added together:
a way to recruit, and a share only for those who hunt.

**World (arm `lean_fat_r7_signal`):** section 7's arm plus:

- **Calls** (`hunter_calls`, `call_radius` 200): a hunter's second action bit, the
  join bit of mammoth hunts and unused without mammoths, becomes a call with no effect
  of its own. Every hunter observes the nearest other hunter that called in the last
  decision within 200 units, in any direction (like a sound, outside the view cone):
  heard, direction and distance. Calls have no built-in meaning.
- **Catching together** (`rabbit_fat_touch_only`, `rabbit_touch_window` 8): a fat
  rabbit (energy 5 or more) is caught only once two or more different hunters have
  each touched it, while ready, within 8 physics steps (one decision), and is shared
  evenly among those touchers only. A hunter merely within 64 units gains nothing. A
  touch with no other toucher in the window fails, with the catch cooldown. Lean and
  middle rabbits as in section 7. The window was chosen by the user over strictly
  simultaneous touching after a paper check: section 7's policies at iteration 570
  replayed with simultaneous touching (20 episodes, never trained under it) made 35
  such catches in 20 episodes, against 3,662 party catches under section 7's rule.

From scratch, 1000 iterations, 14 env runners, one seed. Trained from a separate
checkout (code commit b8bf0763, merged to main after section 7's evaluations) and
started automatically when section 7's training ended, so the two never shared the
GPU.

**Tests, set before training** (final checkpoint):

1. **Coexistence:** hunters and rabbits both alive at step 8000 in at least 45 of 50
   evaluated episodes.
2. **Cooperation:** with the fair measure (`launchers/coop_fair.py`, 20 episodes),
   approaches to fat rabbits have another hunter within 64 units more often than
   approaches to lean rabbits and than chance.
3. **Above section 7:** the fat share in test 2 divided by chance above section 7's
   final value of the same ratio. (Section 7's policies cannot be replayed here as a
   reference, since the call adds to the observation.)
4. **Calls followed by a partner** (`launchers/calls.py`, 20 episodes): when a hunter
   is alone at an open fat rabbit (within 64 units of it, no other hunter within 64 of
   the rabbit), another hunter reaches the rabbit within 4 decisions more often after
   a call than after silence; 95% bootstrap interval over episodes of the difference
   above 0. This is an association, not a causal effect: the policy chooses when to
   call.

Also recorded, not tests: iterations 250 and 500; fat rabbits caught together; call
rate at a fat rabbit against elsewhere.

**Hard stop:** one run, no changes afterwards, whatever the result.

**Caveats, set in advance:** one seed. Calls and the catching rule change together, so
a difference from section 7 cannot be put down to either. Catching together may be too
hard to learn (the paper check above), in which case fat rabbits would drop out of the
diet; that would fail test 2, not show a new kind of cooperation.
