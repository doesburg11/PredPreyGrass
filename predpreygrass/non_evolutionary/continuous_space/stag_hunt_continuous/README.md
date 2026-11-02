# stag_hunt_continuous

**Status (2026-11-02): calibrating (DESIGN.md, plan step 1).** Hunters
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
three species); at ratio 1.5 hunters leave the mammoths. See "Calibration log" below and
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

