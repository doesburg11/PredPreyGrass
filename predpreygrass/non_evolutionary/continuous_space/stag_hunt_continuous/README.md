# stag_hunt_continuous

**Status (2026-11-01): calibrating (DESIGN.md, plan step 1).** Hunters
are viable on rabbits alone. With mammoths, no round so far gives both
coexistence and cooperation: worlds where hunters learn to hunt mammoths
together (joint arrivals up to 2.4x chance) end in a hunter boom that eats
out a prey species, and the brakes tried that keep the prey (a wait
between births, faster rabbits) also remove the hunters' interest in
mammoths. Next candidate: rabbit burrows (DESIGN.md, section 7). See "Calibration log" below and
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
The next candidate is refuges in the terrain, rabbit burrows, which
protect rabbits without making them more plentiful (DESIGN.md, section
7, reviewed by Codex). Accepted earlier: mammoths need
not be the hunters' only food (round 5's ~80-85% share is acceptable,
cooperation stays voluntary), and a forced-join-off test was dropped as
it adds no biological realism. The checkpoints of rounds 1 to 9 are kept
for evaluate.py.

Also found: RLlib's environment check at start-up rejects the final
observation of an agent that dies in its one random step (round 4's
first launch); `train.py` now disables that check
(`disable_env_checking`), the environment tests cover those invariants.
Added for these rounds (defaults keep the earlier behaviour, and older
checkpoints load with them, `checkpoints.ADDED_SETTINGS`):
`reproduction_<species>_interval` (physics steps between an animal's
births) and `truncate_on_prey_extinction` (training only; evaluate.py
always plays whole episodes). The Codex review of the latter found that
an animal dying in the step that ended an episode was flagged both
terminated and truncated; it is now terminated only.

