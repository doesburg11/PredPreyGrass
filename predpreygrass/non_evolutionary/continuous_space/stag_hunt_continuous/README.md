# stag_hunt_continuous

**Status (2026-11-05): stopped, target not met.** Calibration stopped after 20
rounds, and the follow-up with rule-following prey,
[`../stag_hunt_rule_prey/`](../stag_hunt_rule_prey/README.md), was stopped the same
day. The stag hunt line as a whole was closed on 2026-11-06, with a closing summary in
[`../stag_hunt_rule_prey/RESULTS.md`](../stag_hunt_rule_prey/RESULTS.md), section 15. Details of every round,
the reasons for each change and for stopping are in [RESULTS.md](RESULTS.md).

**One more run (2026-11-06, RESULTS section 5):** lean and fat rabbits with both
species learning and hunters seeing rabbit energy. Hunters and rabbits coexisted in 46
of 50 episodes, but approaches to fat rabbits had a partner near no more often than
chance (23.6% against 24.6%; lean 16.8%), so cooperation was not shown. Hard stop: no
changes after this run.

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

- [`RESULTS.md`](RESULTS.md): every calibration round, reasons, analyses.
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
  [`seasons.py`](seasons.py), [`payoffs.py`](payoffs.py): prey switching by
  rabbit count, policies of two checkpoints mixed, populations, food and kills
  by phase of the grass seasons, and the hunters' energy accounts by hunt and
  by what they were doing.
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
- [x] Calibrate viability from scratch: stopped 2026-11-05 after 20 rounds
  without meeting the target ([RESULTS.md](RESULTS.md)).
- [ ] Not run in this module: the main test (hunt arm against a solo control
  and untrained policies, joint arrivals above chance). It needed a world
  that met the calibration target first.

## Results in short

The target was all three species in at least 45 of 50 evaluated episodes, mammoths
about half of hunters' food or more, and cooperation signals, without a tight hunter
cap. No round met it. Worlds with strong cooperation boomed and lost prey, and the
brakes that kept the prey also lost the group hunting. One seed per arm, so this is
calibration and not a test.

| Run | All three species (of 50) | Joint arrivals vs chance | Mammoth share of food |
|---|---|---|---|
| Round 13a, energy rule | 10 | 1.8x | about half |
| Round 14, faster rabbits | 47 | 1.2x | 17% |
| Round 19b at iteration 290, ratio 0.5 | 0 | 2.2x | 87% |
| Round 19c at iteration 250, ratio 0.6 | 30 | 1.3x | 46% |
| Round 19c final | 39 | 1.1x | 24% |
| Round 20 final, ratio 0.5 with birth wait | 50 | 0.95x | 7% |
