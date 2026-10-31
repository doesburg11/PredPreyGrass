# stag_hunt_continuous

**Status (2026-10-31): calibrating (DESIGN.md, plan step 1).** Hunters
are viable on rabbits alone; the mammoth world flips between hunters
ignoring mammoths and a hunter boom that eats out both prey; round 5
(hunter cap) is running. See "Calibration log" below and
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
iterations) unless an evaluation is named. Short episodes mean both prey
were eaten out: an episode ends early only when no rabbits and no
mammoths are left. Mammoth share = mammoth energy / (mammoth + rabbit
energy) gained by hunters.

| Round (arm) | Change | Outcome | Stopped |
| --- | --- | --- | --- |
| 1a `rabbits_only` | no mammoths | early crash (rabbits eaten out by iteration ~50), then recovery to full-length episodes; evaluation of the final checkpoint: hunters and rabbits coexist in 49 of 50 episodes (`results/calibration_rabbits_only_eval.json`) | finished 1,000 |
| 1b `with_mammoths` | configured world (6 hunters, 4 mammoths, carcass 32) | mammoth share ~1%; ~100 solo attempts per episode, join rate on contact 0.49 → 0.25 | iteration 410 |
| 2 `with_mammoths_r2` | 10 hunters, 8 mammoths; struggle 0.02, join cost 0.02 (failed solo attempt −0.66 instead of −1.65) | mammoth share peaked ~9–14% (iteration ~50–150), then fell to ~4%; joining fell, alone faster than with a partner (0.26 vs 0.20 at the end) | iteration 370 |
| 3 `with_mammoths_r3` | round 2 with carcass 64 (struggle 0.01, solo still −0.66) | hunter boom: mammoth share ~90%, both prey eaten out, episodes ~265 decisions from iteration ~150 with no recovery; joint arrival rate 0.43, join with partner 0.74 vs alone 0.60 | iteration 270 |
| 4 `with_mammoths_r4` | carcass 48 | same as round 3, later: mammoth share ~84%, episodes ~246 decisions by iteration ~190; joint arrival rate 0.29, join with partner 0.64 vs alone 0.56 | iteration 190 |
| 5 `with_mammoths_r5` | round 4 with the living-hunter cap 50 → 25 | running (iteration ~120: episodes ~930–1,100 decisions, mammoth share ~65% and rising) | |

Reading so far: hunters live on rabbits; between a carcass of 32 and 48
the mammoth world tips from hunters ignoring mammoths into a hunter boom
that removes both prey within an episode. Tuning the carcass alone would
leave the main test on that edge, so round 5 limits the boom with the
existing population cap instead. The join and joint-arrival figures are
training metrics without a chance comparison; the checkpoints of rounds
2 to 4 are kept for evaluate.py.

Also found: RLlib's environment check at start-up rejects the final
observation of an agent that dies in its one random step (round 4's
first launch); `train.py` now disables that check
(`disable_env_checking`), the environment tests cover those invariants.

