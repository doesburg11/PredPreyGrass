# stag_hunt_continuous

**Status (2026-10-30): built (environment, training, tests); not yet
calibrated or trained.** See [DESIGN.md](DESIGN.md).

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
- [`tests/`](tests/): spaces, determinism, catching, food layers, episode
  ends and the hunt rule.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.train
    python -m pytest predpreygrass/non_evolutionary/continuous_space/stag_hunt_continuous/tests

## TODO

- [x] Decide the open points (DESIGN.md, section 6; 2026-10-30).
- [x] Build: environment, three policies, hunt rule, observations,
  metrics, tests; Codex review (2026-10-30).
- [ ] Evaluation tools: final check (coexistence), the density-matched
  chance level for joint arrivals (the joint arrivals themselves are
  counted by the environment), viewer.
- [ ] Calibrate viability from scratch.
- [ ] Pre-register and run the main test (hunt arm against a solo
  control and untrained policies, joint arrivals above chance).
