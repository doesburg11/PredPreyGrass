# base_environment_continuous

A continuous-space Predator-Prey-Grass environment for RLlib: predators
(hunters) and prey (antelopes) move with steering physics on an 800 × 800
torus, prey graze grass clusters that move as they are grazed, both
species pay energy to live and to move, starve at zero energy and give
birth at an energy threshold. Each agent chooses one of 16 directions at
full speed, half speed or standing still, and sees the nearest animals and
grass patch in its view cone. The only reward is +10 per birth, as in
[`base_environment`](../base_environment/).

The environment was developed in the
[PredPreyGrassContinuous](https://github.com/doesburg11/PredPreyGrassContinuous)
repository, first on top of the physics-based
[Aquarium](https://github.com/michaelkoelle/marl-aquarium) package
(unmaintained since 2024) with a stack of patches (its `add_walls`
module), then as this standalone `MultiAgentEnv` without Aquarium or
PettingZoo. It contains the world of that repository's milestone runs
(AW-17 to AW-22): no walls, 75% of the energy cost at full speed from
movement (Taylor, Heglund & Maloiy 1982), prey a little faster than
predators, and grass in 20 small clusters. In that world predators and prey
keep improving against each other for about 2,000 training iterations, a
Red Queen arms race.

This first version reproduces `add_walls` **exactly**, including
Aquarium's quirks, so the results obtained there remain valid here.
[SPEC.md](SPEC.md) records every rule, and in section 9 the quirks that are
candidates for a later, deliberate cleanup.

## Files

- [`environment.py`](environment.py): `PredPreyGrassContinuous`, an RLlib
  `MultiAgentEnv` (animals, physics, energy, reproduction, observations,
  decisions).
- [`grass.py`](grass.py): grass patches, seed dispersal and overgrazing.
- [`geometry.py`](geometry.py): 2D vectors and torus distances, offsets and
  view cones.
- [`config/config_env.py`](config/config_env.py) and
  [`config/config_ppo.py`](config/config_ppo.py): settings for training.
- [`train.py`](train.py): PPO training, one policy per species.
- [`eval.py`](eval.py) with [`config/config_eval.py`](config/config_eval.py):
  play a trained checkpoint (or random actions) in a window, to an mp4, or
  without rendering.
- [`render.py`](render.py): the pygame viewer (hunters and antelopes facing
  their direction of movement, energy bars, grass, one predator's and one
  prey's view cone, and the populations over the episode). It only reads
  the state, so watching never changes an episode.
- [`tournament.py`](tournament.py): every predator checkpoint of a run
  against every prey checkpoint, to see an arms race.
- [`checkpoints.py`](checkpoints.py): a checkpoint's settings and policies.
- [`analysis/final_check.py`](analysis/final_check.py): final check of a
  trained checkpoint (outcomes, catch risk, populations, speed choices,
  time and catches near grass).
- [`assets/`](assets/): the sprites.
- [`tests/`](tests/): the regression test against `add_walls` and unit
  tests.

## Usage

From the repository root:

```bash
python predpreygrass/non_evolutionary/base_environment_continuous/train.py
python predpreygrass/non_evolutionary/base_environment_continuous/eval.py
python predpreygrass/non_evolutionary/base_environment_continuous/tournament.py \
    ~/simulation_results/ray_results/base_environment_continuous/<run> --every 300
python -m pytest predpreygrass/non_evolutionary/base_environment_continuous/tests
```

Training and evaluation take no command-line arguments: edit the config
files. Runs are saved in
`~/simulation_results/ray_results/base_environment_continuous/`
(`RAY_RESULTS_DIR` in `predpreygrass/global_config.py`), each with a copy of
the code and settings it was started with in `source_code/`. Saving videos
needs `imageio` and `imageio-ffmpeg` (`pip install imageio imageio-ffmpeg`);
on a machine without a display, set `SDL_VIDEODRIVER=dummy`.

## How the equivalence was checked

- In the development repository, `tests/test_equivalence.py` records
  trajectories of `add_walls`'s environment with random actions, replays
  the same actions here and compares every decision bit for bit:
  observations, rewards, flags, infos and the full state, including the
  random-number generators. 61 episodes, about 12,500 decisions, covering
  births, catches, starvation, grass collapse, extinction, the time limit,
  one physics step per decision, population caps and exhausted ID pools.
- Here, [`tests/test_regression.py`](tests/test_regression.py) checks the
  same trajectories without `add_walls` or Aquarium, against a digest of
  every decision of `add_walls`'s results
  ([`tests/data/reference_digests.json`](tests/data/reference_digests.json)).
  It also passes with this repository's NumPy 1.26 and Torch 2.13 (the
  development repository uses NumPy 2.4 and Torch 2.14).
- Run AW-21's trained policies played the same 50 seeded episodes in both
  environments with identical results (55,298 decisions, 21,873 births,
  19,860 catches).

Two deliberate differences, neither affecting the dynamics: with
`reset(seed=None)` each environment has its own spawn stream instead of
Python's process-global one, and the keys of the returned dictionaries come
in a different order.

## Validation runs

BE-01 and BE-02 trained the milestone world from scratch with this
environment (3,000 iterations, 14 simulation workers each, side by side),
as AW-21 and AW-22 did with `add_walls`. They reached the same state: both
species survived in 72 of 100 evaluation episodes (35 and 37 of 50 per
run) against 71 of 100 for AW-21 and AW-22 (33 and 38 of 50), the same catch risk, populations and behaviour
(predators waiting near grass), and the same arms race. Details in
[RESULTS.md](RESULTS.md). Their run folders are in
`~/simulation_results/ray_results/base_environment_continuous/`;
`config/config_eval.py` plays BE-01's final policies by default.

Training took 8–15% less wall time than with `add_walls` (about 9 hours
instead of 10 for 3,000 iterations): the environment plays episodes about
twice as fast, but the learner takes most of the time.

BE-01 and BE-02 were trained in the development repository under NumPy 2.
RLlib cannot restore such checkpoints under this repository's NumPy 1.26,
so `checkpoints.py` then rebuilds RLlib's default PPO network, which
`train.py` always uses, from the run's saved settings and loads the
trained weights, refusing any mismatch in their names or shapes. For BE-01
the rebuilt policies give bit-identical outputs, and its 50-episode final
check gives identical episodes in both repositories.
