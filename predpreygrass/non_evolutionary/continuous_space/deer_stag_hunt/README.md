# Hunting Together: A Stag Hunt with Deer

** In 3 independent runs, hunters and deer
coexisted in 43, 49 and 50 of 50 episodes, and approaches to adults had a partner near
1.20, 1.23 and 1.15 times as often as chance (above fawns too; 95% intervals above
zero). run_1 missed the coexistence bar; the other two met all tests. The effect is
small, and adult catches fell late in training in run_1 and run_3 (see RESULTS.md).
Written up for a general reader, with an animation and a results chart, on the site page
[A Replicated Stag Hunt](https://humanbehaviorpatterns.org/learning-selection-interaction/predpreygrass-continuous/deer-stag-hunt).

A stag hunt with one prey species. Hunters (the predators, standing for humans) and
deer share a continuous torus with clustered grass. A **fawn** can be caught by one
hunter for sure, as a hare in the classic stag hunt; an **adult** deer only by two
or more hunters touching it within one decision, as the stag, and only they share it. A
lone hunter can live on fawns, but two hunters sharing an adult each get more than a
fawn gives one hunter. Both species learn (one PPO policy per species) and are rewarded
only for births. The aim: hunters and deer coexisting, with hunters seeking a partner
for adults more than chance.

## Origin

This module is the world of
[`stag_hunt_continuous`](https://github.com/doesburg11/PredPreyGrass-archive/blob/main/stag_hunt_continuous/RESULTS.md)'s section 8 (that module is now in PredPreyGrass-archive),
the one world of that line that met both coexistence (49 of 50 episodes) and
cooperation above chance (adult approaches with a partner near 15.5% against chance
13.2%), stripped to what that run used. Removed: hunters' calls (unused there:
silencing them changed nothing), mammoths and their hunts, the steppe food layer,
burrows, seasons, birth spacing, and the mammoth-era analysis tools. Renamed: rabbits
to deer, lean to fawn, fat to adult. The world is the same: with the same seed and the
same movement actions an episode here is identical to one there, calls off
([tests/test_equivalence.py](tests/test_equivalence.py)). The networks are slightly
smaller: predators have no unused join output and no always-zero inputs (empty mammoth
slots, hunt cooldown), so training is not draw for draw the same.

## The world

Defaults in [environment.py](environment.py) (`DEFAULT_CONFIG`):

| | Hunters | Deer |
| --- | --- | --- |
| Start | 6, energy 5 | 8, energy 2 |
| Food | deer | grass (+2 per patch) |
| Birth | at energy 12, pays 5 | at energy 7, pays 3 |
| Top speed | 5 | 6 |
| Sees | 4 hunters and 2 deer (with their energy) in a 150° cone of 200 | 3 hunters and 3 deer in a 240° cone of 267 |

Catching, by the deer's energy (a deer grows by grazing and gives birth at 7, which
drops it to 4):

| Deer | Energy | Caught by one hunter | Caught with a partner | Energy gained per hunter |
| --- | --- | --- | --- | --- |
| Fawn (the hare) | 2 or less | always | always, if another hunter is within 64 units; shared by all hunters within 64 | alone up to 1 |
| Middle | 2 to 5 | chance (5 − energy) / 3, falling from 1 to 0 | always, as for a fawn | alone 1 to 2.5 |
| Adult (the stag) | 5 or more | never | only if two or more hunters touch it within one decision (8 physics steps); shared by those who touched it only | 1.25 to 1.75 each, when two share it |

A hunter gains half its share of the deer's energy, so two hunters sharing an adult
each get more than a fawn gives one hunter alone. A failed attempt costs each hunter
that touched the deer a wait of 8 physics steps before it can catch again.

Episodes last 8000 physics steps (about 1,400 decisions, since a birth ends a
decision early).

## Running

From the repository root:

```bash
# A training run from scratch (defaults: 1000 iterations, 14 env runners)
python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/replicate.py NAME

# Coexistence and catch counters over 50 episodes
python -m predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.evaluate \
    predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/runs/NAME/checkpoint 50 out.json --workers 16

# Cooperation against chance over 20 episodes
python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/coop_fair.py \
    predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/runs/NAME/checkpoint 20 10 out.json

# Watch an episode (or --video out.mp4)
python -m predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.watch \
    predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/runs/NAME/checkpoint --seed 0
```

TensorBoard's `env_runners/catches/` holds the per-episode counters, among them
`adult_energy_share` (the share of the hunters' energy from deer that came from
adults) and `adults_caught`.

## Files

- `environment.py`: the world (`DeerStagHunt`); `grass.py`, `geometry.py`: grass
  clusters and torus geometry.
- `train.py`, `config/`: training; `callbacks.py`: TensorBoard counters.
- `evaluate.py`, `launchers/coop_fair.py`: evaluation and the fair cooperation
  measure (with a bootstrap interval over episodes); `launchers/replicate.py`: training
  runs from scratch; `launchers/verdict.py`: RESULTS.md section 1's tests and verdict.
- `checkpoints.py`, `render.py`, `watch.py`: loading checkpoints, viewing episodes.
- `tests/`: the rules, evaluation, and equivalence with `stag_hunt_continuous`
  (that test needs a PredPreyGrass-archive checkout next to this repository, or
  `PREDPREYGRASS_ARCHIVE` set to one; it is skipped otherwise).
- `RESULTS.md`: experiments and results.
