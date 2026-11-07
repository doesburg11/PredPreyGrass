# deer_stag_hunt

**Status (2026-11-07): built; first experiment (replication without calls) set in
[RESULTS.md](RESULTS.md), section 1.**

A stag hunt with one prey species. Hunters (the predators, standing for humans) and
deer share a continuous torus with clustered grass. A **fawn** can be caught by one
hunter for sure, as a hare in the classic stag hunt; an **adult** deer only by two
hunters touching it within one decision, as the stag, and only those two share it. A
lone hunter can live on fawns, but two hunters sharing an adult each get more than a
fawn gives one hunter. Both species learn (one PPO policy per species) and are rewarded
only for births. The aim: hunters and deer coexisting, with hunters seeking a partner
for adults more than chance.

## Origin

This module is the world of
[`stag_hunt_continuous`](../stag_hunt_continuous/README.md)'s RESULTS.md section 8,
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
|---|---|---|
| Start | 6, energy 5 | 8, energy 2 |
| Food | deer | grass (+2 per patch) |
| Birth | at energy 12, pays 5 | at energy 7, pays 3 |
| Top speed | 5 | 6 |
| Sees | 4 hunters and 2 deer (with their energy) in a 150° cone of 200 | 3 hunters and 3 deer in a 240° cone of 267 |

Catching, by the deer's energy:

- **fawn** (2 or less): a lone hunter touching it catches it;
- **middle** (between 2 and 5): a lone hunter with a chance falling linearly from 1
  to 0;
- a fawn or middle deer touched with another hunter within 64 units is always caught,
  shared by every hunter within 64;
- **adult** (5 or more): only once two hunters have touched it within 8 physics steps
  (one decision), shared by those two only.

A failed attempt costs each toucher a wait of 8 physics steps. A hunter gains half its
share of the deer's energy. Episodes last 8000 physics steps (about 1,400 decisions,
since a birth ends a decision early).

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
- `tests/`: the rules, evaluation, and equivalence with `stag_hunt_continuous`.
- `RESULTS.md`: experiments and results.
