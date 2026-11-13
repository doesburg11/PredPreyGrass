# deer_food_sharing

**Status (2026-11-13): CLOSED, not met.** Three runs with big game shared all coexisted
(50 of 50), but none partnered on adults more than every run without sharing (C
1.04-1.12 against 1.12-1.21), and none caught more adults; the theory model predicted
this before training. Per the hard stop, the sharing line ends
([RESULTS.md](RESULTS.md) section 1.6).

Does sharing big game make learned cooperation on big game stronger? This is
[deer_stag_hunt](../deer_stag_hunt/)'s touch world, where hunters that only learn
approach adult deer with a partner near slightly more often than chance, with one rule
added: **an adult catch is shared**. The catchers keep half of the food, and the other
half goes to every other hunter within 128 units of the kill, whether it helped or not.
Young deer are not shared. In human foragers the big, unpredictable packages are the
ones shared widely, and sharing is one of the main proposed reasons cooperative
big-game hunting pays (DESIGN.md).

Three runs with sharing are compared with deer_stag_hunt's touch_1-3, the same world
without it. With sharing off, this world is deer_stag_hunt's episode for episode, and
this module's evaluation reproduces touch_1's exactly.

## Running

From the repository root:

```bash
# The three runs, their evaluations and the verdict (waits while
# eco_evolutionary_deer_stag_hunt's section 3 runs hold the GPU)
systemd-run --user --unit deer-sharing bash $PWD/predpreygrass/non_evolutionary/continuous_space/deer_food_sharing/launchers/run_sharing.sh

# One run
python predpreygrass/non_evolutionary/continuous_space/deer_food_sharing/launchers/replicate.py NAME [ITERATIONS] [ENV_RUNNERS]

# Evaluate a checkpoint; partnering against chance; watch it
python -m predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.evaluate CHECKPOINT 50 OUT.json --workers 16
python predpreygrass/non_evolutionary/continuous_space/deer_food_sharing/launchers/coop_fair.py CHECKPOINT 20 10 OUT.json
python -m predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.watch CHECKPOINT
```

## Files

- `environment.py`: deer_stag_hunt's environment plus the sharing rule
  (`adult_share_fraction`, `adult_share_radius`; 0 is no sharing) and its counters.
- `config/config_env.py`: this module's world (touch rule, s = 0.5, R = 128);
  `config/config_ppo.py`: deer_stag_hunt's PPO settings.
- `launchers/run_sharing.sh`, `launchers/replicate.py`: the runs;
  `launchers/verdict_sharing.py`: RESULTS.md section 1's tests;
  `launchers/coop_fair.py`, `launchers/coop_by_energy.py`: partnering measures.
- `tests/`: deer_stag_hunt's tests, the sharing rule (`test_sharing.py`) and
  episode-for-episode equivalence with deer_stag_hunt without sharing
  (`test_equivalence.py`).
- The rest (`train.py`, `evaluate.py`, `checkpoints.py`, `callbacks.py`, `grass.py`,
  `geometry.py`, `render.py`, `watch.py`) is deer_stag_hunt's, with this module's
  paths.
