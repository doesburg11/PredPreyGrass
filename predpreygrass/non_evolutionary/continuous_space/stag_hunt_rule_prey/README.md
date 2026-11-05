# stag_hunt_rule_prey

**Status (2026-11-05): first calibration training (arms `base` and `lean`), results
pending.** Only hunters learn; rabbits and
mammoths follow fixed rules. A simplification of
[`stag_hunt_continuous`](../stag_hunt_continuous/README.md), whose 20 calibration
rounds are the background.

## Why

In `stag_hunt_continuous` all three species have the same network and learn at
the same rate. That is hard to defend for humans and their prey: prey adapt over
generations, while human hunters learn within a lifetime. It also makes hunters'
targets shift during training, as when hunted mammoths grew fatter (11 to 19 energy
in its round 14). Removing prey learning leaves less to compute, one policy to
train, and a world that holds still.

Fixed prey will not by themselves stop hunters overrunning them, and may make it
worse: in the earlier module lower catch rates went with trained rabbits.

## The world

As `stag_hunt_continuous`'s round 19c, by default:

- **Hunters** (10 at the start, speed 5) learn one shared policy, rewarded only for
  births. An action is [move, join].
- **Rabbits** (8, speed 7) graze grass, and one hunter catches one on contact.
- **Mammoths** (8, speed 4) graze steppe. A hunt succeeds only with at least 2
  contributors (`hunt_min_party`) whose summed energy exceeds 0.6 x the mammoth's
  (`hunt_energy_ratio`). Failure costs energy and starts a cooldown.
- **Prey rules** ([`rules.py`](rules.py)), chosen at each decision: flee the nearest
  hunter in the view cone within the flee distance (rabbits 120, mammoths 60), at
  full speed; otherwise go to the nearest food in view and stop to eat; otherwise
  wander at half speed.

Gone compared with the earlier module: burrows, the chance and condition hunt
rules, prey observations and prey policies. A prey's birth does not cut a decision
short and is not rewarded. Seasons remain available in `grass.py`, off by default.

## Plan

Agreed in advance, to avoid another long series of adjustments:

1. Check prey on their own ([`prey_check.py`](prey_check.py)), done, below.
2. One small calibration, with settings chosen before seeing results, minimum
   party 2: arm `base` (collapse delays 400) and arm `lean`
   (`grass_overgrazing_delay` and `steppe_overgrazing_delay` 800). The plan first
   named `*_respawn_delay`, but prey checks at 200 and 800 changed almost nothing:
   clusters get grazed out and a collapse overwrites that delay. Unhunted prey at
   the collapse delays, 12 episodes each: grass 200/400/800/1600 gives rabbits at
   the cap of 100/83/46/out in 7 of 12; steppe 400/800/1600/3200 gives mammoths
   20 (energy rising to 25)/15 (about 10)/7-9 (6-9)/out in all 12.
3. Stop calibrating, whatever the result, and write up. Safe zones or a larger
   world would be separate decisions, on the evidence.

Targets, as before: all three species in at least 45 of 50 evaluated episodes,
mammoths about half or more of hunters' food, and cooperation signals (joint
arrivals above chance, joining more with a partner than alone).

## Prey on their own

One harmless hunter (it stands still, catches and frightens nothing, and has no
young), default settings, 12 episodes (`results/prey_check_default.json`):

| From step | Rabbits | Mammoths | Mammoth energy | Edible grass | Edible steppe |
|---|---|---|---|---|---|
| 0 | 50 | 16 | 13 | 12 | 5.4 |
| 1000 | 81 | 20 | 15 | 4.7 | 4.0 |
| 3000 | 83 | 20 | 21 | 5.3 | 3.7 |
| 7000 | 83 | 20 | 25 | 4.7 | 3.3 |

No extinctions. Rabbits settle near 83, under their cap of 100, so grass limits them.
Mammoths stay at their cap of 20, where they cannot breed, and keep fattening, so
unhunted mammoths drift out of a party's reach: at ratio 0.6 a mammoth of 25 needs
a party above 15 energy, three typical hunters. Hunting should lower this by making
room for births, not measured yet.

Training, evaluation, accounts and the viewer have been run on a two-iteration
smoke run. All tests pass.

## Files

- [`environment.py`](environment.py): the world; only hunters are RLlib agents.
- [`rules.py`](rules.py): the prey's rules.
- [`prey_check.py`](prey_check.py): prey populations with no hunting.
- [`train.py`](train.py), [`launchers/calibration.py`](launchers/calibration.py),
  [`config/`](config/): training.
- [`evaluate.py`](evaluate.py): coexistence, hunt counters, joint arrivals against
  chance. [`payoffs.py`](payoffs.py): hunters' energy accounts.
- [`watch.py`](watch.py), [`render.py`](render.py): watch an episode or save a video.
- [`geometry.py`](geometry.py), [`grass.py`](grass.py), [`callbacks.py`](callbacks.py),
  [`checkpoints.py`](checkpoints.py): as in the earlier module, without its burrow
  and chance-rule parts where they were removed; `grass.py` still has its unused
  burrow code.

Run from the repository root, for example:

    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_rule_prey/launchers/calibration.py base 1000 14
    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.evaluate <checkpoint> 50 out.json --workers 16
    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.watch <checkpoint> --video out.mp4
