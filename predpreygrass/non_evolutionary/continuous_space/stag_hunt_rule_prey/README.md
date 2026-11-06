# stag_hunt_rule_prey

**Status (2026-11-06): closed, together with the whole stag hunt line.** See
[RESULTS.md](RESULTS.md), section 15. Only hunters learn; rabbits and mammoths follow
fixed rules. A simplification of
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

1. Check prey on their own ([`prey_check.py`](prey_check.py)), done, see
   [RESULTS.md](RESULTS.md).
2. One small calibration, with settings chosen before seeing results, minimum
   party 2: arm `base` (collapse delays 400) and arm `lean`
   (`grass_overgrazing_delay` and `steppe_overgrazing_delay` 800). See
   [RESULTS.md](RESULTS.md).
3. Stop calibrating, whatever the result, and write up. Safe zones or a larger
   world would be separate decisions, on the evidence.

Targets, as before: all three species in at least 45 of 50 evaluated episodes,
mammoths about half or more of hunters' food, and cooperation signals (joint
arrivals above chance, joining more with a partner than alone).

## Results in short

One calibration, two arms, then one later run with harder rabbits, one seed each. Hunters
learned strong group hunting of mammoths, which became their main net food, but no
evaluated episode kept all three species. In `base` and `lean` hunters overran the prey
within about 700 steps, and scarcer food did not help. A larger world, replayed without
training, delayed this only slightly. With harder rabbits most worlds lost the hunters
instead, and rabbits stopped being a real food source. Paying hunters for time alive
instead of births lost the group hunting without bringing restraint. Details are in
[RESULTS.md](RESULTS.md).

| Arm, final | All three species (of 50) | Joint arrivals vs chance | Mammoth share of food | Mammoth hunts vs rabbits, net energy per 1000 steps |
|---|---|---|---|---|
| `base` | 0 | 2.7x | 87% | +18.8 vs +4.2 |
| `lean` | 0 | 3.1x | 87% | +16.8 vs +3.4 |
| `harder_rabbits` | 0 | 3.6x | 92% | +7.6 vs +0.9 |
| `alive_individual` | 0 | 0.6x | 2% | -0.2 vs +5.8 |
| `alive_shared` | 0 | 0.7x | 3% | -0.2 vs +5.5 |

After closing, three rabbits-only arms (two species, no mammoths). With plain rabbits
(`rabbits_only`) hunters overran them. With lean and fat rabbits (`lean_fat_rabbits`: a
lone hunter cannot catch a fat rabbit, two can) hunters and rabbits coexisted in 46 of 50
episodes at 1000 iterations, meeting the target, but hunters did not approach fat rabbits
with a partner more often than chance. Continued to 2000 iterations
(`lean_fat_rabbits_more`), coexistence fell to 29 of 50 and cooperation stayed the same.
Then rabbits with heritable flee distance and breeding threshold
(`evolving_rabbits`, with a neutral-gene reference): all three tests failed. Coexistence
18 of 50, but now with hunters starving in 22, mostly against lineages that stayed fat;
gene differences from drift not significant; no cooperation. See [RESULTS.md](RESULTS.md),
sections 10 to 14, and section 15 for the closing summary: group hunting was easy to learn,
restraint was never learned, and leaving prey for later may be a commons dilemma of its own.

## Files

- [`RESULTS.md`](RESULTS.md): prey checks and calibration results.
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
