# stag_hunt_rule_prey: results

Calibration only, one seed per arm, no confirmatory test (see the [README](README.md)).

## 1. Prey on their own

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

## 2. Prey at other food levels

Same check, 12 episodes per level, not committed as result files. The ordinary
regrowth delays (`*_respawn_delay`, 200 and 800) changed almost nothing, because
clusters get grazed out and a collapse overwrites that delay. The delay after a
collapse (`*_overgrazing_delay`) is what matters:

| Grass collapse delay | Rabbits |
|---|---|
| 200 | 100 (at the cap) |
| 400 (default) | 83 |
| 800 | 46 |
| 1600 | dying out, gone in 7 of 12 |

| Steppe collapse delay | Mammoths | Mammoth energy |
|---|---|---|
| 400 (default) | 20 (at the cap) | rising to 25 |
| 800 | 15 | about 10 |
| 1600 | 7-9 | 6-9 |
| 3200 | gone in all 12 | |

## 3. Calibration (2026-11-05)

Arms `base` (collapse delays 400) and `lean` (grass and steppe 800), trained from
scratch, 1000 iterations, one seed each, 14 env runners, about 3 h 15 min side by side
(about 12-13 s per iteration). Settings were chosen before training. Evaluations are
50 episodes each (`results/calibration_<arm>[_iter<N>]_eval.json`).

**Result: the coexistence target failed in both arms, and the two others were met.** No
evaluated episode at any checkpoint kept all three species.

| Arm, iteration | All three species (of 50) | Outcome of the others | Joint arrivals vs chance | Partner vs alone | Mammoth share | Group kills | Time at hunter cap |
|---|---|---|---|---|---|---|---|
| `base` 250 | 0 | 36 no prey, 14 hunters out and rabbits out | 0.337 vs 0.178 (1.9x) | 0.74 / 0.57 | 88% | 1,324 | 50% |
| `base` 500 | 0 | 48 no prey | 0.417 vs 0.176 (2.4x) | 0.73 / 0.52 | 88% | 931 | 54% |
| `base` 1000 | 0 | 50 no prey | 0.398 vs 0.150 (2.7x) | 0.74 / 0.56 | 87% | 850 | 46% |
| `lean` 250 | 0 | 29 no prey, 15 hunters and rabbits out, 6 hunters out | 0.295 vs 0.153 (1.9x) | 0.75 / 0.60 | 88% | 791 | 34% |
| `lean` 500 | 0 | 45 no prey, 5 hunters out | 0.405 vs 0.144 (2.8x) | 0.69 / 0.53 | 87% | 640 | 48% |
| `lean` 1000 | 0 | 45 no prey, 5 hunters out | 0.442 vs 0.144 (3.1x) | 0.66 / 0.51 | 87% | 649 | 53% |

At the final checkpoints the prey are gone early: rabbits at a median step of 566
(`base`) and 670 (`lean`), mammoths at 653 and 396, in episodes that could last 8,000.
Where hunters die out instead (5 `lean` episodes), it is at a median step of 707.

For comparison, the same world with learning prey, `stag_hunt_continuous`'s round 19c,
ended at 39 of 50, joint 1.1x, mammoths 24%, 276 group kills.

**Energy accounts** (`results/payoffs_<arm>[_iter500].json`, final checkpoints), per
1000 hunter steps:

| Arm | Mammoth hunts, net (95% interval) | Rabbits | Scavenged | Metabolic cost |
|---|---|---|---|---|
| `base` | +18.8 (17.5 to 20.3) | +4.2 | +5.8 | 11.7 |
| `lean` | +16.8 (15.2 to 18.6) | +3.4 | +4.9 | 11.5 |

Per contributing hunter a hunt nets +10.8 to +11.2 in pairs, +10.6 to +10.8 in trios,
+7.7 to +8.5 in parties of four or more, and -0.66 alone. Solo attempts remain common
(4,397 and 3,116, against 1,069 and 864 party hunts). In round 19c the same net was
-0.39 against rabbits' +10.8.

**Reading (one seed per arm, no control).**

- Here mammoth hunting is hunters' main net food, the first time in either module that
  it clearly pays overall. These rates come from short worlds that are overrunning
  their prey, so they do not show what the hunting would yield in a lasting world, and
  they are not yet offspring by party-hunting rate.
- Compared with round 19c, the same settings with rule-following prey went with
  stronger group hunting and no coexistence. Prey learning may have protected the prey,
  but so might any stronger rule: the flee rule is one fixed choice, so this does not
  separate learning from the rule's design.
- Scarcer food in `lean` did not help coexistence, and mammoths went sooner. Fewer and
  thinner prey may be easier to overrun, though one seed cannot settle it.
- Joint arrivals are above chance but there is no solo control, so this is not yet
  evidence of learned coordination as distinct from hunters gathering at the same prey.

**Decision.** As agreed in advance, calibration of this module stops here whatever the
result. A larger world, refuges, or a brake on hunters would each be a separate
decision.

## 4. Larger worlds, replay without training (2026-11-05)

`base`'s final hunters, not retrained, in larger worlds at equal densities
([`launchers/large_world_replay.py`](launchers/large_world_replay.py),
`results/replay_base_area_x<N>.json`). Counts of animals and food, clusters and caps
scale with the area; bodies, speeds, view and hunt rules do not.

| Area | World | Episodes | All three species | Rabbits gone, median step | Mammoths gone, median step | Joint vs chance |
|---|---|---|---|---|---|---|
| x1 | 800 | 50 | 0 | 566 | 653 | 2.7x |
| x2 | 1,131 | 50 | 0 | 603 | 763 | 2.7x |
| x4 | 1,600 | 50 | 0 | 665 | 938 | 2.6x |
| x9 | 2,400 | not finished | | | | |
| x16 | 3,200 | not finished | | | | |

x1 reproduces the evaluation exactly. Prey last somewhat longer with area, in episodes
that could reach 8,000 steps. At x4, 3 episodes kept mammoths after rabbits and then
hunters died out. x9 and x16 hit their time limits twice (50 min with 50 episodes, then
25 min with 20 and 10), so they say nothing about survival, and are costly to play.
These hunters were trained in the small world, so this does not show how hunters
trained in a larger one would do.

## 5. Harder rabbits, replay without training (2026-11-05)

`base`'s final hunters, not retrained, 50 episodes, against rabbits with view 340 degrees
(was 240) and flee distance 200 (was 120), both changed together
([`launchers/rabbit_replay.py`](launchers/rabbit_replay.py),
`results/replay_base_rabbit_fov340_flee200.json`). Prompted by a measurement over 20
episodes of 858 catches: median rabbit speed at the catch 3.7 of a top speed of 7, 72%
below the hunters' top speed, and 19% with the catcher in the rabbit's blind area.

| | Old rabbits | New rabbits |
|---|---|---|
| All three species | 0 | 0 |
| Hunters out, both prey left | 0 | 33 (median step 563) |
| Both prey gone, hunters left | 50 | 17 |
| Rabbits caught | 2,369 | 706 |
| Hunter births | 2,441 | 875 |
| Rabbit births | 1,995 | 1,497 |
| Mammoth share of food | 87% | 88% |
| Joint arrivals vs chance | 2.7x | 2.7x |

Rabbits are caught far less, and worlds now often lose the hunters instead. These hunters
never met such rabbits, so this does not show what retrained hunters would do. Episodes
also ended sooner, so births are not a like-for-like comparison.
