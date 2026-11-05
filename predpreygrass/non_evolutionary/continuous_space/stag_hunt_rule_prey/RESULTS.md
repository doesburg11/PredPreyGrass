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

## 6. One run with harder rabbits (2026-11-05)

Settings fixed before training, after section 5: `base` with `rabbit_fov` 340 and
`rabbit_flee_distance` 200, 800 x 800, from scratch, 1000 iterations, one seed. The
target is unchanged: at least 45 of 50 evaluated episodes with all three species, mammoths
about half of hunters' food or more, and joint arrivals above chance. One run, with no
further rabbit changes after it, whatever the result.

**Result: coexistence target not met, 0 of 50 at every checkpoint. As agreed, no further
rabbit changes.** Whole-episode evaluation, 50 episodes per checkpoint
(`results/calibration_harder_rabbits*_eval.json`, `results/payoffs_harder_rabbits*.json`):

| | Iteration 250 | Iteration 500 | Iteration 1000 | `base`, 1000 |
|---|---|---|---|---|
| All three species | 0 | 0 | 0 | 0 |
| Hunters out, both prey left | 25 | 28 | 29 | 0 |
| Hunters and mammoths out, rabbits left | 25 | 12 | 6 | 0 |
| Both prey out, hunters left | 0 | 10 | 15 | 50 |
| Joint arrivals vs chance | 2.7x | 3.3x | 3.6x | 2.7x |
| Joining, with a partner / alone | .70 / .62 | .68 / .60 | .67 / .60 | .74 / .56 |
| Mammoth share of food | 95% | 93% | 92% | 87% |
| Group kills | 382 | 319 | 296 | 850 |
| Rabbits caught | 460 | 638 | 595 | 2,369 |

Episodes end when no hunters are left or no prey are left, so the counts of kills and
catches come from episodes of different lengths and are not like-for-like.

Final hunters fall into three kinds of episode, all over by step 2,560 of 8,000:

- **29 episodes, early loss:** hunters die out at steps 224 to 457, both prey left.
- **15 episodes, prey overrun:** mammoths go first (steps 140 to 435), then rabbits (748
  to 1,397), hunters left.
- **6 episodes, slow loss:** mammoths go early and hunters die out at steps 1,965 to
  2,560, rabbits left.

Hunters' net energy per 1000 steps, final: mammoth hunts +7.6 (6.6 to 8.4), rabbits +0.9,
scavenged +2.2, metabolic cost 10.4. In `base`: +18.8, +4.2, +5.8, 11.7. Rabbits now give
under a tenth of the metabolic cost, so in this run they do not act as the food a hunter
can live on alone, which point 2 of the stag hunt definition asks for. Income and cost
are close to balance, which may be why so many worlds lose their hunters, though this was
not tested.

Group hunting still pays those who manage it. Party of 2 succeeds 52% of the time, party
of 3 84%, party of 4 97%, against none alone. Net energy per 1000 steps while in a party
attempt is +104, in a solo attempt -26, elsewhere -8.9. Solo attempts remain 81% of all
attempts (1,995 of 2,476), each losing 0.66, and joining at a contact is not tied to
having enough energy: .81 when the party has enough, .87 when it does not. Joint arrivals
are the highest of the module, but the partner effect is smaller than in `base`.

Reading, hedged, one seed: harder rabbits moved the failure rather than removed it. From
`base`, where hunters always overran the prey, most worlds now lose the hunters, and some
still lose the prey, so the world seems to lie between the two failures with no stable
state found here. These results do not separate the rabbit changes from training
differences between runs, and no solo control was run, so they are not evidence that
cooperation emerged or caused anything.

## 7. Is there a sustainable hunting pressure? Replay at lower hunter caps (2026-11-05)

A check before any restraint training: `base`'s final hunters, not retrained, 50 episodes,
with `reproduction_max_predator` lowered from 50
([`launchers/rabbit_replay.py`](launchers/rabbit_replay.py),
`results/replay_base_cap*.json`). The caps are a diagnostic, not a proposed design.

| Hunter cap | All three species | Hunters and mammoths left | Mammoth share | Joint vs chance | Partner / alone | Episode length, mean | Hunter births per episode | Hunter-steps per episode |
|---|---|---|---|---|---|---|---|---|
| 10 | 32 | 11 | 59% | 3.4x | .56 / .47 | 7,658 | 18 | 75,843 |
| 15 | 1 | 49 | 91% | 3.8x | .63 / .52 | 8,000 | 37 | 117,627 |
| 20 | 0 | 34 | 94% | 4.1x | .65 / .53 | 6,906 | 44 | 133,791 |
| 30 | 0 | 0 | 89% | 3.2x | .70 / .57 | 1,789 | 38 | 45,568 |
| 50 (`base`) | 0 | 0 | 87% | 2.7x | .74 / .56 | 845 | 49 | 25,500 |

The remaining cap 10 episodes kept mammoths only (7); cap 20 also had 15 episodes with
hunters only and 1 with mammoths only. Hunters were at the cap about 94 to 98% of the time
at caps 10 to 20.

What this shows, for these hunters, who were not trained at these caps:

- **A sustainable pressure seems to exist, but low.** Cap 10, the starting hunter number,
  gives 32 of 50, short of the 45 target, but with mammoths at 59% and joint arrivals well
  above chance. Rabbits go first: at cap 15 they are out in 49 of 50 episodes (median step
  1,642), while mammoths last in all 50.
- **Births favour the overrun.** The overrun gives the most hunter births per episode even
  counted over whole 8,000-step episodes, and in far less time, so a births reward may
  prefer it even if shared among all hunters and with no discounting. This changes the
  plan, which would have kept births as the reward. At the observed rates, cap 20 passes
  49 births after about 7,700 steps, cap 10 after about 20,000. The caps themselves hold
  births down, so a world restrained some other way could do better than this.
- **Hunter lifetime favours restraint, but not all three species.** Hunter-steps peak at
  caps 15 to 20, where rabbits are nearly always lost, so a reward for time alive might
  still give up the rabbits.

## 8. Restraint: paid for time alive (2026-11-05)

Settings and stop rule fixed before training, after section 7, which suggests births pay
most for the overrun. Two arms on `base` rabbits, 800 x 800, from scratch, 1000 iterations,
one seed each, 14 env runners each, run side by side:

- **`alive_individual`:** 0.002 per physics step to each hunter that lives through it, no
  birth reward, training on whole episodes (`truncate_on_prey_extinction` off), gamma 0.999
  per decision.
- **`alive_shared`:** the same, with the pay multiplied by surviving hunters / 10, so a
  hunter is also paid for the others being alive. Splitting a time-alive total among the
  living would pay each hunter the same as the individual arm, so it is not split.

The target is unchanged: at least 45 of 50 evaluated episodes with all three species,
mammoths about half of hunters' food or more, and joint arrivals above chance.

Stop rule:

- `alive_individual` meets the target: no more tuning; next are a solo control and more seeds.
- Only `alive_shared` meets it: a dilemma between individual and group is suggested, and
  how to solve it is a separate decision.
- Neither meets it: restraint is not shown to be learnable in this setup, the module closes,
  and there are no further variations.

Expected risks, stated in advance: hunter-steps in section 7 peak where rabbits are lost,
so both arms may settle on hunters and mammoths only. Hunters see only nearby animals, so
they may be unable to sense depletion. `gae_lambda` 0.95 stays as it was, so long-range
credit rests on the critic.

**Result: neither arm met the target. Under the stop rule the module closes.** Whole
episodes, 50 per checkpoint (`results/calibration_alive_*_eval.json`,
`results/payoffs_alive_*.json`):

| | Individual 250 / 500 / 1000 | Shared 250 / 500 / 1000 |
|---|---|---|
| All three species | 7 / 0 / 0 | 7 / 0 / 0 |
| Only mammoths left | 40 / 49 / 50 | 40 / 49 / 50 |
| Rabbits gone, median step | 2,527 / 1,424 / 1,518 | 2,675 / 1,605 / 1,225 |
| Hunters gone, median step | 4,249 / 3,330 / 3,695 | 4,210 / 3,672 / 3,268 |
| Mammoth share of food | 1 / 3 / 2% | 2 / 3 / 3% |
| Group hunts (solo) | 9 (2,291) / 10 (1,429) / 6 (1,039) | 15 (2,615) / 13 (1,445) / 8 (1,014) |
| Joint arrivals vs chance | 0.59 / 0.64 / 0.62x | 0.57 / 0.57 / 0.67x |
| Hunter-steps per episode | 65,036 / 40,510 / 43,215 | 67,549 / 43,075 / 34,453 |

Hunters gave up mammoth hunting and joining almost entirely (joining .05 to .09 at
contacts), lived on rabbits until the rabbits were gone, then starved. Accounts, final,
per 1000 hunter steps: rabbits +5.8 and +5.5, hunts -0.2, metabolic cost 6.8 and 6.7.
The shared arm looks no different from the individual one.

Hunter-steps per episode, the quantity the reward pays for, fell from iteration 250 to
1000 in both arms, so training moved away from what the reward pays for in whole
episodes. Why was not tested. One possibility: a failed solo hunt costs energy and
survival is paid at once, so hunters learn early not to join, before pairs could find out
that mammoth hunts pay. Another: hunters see only nearby animals and cannot sense that
rabbits are running out.

## 9. Where it ended (2026-11-05)

The module is closed, together with
[`stag_hunt_continuous`](../stag_hunt_continuous/README.md); the stag hunt line is
paused. One seed per run, no solo control, so none of this is a confirmatory result.

- **Cooperation in the hunt was easy, restraint was hard.** With rule-following prey and a
  births reward, hunters learned strong mammoth hunting in groups (joint arrivals 2.7 to
  3.6x chance, mammoths 87 to 92% of food), far above the learning-prey module. In every
  such world they overran the prey: 0 of 50 coexisting in all five trained arms.
- **Changing the prey moved the failure, not removed it.** Scarcer food, a larger world
  (replay) and harder rabbits each led to hunters overrunning the prey or starving.
- **A sustainable pressure seems to exist, but births do not pay for it.** At a cap of 10
  hunters, 32 of 50 kept all three species with cooperation signals intact (replay), but
  the overrun gave the most hunter births.
- **Paying for time alive lost cooperation without bringing restraint.**
- **Across both modules**, about 25 runs lay along one trade-off: strong cooperation where
  hunters could overrun the prey, coexistence where they had little need to cooperate. One
  reading, not tested: as in the classic stag hunt, hunters settle on the safe solo prey
  while it is plentiful and turned to mammoths only under a scarcity this world could not
  sustain.

Ideas on record, not tried: rabbits that keep hunters alive but give too little energy for
births, so only mammoth hunts lead to offspring (point 2 of the stag hunt directly in the
rules); a trained run at a hunter cap of 10 to 12, read as cooperation in a world where
something else limits hunter numbers, not as emerged restraint; refuges or territories.


## 10. Two species only: hunters and rule rabbits (2026-11-05)

One run added after the module closed, at the user's request, to ask whether the third
species was the obstacle to coexistence. With one predator and two prey, hunters can live
on one prey while eating out the other (apparent competition), so the scarcity of either
prey no longer brakes them. Settings fixed before training: `base` with no mammoths and
no steppe (arm `rabbits_only`), births reward, 800 x 800, from scratch, 1000 iterations,
one seed.

How it differs from earlier runs: `stag_hunt_continuous` round 1a had the same two
species with learning rabbits (49 of 50 coexisting); `base` had mammoths; the time-alive
arms had mammoths present, though ignored, and another reward.

Stop rule: coexistence (hunters and rabbits both left at step 8000; evaluate.py labels
this `only_predator_rabbit`) in at least 45 of 50 evaluated episodes counts as success.
Success points to the second prey as the obstacle; failure points away from species
count, towards rule prey that never adapt and gains turned into more hunters. One run,
no variations after it, whatever the result.

**Result: stopped early by the user at iteration 290, on clear evidence of an
overrun, so the stop rule's 50-episode evaluation was not run.** Training episodes (which
end when rabbits or hunters die out) grew to about 3,450 physics steps by iteration 60,
then fell to about 1,300 by iteration 185, while rabbits caught per episode rose from 40 to
75 and the mean number of rabbits fell from about 60 to 39. Twelve whole episodes from the
iteration 140 checkpoint (`results/rabbits_only_iter140_peek_eval.json`): rabbits eaten
out in all 12, median step about 480, hunters left in every one.

Reading, hedged, one seed: with two species the rabbits are overrun as well, so the third
species does not seem to be the main obstacle. Two explanations remain and this run does
not separate them: rule rabbits that never adapt, which hunters can keep exploiting, and
gains in hunting turned into more hunters. `stag_hunt_continuous` round 1a, with learning
rabbits, kept both species in 49 of 50, but in a world that differed in more than rabbit
learning. A run in this world with learning rabbits would separate the two; not started.


## 11. Lean and fat rabbits: replay without training (2026-11-05)

The user's idea: a stag hunt with rabbits only, where lean rabbits are the hare and fat
rabbits the stag. Rule (`rabbit_fat_energy` > 0, off by default): a hunter touching a
rabbit with no other hunter within 64 units catches it with probability 1 at energy 3 or
less, falling linearly to 0 at energy 6 (4: 67%, 5: 33%); with at least one more hunter
within 64 units, always. The catch is shared equally among the hunters within 64 units. A
failed attempt lets the rabbit go on and blocks the hunter's catching for 8 physics steps.

Cut-offs chosen from rabbits with a harmless hunter (6 episodes): energy 3 or less is
31% of rabbits at fewer than 10 rabbits and 46% at 80 or more; energy 6 or more is 25%
and 15%. Energy depends on rabbit numbers only weakly, because each rabbit's own breeding
cycle (birth at 8, back to 5) dominates. Fat rabbits are mostly rabbits about to breed.

Check, with the stop rule fixed in advance (training only if hunters and rabbits both
survive in at least 25 of 50): the `rabbits_only` hunters from iteration 290 (section 10),
not retrained, 50 whole episodes each (`results/replay_rabbits_only_iter290_*.json`).

| | Without the rule | With the rule |
|---|---|---|
| Hunters and rabbits both left at step 8000 | 0 | **38** |
| Rabbits gone | 50 (median step 510) | 12 (median step 4,176) |
| Rabbits caught | 2,228 | 23,050 |
| Of which with another hunter within 64 units | - | 10,211 (44%) |
| Failed solo attempts | - | 23,922 |
| Hunter births | 186 | 1,031 |
| Hunters, mean | 11.4 | 9.7 |

The check passes. These hunters were trained without the rule, so the 44% catches with
another hunter near come from hunters close together, not from learned cooperation; any
cooperation claim needs a comparison with chance.

## 12. Lean and fat rabbits, trained (2026-11-05)

Arm `lean_fat_rabbits`: `rabbits_only` with the rule of section 11, births reward, 800 x
800, from scratch, 1000 iterations, one seed. Fixed before training:

- **Coexistence:** hunters and rabbits both left at step 8000 in at least 45 of 50
  evaluated episodes.
- **Cooperation:** catch attempts made with another hunter within 64 units more often than
  hunters spend time with another hunter within 64 units (chance), measured afterwards on
  the evaluated hunters.

One run, no variations after it, whatever the result. Risk stated in advance: hunters
that learn to move in pairs make every fat rabbit catchable, which may remove the refuge.

**Result: coexistence met (46 of 50), cooperation against chance not met.** Whole
episodes, 50 per checkpoint (`results/calibration_lean_fat_rabbits*_eval.json`):

| | Iteration 250 | Iteration 500 | Iteration 1000 |
|---|---|---|---|
| Hunters and rabbits both left at step 8000 | 39 | **50** | **46** |
| Hunters gone, rabbits left | 11 (median step 3,447) | 0 | 0 |
| Rabbits gone, hunters left | 0 | 0 | 4 (median step 5,454) |
| Hunters, mean | 7.0 | 10.5 | 11.6 |
| Rabbits, mean | 40.4 | 19.4 | 14.0 |
| Hunter births | 790 | 1,255 | 1,217 |
| Catches alone / with another hunter near | 12,070 / 5,384 | 12,889 / 11,736 | 11,555 / 12,586 |
| Failed solo attempts | 16,064 | 21,770 | 21,665 |

This is the first world in either stag hunt module where trained hunters and prey kept
coexisting at the target level. Mean rabbit numbers fell from 40 to 14 over training,
so the margin may be narrowing; the run is being continued (section 13).

Cooperation, as fixed before training ([`launchers/coop_check.py`](launchers/coop_check.py),
20 episodes each): catch attempts made with another hunter within 64 units, against the
share of hunter time spent with another hunter within 64 units.

| | Iteration 250 | Iteration 1000 |
|---|---|---|
| Attempts with a partner near | 16.1% | 26.9% |
| Hunter time with a partner near (chance) | 24.2% | 33.4% |
| Ratio | 0.67x | 0.81x |
| Episodes above chance | 0 of 20 | 0 of 20 |

Attempts with a partner near are below chance, so the target is not met. Both measures
are imperfect, which was not foreseen when the target was set: a lone hunter at a fat
rabbit fails and retries after each cooldown, so solo attempts are counted many times and
the attempt share is pushed down. Counting catches instead (52% with a partner near at
iteration 1000, 1.6x the time share) pushes the other way, since a party always succeeds
and a lone hunter often does not. A fair measure would count each hunter's approach to a
rabbit once. Either way, the evidence for learned cooperation is weak: the rise in
catches with a partner near (5,384 to 12,586) goes with more hunters close together
(time share 24% to 33%).

## 13. Lean and fat rabbits, continued to 2000 iterations (2026-11-06, running)

At the user's request, after the training curve was still rising at iteration 1000:
`lean_fat_rabbits_more`, 1000 more iterations from the final weights of section 12 (the
optimizer state starts fresh), same world and rule, evaluated at 1250, 1500 and 2000
iterations overall with the same targets. Section 12's iteration 1000 stays the result
fixed in advance.

