# Hunting Together: A Stag Hunt with Deer

## The idea

In the classic stag hunt, a hunter alone can always catch a hare, a small but safe
meal. A stag is a much bigger meal, but one hunter cannot take it: it needs partners.
Hunting together pays more, but only if a partner shows up. The game is a standard
model of cooperation between humans.

This module puts that choice into a simulated ecosystem. The hunters stand for
humans. The deer are the prey, and how a deer can be caught depends on how big it is:
a **fawn** is the hare, which one hunter can catch on its own, and an **adult** deer is
the stag, which only two hunters together can catch. Nobody tells the hunters to
cooperate. The question is whether they learn to.

## The world

Hunters and deer live on an 800 by 800 continuous torus (no edges) with clustered
grass. Deer eat grass, hunters eat deer, and both give birth when they have enough
energy. Defaults in [environment.py](environment.py) (`DEFAULT_CONFIG`):

| | Hunters | Deer |
| --- | --- | --- |
| Start | 6, energy 5 | 8, energy 2 |
| Food | deer | grass (+2 per patch) |
| Birth | at energy 12, pays 5 | at energy 7, pays 3 |
| Top speed | 5 | 6 |
| Sees | 4 hunters and 2 deer (with their energy) in a 150° cone of 200 | 3 hunters and 3 deer in a 240° cone of 267 |

Catching, by the deer's energy (a deer grows by grazing and gives birth at 7, which
drops it to 4):

| Deer | Energy | Hunter alone (no other hunter within 64 units) | Another hunter within 64 units of the deer |
| --- | --- | --- | --- |
| Fawn (the hare) | 2 or less | always caught; gains up to 1 | always caught, but shared by every hunter within 64 (the others need not touch it) |
| Middle | 2 to 5 | caught with chance (5 − energy) / 3, falling from 1 to 0; gains 1 to 2.5 | always caught, shared as for a fawn |
| Adult (the stag) | 5 or more | never caught | caught only if two or more hunters touch it within one decision (8 physics steps); shared by those who touched it only; 1.25 to 1.75 each when two share it |

A hunter gains half its share of the deer's energy. A failed attempt costs each hunter
that touched the deer a wait of 8 physics steps before it can catch again.

The rule for young deer is a sharing rule, kept from the world this module copies
(see Origin): a hunter nearby makes the catch certain but takes a share without
touching the deer. With one partner near, the expected gain per hunter is:

| Deer | Alone | With one partner near (each) | Better |
| --- | --- | --- | --- |
| Fawn, energy 2 | 1.0 | 0.5 | alone |
| Middle, energy 3 | 1.0 | 0.75 | alone |
| Middle, energy 3.5 | 0.875 | 0.875 | equal |
| Middle, energy 4.5 | 0.375 | 1.125 | partner |
| Adult, energy 5 to 7 | 0 | 1.25 to 1.75 | partner (required) |

So the value of a partner grows with the deer's size: it costs half a fawn, pays for
middle deer above energy 3.5, and is required for an adult.

Episodes last 8000 physics steps (about 1,400 decisions, since a birth ends a
decision early).

**Grass regrowth (`grass_regrowth`).** Sections 1 and 2 of RESULTS.md ran with
`"instant"` grass: an eaten patch is gone for 400 physics steps, then back whole.
Since 2026-11-09 the default is `"logistic"`: each patch's biomass regrows
gradually, slowly when grazed short, fastest at half grown, slowly again near full
(bare to 95% in about 400 physics steps). A deer eats all of a patch and gains that
share of its energy (2 for a full patch), and deer see how grown the nearest patch is.
Grass kept short barely regrows, so a deer sitting on a bare patch gains less than it
spends at rest. Runs saved before the setting existed load with `"instant"` and
reproduce exactly ([grass.py](grass.py), [tests/test_grass.py](tests/test_grass.py)).
No training run has used logistic grass yet.

**A second catching rule (`deer_catch_rule: "touch"`, tested in RESULTS.md
section 2).** Only touching counts, with one rule for every deer: each touch is a
solo attempt with the chance above (0 for an adult), a failed attempt costs a wait,
and two or more hunters touching a deer within 8 physics steps catch it and share it.
Hunters merely near a deer get nothing. The default stays `"party"`, the world above,
so section 1's runs reproduce exactly.

## How the animals learn

Hunters and deer start with no knowledge. Each species has one neural network
policy, shared by all its members, trained with PPO (reinforcement learning). The
only reward is a birth: an animal gets a reward each time it gives birth, and nothing
for eating, catching or surviving as such. To breed, a hunter must eat enough, so
training pushes the hunters' policy toward hunting that feeds them well. Training
runs for 1000 iterations, each a batch of experience from 14 worlds run in parallel.

## The experiment

An earlier module found both coexistence and cooperation in this world once, in a
single training run, and there the hunters could also make calls to each other. This experiment asks: **does the result
hold without calls, and in more than one run?** Three runs were trained from scratch
(`run_1`, `run_2`, `run_3`), differing only by chance. Each final policy was then
evaluated:

- **Coexistence:** in how many of 50 test episodes are hunters and deer both still
  alive at the end? Hunters that eat all the deer, or starve, fail.
- **Cooperation against chance:** when a hunter goes for a deer, is another hunter
  within 64 units? If hunters seek partners for adults, this should happen more often
  for adults than for fawns, and more often than chance. Chance is the share of all
  hunter time spent with another hunter that close, so hunters that simply stay
  together do not count as cooperating.

The tests, set before training (RESULTS.md, section 1): at least 45 of 50 episodes
coexisting; adults approached with a partner near more often than fawns and than
chance; and that ratio above the value of a weaker earlier world. The result counts
as replicated if all three tests are met in at least 2 of the 3 runs.

## Outcome

**Replicated.** In the 3 runs, hunters and deer coexisted in 43, 49 and 50 of 50
episodes. Adults were approached with a partner near 1.20, 1.23 and 1.15 times as often
as chance, and more often than fawns, with 95% intervals above zero in every run.
run_1 missed the coexistence bar; the other two met all tests. So calls were not
needed for this result.

Limits: the effect is small (adults are 1 to 4% of the deer caught), and adult catches
fell late in training in run_1 and run_3. The rule makes a partner necessary for an
adult, so this shows coordination where the task requires two, not proof that hunters
seek each other out. And a partner near a fawn takes half of it, so hunters have a
reason to keep away from others near fawns. That lowers the fawn share and makes "adults
above fawns" easier to pass; the comparison with chance is not affected this way, and
adults were above chance in all three runs, but part of the gap between adults and
fawns may come from avoiding sharing rather than seeking partners. Details in [RESULTS.md](RESULTS.md); written up for a general
reader, with an animation and a results chart, on the site page
[A Replicated Stag Hunt](https://humanbehaviorpatterns.org/learning-selection-interaction/predpreygrass-continuous/deer-stag-hunt).

**With only touching counting (RESULTS.md section 2): replicated, in all three runs.**
Under the touch rule (`touch_1` to `touch_3`), hunters and deer coexisted in 50 of 50
episodes in every run, and adults were approached with a partner near 1.12, 1.21 and
1.15 times as often as chance, intervals above zero. So the party rule for young deer
was not needed for this result. The same limits apply: the effect is small, and adult
catches again fell late in training (by 25 to 44%).

**Partnering where it pays but is not required (section 2.1): pattern shown, in 2 of 3
runs.** Under the touch rule a middle deer can be caught alone by chance or by two
hunters for certain, and a partner pays more for the larger ones. Hunters approached
the largest middle deer (energy 4.5 to 5) with a partner near slightly above chance
(1.09 to 1.15 times) and teamed up on them nearly as often as on adults; under the
party rule they did neither. This is consistent with the stag hunt in its strict sense,
pairing without being forced, but with hedges: the tests were set after a look at
interim checkpoints of two of the runs (the third, not looked at, met them), the margins
are 1 to 2 percentage points, and big middle deer may share places with adults or be
treated like them; a control world in which adults can be caught alone was not run.

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

## Running

From the repository root:

```bash
# A training run from scratch (defaults: 1000 iterations, 14 env runners)
python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/replicate.py NAME
# ... with the touch rule (RESULTS.md section 2); all three runs and evaluations: launchers/run_touch.sh
python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/replicate.py NAME --catch-rule touch

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

In the viewer, deer show their stage: fawns are small with white spots, middle deer
medium-sized and plain, adults full size with antlers. (Until 2026-11-08 adults were
marked by a ring instead; `render.Viewer(..., deer_style="ring")` still draws that.)

TensorBoard's `env_runners/catches/` holds the per-episode counters, among them
`adult_energy_share` (the share of the hunters' energy from deer that came from
adults) and `adults_caught`.

## Files

- `environment.py`: the world (`DeerStagHunt`); `grass.py`, `geometry.py`: grass
  clusters (instant or logistic regrowth) and torus geometry.
- `train.py`, `config/`: training; `callbacks.py`: TensorBoard counters.
- `evaluate.py`, `launchers/coop_fair.py`: evaluation and the fair cooperation
  measure (with a bootstrap interval over episodes); `launchers/replicate.py`: training
  runs from scratch; `launchers/verdict.py`: RESULTS.md section 1's tests and verdict.
- `launchers/coop_by_energy.py`, `launchers/energy_verdict.py`, `launchers/run_energy.sh`:
  partnering by the deer's energy, RESULTS.md section 2.1.
- `checkpoints.py`, `render.py`, `watch.py`: loading checkpoints, viewing episodes.
- `tests/`: the rules, evaluation, and equivalence with `stag_hunt_continuous`
  (that test needs a PredPreyGrass-archive checkout next to this repository, or
  `PREDPREYGRASS_ARCHIVE` set to one; it is skipped otherwise).
- `RESULTS.md`: experiments and results.
