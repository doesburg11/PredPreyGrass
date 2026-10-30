# stag_hunt_continuous: design

Status: **draft 2, 2026-10-30; not built.** Draft 1 was revised after a
Codex review (decisive test, timing, mammoth economics, grass, action
space, party geometry, control arm, facts). Decisions marked *(open)*
are for the user before building; values marked *(calibrate)* are set in
step 1 of the plan and frozen in the pre-registration.

## 1. Purpose

The project's ultimate aim is to let human behaviour, and above all human
cooperation, emerge; the predators are the humans. This module asks
whether hunters in the continuous world **learn to hunt together**: to
bring about joint hunts of prey that one hunter cannot kill, more than
chance meetings would, and more than the capture rule alone explains.

Its starting point is the discrete
[`stag_hunt_forward_view`](../../discrete_space/project_cooperation/stag_hunt_forward_view/README.md),
which keeps three elements that resemble human hunting:

1. **A choice between big, risky prey and small, reliable prey** (the stag
   hunt): mammoths need several hunters, rabbits can be caught alone.
2. **A temptation to free-ride**: joining costs; hunters nearby who do not
   join still get a scavenger share.
3. **Forward-facing predator eyes**, as in the continuous field-of-view
   work (eco_evolutionary_continuous_fov).

What that module shows, read conservatively: in one evaluation per
scavenger fraction (five fractions, one training each, 30 evaluation runs
each), hunters joined more often with a mammoth adjacent than with only a
rabbit adjacent (0.848 against 0.742 in one evaluation), and hunters
labelled by their own event histories as group hunters reproduced more
than solo hunters and free-riders. Those labels are descriptive and
post-treatment (hunters that live longer meet more mammoths), so they do
not show that group hunting caused the higher reproduction. The checked
implementation charges its join cost to every joiner, solo or not, on
success and failure.

## 2. Lessons that shape the design

- `predator_bands` (mammoth tests, one seed each): success and death by
  party size 1 / 2 / 3 / 4+ of 0.02 / 0.25 / 0.6 / 0.9 and 0.30 / 0.15 /
  0.05 / 0.02 did not produce group hunts in 150 iterations (solo
  attempts frequent, deadly, nearly useless). With solo attempts harmless
  and useless, pair attempts and kills occurred with and without a band
  compass, but only grew through training with it.
- foraging_swarms (continuous, closed 2026-10-22, archived): five attempts
  at predator cooperation found none learned. In the large-prey test
  (prey with energy at least 6 needed a helper within 128), training from
  scratch gave no predator births, so policies were warm-started;
  predators could not see which prey were large; and the helper benefits
  measured were those of the rule: untrained policies showed them too.
  **A helper-removal effect measures the rule, not learning.**

Requirements that follow:

1. **Viable from scratch**: rabbits alone sustain the hunters; mammoths
   are an extra, richer food source, without taking the rabbits' food.
2. **Visible prey type**: mammoths and rabbits in separate observation
   slots.
3. **Cooperation as a measurable choice**: a join decision in its own
   action head.
4. **Learned against imposed**: the decisive measure is whether trained
   policies *create* joint-hunt states more often than untrained policies
   and a control trained without the need to cooperate, per encounter and
   against a density-matched chance level. Helper removal is only a
   manipulation check.

## 3. World

A copy of [`base_environment_continuous`](../base_environment_continuous/README.md)
with every quirk fixed (`environment.FIXES`, the BE-04/BE-05 world):
800 × 800 torus, 16 directions × 3 speeds, 8 physics steps per decision,
energy, starvation, births as the only reward (+10), clustered moving
grass. Changes follow.

### 3.1 Two prey species, two food layers

| | Rabbit (the base prey) | Mammoth (new) |
| --- | --- | --- |
| Policy | `rabbit` | `mammoth` |
| Radius | 16 | 32 |
| Top speed | 6 (faster than hunters, 5) | 4 *(open: 4 or 5)* |
| Mass, max acceleration | 3, 1 | 30, 0.3 |
| Energy: initial, cap, birth threshold | 3, 16, 8 (base) | 10, 40, 24 *(calibrate)* |
| Food | the base grass | its own **steppe** layer (a second grass layer, fewer, larger clusters) |
| Caught | by one hunter on contact (base rule) | only by a hunt (3.2) |
| Initial count, cap | 8, 100 (base) | 4, 20 *(calibrate)* |

Mammoths graze a separate layer so they do not take the rabbits' grass:
the rabbits are the hunters' safety net while they learn, and a second
food layer keeps that net independent of mammoth numbers (different
diets, as for grazers and browsers).

### 3.2 Mammoth hunts

Resolved **once per decision**, at the decision boundary (after the
physics steps of the decision, or after an early end by a birth), from
the positions then:

- **Contributors**: hunters whose join head is on, that touch the mammoth
  (centre distance below the sum of radii, 48) and are not in a hunt
  cooldown. **Bystanders**: other hunters within a centre distance of
  48 + 48 = 96 (including touching hunters with join off).
- A hunt happens if a mammoth has at least one contributor. A hunter
  touching several mammoths contributes to the nearest (ties by agent
  ID); each mammoth is hunted at most once per decision.
- **Success** with probability `success_by_party[min(n, 4)]` for n
  contributors; initially 0.0 / 0.4 / 0.75 / 0.9 for 1 / 2 / 3 / 4+
  *(calibrate)*, i.e. predator_bands' no-solo curve with a pair chance
  60% higher (0.25 to 0.4), because there pair hunts did not grow without
  a compass.
- **On success**: the mammoth is removed; a carcass of fixed value
  `carcass_energy` *(calibrate; not the mammoth's current energy, so a
  thin mammoth is not worthless)* goes `(1 − s)` to the contributors in
  equal shares and `s` to the bystanders in equal shares (`s` = 0.2;
  all to contributors without bystanders), each capped at the hunter's
  energy cap.
- **On failure**: the mammoth escapes; each contributor loses a struggle
  cost (a fraction of the carcass value split among them, borrowed from
  the discrete `stag_hunt`/`shared_prey` struggle cost) and starts a hunt
  cooldown of 2 decisions. No death in the first version: a hunter death
  ends a sparse-reward trajectory and draft calculations do not yet show
  a viable death rate; death by party size is a later variant.
- **Join cost**: every contributor pays `join_cost` per hunt, success or
  failure (as the checked discrete implementation).

**Expected-value table** (pre-registered, from the calibrated values):
for party sizes 1 to 4, the expected net energy per contributor per hunt
(success chance × share − struggle cost × failure chance − join cost), and
per bystander; next to the expected energy of a rabbit catch (catch
efficiency 0.5 × rabbit energy at capture, measured in calibration). The
calibration target: a pair hunt worth at least twice a rabbit catch per
hunter, a solo hunt worth less than nothing.

### 3.3 Actions

Hunters: `MultiDiscrete([48, 2])`, the base's 48 movement actions and a
separate join head (0 = stand by, 1 = join), as the discrete
`MultiDiscrete([move, join_hunt])`. Prey keep `Discrete(48)`. The join
head acts only in mammoth hunts.

*(Later variant)* `mammoths_defection`'s movement-based defection (no join
head).

### 3.4 Observations

Nearest first in the view cone (ties by agent ID); each animal slot
`[1, dx/R, dy/R, distance/R, vx/its top speed, vy/its top speed]`, unused
slots zero; `R` the observer's view distance.

- Hunters (58 inputs): self (4), **4 hunter slots** (24; party sizes up
  to 4+ can be seen), **2 mammoth slots** (12), **2 rabbit slots** (12),
  grass (4: the nearest rabbit grass, which hunters only use as a
  landmark), own energy (1), hunt cooldown (1, remaining / 2).
- Rabbits: self, 3 hunter slots, 3 rabbit slots, grass, energy.
- Mammoths: self, 3 hunter slots, 2 mammoth slots, steppe, energy.
- Prey do not see the other prey species (first version).

View cones and distances as the base (hunters 150°, 200; prey 240°, 267).
Monocular distance noise and strike catching from the FOV module are off.
Hunters do not see each other's join decisions in the first version;
seeing them is a form of signalling and would be its own pre-registered
arm.

## 4. Measures

Every rate has an explicit denominator.

- **Mammoth encounter** (the unit for cooperation measures): a decision at
  which a hunter is within 96 of a mammoth's centre and sees it.
- **Joint arrival** (primary): within one encounter episode of a mammoth
  (from the first hunter within 96 until no hunter is), at least two
  hunters touch it within 2 decisions of each other. Rate = joint
  arrivals per mammoth encounter episode.
- **Chance level** for joint arrival: the same hunters' trajectories with
  their positions rotated around the torus relative to the mammoth by
  random offsets (a density-matched spatial null, same population, same
  movement statistics), so more hunters alone do not count as more
  cooperation.
- Join probability at mammoth contact, with a potential partner (another
  hunter within 96) present versus absent.
- Hunts and kills by party size; share of hunter energy from mammoths;
  births per hunter.
- **Manipulation check**: in states with two contributors next to a
  mammoth, replacing one's join by 0 (deep-copied world, same RNG state)
  must lower the kill chance as the rule says.
- Ecology: coexistence of the three species in 50 final-check episodes;
  population sizes; rabbit-only hunter viability (3.1).

## 5. Plan

0. **Build**: the environment, three policies, hunts, observations,
   metrics; tests for the invariants (one hunt per mammoth per decision,
   one party per hunter, cooldown, torus distances, energy accounting:
   carcass in = shares out); Codex review.
1. **Calibration** (not confirmatory; its seeds are not reused):
   (a) hunters with rabbits only, trained from scratch: viable;
   (b) with mammoths and the expected-value table above: viable from
   scratch, mammoths a substantial but not necessary food source. Frozen
   values go into the pre-registration.
2. **Main test, pre-registered before the runs.** Three training arms,
   the same seeds (at least 3; the training seed is the unit):
   - **hunt**: the rules of 3.2;
   - **solo control**: mammoths catchable by one contributor, with the
     success chance set so the expected carcass energy per encounter
     matches the hunt arm's; the same ecology without the need to
     cooperate;
   - evaluations of the **untrained** (iteration 0) and early policies in
     the hunt world.
   Primary contrast: joint arrivals per mammoth encounter above the
   density-matched chance level, hunt arm against both the solo control
   and the untrained policies, in the same direction in every seed, with
   a threshold fixed in the pre-registration. Secondary: join probability
   with a partner present minus absent (hunt arm against untrained). The
   pre-registration also fixes: every parameter, PPO settings and network,
   seeds, iterations and checkpoint choice, evaluation seeds, denominators,
   the handling of extinctions and newborns, primary against exploratory
   measures, and the consequence of failing the coexistence guard (45 of
   50): a run that fails it is reported and counts as not supporting
   learned cooperation, it is not excluded.
3. **Nature + nurture** (later, in evolutionary/continuous_space): a
   heritable cooperation trait as in
   `stag_hunt_forward_view_nature_nurture`, joining the Baldwin line of
   ERL and continuous_traits.
4. **Society** (later): band-wide meat sharing (predator_bands) and
   reputation (stag_hunt_reputation), where free-riding becomes a band
   problem.

## 6. Open decisions

1. Mammoth top speed: 4 (slower than hunters, default) or 5 (equal).
2. A separate steppe layer for mammoths (default) or shared grass.
3. No hunter death in hunts in the first version (default), or death by
   party size from the start.
4. A memory (LSTM) for hunters, or not (default not): joint arrival
   needs hunters to wait for or track partners they may lose from view.
5. The summed-energy capture rule of the discrete module as an extra arm,
   or not (default not).
