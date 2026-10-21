# foraging_swarms: specification

This module's environment is a copy of `base_environment_continuous`, and this
is that module's specification. The one addition is in section 7.1 (social
information settings). With those settings at their defaults the environment
is identical to `base_environment_continuous`: same observations, same
dynamics (`tests/test_social_observation.py`).

---


A standalone continuous-space Predator-Prey-Grass environment, replacing the
`marl-aquarium` package (unmaintained since 2024) and the layers of patches
that `add_walls` (in the
[PredPreyGrassContinuous](https://github.com/doesburg11/PredPreyGrassContinuous)
development repository) puts on top of it. This document records exactly what the
current simulation does (Aquarium's code as installed, version 0.1.10, plus
every patch in `add_walls`), so the new environment can reproduce it.

**Scope.** Only the world of the milestone runs (AW-17 to AW-22): no walls,
target-speed actions, energy with a movement cost, moving grass clusters,
energy-threshold reproduction, egocentric observations, predators observing
grass, sparse birth rewards. Walls, group hunting, reward shaping,
observation stacking, Aquarium's own observations and rewards and its
respawn mode are left out.

**Rule for this version.** Everything that affects the dynamics is
reproduced exactly, including Aquarium's quirks (marked **quirk** below), so
that results obtained with `add_walls` remain valid for the new environment.
Each quirk can be removed with its own `fix_*` setting, all off by
default (section 9); with every setting off the environment stays exact.

## 1. Arena, time and seeding

- Torus of width 800 and height 800.
- One **physics step** moves every animal once. One **decision** (an RLlib
  step) repeats each agent's action for `action_repeat = 8` physics steps,
  ending early on a birth or at the end of the episode (section 7).
- The episode is truncated when, at the start of the bookkeeping of a
  physics step, the step counter exceeds `max_time_steps` (8000), so it lasts
  at most 8,002 physics steps (**quirk**: counter checked before it is
  incremented, with `>`).
- `reset(seed)`: three independent random streams, all derived from `seed`:
  - **spawn stream**: Python's `random` seeded with `seed` (in `add_walls` the
    process-global module), used only for the initial positions;
  - **grass stream**: `random.Random(seed)`;
  - **birth stream**: `random.Random(seed + 1)`.

  With `seed=None` all three are unseeded. (Deliberate difference: here
  each environment has its own spawn stream. In `add_walls` an unseeded
  reset draws from the process-global `random`, so it continues whatever
  stream that module is in, shared by every environment in the process.
  Seeded resets are identical.)

## 2. Animals and initial state

Each animal has a position, velocity and acceleration (2D vectors), radius 16,
species-specific top speed, maximum acceleration and steering force, a mass,
an orientation angle, an energy store and an ID `predator_<n>` or
`prey_<n>`.

| | Predator | Prey |
|---|---:|---:|
| Mass | 10 | 3 |
| Top speed (`*_max_velocity`) | 5 | 6.0 (milestone; 5 in `add_walls`'s config) |
| Maximum acceleration | 0.6 | 1.0 |
| Maximum steering force | 0.6 | 0.6 |
| View distance / field of view | 200 / 150° | 267 / 240° |
| Initial velocity | (0, -1) | (-1, 0) |

**Initial placement**, in this order from the spawn stream: for each of the
`prey_count` (8) prey, `x = randint(20, 780)`, `y = randint(20, 780)`, then
one further `randint(150, 250)` (Aquarium's unused replication age;
**quirk**, consumed only to keep the stream); then for each of the
`predator_count` (6) predators, `x = randint(0, 800)`, `y = randint(0, 800)`.
Prey get IDs `prey_0 ...`, predators `predator_0 ...` in creation order.

**Orientation** is `-atan2(vy, vx)` in degrees: initially -180 for prey
and 90 for predators (not rounded); after every movement it is set to
`round(-atan2(vy, vx) in degrees, 1)`, so a stopped animal faces 0°
(**quirk**: standing still resets the heading, and with it the view cone,
to the +x direction).

Energy starts at 5.0 (predators) and 3.0 (prey).

## 3. One physics step

Animals move **one at a time**, in the order of the entity list: at reset
all prey, then all predators; newborns are appended at the end in birth
order. Each animal sees the positions of those already moved in this step
(**quirk**: sequential, not simultaneous, updates).

### 3.1 Desired velocity from the action

48 actions: `direction = a % 16`, `speed index = a // 16` with factors
(1.0, 0.5, 0.0). The direction angle is `360 / 16 × direction` degrees, and
the unit vector is `(sin(angle), -cos(angle))` (action 0 points to -y). The
desired velocity is that unit vector times the species' top speed times the
speed factor.

### 3.2 Prey update

For prey `q` with desired velocity `d`:

1. **Caught?** If any predator, in predator-list order, is at torus distance
   strictly less than the sum of radii (32), `q` is caught: the first such
   predator is its catcher, recorded with `q`'s energy at that moment. The
   prey still moves in this step (**quirk**) and is removed at the end of
   its own update.
2. Acceleration is reset to zero. Let `e = d - v`; the effective maximum
   acceleration is `min(prey max acceleration, |e|)`.
3. Steering force `s = e` limited to magnitude 0.6; acceleration `a = s / 3`
   (mass), then normalised to unit length (unchanged if zero) and multiplied
   by the effective maximum acceleration.
4. **Bumping into prey.** If another prey (first in prey-list order, the
   caught prey itself still listed) is at torus distance below 32: `v`
   gains the unit vector pointing away from it, *instead of* `a` (**quirk**:
   the bump replaces acceleration for that step). Otherwise `v += a`.
5. `v` is limited to the top speed; position `+= v`.
6. **Borders** (**quirk**: not a modulo): if `x > 800` then `x = 0`, else if
   `x < 0` then `x = 800`; the same for `y`.
7. Orientation update (section 2).
8. If caught: the prey is removed from the prey and entity lists.

"Limited to magnitude m" means: if the length exceeds `m`, rescale to
length `m`. The away-vector is the wrapped offset to the other animal's
centre (each component shifted by ±width or ±height if its absolute value
exceeds half the arena), negated, normalised.

### 3.3 Predator update

For predator `p` with desired velocity `d`:

1. Steps 2 and 3 as for prey, with mass 10 and the predator's maximum
   acceleration (0.6).
2. **Bumping into predators.** If another predator (first in list order) is
   at torus distance below 32, `a` gains the away-vector scaled to length
   `5 / 8` (top speed / 8). Unlike prey, this adds to the acceleration.
3. `v += a`, limited to the top speed; position `+= v`; borders; orientation.

Predators never die in this update (Aquarium's age limit is disabled with
`predator_max_age = 10^9`; catches only reset an unused age counter).

### 3.4 Bookkeeping after movement (in this order)

1. **Episode end check** (Aquarium): all agents are truncated if the step
   counter exceeds `max_time_steps`, or no prey or no predator is left. The
   counter is then incremented.
2. **Grass** (section 4): regrowth placement, grazing, overgrazing.
3. **Energy** (section 5): catch gains, grass gains, metabolic costs,
   starvation.
4. **Reproduction** (section 6), unless the episode ended in this step.

## 4. Grass

100 patches in 20 clusters of 5 (milestone; `add_walls`'s config has 5
clusters of 20), consumption radius 12, regrowth delay 400 physics steps,
dispersal distance 24, overgrazing on with threshold 0 and delay 400.

- **Cluster grid:** `columns = ceil(sqrt(clusters × width / height))`,
  `rows = ceil(clusters / columns)`; for 20 clusters 5 × 4.
- **Reset:** cluster centres in distinct grid cells, chosen with
  `sample(range(cells), clusters)` from the grass stream; each centre is
  `((cell % columns + uniform(0.25, 0.75)) × cell width, (cell // columns +
  uniform(0.25, 0.75)) × cell height)`. Patch `i` belongs to cluster
  `i % clusters` and is placed at its centre plus `gauss(0, spread)` in x
  and in y (spread 24), wrapped with modulo.
- **Each physics step** (`advance`): the grass tick is incremented; every
  patch whose regrowth is due and is marked for relocation is placed
  (*disperse*: next to a random available patch of its cluster, `gauss(0,
  24)` offsets, else at a fresh spot in its cluster; *centre*: at a fresh
  spot around the cluster centre). Then, for every available patch in list
  order, the closest living prey within 12 (ties broken by agent ID) eats
  it: the patch becomes unavailable until `tick + 400` and is marked
  *disperse*. Then overgrazing: a cluster with no available patch left
  collapses: its centre moves to a random grid cell no other cluster uses
  (any cell but its own if none is free) and all its patches regrow there
  together after 400 steps; it re-arms once a patch is available again.
- The exact procedures, including the order of random draws, are those of
  `add_walls/grass.py` (`GrassLayer`), without the wall branches.

## 5. Energy

| | Predator | Prey |
|---|---:|---:|
| Initial energy (also a newborn's) | 5.0 | 3.0 |
| Cap | 24.0 | 16.0 |
| Resting cost per physics step | 0.25 × 0.15 / 8 | 0.25 × 0.05 / 8 |
| Speed cost per unit of speed per step | 0.75 × 0.15 / 8 / 5 | 0.75 × 0.05 / 8 / 5 |
| Gain | 0.5 × caught prey's energy | 2.0 per grass patch |

(Milestone: 75% of the cost from movement. `add_walls`'s config default is
10%.) Each physics step, after grass:

1. Each catch feeds its catcher, if still alive, by 0.5 × the prey's energy
   at the moment of the catch (capped).
   (foraging_swarms only: with `energy_catch_share_radius` > 0, that energy
   is split equally among the catcher, if alive, and every other living
   predator within that distance of the spot where the prey was caught,
   positions after this physics step's movement; each share is capped
   separately. Default 0: the catcher alone.)
   (foraging_swarms only: with `capture_assist_radius` > 0, a prey with at
   least `capture_assist_threshold` energy is caught only if another predator
   is within that radius of it, found from the same positions as the
   contact; the energy is then split among the catcher and those helpers,
   plus any `energy_catch_share_radius` predators, each counted once.
   `observe_energy_in_slots` adds the observed animal's energy / its cap as a
   seventh value to every slot.)
2. Each prey gains 2.0 per patch eaten (capped).
3. Each living animal pays `resting + speed cost × |v|`, with `v` its
   velocity after movement.
4. Animals at or below zero energy starve: removed and terminated. If this
   empties a species, every remaining agent is truncated and the episode
   ends.

## 6. Reproduction

After energy, unless the episode ended this step: for every living animal,
predators first then prey in list order, with energy at or above its
species' threshold (12 for predators, 8 for prey), while its species is
below its cap (50 predators, 100 prey) and the ID pool (2000 per species) is
not used up: one offspring with the initial energy, paid by the parent,
placed `2 × radius` from the parent in direction `uniform(0, 2π)` from the
birth stream (wrapped with modulo), initial velocity as in section 2. IDs
continue per species (`prey_8`, `prey_9`, ...). The parent is rewarded +10.
Newborns act from the next physics step.

## 7. Decisions, observations and the RLlib interface

- **Rewards:** +10 to the parent per birth; nothing else.
- **A decision** applies each agent's action for up to 8 physics steps,
  summing rewards. It ends early after a physics step with a birth (so RLlib
  registers newborns while their parents live) or at the end of the
  episode. Agents that finish mid-decision drop out of the remaining steps.
- **Observations** are computed once, after the decision (33 inputs as
  below), for every animal still in the arena, including animals truncated
  at the time limit or when the other species died out; caught and starved
  animals get zeros.
- **Actions** are needed for every agent still acting; actions for agents
  that no longer act are ignored.
- **Termination flags:** a caught or starved animal is terminated. When the
  last prey dies, every agent is terminated (not truncated). When the last
  predator dies, the remaining prey are truncated. At the time limit all are
  truncated.
- `infos[agent]` carries `grass_eaten`, `births`, `born`, `parent`,
  `starved` and `energy` as in `add_walls`.

### 7.1 Observation of one animal (33 inputs)

All values in [-1, 1], float32. `R` is the observer's own view distance.

1. Self (4): `vx / top speed`, `vy / top speed`, `sin(h)`, `cos(h)` with `h`
   the orientation in radians.
2. One predator slot and three prey slots (6 each), the nearest other
   animals of that species in the observer's view cone, nearest first,
   unused slots zero: `[1, dx / R, dy / R, distance / R, vx / its top speed,
   vy / its top speed]`, with `dx, dy` the wrapped offset.
3. Grass (4): the nearest available patch in the view cone, `[1, dx / R,
   dy / R, distance / R]`, or zeros. Both species.
4. Own energy (1): energy / cap, clipped to [0, 1].

**Social information (foraging_swarms only).** Per species,
`<species>_observe_neighbors` (default True) and
`<species>_observe_neighbor_movement` (default True) apply to the
observer's **own-species** slots (prey slots for prey, predator slots for
predators). With `observe_neighbors` False those slots are always zeros
(the animals are still present and still bump). With
`observe_neighbor_movement` False the slots keep `[1, dx, dy, distance]`
but `vx` and `vy` are 0. The other species' slots, grass, self and energy
are never changed, and neither are the dynamics. The observation length is
the same in every case.

**View cone:** visible if the wrapped distance is at most `R` and the
wrapped angular difference between `-atan2(dy, dx)` (degrees) and the
observer's orientation is at most half the field of view (`add_walls`'s
corrected view check, not Aquarium's original).

## 8. What is not part of the dynamics

Rendering (pygame view, sprites, view cones, grass, the
population chart) only draws the state and is reimplemented separately
(`render.py`); it must not draw random numbers or change the state
(`tests/test_environment.py` checks that an episode is the same with and
without drawing).

## 9. Quirks and their fixes

Each quirk has a setting that removes it. All are off by default, which
reproduces `add_walls` exactly; `environment.FIXES` turns them all on.

| Quirk | Effect | Setting | With the fix |
|---|---|---|---|
| Sequential updates in list order | Later animals in the list react to positions already updated in this step | `fix_simultaneous_movement` | Catches and bumps of every animal are found from the positions at the start of the step (with the push directions), then every animal moves. A prey touched by several predators still goes to the first in the predator list (birth order) |
| A caught prey still moves in its last step | Small; it can still bump other prey | `fix_caught_prey_does_not_move` | A caught prey is removed without moving |
| A stopped animal's heading resets to 0° | Its view cone jumps to +x when it stops; a leftover speed of about 1e-9 points it in an arbitrary direction | `fix_keep_heading_when_stopped` | Below a speed of 1e-6 the heading stays as it was |
| Borders set to 0 or 800 instead of wrapping | An animal crossing an edge loses the overshoot | `fix_wrap_borders` | Position modulo the arena size |
| A prey bump replaces its acceleration; a predator bump adds to it | Asymmetric collisions | `fix_same_collisions_for_both` | Prey use the predators' rule: a push of top speed / 8 added to the acceleration |
| Only the first colliding animal counts for a bump | Crowds resolve unevenly | `fix_all_contacts` | One push away from every animal of its species it touches, summed |
| Episodes last up to 8,002 physics steps | Off by two | `fix_episode_length` | Truncated after exactly `max_time_steps` physics steps |
| A spare random draw per initial prey | None (keeps the stream) | `fix_no_spare_spawn_draw` | Dropped; initial positions differ for the same seed |

`tests/test_fixes.py` checks what each setting changes.

## 10. Verification

- **Exact equivalence:** for the same seed and the same actions, the new
  environment must return the same observations, rewards, termination flags
  and infos as `add_walls`'s `make_env` with the matching configuration,
  decision by decision, and the same state: animal positions, velocities,
  accelerations, orientations and energies, grass patches, cluster centres
  and collapse flags, the grass and birth random streams, ID counters, the
  step counter, the movement order and the acting agents (bit for bit,
  including the sign of zero and int versus float). The development
  repository's `tests/test_equivalence.py` checks this over 61 episodes
  with random actions (about 12,500 decisions): births, catches,
  starvation, grass collapse, extinction, the time limit,
  `action_repeat = 1`, population caps and exhausted ID pools. Here,
  `tests/test_regression.py` checks the same decisions against stored
  digests of `add_walls`'s results.
- **Trained policies:** the policies of run AW-21 (iteration 2,990) played
  the same 50 seeded episodes in both environments (`tournament.py`, with
  each species' actions sampled in sorted agent order) with identical
  results: 55,298 decisions, 21,873 births, 19,860 catches. The order of the
  keys in the returned dictionaries is not reproduced (it does not matter to
  RLlib), so a script that hands out sampled actions in dictionary order
  gives agents different actions in the two environments.
- **Validation runs:** BE-01 and BE-02 trained the milestone world from
  scratch with the new environment and reached the same state as AW-21 and
  AW-22: 72 against 71 of 100 episodes coexisting (two runs of 50 each),
  the same behaviour and arms race (RESULTS.md).
