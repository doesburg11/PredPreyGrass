# stag_hunt_continuous: design

Status: **draft 2, 2026-10-30; decisions taken (section 6); not built.** Draft 1 was revised after a
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

## 6. Decisions (2026-10-30)

Taken as best guesses at the user's request; each can be revisited.

1. **Mammoth top speed 4**, slower than hunters (5). A lone hunter can
   reach a mammoth easily, so the only obstacle is the kill: the design
   then isolates cooperation in killing, not in chasing.
2. **A separate steppe layer** for mammoths. It keeps the rabbits, the
   hunters' safety net while they learn, independent of mammoth numbers;
   shared grass is a later variant once the world is viable.
3. **No hunter death in hunts** in the first version. Deaths early in
   training can make the world non-viable before anything is learned
   (predator_bands' first mammoth pilot); death by party size is a later
   variant once cooperation is seen.
4. **No memory (LSTM)** in the first version. The base world's arms race
   was learned without it, a hunter sees 4 others, and memory roughly
   doubles training cost. If joint arrivals stay at chance, memory is
   the first lever to try.
5. **No summed-energy arm.** The GPU fits two or three learners: three
   arms × three seeds is already the budget.

In the text above, the *(open)* marks on these points are settled by
this section.

## 7. Rabbit burrows (2026-11-01; built, settings `burrow_count`, `burrow_radius`, `burrow_placement`, `burrow_hides_rabbits`)

Proposed after calibration rounds 1-9 (README, calibration log): every
brake on the hunters tried so far (a cap, costlier offspring, a fixed
birth wait, faster rabbits) ended with the hunters living on one prey.
Either rabbits were eaten out (a mammoth payoff big enough for group
hunting fed a hunter boom), or the hunters ignored the mammoths. Faster
rabbits (round 9) multiplied instead of escaping and became an easy
staple. A burrow protects rabbits without making them more plentiful or
easier to find: it caps how many can be caught, not how many there are.
The hope (untested) is that rabbits then survive a hunter boom, while
rabbits stay too scarce in the open for hunters to live on them alone,
leaving mammoths as the main food and the cap unnecessary.

1. **Burrows**: `burrow_count` fixed discs of radius `burrow_radius`,
   drawn uniformly on the torus at reset from their own random stream
   (seed + 5), so the grass, spawn and policy draws are those of the
   same seed without burrows. Default 0 = the current world (old
   checkpoints keep their observation shape).
2. **Protection**: a rabbit whose centre is inside a burrow cannot be
   caught. Hunters can walk over burrows; nothing else changes for
   hunters or mammoths. No capacity limit in the first version. Contacts
   are tested before movement, so a hunter waiting at the edge can catch
   a rabbit whose centre has just left: ambush at the exit is allowed
   (as in nature) and measured (8).
3. **No food in burrows**: a grass patch whose centre is inside a burrow
   cannot be eaten. Patches are not redrawn, so the grass layer's own
   random stream, dispersal and overgrazing are unchanged; the blocked
   patches still count for dispersal and overgrazing. A rabbit must
   leave to eat, and since grass clusters move while burrows stay put,
   the distance from safety to food changes during an episode: a safety
   versus feeding trade-off (a "landscape of fear").
4. **Births and spawns** are not moved: initial rabbits and newborns may
   start inside or outside a burrow. Births inside are counted (8).
5. **Observation**:
   - rabbits: the nearest burrow regardless of the view cone (a rabbit
     knows its warrens), `[dx/R, dy/R, distance/R, inside]`, offsets
     clipped to the view distance (4 inputs);
   - hunters: the nearest burrow in view, `[1, dx/R, dy/R, distance/R]`
     (4), and an `inside` bit on each rabbit slot (2), since the nearest
     burrow need not be the rabbit's.
   Rabbits 45 → 49 inputs, hunters 58 → 64.
6. **Starting values** (to calibrate): 10 burrows of radius 24 on the
   800 × 800 arena (about 3% of the area). Burrows cut how many rabbits
   can be caught, not how many there are; if rabbits again multiply into
   a staple (as in round 9), fewer or smaller burrows, or a capacity.
7. **Calibration rounds** (truncated at the first prey extinction in
   training; coexistence judged by `evaluate.py` on 50 whole episodes):
   - r10a: round 5 (cap 25) + burrows: does the world with cooperation
     keep its rabbits?
   - r10b: round 4 (cap 50) + burrows, towards the user's aim of no
     tight cap; the share of time at the cap is recorded, and a cap that
     binds often is not counted as "no cap".
   A calibration round only passes into a pre-registered test with:
   at least 45/50 coexisting, mammoths at least half of the hunters'
   energy, and cooperation signals beyond joint arrivals alone (partner
   versus alone joining, kill party sizes, contrast with untrained
   policies); the main test then needs its effect margin and per-seed
   reporting fixed in advance (DESIGN 4-5).
8. **Counters** (TensorBoard `hunts/*`, summed over physics steps, and
   rates from them; `evaluate.py` also per episode): rabbit steps and
   those inside a burrow (`rabbit_inside_share`, `open_rabbits_mean`);
   `catches_per_open_rabbit_step`; `catches_blocked` (rabbit steps inside
   a burrow touching a predator); `burrow_exits` and `catches_after_exit`
   (within 8 physics steps); `predator_burrow_edge_share` (within a
   predator radius of an edge: camping); `births_rabbit_in_burrow`;
   `grass_blocked_patch_steps` (grown patches a burrow keeps from being
   eaten); `predator_cap_share` (steps at the hunter cap). Evaluation
   only: each rabbit's share of time inside (quartiles) and the physics
   step at which each species died out.
9. **Warrens beside the grass** (2026-11-02, setting `burrow_placement`,
   round 12). In round 10 rabbits spent ~3% of their time inside, the
   burrows' share of the area, and more than half never entered one
   (evaluations of r10a and r10b). A likely reason (measured, not
   tested): uniform burrows lie far from the clustered grass, the median
   grass patch 115 units from the nearest burrow centre (98 from its
   edge), with hunters seeing 200 units, so a feeding rabbit seldom has
   a refuge it can reach. (Blocked grass inside is a weaker reason: a
   rabbit just inside the rim reaches grass within 12 units outside.)
   `burrow_placement = "cluster"` puts one warren per grass cluster
   (`burrow_count` must equal `grass_cluster_count`), its edge one
   cluster spread (24) from the cluster's centre, in a direction drawn
   from the burrows' own stream; when a cluster collapses and its centre
   moves (overgrazing), its warren is redrawn beside the new centre at
   the start of the next physics step (rabbits dig near their pasture),
   though the new cluster's grass regrows only after the overgrazing
   delay. A rabbit in the abandoned warren is then in the open (an exit
   is counted). Grass inside stays uneatable. Untrained, the median
   edible patch is then 31 units from a warren's edge (98 with uniform
   burrows), about 5 physics steps at a rabbit's top speed; warrens
   cover ~9% of the grass patches (2%), a little less food for rabbits.
   New counters: `grass_near_burrow_share` (edible patches within reach
   of a rabbit inside a burrow) and `burrow_moves`. "random" (default)
   keeps the earlier placement and draws.
10. **Hidden rabbits** (2026-11-02, setting `burrow_hides_rabbits`).
   Without mammoths and with these warrens (check `rabbits_only_warrens`)
   rabbits learned to hide (35% of their time inside after 1,000
   iterations), but hunters spent 22% of their steps at a warren's edge,
   stood on hidden rabbits, and no evaluated episode kept both species
   (49 of 50 without warrens). Hunters saw a rabbit inside (its slot and
   the inside bit of point 5), which real predators mostly cannot: an
   animal underground is out of sight (smell aside). With
   `burrow_hides_rabbits`, a rabbit inside a burrow fills no hunter slot
   (judged from current positions, as the catch test) and the inside
   bits are dropped (hunters 64 → 62 inputs); hunters still see the
   nearest warren. Default False keeps point 5.

Realism: rabbits (and the hares and marmots Pleistocene hunters took) do
shelter in burrows; the food-versus-safety trade-off is standard
predator-prey ecology. A burrow is the cheapest refuge to build: a disc
test in the catch rule, a redraw in grass placement, one observation
slot.

## 8. Well-fed mammoths are harder to kill (prepared 2026-11-01, setting `hunt_condition_factor`)

Proposed after asking why the discrete stag hunt (stag_hunt_forward_view)
seemed to work where this world does not. Its capture rule (joiners'
summed energy must exceed the prey's) brakes a boom by itself; its own
notes show rabbits dying out there too, so it never met this module's
coexistence test. Its literal rule is not realistic (body reserves are not
hunting force; moderate hunger raises hunting effort). The prey half has
some support, with caveats: predators often select vulnerable prey
(young, old, malnourished), but how strongly varies with predator, prey
size, season and demography (e.g. lions and cheetahs), and mammoth kill
sites differ (some suggest vulnerable juveniles, others show no strict age
selection). Here "condition" is a deliberately simplified proxy, the
mammoth's energy balance, not age, injury or escape ability; one artefact
is that a mammoth that has just given birth (losing its offspring's 10
energy) is easier to kill. The rule is intended to create a density
feedback (scarce mammoths eat well, so are harder to kill, so hunting
falls off before they are gone); scarcity does not by itself guarantee
well-fed mammoths, and whether the feedback appears is what round 11
tests.

1. **Rule**: success probability = `hunt_success_by_party[n]` × (1 − k ×
   condition), condition = mammoth energy / its reproduction threshold
   (24), at most 1. One random draw per hunt as before. k = 0 (default) is
   the current world.
2. **Observation**: predators see the condition of each mammoth in their
   slots (2 inputs), so they can pick weak ones. Hunter condition is left
   out.
3. **Counters**: `hunted_condition_mean`, `killed_condition_mean`.
4. **Calibration**: r11a k = 0.5, r11b k = 0.8, in round 4 (cap 50, no
   burrows, to test the rule alone), training truncated at the first prey
   extinction. Risk: a lower mean success may drop the mammoth payoff into
   the "ignore mammoths" regime of rounds 7-9; the counters show which
   mammoths are hunted and killed.

## 9. Summed-energy capture rule (2026-11-02, settings `hunt_rule`, `hunt_energy_ratio`)

Proposed after rounds 1-12 and the burrow checks (README, calibration
log): every world either booms (mammoths pay, hunters multiply and eat
out a prey) or the hunters ignore the mammoths, and refuges for rabbits
(burrows) cut rabbit breeding more than they cut catches. The discrete
`stag_hunt_forward_view`, the source of this module, resolves a hunt by
the hunters' state instead of a draw: it succeeds when the joiners'
summed energy exceeds the prey's energy (`team_capture_margin` 0). This
was the "summed-energy arm" left out on 2026-10-30 (section 6).

1. **Rule**: with `hunt_rule = "energy"` a hunt succeeds when the
   contributors' summed energy (before the join cost) exceeds
   `hunt_energy_ratio` x the mammoth's energy; no random draw. Costs,
   cooldown, carcass (fixed `hunt_carcass_energy`), scavenger share and
   bystanders are unchanged. Default "chance" keeps the party-size draw;
   `hunt_condition_factor` applies to it only.
2. **Why it may brake a boom** (the hope, untested): hunters that have
   outgrown their food are hungry, and hungry parties cannot kill, so
   mammoth hunting stops as the hunters multiply and resumes as they
   thin out. The brake grows with the hunters' hunger, not with a fixed
   number (unlike the cap). Well-fed mammoths are also harder to kill,
   which in round 11 turned hunters away from mammoths; here the hunters'
   own energy is the other side of the scale.
3. **Scale** (measured in round 4's trained checkpoint, iteration 190,
   4 episodes): in its hunts the contributors' summed energy exceeded the
   mammoth's in 15% of solo, 45% of pair and 84% of trio attempts (6%,
   33%, 66% at 1.5 x), against the chance rule's 0, 40 and 75%: about
   the same kill rates on average, but tied to hunger. A well-fed hunter
   can kill a lean mammoth alone (as in the discrete module), so the
   stag-hunt structure is weaker at ratio 1. Untrained hunters are lean
   (median energy 3, mammoths 8): early kills will be rare, a risk that
   hunters learn to ignore mammoths before they learn to hunt them.
4. **Observation**: predators see the energy of the mammoths in their
   slots (/ `energy_mammoth_max`) and of the predators in their slots
   (/ `energy_predator_max`), 2 + 4 inputs (58 → 64), since the rule
   depends on them (the discrete module's grid holds every agent's
   energy). Whether a hunter can judge a partner's or a mammoth's
   condition by sight is a simplification.
5. **Counters**: summed party energy and mammoth energy per hunt
   (`hunted_party_energy_mean`, `hunted_mammoth_energy_mean`).
6. **Calibration round 13**: round 4 (cap 50, no burrows), training
   truncated at the first prey extinction, ratio 1 (r13a) and 1.5 (r13b).
   Judged as before: coexistence in at least 45 of 50 evaluated episodes,
   mammoths at least half of the hunters' food, cooperation signals.
