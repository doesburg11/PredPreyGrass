# predator_sexual_reproduction

A copy of [`base_environment_step_energy`](../base_environment_step_energy/) that
replaces asexual predator reproduction with **sexual (two-parent) reproduction**,
and splits predators into two sexed policies with different foraging roles.

## Research motivation

Predators, not prey, are the human analog in this module. The project's
broader interest is human cooperation and behavior; predators are cast as
"humans" here because humans hunt *and* gather fruit, whereas herbivore prey
don't map onto that. Two features are specifically included as human-like
traits under study, not just mechanics:

- **Sexual reproduction** (a mate requirement, not solo energy-threshold
  reproduction), with asymmetric parental investment (see below), as a
  substrate for pair-bonding-like dynamics.
- **Risk-driven division of labor by sex** -- both sexes can hunt, but face
  very different odds (see "Why hunting is risky, and riskier for females"
  below), so any resulting specialization is meant to emerge from training,
  not be hardcoded by fiat -- a precondition many theories of human
  cooperation lean on.

This is why predators were chosen over prey for this feature despite prey
being the easier population to test it on (denser, more frequent mating
opportunities, no interaction with the hunting/engagement code path) --
prey have no analogous role in a human-cooperation framing.

## The mechanism

- **predator_male**: low-risk/high-success hunter (90% success / 5% death per
  attempt by default), also gathers fruit.
- **predator_female**: high-risk/low-success hunter (20% success / 10% death
  per attempt by default), also gathers fruit.
- **prey**: unchanged from `base_environment_step_energy` -- only eats grass,
  reproduces asexually (solo, energy-threshold triggered).

Grass is prey-exclusive food; fruit is predator-exclusive food (both sexes can
gather it). This keeps predator and prey foraging from ever competing over
the same patches, unlike `eco_evolutionary_nuptial_gift`'s predator_female
(which deliberately competes with prey for grass).

### Sexual reproduction

A `predator_male` and a `predator_female` reproduce together when:

1. Both **independently** clear `predator_creation_energy_threshold`.
2. They are within `mate_search_radius` (Chebyshev distance) of each other.

This can never be an exact-cell match, unlike predator-catches-prey or
prey-eats-grass: `predator_male`/`predator_female` share one grid layer
(channel 1, "predator"), and movement collision already forbids two
predators -- of either sex -- from ever occupying the same cell. A radius
check is therefore structurally required, not just a design preference.
This is the same constraint `eco_evolutionary_nuptial_gift`'s
`cooperation_range` (male → female nuptial gift) faces, and is modeled on it.

Unlike the nuptial-gift module, reproduction itself is genuinely two-parent
here: both parents pay energy, both get the reproduction reward, and a single
offspring (sex assigned by an unbiased coin flip) is spawned adjacent to the
**female** (the mate). There is no genome in this module (it's a
`non_evolutionary/` module, mirroring `base_environment_step_energy`) --
"sexual" here means the two-parent *mechanic* (mate-finding, joint energy
cost, joint reward), not genetic recombination.

**Birth cost is split asymmetrically**: the female pays 90% and the male
pays 10% of the offspring's starting energy by default
(`predator_birth_cost_share_female`/`_male`), modeled on parental investment
theory (Trivers, 1972) -- the female bears the larger share of the shared
reproductive cost, consistent with her also being the structurally riskier
forager (see below).

### Male provisioning (offsetting the female's post-birth energy deficit)

A `predator_female` pays the larger share of birth cost (90% by default) but
her only reliable income is fruit-gathering -- a shared, depletable resource,
and a much weaker income stream than hunting is for the male (90% success vs.
her 20%). Left alone, this makes her a recovery bottleneck after even one
successful birth. To offset this, a `predator_male`'s successful hunt donates
`male_gift_donation_rate` (default `0.3`) of the energy gained to **his
recorded mate only** (`self.agent_mate`), provided she's currently within
`predator_gift_range` (Chebyshev distance, default `3`) -- not to any nearby
female. This is exclusive/pair-bonded (more biologically apt for the human
pair-bonding this module studies than broadcasting to whoever happens to be
nearby, which is what `eco_evolutionary_nuptial_gift` does for its
non-pair-bonded species), unidirectional (male → female only), and
mechanically executed -- not a learned action -- for the same
credit-assignment rationale as `eco_evolutionary_nuptial_gift`'s
`male_donation_rate` (a donor's own reward stream never reflects a
recipient's downstream fitness, so a learned "donate" action would face a
real credit-assignment gap).

The mate bond (`self.agent_mate[male] = female` and vice versa) is recorded
the first time a pair successfully reproduces together, and updated (not
accumulated) on any later reproduction -- serial monogamy, i.e. his most
recent partner, not lifetime exclusivity. Remating severs both former
partners' reverse pointers first, so an ex never keeps receiving gifts
indefinitely. A male who has never reproduced has no recorded mate and
gives no gifts at all. See `_apply_male_gift`.

### Parental care (both parents feed their own nearby offspring)

Alongside mate provisioning, **both** parents (not just the mother) share
`parent_offspring_share_rate` (default `0.2`) of any successful forage --
hunt *or* fruit -- with their own nearby living offspring, recorded at birth
in `self.agent_parents[child] = (father, mother)`. Direct paternal
provisioning of offspring (not just of the mother) is a documented
human-specific trait in the cooperative-breeding literature -- most mammals
rely on the mother alone, which is part of why this module casts predators
as humans in the first place. Unlike the exclusive, single-recipient mate
gift, this splits evenly across however many of a parent's own children are
currently nearby, since (unlike mates) a parent can have several living
offspring at once. Reuses `predator_gift_range` for proximity rather than
adding a separate knob, and needs no explicit "weaning age" cutoff -- a
grown offspring that wanders off simply falls out of range on its own, so
care tapers off for free. See `_share_energy_with_offspring`.

### Why hunting is risky, and riskier for females

Hunting is a 3-outcome stochastic contest for **both** sexes, resolved
whenever a predator shares a cell with a prey (`_resolve_hunting_attempt`):

1. **success** -- the predator eats the prey (same bookkeeping as a
   deterministic catch would have).
2. **predator dies** -- the prey is left completely unharmed; the predator
   itself is removed, using the same bookkeeping as starvation.
3. **failure** -- nothing happens; the predator can still separately gather
   fruit that step.

`predator_male` gets favorable odds (90% success / 5% death by default);
`predator_female` gets unfavorable odds (20% success / 10% death by default).
This reverses the module's original design, where `predator_female` was
*structurally incapable* of touching prey at all -- now both sexes have the
same capability, and any tendency for females to lean on fruit-gathering
instead of hunting is meant to be an **emergent consequence of RL training**
under this risk asymmetry, not a hardcoded rule. Whether training actually
produces that specialization is an open empirical question (see Status).

## Grid channels

`num_obs_channels = 5`: Border, Predator (both sexes share this layer),
Prey, Grass, Fruit.

## Config (see `config_env.py`)

Energy costs (`homeostatic_energy_cost_per_step_*`, `move_energy_cost_per_step_*`)
are inherited unchanged from `base_environment_step_energy`'s validated
defaults. New keys: `mate_search_radius` (default `3`), `n_possible_predator_male`/
`n_possible_predator_female` (replacing the single `n_possible_predators` pool),
`n_initial_active_predator_male`/`n_initial_active_predator_female`,
`initial_energy_predator_male`/`initial_energy_predator_female`, the
`initial_num_fruit`/`initial_energy_fruit`/`energy_gain_per_step_fruit` fruit
settings (same shape as the grass settings),
`predator_birth_cost_share_female`/`_male` (default `0.9`/`0.1`),
`penalty_predator_death_in_combat` (default `0.0`), the hunting-odds keys
`prey_vs_predator_male_success_prob`/`_death_prob` (default `0.90`/`0.05`)
and `prey_vs_predator_female_success_prob`/`_death_prob` (default
`0.20`/`0.10`), `male_gift_donation_rate`/`predator_gift_range` (default `0.3`/`3`) for male
provisioning, and `parent_offspring_share_rate` (default `0.2`, reuses
`predator_gift_range`) for parental care. `__init__` raises `ValueError` if
a sex's success+death probabilities exceed 1.0, if the birth-cost shares
don't sum to exactly 1.0, or if `male_gift_donation_rate`/
`parent_offspring_share_rate` is outside `[0, 1]`, or if their sum exceeds
`1.0` (a male's successful hunt applies both donations to the same gross
gain, not sequentially off a shrinking remainder, so an unchecked sum above
1.0 could deduct more energy than the hunt actually gained).

## Running

Train:
```
python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.tune_ppo_predator_sexual_reproduction --seed 42 --max-iters 500
```

Watch a random policy (sanity check, no training needed):
```
python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.random_policy
```

Evaluate a checkpoint interactively:
```
python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.evaluate_ppo_from_checkpoint_debug
```
(update the hardcoded `checkpoint_path` first).

Run the validation test suite (not auto-discovered by the repo's pytest
`testpaths` -- must be invoked explicitly, same as every other module's tests):
```
pytest predpreygrass/non_evolutionary/predator_sexual_reproduction/tests/ -v
```

## Status

Freshly scaffolded (2026-09-17); probabilistic combat + asymmetric
reproduction added (2026-09-18); male provisioning added the same day after
identifying that the 90/10 birth-cost split left females with no viable way
to recover the energy spent on a birth (later revised to be exclusive to a
recorded mate rather than broadcast to any nearby female); parental care
(both parents feeding their own nearby offspring) added the same day.
Smoke-tested via the unit test suite, a random-policy run, and a local PPO
build/train iteration, not yet trained for real. Open questions for a first real run: whether
`mate_search_radius=3` gives frequent-enough mating opportunities at this
module's population density (see `base_environment_step_energy/RESULTS.md`
for the equilibrium ~24-29 prey / ~12-14 predators baseline this module
inherits its costs from -- predator count there wasn't split by sex, so the
per-sex equilibrium here is untested); whether the 90/10 birth-cost split
and `male_gift_donation_rate=0.3` need retuning against each other (does the
gift actually offset the birth cost, or just soften it); and the central
empirical question this module now exists to test -- **does training
actually produce male-hunts/female-gathers specialization from the 90/5 vs
20/10 success/death odds, or does the risk asymmetry turn out not to be
enough to drive it?**
