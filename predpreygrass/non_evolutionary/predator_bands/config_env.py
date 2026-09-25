config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 8,  # Border, Predator (total energy), Prey, Grass, Fruit, Fruit store, Same-band, Other-band
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Rewards
    "reward_predator_catch_prey": 0.0,
    "reward_predator_gather_fruit": 0.0,
    # Energy-proportional forage reward, applied to BOTH fruit and prey at the same rate:
    # reward += reward_predator_per_energy * (gross energy gained from that fruit / catch),
    # ADDED to the flat per-event rewards above. Default 0.0 = off (old behaviour). Motivation:
    # a flat per-event reward pays a nearly empty regrowing fruit as much as a full one and
    # (measured) made fruit ~4.6x more rewarding per unit of energy than prey. Set the two flat
    # rewards to 0.0 and this to e.g. 0.2 for a purely energy-proportional reward.
    "reward_predator_per_energy": 0.0,
    "reward_prey_eat_grass": 0.0,
    "reward_predator_step": 0.0,
    "reward_prey_step": 0.0,
    "penalty_prey_caught": 0.0,
    # Reward when a predator dies from a failed hunting attempt (see the
    # Hunting section below). Kept as its own key, distinct from
    # penalty_prey_caught (which is the prey's reward on a successful catch),
    # so it's separately tunable.
    "penalty_predator_death_in_combat": 0.0,
    "reproduction_reward_predator": 10.0,
    "reproduction_reward_prey": 10.0,
    # Energy settings: same additive homeostatic (always charged) + move
    # (charged on top, only when the action isn't noop) split validated in
    # base_environment_step_energy -- see that module's config_env.py/RESULTS.md
    # for the full rationale. Applies uniformly to predator_male/predator_female;
    # only reproduction and foraging are sex-differentiated here.
    "homeostatic_energy_cost_per_step_predator": 0.10,
    "homeostatic_energy_cost_per_step_prey": 0.035,
    "move_energy_cost_per_step_predator": 0.08,
    "move_energy_cost_per_step_prey": 0.035,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    # Sexual reproduction: a predator_male and a predator_female must BOTH
    # independently clear predator_creation_energy_threshold AND be within
    # mate_search_radius (Chebyshev distance) of each other. This can never be
    # an exact-cell match (unlike predator-catches-prey or prey-eats-grass):
    # predator_male/predator_female share one grid layer (channel 1), so
    # movement collision already forbids two predators -- of either sex --
    # from ever occupying the same cell. Modeled on
    # eco_evolutionary_nuptial_gift's cooperation_range, which faces the
    # identical constraint for its male-female proximity check.
    "mate_search_radius": 3,
    # Predator population cap: when set (an int), blocks predator reproduction for the rest of a
    # step once current_num_predator_male + current_num_predator_female is already at or above this
    # ceiling (a birth that brings the population up TO the cap is still allowed) -- the mirror
    # image of FixedPreyDensityEnv's prey floor (fixed_prey_density_env.py), which replenishes prey
    # losses instead of blocking predator gains. Added to test whether the fixed-density odds
    # factorial's behavioral effects (RESULTS.md, Iteration 12) survive when predator population
    # size is also held closer to constant across odds conditions -- that iteration found predator
    # population size is NOT held fixed by the prey floor alone and roughly triples between the
    # lowest- and highest-success conditions, confounding the odds comparison with a
    # population-size/crowding comparison. Note this caps growth, it does not hold population size
    # exactly matched: condition-dependent death rates can still keep different conditions at
    # different levels below the cap, so call this "capped," not "fixed/matched," predator
    # population unless per-run measurements confirm they land close to the cap in practice. None
    # (default) disables the cap entirely; all existing runs are unaffected. __init__ validates it
    # is a non-negative int (bool and non-integral values, e.g. 3.9, are rejected) when set.
    "predator_population_cap": None,
    # Predator density target (FixedPredatorDensityEnv only, fixed_predator_density_env.py): a
    # cleaner alternative to predator_population_cap that never blocks reproduction (removing the
    # cap's own reproductive-selection confound -- who gets to reproduce near the ceiling was decided
    # by an arbitrary agent-ID tie-break). Instead, population is pushed back toward this target after
    # every step: overflow is corrected by culling predators chosen UNIFORMLY AT RANDOM across both
    # sexes (not by reproductive eligibility or fitness), and shortfall by spawning random-sex
    # replacements, mirroring FixedPreyDensityEnv's own prey-floor pattern. Trades the reproduction-
    # blocking confound for a different, smaller one (an exogenous, policy-independent random death
    # risk) -- not confound-free, just cleaner on the specific axis (reproduction) these experiments
    # care about. None (default) disables it entirely. __init__ validates it is a non-negative int
    # (bool and non-integral values rejected) when set, and raises ValueError if
    # predator_population_cap is ALSO set, since the cap would otherwise silently still block
    # reproduction here too (it lives in the shared base class, not overridden by this class).
    "predator_density_target": None,
    # Male provisioning (unidirectional male -> female energy gift on a
    # successful hunt, exclusive to his recorded mate -- see self.agent_mate,
    # not broadcast to any nearby female): offsets predator_female's
    # post-birth energy deficit -- she pays the larger share of birth cost
    # (predator_birth_cost_share_female below) but her only reliable income
    # (fruit) is weak, shared, and depleting, unlike the male's much higher
    # hunting success rate. Modeled on eco_evolutionary_nuptial_gift's
    # male_donation_rate/cooperation_range, mechanically executed (not a
    # learned action) for the same credit-assignment reasons documented in
    # that module's README.
    # __init__ raises ValueError unless the rate is in [0, 1].
    "male_gift_donation_rate": 0.0,  # band sharing (below) replaces the mate gifts in this module
    "predator_gift_range": 3,
    # Parental care: both parents (not just the mother) share a fraction of
    # ANY successful forage -- hunt or fruit -- with their own nearby living
    # offspring (self.agent_parents), reusing predator_gift_range rather
    # than adding a separate proximity knob. Split evenly if a parent has
    # multiple living offspring nearby at once (unlike the exclusive,
    # single-recipient mate gift above). No explicit weaning-age cutoff:
    # spatial dispersion already makes care taper off for free once a grown
    # offspring wanders out of range. Mechanically executed, same
    # credit-assignment rationale as male_gift_donation_rate above.
    # __init__ raises ValueError unless the rate is in [0, 1], AND unless
    # male_gift_donation_rate + parent_offspring_share_rate <= 1.0 -- a
    # male's successful hunt applies both donations to the same gross gain
    # (not sequentially off a shrinking remainder), so an unchecked sum
    # above 1.0 could deduct more energy than the hunt actually gained.
    "parent_offspring_share_rate": 0.2,
    # Complementary diet (the point of this module, a clone of predator_sexual_reproduction; nothing in
    # that module changes). Every predator keeps its total energy (agent_energies, exactly as before) plus
    # a FRUIT STORE (the fruit-derived part of it); the MEAT STORE is the rest (E - F). Fruit adds to the
    # fruit store, prey to the meat store. Running costs (homeostatic, movement, birth) are drawn from the
    # two stores in fixed shares -- a fraction diet_meat_cost_share from meat, the rest from fruit -- so
    # neither food can stand in for the other. With diet_required a predator DIES when either store reaches
    # zero (even with plenty of total energy), and can only reproduce when both stores are at least
    # predator_min_store_fraction_for_reproduction * predator_creation_energy_threshold. Newborns and
    # founders start with diet_initial_fruit_share of their energy in the fruit store. Set
    # diet_required=False for the no-requirement control (stores still tracked and observed, but nothing
    # is gated on them). Both stores are visible: channel 1 shows total energy, channel 5 the fruit store.
    # Scripted prey: prey are moved by a fixed rule inside the env instead of by a learned policy (flee a predator
    # within prey_flee_radius, else walk to the nearest visible grass). They then need no observations or policy,
    # which removes most of the learner's data (prey are ~80% of sampled agent steps). Default False = learned prey.
    "scripted_prey": True,
    "prey_flee_radius": 2,
    "diet_required": True,
    "diet_meat_cost_share": 0.25,
    "diet_initial_fruit_share": 0.5,
    "predator_min_store_fraction_for_reproduction": 0.25,
    # Reciprocal exchange (the male's meat gift, male_gift_donation_rate above, has a fruit counterpart):
    # a female passes this fraction of any fruit she eats to her recorded mate (within predator_gift_range).
    # Mechanical, exclusive to the recorded mate, same credit-assignment rationale. 0.0 = no fruit gift.
    # __init__ raises ValueError unless in [0, 1] and female_gift_donation_rate +
    # parent_offspring_share_rate <= 1.0.
    "female_gift_donation_rate": 0.0,
    # Birth cost is split asymmetrically between parents, modeled on parental
    # investment theory (Trivers, 1972): the female bears the larger share of
    # the shared reproductive cost because she is also the structurally
    # riskier forager here (see prey_vs_predator_female_* below -- her
    # hunting attempts have a much higher death probability and lower success
    # probability than the male's). Unlike an even 50/50 split, this
    # deliberately introduces a second, related asymmetry between the sexes
    # rather than isolating mate-finding as the sole new variable under test.
    # __init__ raises ValueError unless the two shares are non-negative and
    # sum to exactly 1.0 (an invalid split would silently create or destroy
    # energy at every birth).
    "predator_birth_cost_share_female": 0.9,
    "predator_birth_cost_share_male": 0.1,
    # Hunting is a 3-outcome stochastic contest for BOTH predator sexes (not a
    # deterministic, male-only catch): "success" (predator eats the prey,
    # exactly as a deterministic catch would have), "predator_dies" (the prey
    # is left completely unharmed; the predator itself is removed, using the
    # same bookkeeping as starvation), or "failure" (nothing happens -- same
    # as an ordinary non-engaged step). Failure probability is implied
    # (1 - success - death), not stored as its own key, to avoid a redundant
    # value that could silently drift out of sync with the other two.
    # predator_male is a low-risk/high-success hunter; predator_female is a
    # high-risk/low-success hunter -- this asymmetry is the mechanism meant to
    # produce emergent (not hardcoded) sex-based foraging specialization.
    # __init__ raises ValueError if success + death > 1.0 for either sex: an
    # invalid probability here is a correctness bug (it implies a negative
    # failure probability and silently wrong simulation dynamics), not just a
    # tuning choice like this module's otherwise-unvalidated population-size
    # settings, so it gets an explicit invariant check.
    "prey_vs_predator_male_success_prob": 0.90,
    "prey_vs_predator_male_death_prob": 0.05,
    "prey_vs_predator_female_success_prob": 0.20,
    "prey_vs_predator_female_death_prob": 0.10,
    #
    # Learning agents. IDs are never reused within an episode (RLlib requires
    # each agent-ID string to map to exactly one trajectory per episode), so
    # these must comfortably cover cumulative births over a whole episode, not
    # just concurrent population size.
    "n_possible_predator_male": 1000,
    "n_possible_predator_female": 1000,
    "n_possible_prey": 2000,
    # Doubled from 3/3 to 6/6 (2026-09-18): a small starting population
    # means one sex hitting zero (which ends the episode -- reproduction
    # structurally needs both) can happen from a handful of unlucky combat
    # deaths or starvation events, especially early in training before
    # either policy has learned anything. More individuals per sex means
    # more must die simultaneously to trigger that, buying more steps of
    # experience per episode without changing any per-individual risk.
    #
    # Female bumped further, to 10 (same day): a symmetric 6/6 smoke test
    # showed predator_female reaching 0 while predator_male still had 4-6
    # left in every one of 5 random-policy seeds tried -- she absorbs both
    # the higher per-attempt combat-death risk (prey_vs_predator_female_death_prob
    # 0.10 vs male's 0.05) and the larger birth-cost share
    # (predator_birth_cost_share_female 0.9), so she's structurally the
    # bottleneck sex, not male. Biasing the starting population toward her
    # targets that asymmetry directly instead of just buying more time
    # symmetrically.
    "n_initial_active_predator_male": 6,
    "n_initial_active_predator_female": 10,
    "n_initial_active_prey": 8,
    "initial_energy_predator_male": 5.0,
    "initial_energy_predator_female": 5.0,
    "initial_energy_prey": 3.0,
    # Grass settings -- prey food only. Predators cannot eat grass in this module.
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    # Fruit settings -- predator food only (both sexes gather it; prey cannot
    # eat fruit). Same regrowth shape as grass, kept as a separate resource
    # pool/grid channel so predator and prey foraging never compete for the
    # same patches.
    "initial_num_fruit": 100,
    "initial_energy_fruit": 2.0,
    "energy_gain_per_step_fruit": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_spawning": False,
}

# --- Bands (this module) -------------------------------------------------------------------------------
# Initial structure: num_bands bands, each with `band_couples` founding couples (recorded mates) each with
# `band_children_per_couple` dependent children (alternating male, female), plus band_singles_male / _female unpaired
# adults; members start on the free cells nearest a band centre (centres spread out by farthest-point sampling).
# The default 5 x (1 couple + 2 children + 1 single male + 1 single female) = 30 predators. num_bands = 0 turns every
# band feature off (random initial layout, n_initial_active_* as configured), i.e. predator_complementary_diet.
# Sharing (mechanical, like the gifts; no reward term): a fraction band_share_rate of ANY forage (prey or fruit) is split
# equally among the forager's living band members within band_share_range; meat stays meat and fruit stays fruit.
# Mating: a male and female within mate_search_radius pair as before (any band); with kin_exclusion, parent-child and
# sibling pairs are blocked. A pairing across bands is a marriage: per marriage_rule one partner joins the other's
# band, with their unreproduced children; newborns join the father's (post-marriage) band.
# __init__ raises ValueError unless male_gift + care + band_share <= 1 and female_gift + care + band_share <= 1.
config_env.update(
    {
        "num_bands": 5,
        "band_couples": 1,
        "band_children_per_couple": 2,
        "band_singles_male": 1,
        "band_singles_female": 1,
        "band_spawn_radius": 3,
        "band_share_rate": 0.3,
        "band_share_range": 5,
        "kin_exclusion": True,
        "marriage_rule": "female_joins_male",
        "n_initial_active_prey": 40,
    }
)
