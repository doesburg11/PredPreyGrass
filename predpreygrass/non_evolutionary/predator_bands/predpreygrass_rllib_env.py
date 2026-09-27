"""
A copy of base_environment_step_energy that replaces asexual predator
reproduction with sexual (two-parent) reproduction, and splits predators into
two sexed policies with different foraging roles:

- predator_male: high-success/low-risk hunter (also gathers fruit) -- see
  prey_vs_predator_male_* in config_env.py.
- predator_female: low-success/high-risk hunter (also gathers fruit) -- see
  prey_vs_predator_female_* in config_env.py. Both sexes can attempt to hunt
  prey and both can die doing so; see _resolve_hunting_attempt. Any resulting
  specialization is meant to emerge from RL training under this risk
  asymmetry, not from a hardcoded incapability.
- prey: unchanged -- only eats grass, reproduces asexually (solo, energy-
  threshold triggered), exactly as in base_environment_step_energy.

Grass is prey-exclusive food; fruit is predator-exclusive food (both sexes).
Sexual reproduction requires a predator_male and a predator_female to each
independently clear predator_creation_energy_threshold AND be within
mate_search_radius of each other (Chebyshev distance) -- an exact-cell match
is impossible since both sexes share one grid layer and movement collision
already forbids two predators occupying the same cell. The offspring spawns
near the female and birth cost splits asymmetrically (see config_env.py for
the parental-investment rationale).

Because the female pays the larger share of birth cost but has only a weak,
shared, depleting income source (fruit) to recover with, a predator_male also
donates a fraction of each successful hunt's energy gain to HIS RECORDED MATE
ONLY (_apply_male_gift, unidirectional, mechanically executed like
eco_evolutionary_nuptial_gift's male_donation_rate, but exclusive/pair-bonded
rather than broadcast to any nearby female -- see self.agent_mate) -- meant
to offset her post-birth energy deficit.

Both parents also share a fraction of ANY successful forage (hunt or fruit)
with their own nearby living offspring (_share_energy_with_offspring, see
self.agent_parents) -- direct parental care by both sexes, not just the
mother, matching the cooperative-breeding literature's account of human
parenting as distinctive among mammals. Split evenly across however many of
a parent's own children are currently nearby (unlike the exclusive,
single-recipient mate gift). Care stops once the offspring has reproduced
itself (self.has_reproduced) -- a reproduction-based, not age-based,
independence cutoff (this module tracks no per-agent age at all). See
config_env.py for the mate-search/cost-split/gift/parental-care rationale
and RESULTS.md (once populated) for empirical findings.
"""
from predpreygrass.non_evolutionary.predator_bands.config_env import config_env

# standard library
import numbers

# external libraries
import numpy as np

from numpy.typing import NDArray
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or config_env  # Use provided config or default config_env

        self.verbose_engagement = config.get("verbose_engagement", False)
        self.verbose_movement = config.get("verbose_movement", False)
        self.verbose_spawning = config.get("verbose_spawning", False)

        self.max_steps = config.get("max_steps", 10000)

        # Rewards
        self.reward_predator_catch_prey = config.get("reward_predator_catch_prey", 0.0)
        self.reward_predator_gather_fruit = config.get("reward_predator_gather_fruit", 0.0)
        # Energy-proportional forage reward (fruit and prey alike); see config_env.py.
        self.reward_predator_per_energy = config.get("reward_predator_per_energy", 0.0)
        if not (0.0 <= self.reward_predator_per_energy < float("inf")):
            raise ValueError(
                f"reward_predator_per_energy must be finite and >= 0 (got {self.reward_predator_per_energy})"
            )
        self.reward_prey_eat_grass = config.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = config.get("reward_predator_step", 0.0)
        self.reward_prey_step = config.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = config.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator = config.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = config.get("reproduction_reward_prey", 10.0)

        # Energy settings: homeostatic (always charged) and move (charged on
        # top, only when the agent's action isn't noop) are independent,
        # additive costs -- see base_environment_step_energy for the
        # rationale. Applies uniformly to predator_male/predator_female.
        self.homeostatic_energy_cost_per_step_predator = config.get("homeostatic_energy_cost_per_step_predator", 0.10)
        self.homeostatic_energy_cost_per_step_prey = config.get("homeostatic_energy_cost_per_step_prey", 0.035)
        self.move_energy_cost_per_step_predator = config.get("move_energy_cost_per_step_predator", 0.08)
        self.move_energy_cost_per_step_prey = config.get("move_energy_cost_per_step_prey", 0.035)
        self.predator_creation_energy_threshold = config.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = config.get("prey_creation_energy_threshold", 8.0)

        # Sexual reproduction: Chebyshev-distance radius for mate-finding
        # (see config_env.py for why this can't be an exact-cell match).
        self.mate_search_radius = int(config.get("mate_search_radius", 3))

        # Predator population cap: None (default) leaves reproduction unbounded, unchanged from
        # every existing run. When set, Step 5b below stops producing predator offspring for the
        # rest of a step once the population is already at or above this ceiling -- a birth that
        # brings the population up TO the cap is still allowed, one that would push past it is not
        # (see config_env.py for the full rationale).
        self.predator_population_cap = config.get("predator_population_cap", None)
        if self.predator_population_cap is not None:
            # Codex review: int(...) before validating would silently accept 3.9 (truncated to 3),
            # True/False (bool is an int subclass), or "5" (numeric strings) despite the documented
            # "an int" contract. Reject anything that isn't already a plain int (or a numbers.Integral
            # that isn't a bool) up front instead.
            if isinstance(self.predator_population_cap, bool) or not isinstance(self.predator_population_cap, numbers.Integral):
                raise ValueError(
                    f"predator_population_cap must be an int (got {self.predator_population_cap!r})"
                )
            self.predator_population_cap = int(self.predator_population_cap)
            if self.predator_population_cap < 0:
                raise ValueError(f"predator_population_cap must be >= 0 (got {self.predator_population_cap})")

        # Male provisioning: unidirectional energy transfer from
        # predator_male to nearby predator_female neighbors on a successful
        # hunt -- offsets her post-birth energy deficit (see config_env.py
        # for the full rationale).
        self.male_gift_donation_rate = config.get("male_gift_donation_rate", 0.3)
        if not (0.0 <= self.male_gift_donation_rate <= 1.0):
            raise ValueError(f"male_gift_donation_rate must be in [0, 1] (got {self.male_gift_donation_rate})")
        self.predator_gift_range = int(config.get("predator_gift_range", 3))
        if self.predator_gift_range < 0:
            raise ValueError(f"predator_gift_range must be non-negative (got {self.predator_gift_range})")

        # Parental care: both parents share a fraction of any successful
        # forage (hunt or fruit) with their own nearby living offspring
        # (self.agent_parents), reusing predator_gift_range for proximity --
        # see config_env.py for the full rationale.
        self.parent_offspring_share_rate = config.get("parent_offspring_share_rate", 0.2)
        if not (0.0 <= self.parent_offspring_share_rate <= 1.0):
            raise ValueError(
                f"parent_offspring_share_rate must be in [0, 1] (got {self.parent_offspring_share_rate})"
            )
        # A male's successful hunt applies both donations to the SAME gross
        # gain (not sequentially off a shrinking remainder), so their sum
        # must not exceed 1.0 -- otherwise a hunt could deduct more energy
        # than it gained, pushing the male negative regardless of how much
        # energy he had banked beforehand.
        if self.male_gift_donation_rate + self.parent_offspring_share_rate > 1.0:
            raise ValueError(
                "male_gift_donation_rate + parent_offspring_share_rate must be <= 1.0 "
                f"(got {self.male_gift_donation_rate} + {self.parent_offspring_share_rate} "
                f"= {self.male_gift_donation_rate + self.parent_offspring_share_rate})"
            )

        # Complementary diet (this module's addition; see config_env.py for the rationale). Every predator keeps
        # its total energy (agent_energies, exactly as in predator_sexual_reproduction) PLUS a fruit store
        # (agent_fruit_store, the part of that energy that came from fruit); the meat store is the remainder,
        # E - F. Running costs are drawn from the two stores in fixed shares, so neither food can substitute
        # for the other, and a predator dies when EITHER store reaches zero (diet_required).
        # Scripted (non-learning) prey: when True the prey are moved by a fixed rule inside the env (flee nearby
        # predators, otherwise walk to the nearest grass), are NOT in self.agents / possible_agents, and get no
        # observations, rewards or policy -- most agent steps in this ecology are prey steps, so this removes the bulk
        # of the learner's data. Everything else about prey (energy, grazing, hunting, reproduction) is unchanged.
        self.scripted_prey = bool(config.get("scripted_prey", False))
        self.prey_flee_radius = int(config.get("prey_flee_radius", 2))
        self._scripted_prey_ids: List[AgentID] = []
        # Richer prey (default 1.0 = the old behaviour): a predator that catches prey gains prey_energy_yield x the prey's energy. It changes only what
        # a catch is worth (energy, reward and shared meat), not the prey's own ecology, so prey can be made a larger share of predator food
        # without raising the meat cost share of the diet.
        self.prey_energy_yield = float(config.get("prey_energy_yield", 1.0))
        if self.prey_energy_yield < 0:
            raise ValueError(f"prey_energy_yield must be >= 0 (got {self.prey_energy_yield})")
        self.diet_required = bool(config.get("diet_required", True))
        self.diet_meat_cost_share = config.get("diet_meat_cost_share", 0.25)
        self.diet_initial_fruit_share = config.get("diet_initial_fruit_share", 0.5)
        self.predator_min_store_fraction_for_reproduction = config.get(
            "predator_min_store_fraction_for_reproduction", 0.25
        )
        for _name in ("diet_meat_cost_share", "diet_initial_fruit_share", "predator_min_store_fraction_for_reproduction"):
            if not (0.0 <= getattr(self, _name) <= 1.0):
                raise ValueError(f"{_name} must be in [0, 1] (got {getattr(self, _name)})")
        # Reciprocal exchange: a female donates female_gift_donation_rate of any fruit she eats to her recorded
        # mate (within predator_gift_range), mirroring the male's meat gift (male_gift_donation_rate).
        self.female_gift_donation_rate = config.get("female_gift_donation_rate", 0.3)
        if not (0.0 <= self.female_gift_donation_rate <= 1.0):
            raise ValueError(f"female_gift_donation_rate must be in [0, 1] (got {self.female_gift_donation_rate})")
        # A female's fruit gift and her parental-care share apply to the same gross gain (not off a shrinking
        # remainder), so their sum must not exceed 1.0 -- checked after parent_offspring_share_rate is read.

        if self.female_gift_donation_rate + self.parent_offspring_share_rate > 1.0:
            raise ValueError(
                "female_gift_donation_rate + parent_offspring_share_rate must be <= 1.0 "
                f"(got {self.female_gift_donation_rate} + {self.parent_offspring_share_rate})"
            )

        # Birth cost split: parental-investment-theory rationale (Trivers,
        # 1972) -- the female also bears the larger share of the shared
        # reproduction cost, consistent with her being the structurally
        # riskier forager (see prey_vs_predator_female_* below).
        self.predator_birth_cost_share_female = config.get("predator_birth_cost_share_female", 0.9)
        self.predator_birth_cost_share_male = config.get("predator_birth_cost_share_male", 0.1)
        if (
            not (0.0 <= self.predator_birth_cost_share_female <= 1.0)
            or not (0.0 <= self.predator_birth_cost_share_male <= 1.0)
            or abs(self.predator_birth_cost_share_female + self.predator_birth_cost_share_male - 1.0) > 1e-9
        ):
            raise ValueError(
                "predator_birth_cost_share_female/_male must be non-negative and sum to exactly 1.0 "
                f"(got female={self.predator_birth_cost_share_female}, male={self.predator_birth_cost_share_male})"
            )

        self.penalty_predator_death_in_combat = config.get("penalty_predator_death_in_combat", 0.0)

        # Hunting is a 3-outcome stochastic contest for BOTH sexes (not a
        # deterministic, male-only catch): "success" (kill, as a deterministic
        # catch would have), "predator_dies" (the prey is left completely
        # unharmed; the predator itself is removed), or "failure" (nothing
        # happens). Failure probability is implied (1 - success - death), not
        # stored as its own key, to avoid a redundant value that could
        # silently drift out of sync with the other two. See config_env.py
        # for the full rationale.
        self.prey_vs_predator_male_success_prob = config.get("prey_vs_predator_male_success_prob", 0.90)
        self.prey_vs_predator_male_death_prob = config.get("prey_vs_predator_male_death_prob", 0.05)
        self.prey_vs_predator_female_success_prob = config.get("prey_vs_predator_female_success_prob", 0.20)
        self.prey_vs_predator_female_death_prob = config.get("prey_vs_predator_female_death_prob", 0.10)
        for _sex, _success, _death in (
            ("predator_male", self.prey_vs_predator_male_success_prob, self.prey_vs_predator_male_death_prob),
            ("predator_female", self.prey_vs_predator_female_success_prob, self.prey_vs_predator_female_death_prob),
        ):
            if not (0.0 <= _success <= 1.0) or not (0.0 <= _death <= 1.0) or _success + _death > 1.0:
                raise ValueError(
                    f"{_sex} hunting probabilities must be non-negative and sum to <= 1.0 "
                    f"(got success={_success}, death={_death})"
                )

        # Learning agents
        self.n_possible_predator_male = config.get("n_possible_predator_male", 1000)
        self.n_possible_predator_female = config.get("n_possible_predator_female", 1000)
        self.n_possible_prey = config.get("n_possible_prey", 2000)
        self.n_initial_active_predator_male = config.get("n_initial_active_predator_male", 3)
        self.n_initial_active_predator_female = config.get("n_initial_active_predator_female", 3)
        self.n_initial_active_prey = config.get("n_initial_active_prey", 8)

        # Bands (this module's addition; see config_env.py). num_bands = 0 turns every band feature off and gives the
        # predator_complementary_diet behaviour (random initial layout, n_initial_active_* as configured).
        self.num_bands = int(config.get("num_bands", 0))
        self.band_couples = int(config.get("band_couples", 1))
        self.band_children_per_couple = int(config.get("band_children_per_couple", 2))
        self.band_singles_male = int(config.get("band_singles_male", 1))
        self.band_singles_female = int(config.get("band_singles_female", 1))
        self.band_spawn_radius = int(config.get("band_spawn_radius", 3))
        self.band_share_rate = config.get("band_share_rate", 0.3)
        # Separate sharing rates by food type (real bands share meat far more widely than gathered plant food). Each
        # defaults to band_share_rate when not given, so the single-rate configs behave exactly as before.
        _meat_rate, _fruit_rate = config.get("band_meat_share_rate"), config.get("band_fruit_share_rate")
        self.band_meat_share_rate = self.band_share_rate if _meat_rate is None else _meat_rate
        self.band_fruit_share_rate = self.band_share_rate if _fruit_rate is None else _fruit_rate
        for _name in ("band_meat_share_rate", "band_fruit_share_rate"):
            if not (0.0 <= getattr(self, _name) <= 1.0):
                raise ValueError(f"{_name} must be in [0, 1] (got {getattr(self, _name)})")
        self.band_share_range = int(config.get("band_share_range", 5))
        # Distance decay of band sharing (0 = off, the flat equal split). Each in-range recipient's equal share is scaled by
        # 1 - band_share_distance_decay * d / (band_share_range + 1) (d = Chebyshev distance to the forager), and the donor pays
        # only what is delivered, so being far from a forager costs energy without any hard cutoff being crossed.
        self.band_share_distance_decay = config.get("band_share_distance_decay", 0.0)
        if not (0.0 <= self.band_share_distance_decay <= 1.0):
            raise ValueError(f"band_share_distance_decay must be in [0, 1] (got {self.band_share_distance_decay})")
        self.kin_exclusion = bool(config.get("kin_exclusion", True))
        self.band_compass = bool(config.get("band_compass", False))  # see the observation settings below
        # Band fission / fusion / drift-out (all default off): membership follows co-residence. See config_env.py and BANDS_DESIGN.md s.9.
        self.band_fission = bool(config.get("band_fission", False))
        self.band_fusion = bool(config.get("band_fusion", False))
        self.band_max_size = int(config.get("band_max_size", 12))
        self.band_min_split_size = int(config.get("band_min_split_size", 3))
        self.band_fuse_distance = int(config.get("band_fuse_distance", 3))
        self.band_fuse_steps = int(config.get("band_fuse_steps", 30))
        self.band_drift_steps = int(config.get("band_drift_steps", 0))
        self.band_drift_join = bool(config.get("band_drift_join", False))
        self.band_check_interval = int(config.get("band_check_interval", 10))
        if self.band_fission or self.band_fusion or self.band_drift_steps > 0:
            if self.num_bands == 0:
                raise ValueError("band fission / fusion / drift-out need num_bands > 0")
            if self.band_check_interval < 1 or self.band_max_size < 2 or self.band_min_split_size < 1:
                raise ValueError("band_check_interval >= 1, band_max_size >= 2 and band_min_split_size >= 1 are required")
            if self.band_fission and 2 * self.band_min_split_size > self.band_max_size + 1:
                raise ValueError("band_min_split_size is too large for band_max_size (a split must leave two viable parts)")
            if min(self.band_fuse_distance, self.band_fuse_steps, self.band_drift_steps) < 0:
                raise ValueError("band_fuse_distance, band_fuse_steps and band_drift_steps must be >= 0")
        # Threats (roaming non-learning animals that kill predators; group defense makes lone wandering dangerous). num_threats = 0
        # turns the whole feature off. See config_env.py.
        self.num_threats = int(config.get("num_threats", 0))
        self.threat_sense_radius = int(config.get("threat_sense_radius", 4))
        self.threat_kill_prob = config.get("threat_kill_prob", 0.5)
        self.threat_defense_radius = int(config.get("threat_defense_radius", 2))
        self.threat_defenders_to_repel = int(config.get("threat_defenders_to_repel", 3))
        self.threat_defense_by = config.get("threat_defense_by", "band")
        self.threat_flee_distance = int(config.get("threat_flee_distance", 8))
        # False (default, the old behaviour): a threat attacks only the single nearest adjacent predator. True: it attacks every
        # viable predator within distance 1 of it at once (see _threats_act) -- several band-mates standing together next to a
        # threat all face the kill roll in the same step and can be each other's defenders, instead of only the nearest one
        # being at risk. Tests whether a threat that menaces the whole group at once (not one member at a time) is what a
        # learned pull toward band-mates needs (see RESULTS.md's common-enemy-effect discussion).
        self.threat_attack_all_adjacent = bool(config.get("threat_attack_all_adjacent", False))
        # Rest after an attack (0 = off, the old behaviour): a threat that KILLS is sated and does not attack for
        # threat_satiation_steps; one whose attack does not kill (survived roll, or driven off) does not attack again for
        # threat_cooldown_steps. Resting threats wander and never chase.
        self.threat_satiation_steps = int(config.get("threat_satiation_steps", 0))
        self.threat_cooldown_steps = int(config.get("threat_cooldown_steps", 0))
        if min(self.threat_satiation_steps, self.threat_cooldown_steps) < 0:
            raise ValueError("threat_satiation_steps and threat_cooldown_steps must be >= 0")
        self.threat_rest_until: Dict[str, int] = {}
        # Free-rider punishment (band_ostracism, default off). threat_defense_radius (checked by _defender_ids) is wider than
        # the distance-1 range a threat actually attacks at, so a predator can sit just inside the defense radius but outside
        # the attack range and get full credit as a "defender" at zero personal risk (see RESULTS.md's free-rider discussion).
        # This tracks, per predator, how many times it was counted as a defender in a genuine attack (an "opportunity") versus
        # how many of those times it was itself within distance 1 of the threat ("exposure"). Once a predator has at least
        # ostracism_min_opportunities recorded and its exposure ratio is below ostracism_exposure_threshold, it is ostracized
        # for ostracism_duration steps: excluded from counting as anyone's defender (_defender_ids) and from receiving band
        # shares (_apply_band_share); its counts then reset so judgment restarts fresh once the ostracism ends. Modeled loosely
        # on the reputation-based sanctioning (ridicule, exclusion) anthropologists (Boehm) describe enforcing real
        # hunter-gatherer sharing norms; mechanically executed like every other designed payoff here, not a learned
        # "ostracize" action, for the same credit-assignment reasons as the male/female gifts.
        self.band_ostracism = bool(config.get("band_ostracism", False))
        self.ostracism_min_opportunities = int(config.get("ostracism_min_opportunities", 5))
        self.ostracism_exposure_threshold = float(config.get("ostracism_exposure_threshold", 0.34))
        self.ostracism_duration = int(config.get("ostracism_duration", 200))
        if self.ostracism_min_opportunities < 1:
            raise ValueError("ostracism_min_opportunities must be >= 1")
        if not (0.0 <= self.ostracism_exposure_threshold <= 1.0):
            raise ValueError(f"ostracism_exposure_threshold must be in [0, 1] (got {self.ostracism_exposure_threshold})")
        if self.ostracism_duration < 0:
            raise ValueError("ostracism_duration must be >= 0")
        self.defense_opportunities: Dict[str, int] = {}
        self.defense_exposures: Dict[str, int] = {}
        self.ostracized_until: Dict[str, int] = {}
        if self.num_threats < 0:
            raise ValueError(f"num_threats must be >= 0 (got {self.num_threats})")
        if not (0.0 <= self.threat_kill_prob <= 1.0):
            raise ValueError(f"threat_kill_prob must be in [0, 1] (got {self.threat_kill_prob})")
        if self.threat_defenders_to_repel < 1:
            raise ValueError("threat_defenders_to_repel must be >= 1")
        if min(self.threat_sense_radius, self.threat_defense_radius, self.threat_flee_distance) < 0:
            raise ValueError("threat radii and distances must be non-negative")
        if self.num_threats > 0 and self.threat_flee_distance > config.get("grid_size", 25) - 1:
            raise ValueError("threat_flee_distance cannot exceed grid_size - 1")
        if self.threat_defense_by not in ("band", "any"):
            raise ValueError(f"threat_defense_by must be 'band' or 'any' (got {self.threat_defense_by!r})")
        self.threat_positions: Dict[str, Tuple[int, int]] = {}
        # Mammoths (group big game; num_mammoths = 0 turns it off). See config_env.py and MAMMOTH_DESIGN.md.
        self.num_mammoths = int(config.get("num_mammoths", 0))
        self.mammoth_energy = float(config.get("mammoth_energy", 20.0))
        self.mammoth_move_prob = config.get("mammoth_move_prob", 0.3)
        self.mammoth_respawn_steps = int(config.get("mammoth_respawn_steps", 80))
        self.mammoth_party_radius = int(config.get("mammoth_party_radius", 1))
        self.mammoth_success_by_party = list(config.get("mammoth_success_by_party", [0.02, 0.25, 0.6, 0.9]))
        self.mammoth_death_by_party = list(config.get("mammoth_death_by_party", [0.30, 0.15, 0.05, 0.02]))
        if self.num_mammoths < 0 or self.mammoth_energy <= 0 or self.mammoth_respawn_steps < 0 or self.mammoth_party_radius < 0:
            raise ValueError("num_mammoths, mammoth_respawn_steps and mammoth_party_radius must be >= 0 and mammoth_energy > 0")
        if not (0.0 <= self.mammoth_move_prob <= 1.0):
            raise ValueError(f"mammoth_move_prob must be in [0, 1] (got {self.mammoth_move_prob})")
        for name in ("mammoth_success_by_party", "mammoth_death_by_party"):
            values = getattr(self, name)
            if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
                raise ValueError(f"{name} must be 4 probabilities in [0, 1] (party sizes 1, 2, 3, 4+)")
        if self.num_mammoths > 0 and self.num_bands == 0:
            raise ValueError("mammoth hunts are band hunts: num_mammoths > 0 needs num_bands > 0")
        self.mammoth_positions: Dict[str, Tuple[int, int]] = {}
        self.mammoth_respawn_at: Dict[str, int] = {}
        self._pending_rewards: Dict[str, float] = {}
        # Channels: 8 base, +4 with band_compass, +1 with threats (the threat channel comes last).
        required_channels = 8 + (4 if self.band_compass else 0) + (1 if self.num_threats > 0 else 0) + (1 if self.num_mammoths > 0 else 0)
        if config.get("num_obs_channels") is not None and config["num_obs_channels"] < required_channels:
            raise ValueError(
                f"num_obs_channels must be >= {required_channels} for this configuration (8, +4 with band_compass, +1 with threats, +1 with mammoths)"
            )
        self.marriage_rule = config.get("marriage_rule", "female_joins_male")
        if self.num_bands < 0:
            raise ValueError(f"num_bands must be >= 0 (got {self.num_bands})")
        if not (0.0 <= self.band_share_rate <= 1.0):
            raise ValueError(f"band_share_rate must be in [0, 1] (got {self.band_share_rate})")
        if self.band_share_range < 0 or self.band_spawn_radius < 0:
            raise ValueError("band_share_range and band_spawn_radius must be non-negative")
        if min(self.band_couples, self.band_children_per_couple, self.band_singles_male, self.band_singles_female) < 0:
            raise ValueError("band composition counts must be non-negative")
        if self.marriage_rule not in ("female_joins_male", "male_joins_female"):
            raise ValueError(f"marriage_rule must be 'female_joins_male' or 'male_joins_female' (got {self.marriage_rule!r})")
        if self.num_bands > 0:
            kids_male = (self.band_children_per_couple + 1) // 2  # children alternate male, female, male, ...
            kids_female = self.band_children_per_couple // 2
            self.n_initial_active_predator_male = self.num_bands * (
                self.band_couples * (1 + kids_male) + self.band_singles_male
            )
            self.n_initial_active_predator_female = self.num_bands * (
                self.band_couples * (1 + kids_female) + self.band_singles_female
            )
        self.agent_band: Dict[AgentID, int] = {}
        if self.num_bands > 0:
            if self.n_initial_active_predator_male > config.get("n_possible_predator_male", 1000) or (
                self.n_initial_active_predator_female > config.get("n_possible_predator_female", 1000)
            ):
                raise ValueError("band-derived initial predator counts exceed n_possible_predator_male/female")
            cells_needed = (
                self.n_initial_active_predator_male
                + self.n_initial_active_predator_female
                + self.n_initial_active_prey
                + config.get("initial_num_grass", 100)
                + config.get("initial_num_fruit", 100)
                + self.num_threats
                + self.num_mammoths
            )
            if cells_needed > config.get("grid_size", 25) ** 2:
                raise ValueError(f"the initial layout needs {cells_needed} cells but the grid has fewer")
        # Every donation from one gain is a share of that same gross gain, so they must not sum past 1.0.
        active_band_share = (
            max(self.band_meat_share_rate, self.band_fruit_share_rate) if self.num_bands > 0 else 0.0
        )  # inactive when bands are off; the larger of the two rates bounds the donations from one gain
        if self.male_gift_donation_rate + self.parent_offspring_share_rate + active_band_share > 1.0:
            raise ValueError("male_gift_donation_rate + parent_offspring_share_rate + band_share_rate must be <= 1.0")
        if self.female_gift_donation_rate + self.parent_offspring_share_rate + active_band_share > 1.0:
            raise ValueError("female_gift_donation_rate + parent_offspring_share_rate + band_share_rate must be <= 1.0")

        self.initial_energy_predator_male = config.get("initial_energy_predator_male", 5.0)
        self.initial_energy_predator_female = config.get("initial_energy_predator_female", 5.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 3.0)

        # Grid and Observation Settings
        self.grid_size = config.get("grid_size", 25)
        # 8 channels: Border, Predator (total energy), Prey, Grass, Fruit, Fruit store (of predators),
        # Same-band predators, Other-band predators. With band_compass, 4 more constant planes (channels 8-11) tell a predator where its
        # nearest band-mate is even when that band-mate is outside the observation window: [has band-mate, (dx + 1) / 2, (dy + 1) / 2,
        # Chebyshev distance / grid_size], dx/dy being the unit vector toward the nearest same-band predator (zeros if there is none).
        self.band_compass = bool(config.get("band_compass", False))
        self.num_obs_channels = config.get("num_obs_channels")  # None/absent = auto
        if self.num_obs_channels is None:
            self.num_obs_channels = (
                8 + (4 if self.band_compass else 0) + (1 if self.num_threats > 0 else 0) + (1 if self.num_mammoths > 0 else 0)
            )
        self.predator_obs_range = config.get("predator_obs_range", 7)
        self.prey_obs_range = config.get("prey_obs_range", 9)

        # Grass settings (prey food only)
        self.initial_num_grass = config.get("initial_num_grass", 100)
        self.initial_energy_grass = config.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = config.get("energy_gain_per_step_grass", 0.04)

        # Fruit settings (predator food only, both sexes)
        self.initial_num_fruit = config.get("initial_num_fruit", 100)
        self.initial_energy_fruit = config.get("initial_energy_fruit", 2.0)
        self.energy_gain_per_step_fruit = config.get("energy_gain_per_step_fruit", 0.04)

        self.cumulative_rewards = {}  # Track total rewards per agent

        # Mate tracking for exclusive (pair-bonded) male provisioning -- see
        # _apply_male_gift. Bidirectional: agent_mate[male] = female and
        # agent_mate[female] = male, set on a successful reproduction event
        # and overwritten on any later one (serial monogamy: the most recent
        # partner, not lifetime exclusivity). No entry until an agent has
        # reproduced at least once.
        self.agent_mate: Dict[AgentID, AgentID] = {}
        # Fruit store per predator (the fruit-derived part of agent_energies); see the diet block above.
        self.agent_fruit_store: Dict[AgentID, float] = {}

        # Lineage tracking for parental care -- see
        # _share_energy_with_offspring. child -> (father, mother), set once
        # at birth, never updated or cleaned up on death: a dead child
        # simply never appears in predator_positions again, so a stale
        # entry here is inert rather than a dangling reference (unlike
        # agent_mate, which is looked up FROM the parent and therefore
        # needed the remating fix).
        self.agent_parents: Dict[AgentID, Tuple[AgentID, AgentID]] = {}

        # Reproduction-based independence cutoff for parental care: once an
        # agent has reproduced itself, it's a breeding adult, not a
        # dependent offspring, so _share_energy_with_offspring stops paying
        # it out regardless of proximity to its own parents. Set once,
        # alongside agent_mate/agent_parents, and never removed -- unlike
        # agent_mate (which the remating fix can clear from an abandoned
        # partner), "has this individual ever reproduced" is permanent, so
        # it needs its own flag rather than reusing agent_mate membership.
        self.has_reproduced: set[AgentID] = set()

        self._pending_removal: List[AgentID] = []
        self._next_predator_male_idx = self.n_initial_active_predator_male
        self._next_predator_female_idx = self.n_initial_active_predator_female
        self._next_prey_idx = self.n_initial_active_prey

        self.possible_agents: List[AgentID] = (
            [f"predator_male_{i}" for i in range(self.n_possible_predator_male)]
            + [f"predator_female_{i}" for i in range(self.n_possible_predator_female)]
            + ([] if self.scripted_prey else [f"prey_{j}" for j in range(self.n_possible_prey)])
        )
        self.agents: List[AgentID] = (
            [f"predator_male_{i}" for i in range(self.n_initial_active_predator_male)]
            + [f"predator_female_{i}" for i in range(self.n_initial_active_predator_female)]
            + ([] if self.scripted_prey else [f"prey_{j}" for j in range(self.n_initial_active_prey)])
        )

        # Non-learning agents (grass, fruit); not included in 'possible_agents' or 'agents'
        self.grass_agents: List[AgentID] = [f"grass_{k}" for k in range(self.initial_num_grass)]
        self.fruit_agents: List[AgentID] = [f"fruit_{k}" for k in range(self.initial_num_fruit)]

        # Gymnasium Spaces
        predator_obs_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        prey_obs_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)

        predator_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=predator_obs_shape, dtype=np.float64)
        prey_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=prey_obs_shape, dtype=np.float64)

        # Assign spaces based on agent type (both predator sexes share the
        # same observation space; only the species -- predator vs prey --
        # determines the space, same convention as base_environment_step_energy).
        self.observation_spaces = {
            agent: predator_obs_space if "predator" in agent else prey_obs_space for agent in self.possible_agents
        }

        self.action_to_move_tuple: Dict[int, Tuple[int, int]] = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1),
        }
        self.noop_action_id = next(a for a, move in self.action_to_move_tuple.items() if move == (0, 0))
        action_space = gymnasium.spaces.Discrete(len(self.action_to_move_tuple))
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

        # Initialize grid_world_state and agent positions
        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.grass_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.fruit_positions: Dict[AgentID, Tuple[int, int]] = {}

        self.agent_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}
        self.fruit_energies: Dict[AgentID, float] = {}
        self.grid_world_state_shape: Tuple[int, int, int] = (
            self.num_obs_channels,
            self.grid_size,
            self.grid_size,
        )
        self.initial_grid_world_state: NDArray[np.float64] = np.zeros(self.grid_world_state_shape, dtype=np.float64)
        self.grid_world_state: NDArray[np.float64] = self.initial_grid_world_state.copy()
        self.num_actions = len(self.action_to_move_tuple)
        self.agents_just_ate = set()  # agent_id → shows green ring this step

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        # Per the Gymnasium reset(seed=...) contract: only reseed when an
        # explicit seed is given -- see base_environment_step_energy for why.
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)

        self.grid_world_state = self.initial_grid_world_state.copy()

        self.possible_agents: List[AgentID] = (
            [f"predator_male_{i}" for i in range(self.n_possible_predator_male)]
            + [f"predator_female_{i}" for i in range(self.n_possible_predator_female)]
            + ([] if self.scripted_prey else [f"prey_{j}" for j in range(self.n_possible_prey)])
        )
        male_ids = [f"predator_male_{i}" for i in range(self.n_initial_active_predator_male)]
        female_ids = [f"predator_female_{i}" for i in range(self.n_initial_active_predator_female)]
        prey_ids = [f"prey_{j}" for j in range(self.n_initial_active_prey)]
        self.agents = male_ids + female_ids + ([] if self.scripted_prey else prey_ids)
        self._scripted_prey_ids = list(prey_ids) if self.scripted_prey else []

        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.agent_energies: Dict[AgentID, float] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]] = {}

        self.cumulative_rewards: Dict[AgentID, float] = {agent_id: 0 for agent_id in self.agents + self._scripted_prey_ids}

        self.agent_mate: Dict[AgentID, AgentID] = {}
        self.agent_fruit_store: Dict[AgentID, float] = {}
        self.agent_band: Dict[AgentID, int] = {}
        self.agent_parents: Dict[AgentID, Tuple[AgentID, AgentID]] = {}
        self.has_reproduced: set[AgentID] = set()

        self._pending_removal = []
        self._next_predator_male_idx = self.n_initial_active_predator_male
        self._next_predator_female_idx = self.n_initial_active_predator_female
        self._next_prey_idx = self.n_initial_active_prey

        self.episode_births = {"predator_male": 0, "predator_female": 0, "prey": 0}
        self.episode_deaths = {"predator_male": 0, "predator_female": 0, "prey": 0}

        # Training-time observability for the hunting/provisioning
        # mechanics -- surfaced via _build_episode_training_metrics, since
        # episode_births/episode_deaths alone can't distinguish "no
        # reproduction because no one hunts" from "no reproduction despite
        # hunting succeeding" from "hunting succeeds but mates never meet."
        self.hunting_attempts = {"predator_male": 0, "predator_female": 0}
        self.hunting_successes = {"predator_male": 0, "predator_female": 0}
        # Deaths specifically caused by a failed hunting attempt (a subset
        # of episode_deaths, which doesn't distinguish cause -- the
        # remainder, episode_deaths[sex] - hunting_deaths[sex], is
        # starvation).
        self.hunting_deaths = {"predator_male": 0, "predator_female": 0}
        self.mate_gift_events = 0
        self.mate_gift_energy_total = 0.0
        self.female_gift_events = 0
        self.female_gift_energy_total = 0.0
        self.parental_care_events = 0
        self.parental_care_energy_total = 0.0
        # Band mechanics: sharing within a band, pairings (cross-band ones are marriages), and pairings the kin rule blocked.
        self.band_share_events = 0
        self.band_share_meat_total = 0.0
        self.band_share_fruit_total = 0.0
        self.marriages = 0
        self.within_band_pairings = 0
        self.kin_blocked_checks = 0
        # Threat mechanics: encounters (a threat adjacent to a predator), kills by sex, kills of a predator with no defender, repelled.
        self.threat_encounters = 0
        self.threat_kills = {"predator_male": 0, "predator_female": 0}
        self.threat_kills_alone = 0
        self.threat_repelled = 0
        self.ostracism_events = 0
        # Deaths from a diet deficiency (a store hit zero while total energy was still positive), by sex and store.
        self.diet_deaths = {
            "predator_male": {"fruit": 0, "meat": 0},
            "predator_female": {"fruit": 0, "meat": 0},
        }

        def generate_random_positions(grid_size: int, num_positions: int, exclude=frozenset()):
            if num_positions + len(exclude) > grid_size * grid_size:
                raise ValueError("Cannot place more unique positions than grid cells.")
            positions = set()
            while len(positions) < num_positions:
                pos = tuple(self.rng.integers(0, grid_size, size=2))
                if pos not in exclude:
                    positions.add(pos)
            return list(positions)

        if self.num_bands > 0:
            layout = self._build_band_layout(male_ids, female_ids)
            self.agent_band = dict(layout["band"])
            self.agent_mate.update(layout["mates"])
            self.agent_parents.update(layout["parents"])
            self.has_reproduced.update(layout["mates"])  # the founding couples are breeding adults
            other_positions = generate_random_positions(
                self.grid_size,
                self.n_initial_active_prey + self.initial_num_grass + self.initial_num_fruit,
                exclude=set(layout["positions"].values()),
            )
            predator_male_positions = [layout["positions"][a] for a in male_ids]
            predator_female_positions = [layout["positions"][a] for a in female_ids]
            idx = 0
            prey_positions = other_positions[idx : idx + self.n_initial_active_prey]
            idx += self.n_initial_active_prey
            grass_positions = other_positions[idx : idx + self.initial_num_grass]
            idx += self.initial_num_grass
            fruit_positions = other_positions[idx : idx + self.initial_num_fruit]
        else:
            total_entities = (
                self.n_initial_active_predator_male
                + self.n_initial_active_predator_female
                + self.n_initial_active_prey
                + self.initial_num_grass
                + self.initial_num_fruit
            )
            all_positions = generate_random_positions(self.grid_size, total_entities)

            idx = 0
            predator_male_positions = all_positions[idx : idx + self.n_initial_active_predator_male]
            idx += self.n_initial_active_predator_male
            predator_female_positions = all_positions[idx : idx + self.n_initial_active_predator_female]
            idx += self.n_initial_active_predator_female
            prey_positions = all_positions[idx : idx + self.n_initial_active_prey]
            idx += self.n_initial_active_prey
            grass_positions = all_positions[idx : idx + self.initial_num_grass]
            idx += self.initial_num_grass
            fruit_positions = all_positions[idx : idx + self.initial_num_fruit]
            idx += self.initial_num_fruit

        for i, agent in enumerate(male_ids):
            self.agent_positions[agent] = predator_male_positions[i]
            self.predator_positions[agent] = predator_male_positions[i]
            self.agent_energies[agent] = self.initial_energy_predator_male
            self.agent_fruit_store[agent] = self.initial_energy_predator_male * self.diet_initial_fruit_share
            self.grid_world_state[1, *predator_male_positions[i]] = self.initial_energy_predator_male
        for i, agent in enumerate(female_ids):
            self.agent_positions[agent] = predator_female_positions[i]
            self.predator_positions[agent] = predator_female_positions[i]
            self.agent_energies[agent] = self.initial_energy_predator_female
            self.agent_fruit_store[agent] = self.initial_energy_predator_female * self.diet_initial_fruit_share
            self.grid_world_state[1, *predator_female_positions[i]] = self.initial_energy_predator_female
        for i, agent in enumerate(prey_ids):
            self.agent_positions[agent] = prey_positions[i]
            self.prey_positions[agent] = prey_positions[i]
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[2, *prey_positions[i]] = self.initial_energy_prey

        self.grass_positions = {}
        self.grass_energies = {}
        for i, grass in enumerate(self.grass_agents):
            self.grass_positions[grass] = grass_positions[i]
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *grass_positions[i]] = self.initial_energy_grass

        self.fruit_positions = {}
        self.fruit_energies = {}
        for i, fruit in enumerate(self.fruit_agents):
            self.fruit_positions[fruit] = fruit_positions[i]
            self.fruit_energies[fruit] = self.initial_energy_fruit
            self.grid_world_state[4, *fruit_positions[i]] = self.initial_energy_fruit

        self._next_band_id = self.num_bands
        self._band_contact: Dict[Tuple[int, int], int] = {}
        self._band_out_steps: Dict[str, int] = {}
        self.band_fissions = 0
        self.band_fusions = 0
        self.band_drift_outs = 0
        self.threat_positions = {}
        self.threat_rest_until = {}
        self.defense_opportunities = {}
        self.defense_exposures = {}
        self.ostracized_until = {}
        self.mammoth_positions = {}
        self.mammoth_respawn_at = {}
        self._pending_rewards = {}
        # Mammoth counters by party size 1, 2, 3, 4+ (index 0..3)
        self.mammoth_attempts = [0, 0, 0, 0]
        self.mammoth_kills = [0, 0, 0, 0]
        self.mammoth_party_deaths = [0, 0, 0, 0]
        self.mammoth_blocked = 0
        self.mammoth_energy_distributed = 0.0
        if self.num_threats > 0:
            taken = set(self.agent_positions.values())
            for i in range(self.num_threats):
                while True:
                    pos = tuple(int(v) for v in self.rng.integers(0, self.grid_size, size=2))
                    if pos not in taken:
                        break
                taken.add(pos)
                self.threat_positions[f"threat_{i}"] = pos

        if self.num_mammoths > 0:
            taken = set(self.agent_positions.values()) | set(self.threat_positions.values())
            for i in range(self.num_mammoths):
                while True:
                    pos = tuple(int(v) for v in self.rng.integers(0, self.grid_size, size=2))
                    if pos not in taken:
                        break
                taken.add(pos)
                self.mammoth_positions[f"mammoth_{i}"] = pos

        self.current_num_prey = self.n_initial_active_prey
        self.current_num_predator_male = self.n_initial_active_predator_male
        self.current_num_predator_female = self.n_initial_active_predator_female
        self.current_num_grass = self.initial_num_grass
        self.current_num_fruit = self.initial_num_fruit

        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent in self._pending_removal:
            if agent in self.agents:
                self.agents.remove(agent)
        self._pending_removal = []

        # step 0: check for truncation
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                observations[agent] = self._get_observation(agent)
                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False

            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos

        self.agents_just_ate.clear()

        if self.scripted_prey:
            self._scripted_prey_ids = [p for p in self._scripted_prey_ids if p in self.agent_positions]
            action_dict = dict(action_dict)
            for prey in self._scripted_prey_ids:
                action_dict[prey] = self._scripted_prey_action(prey)

        # Step 1: homeostatic energy depletion (always charged, every step).
        # Guard membership (mirrors Step 2's guard below): a stale action for
        # an agent already removed in a prior step would otherwise KeyError.
        for agent, action in action_dict.items():
            if agent not in self.agent_positions:
                continue
            if "predator" in agent:
                self._debit(agent, self.homeostatic_energy_cost_per_step_predator)
                self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
            elif "prey" in agent:
                self.agent_energies[agent] -= self.homeostatic_energy_cost_per_step_prey
                self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

        for grass, grass_position in self.grass_positions.items():
            self.grass_energies[grass] = min(
                self.grass_energies[grass] + self.energy_gain_per_step_grass, self.initial_energy_grass
            )
            self.grid_world_state[3, *grass_position] = self.grass_energies[grass]

        for fruit, fruit_position in self.fruit_positions.items():
            self.fruit_energies[fruit] = min(
                self.fruit_energies[fruit] + self.energy_gain_per_step_fruit, self.initial_energy_fruit
            )
            self.grid_world_state[4, *fruit_position] = self.fruit_energies[fruit]

        # Step 2: movement
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                old_position = self.agent_positions[agent]
                new_position = self._get_move(agent, action)
                self.agent_positions[agent] = new_position
                move_cost = self._get_movement_energy_cost(agent, action)
                if "predator" in agent:
                    self._debit(agent, move_cost)
                else:
                    self.agent_energies[agent] -= move_cost
                if "predator" in agent:
                    self.predator_positions[agent] = new_position
                    self.grid_world_state[1, *old_position] = 0
                    self.grid_world_state[1, *new_position] = self.agent_energies[agent]
                elif "prey" in agent:
                    self.prey_positions[agent] = new_position
                    self.grid_world_state[2, *old_position] = 0
                    self.grid_world_state[2, *new_position] = self.agent_energies[agent]

                if self.verbose_movement:
                    print(f"[MOVE] Agent {agent} moved: {old_position} -> {new_position}.")

        # Step 2b: threats move and attack (a killed predator is removed in Step 3 below, like a starved one)
        if self.num_threats > 0:
            self._threats_act()

        # Step 2c: mammoths wander, respawn, and are hunted by band parties (deaths are removed in Step 3; shares are credited as rewards)
        if self.num_mammoths > 0:
            self._mammoths_act()

        # Step 3: removals (starvation) and engagements (hunting/foraging/grazing)
        for agent in self._engagement_order():
            if agent not in self.agent_positions:
                continue
            diet_cause = self._diet_death_cause(agent)
            if self.agent_energies[agent] <= 0 or diet_cause is not None:
                if diet_cause is not None and self.agent_energies[agent] > 0:
                    self.diet_deaths["predator_male" if "predator_male" in agent else "predator_female"][diet_cause] += 1
                if self.verbose_movement:
                    print(f"[MOVE] {agent} at {self.agent_positions[agent]} ran out of energy and is removed.")
                observations[agent] = self._get_observation(agent)
                rewards[agent] = self._pending_rewards.pop(agent, 0)  # a share earned earlier this step is still credited
                terminations[agent] = True
                truncations[agent] = False
                if "predator_male" in agent:
                    self.current_num_predator_male -= 1
                    self.episode_deaths["predator_male"] += 1
                    self.grid_world_state[1, *self.agent_positions[agent]] = 0
                    del self.predator_positions[agent]
                elif "predator_female" in agent:
                    self.current_num_predator_female -= 1
                    self.episode_deaths["predator_female"] += 1
                    self.grid_world_state[1, *self.agent_positions[agent]] = 0
                    del self.predator_positions[agent]
                elif "prey" in agent:
                    self.current_num_prey -= 1
                    self.episode_deaths["prey"] += 1
                    self.grid_world_state[2, *self.agent_positions[agent]] = 0
                    del self.prey_positions[agent]
                del self.agent_positions[agent]
                del self.agent_energies[agent]
                self.agent_fruit_store.pop(agent, None)
                self.agent_band.pop(agent, None)
                self.defense_opportunities.pop(agent, None)
                self.defense_exposures.pop(agent, None)
                self.ostracized_until.pop(agent, None)
                continue
            elif "predator_male" in agent:
                predator_position = self.agent_positions[agent]
                reward = 0.0
                pending_reward = self._pending_rewards.pop(agent, 0.0)  # e.g. a share of a mammoth kill this step
                ate_something = pending_reward > 0.0  # a mammoth share counts as eating for reward_predator_step

                caught_prey = next(
                    (
                        prey
                        for prey, prey_position in self.agent_positions.items()
                        if "prey" in prey and np.array_equal(predator_position, prey_position)
                    ),
                    None,
                )
                if caught_prey:
                    outcome, reward_delta, energy_gained = self._resolve_hunting_attempt(
                        agent,
                        predator_position,
                        caught_prey,
                        self.prey_vs_predator_male_success_prob,
                        self.prey_vs_predator_male_death_prob,
                        observations,
                        rewards,
                        terminations,
                        truncations,
                    )
                    if outcome == "predator_dies":
                        continue
                    if outcome == "success":
                        ate_something = True
                        reward += reward_delta
                        self._apply_male_gift(agent, energy_gained)
                        self._apply_band_share(agent, energy_gained, is_fruit=False)
                        self._share_energy_with_offspring(agent, energy_gained)

                caught_fruit = next(
                    (
                        fruit
                        for fruit, fruit_position in self.fruit_positions.items()
                        if np.array_equal(predator_position, fruit_position)
                    ),
                    None,
                )
                if caught_fruit:
                    ate_something = True
                    self.agents_just_ate.add(agent)
                    fruit_gain = self.fruit_energies[caught_fruit]
                    reward += self.reward_predator_gather_fruit + self.reward_predator_per_energy * fruit_gain
                    self.agent_energies[agent] += fruit_gain
                    self.agent_fruit_store[agent] += fruit_gain
                    self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
                    self.grid_world_state[4, *self.fruit_positions[caught_fruit]] = 0
                    self.fruit_energies[caught_fruit] = 0
                    self._apply_band_share(agent, fruit_gain, is_fruit=True)
                    self._share_energy_with_offspring(agent, fruit_gain, is_fruit=True)

                if not ate_something:
                    reward = self.reward_predator_step
                reward += pending_reward

                rewards[agent] = reward
                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] += rewards[agent]
                terminations[agent] = False
                truncations[agent] = False
            elif "predator_female" in agent:
                predator_position = self.agent_positions[agent]
                reward = 0.0
                pending_reward = self._pending_rewards.pop(agent, 0.0)  # e.g. a share of a mammoth kill this step
                ate_something = pending_reward > 0.0  # a mammoth share counts as eating for reward_predator_step

                caught_prey = next(
                    (
                        prey
                        for prey, prey_position in self.agent_positions.items()
                        if "prey" in prey and np.array_equal(predator_position, prey_position)
                    ),
                    None,
                )
                if caught_prey:
                    outcome, reward_delta, energy_gained = self._resolve_hunting_attempt(
                        agent,
                        predator_position,
                        caught_prey,
                        self.prey_vs_predator_female_success_prob,
                        self.prey_vs_predator_female_death_prob,
                        observations,
                        rewards,
                        terminations,
                        truncations,
                    )
                    if outcome == "predator_dies":
                        continue
                    if outcome == "success":
                        ate_something = True
                        reward += reward_delta
                        self._apply_band_share(agent, energy_gained, is_fruit=False)
                        self._share_energy_with_offspring(agent, energy_gained)

                caught_fruit = next(
                    (
                        fruit
                        for fruit, fruit_position in self.fruit_positions.items()
                        if np.array_equal(predator_position, fruit_position)
                    ),
                    None,
                )
                if caught_fruit:
                    if self.verbose_engagement:
                        print(f"[ENGAGE] {agent} gathered {caught_fruit} at {predator_position}!")
                    self.agents_just_ate.add(agent)
                    ate_something = True
                    fruit_gain = self.fruit_energies[caught_fruit]
                    reward += self.reward_predator_gather_fruit + self.reward_predator_per_energy * fruit_gain
                    self.agent_energies[agent] += fruit_gain
                    self.agent_fruit_store[agent] += fruit_gain
                    self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
                    self.grid_world_state[4, *self.fruit_positions[caught_fruit]] = 0
                    self.fruit_energies[caught_fruit] = 0
                    self._apply_female_gift(agent, fruit_gain)
                    self._apply_band_share(agent, fruit_gain, is_fruit=True)
                    self._share_energy_with_offspring(agent, fruit_gain, is_fruit=True)

                if not ate_something:
                    reward = self.reward_predator_step
                reward += pending_reward

                rewards[agent] = reward
                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] += rewards[agent]
                terminations[agent] = False
                truncations[agent] = False
            elif "prey" in agent:
                if terminations.get(agent) is None or not terminations[agent]:
                    prey_position = self.agent_positions[agent]
                    caught_grass = next(
                        (
                            grass
                            for grass, grass_position in self.grass_positions.items()
                            if "grass" in grass and np.array_equal(prey_position, grass_position)
                        ),
                        None,
                    )
                    if caught_grass:
                        if self.verbose_engagement:
                            print(f"[ENGAGE] {agent} caught grass at {prey_position}!")
                        self.agents_just_ate.add(agent)
                        rewards[agent] = self.reward_prey_eat_grass
                        self.agent_energies[agent] += self.grass_energies[caught_grass]
                        self.grid_world_state[2, *prey_position] = self.agent_energies[agent]

                        self.grid_world_state[3, *self.grass_positions[caught_grass]] = 0
                        self.grass_energies[caught_grass] = 0
                    else:
                        rewards[agent] = self.reward_prey_step

                    observations[agent] = self._get_observation(agent)
                    self.cumulative_rewards[agent] += rewards[agent]
                    terminations[agent] = False
                    truncations[agent] = False

        # Step 4: schedule agent removals for the next step
        self._pending_removal = [agent for agent in self.agents if terminations.get(agent)]
        if self.verbose_engagement:
            for agent in self._pending_removal:
                print(f"[ENGAGE] Agent {agent} terminated!")

        # Step 5a: prey reproduction (asexual, unchanged from base_environment_step_energy)
        for agent in self._engagement_order():
            if agent in self._pending_removal:
                continue
            if "prey" in agent:
                if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
                    if self._next_prey_idx < self.n_possible_prey:
                        occupied_positions = set(self.agent_positions.values())
                        new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
                        if new_position is None:
                            if self.verbose_spawning:
                                print(f"No free spawn position available for prey offspring of {agent}")
                        else:
                            new_agent = f"prey_{self._next_prey_idx}"
                            self._next_prey_idx += 1
                            if self.scripted_prey:
                                self._scripted_prey_ids.append(new_agent)
                            else:
                                self.agents.append(new_agent)
                            self.agent_positions[new_agent] = new_position
                            self.prey_positions[new_agent] = new_position
                            self.agent_energies[new_agent] = self.initial_energy_prey
                            self.agent_energies[agent] -= self.initial_energy_prey
                            self.grid_world_state[2, *self.agent_positions[new_agent]] = self.initial_energy_prey
                            self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]
                            self.current_num_prey += 1
                            self.episode_births["prey"] += 1
                            rewards[new_agent] = 0
                            rewards[agent] = rewards.get(agent, 0.0) + self.reproduction_reward_prey
                            self.cumulative_rewards[agent] += self.reproduction_reward_prey
                            self.cumulative_rewards[new_agent] = 0
                            observations[new_agent] = self._get_observation(new_agent)
                            terminations[new_agent] = False
                            truncations[new_agent] = False
                            if self.verbose_spawning:
                                print(f"New prey {new_agent} spawned at {self.agent_positions[new_agent]}")
                    else:
                        if self.verbose_spawning:
                            print("No new prey agent IDs left in the pool this episode")

        # Step 5b: predator sexual reproduction. A predator_male pairs with a
        # nearby, independently-eligible predator_female (see config_env.py's
        # mate_search_radius comment for why an exact-cell match is
        # impossible). Each male can only be matched to one mate per step and
        # vice versa (paired_this_step), and one offspring is produced per pair.
        paired_this_step: set = set()
        radius = self.mate_search_radius
        eligible_males = [
            a
            for a in self.agents
            if "predator_male" in a
            and a not in self._pending_removal
            and self._can_reproduce(a)
        ]
        # Snapshot female candidates and their positions before pairing starts:
        # a same-step newborn female (if her starting energy happens to clear
        # predator_creation_energy_threshold) must not be scanned as a mate
        # candidate for a later male in this same pass.
        female_snapshot = {
            female: pos
            for female, pos in self.predator_positions.items()
            if "predator_female" in female
            and female not in self._pending_removal
            and self._can_reproduce(female)
        }
        for male in eligible_males:
            if (
                self.predator_population_cap is not None
                and self.current_num_predator_male + self.current_num_predator_female >= self.predator_population_cap
            ):
                # Population already at or above the cap (checked before this birth, not after --
                # a birth that brings the population up TO the cap is still allowed, one that would
                # push it past is not): no further predators reproduce this step. Checked fresh on
                # every iteration (not just once before the loop) so a birth earlier in this same
                # step that reaches the cap stops any later pair too. break, not continue: once the
                # cap is hit nothing later in this loop can reproduce either, so this also skips the
                # mate-search/spawn-position work for them. Whichever pairs are processed first get
                # priority for the remaining slots in a step where the cap is reached mid-loop -- a
                # documented tie-break, not an attempt at fairness. Note eligible_males is in
                # self.agents' sort order, which is lexicographic, not numeric (predator_male_10
                # sorts before predator_male_2), so "processed first" is deterministic but not the
                # same as "created first" once indices reach double digits.
                break
            if male in paired_this_step:
                continue
            male_position = self.agent_positions[male]
            mate = next(
                (
                    female
                    for female, pos in female_snapshot.items()
                    if female not in paired_this_step
                    and max(abs(pos[0] - male_position[0]), abs(pos[1] - male_position[1])) <= radius
                    and not self._kin_blocked(male, female)
                ),
                None,
            )
            if mate is None:
                continue
            mate_position = self.agent_positions[mate]

            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(mate_position, occupied_positions)
            if new_position is None:
                if self.verbose_spawning:
                    print(f"No free spawn position available for offspring of {male} and {mate}")
                continue

            child_sex = "predator_male" if self.rng.random() < 0.5 else "predator_female"
            if child_sex == "predator_male":
                if self._next_predator_male_idx >= self.n_possible_predator_male:
                    if self.verbose_spawning:
                        print("No new predator_male agent IDs left in the pool this episode")
                    continue
                new_agent = f"predator_male_{self._next_predator_male_idx}"
                self._next_predator_male_idx += 1
                offspring_energy = self.initial_energy_predator_male
            else:
                if self._next_predator_female_idx >= self.n_possible_predator_female:
                    if self.verbose_spawning:
                        print("No new predator_female agent IDs left in the pool this episode")
                    continue
                new_agent = f"predator_female_{self._next_predator_female_idx}"
                self._next_predator_female_idx += 1
                offspring_energy = self.initial_energy_predator_female

            paired_this_step.add(male)
            paired_this_step.add(mate)

            # Record (or update, serial-monogamy-style) the pair bond for
            # exclusive male provisioning -- see _apply_male_gift. Sever any
            # PREVIOUS partner's reverse pointer first: e.g. after M1<->F,
            # if F later re-mates with M2, simply overwriting
            # agent_mate[M2]/agent_mate[F] would leave agent_mate[M1] == F
            # dangling, still pointing at F even though F's own record now
            # says M2 -- M1 would keep donating to an ex indefinitely (until
            # M1 himself next reproduces). Remating with a different partner
            # is the ordinary case here (mate selection has no memory of
            # prior pairing), not a rare edge case.
            old_male_mate = self.agent_mate.get(male)
            if old_male_mate is not None and old_male_mate != mate:
                self.agent_mate.pop(old_male_mate, None)
            old_mate_mate = self.agent_mate.get(mate)
            if old_mate_mate is not None and old_mate_mate != male:
                self.agent_mate.pop(old_mate_mate, None)
            self.agent_mate[male] = mate
            self.agent_mate[mate] = male
            if self.num_bands > 0:
                self._record_pairing(male, mate)

            self.agents.append(new_agent)
            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position
            self.agent_energies[new_agent] = offspring_energy
            self.agent_fruit_store[new_agent] = offspring_energy * self.diet_initial_fruit_share
            self.agent_parents[new_agent] = (male, mate)
            if self.num_bands > 0:
                self.agent_band[new_agent] = self.agent_band[male]  # after any marriage, both parents share a band
            # Both parents are now breeding adults -- see has_reproduced's
            # docstring for why this ends their own eligibility for
            # parental care from THEIR parents (_share_energy_with_offspring).
            self.has_reproduced.add(male)
            self.has_reproduced.add(mate)

            # mate is always the predator_female here (drawn from
            # female_snapshot, already filtered to "predator_female" in female).
            cost_female = offspring_energy * self.predator_birth_cost_share_female
            cost_male = offspring_energy * self.predator_birth_cost_share_male
            self._debit(mate, cost_female)
            self._debit(male, cost_male)
            self.grid_world_state[1, *new_position] = offspring_energy
            self.grid_world_state[1, *male_position] = self.agent_energies[male]
            self.grid_world_state[1, *mate_position] = self.agent_energies[mate]

            if child_sex == "predator_male":
                self.current_num_predator_male += 1
            else:
                self.current_num_predator_female += 1
            self.episode_births[child_sex] += 1

            rewards[new_agent] = 0
            rewards[male] = rewards.get(male, 0.0) + self.reproduction_reward_predator
            rewards[mate] = rewards.get(mate, 0.0) + self.reproduction_reward_predator
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[male] += self.reproduction_reward_predator
            self.cumulative_rewards[mate] += self.reproduction_reward_predator

            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False

            if self.verbose_spawning:
                print(f"New {child_sex} {new_agent} spawned at {new_position} (parents: {male}, {mate})")

        # Step 5c: band fission / fusion / drift-out (membership follows co-residence), every band_check_interval steps
        if (self.band_fission or self.band_fusion or self.band_drift_steps > 0) and (self.current_step + 1) % self.band_check_interval == 0:
            self._update_bands()

        # Step 6: generate observations for all agents AFTER all engagements/spawning
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)

        terminations["__all__"] = (
            self.current_num_prey <= 0 or self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0
        )

        observations = {agent: observations[agent] for agent in self.agents if agent in observations}
        rewards = {agent: rewards[agent] for agent in self.agents if agent in rewards}
        terminations = {agent: terminations[agent] for agent in self.agents if agent in terminations}
        truncations = {agent: truncations[agent] for agent in self.agents if agent in truncations}
        truncations["__all__"] = False

        terminations["__all__"] = (
            self.current_num_prey <= 0 or self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0
        )

        self._pending_rewards.clear()  # nothing earned this step is carried over
        self.agents.sort()

        self.current_step += 1

        return observations, rewards, terminations, truncations, infos

    def _build_episode_training_metrics(self) -> Dict[str, float]:
        """
        Ecology metrics for TensorBoard, surfaced via the EpisodeReturn callback.
        """
        return {
            "episode_length": float(self.current_step),
            "births_predator_male": float(self.episode_births["predator_male"]),
            "births_predator_female": float(self.episode_births["predator_female"]),
            "births_prey": float(self.episode_births["prey"]),
            "deaths_predator_male": float(self.episode_deaths["predator_male"]),
            "deaths_predator_female": float(self.episode_deaths["predator_female"]),
            "deaths_prey": float(self.episode_deaths["prey"]),
            "final_num_predator_male": float(self.current_num_predator_male),
            "final_num_predator_female": float(self.current_num_predator_female),
            "final_num_prey": float(self.current_num_prey),
            "extinct_predator_male": float(self.current_num_predator_male <= 0),
            "extinct_predator_female": float(self.current_num_predator_female <= 0),
            "extinct_prey": float(self.current_num_prey <= 0),
            # Hunting behavior/effectiveness by sex -- lets you distinguish
            # "never attempts" (avoidance) from "attempts but fails/dies"
            # from "never gets the chance," which births/deaths alone
            # can't. hunting_deaths is combat-caused only; the remainder,
            # deaths_predator_*[sex] - hunting_deaths_predator_*[sex], is
            # starvation.
            "hunting_attempts_predator_male": float(self.hunting_attempts["predator_male"]),
            "hunting_attempts_predator_female": float(self.hunting_attempts["predator_female"]),
            "hunting_successes_predator_male": float(self.hunting_successes["predator_male"]),
            "hunting_successes_predator_female": float(self.hunting_successes["predator_female"]),
            "hunting_success_rate_predator_male": (
                self.hunting_successes["predator_male"] / self.hunting_attempts["predator_male"]
                if self.hunting_attempts["predator_male"] > 0
                else 0.0
            ),
            "hunting_success_rate_predator_female": (
                self.hunting_successes["predator_female"] / self.hunting_attempts["predator_female"]
                if self.hunting_attempts["predator_female"] > 0
                else 0.0
            ),
            "hunting_deaths_predator_male": float(self.hunting_deaths["predator_male"]),
            "hunting_deaths_predator_female": float(self.hunting_deaths["predator_female"]),
            # Provisioning mechanics: how much energy is actually flowing
            # through the mate-gift and parental-care mechanisms this
            # episode (both are 0 if no one ever reproduces).
            "mate_gift_events": float(self.mate_gift_events),
            "mate_gift_energy_total": float(self.mate_gift_energy_total),
            "band_share_events": float(self.band_share_events),
            "band_share_meat_total": float(self.band_share_meat_total),
            "band_share_fruit_total": float(self.band_share_fruit_total),
            "marriages": float(self.marriages),
            "within_band_pairings": float(self.within_band_pairings),
            "kin_blocked_checks": float(self.kin_blocked_checks),
            **{f"mammoth_attempts_n{i + 1}": float(self.mammoth_attempts[i]) for i in range(4)},
            **{f"mammoth_kills_n{i + 1}": float(self.mammoth_kills[i]) for i in range(4)},
            **{f"mammoth_party_deaths_n{i + 1}": float(self.mammoth_party_deaths[i]) for i in range(4)},
            "mammoth_blocked": float(self.mammoth_blocked),
            "mammoth_energy_distributed": float(self.mammoth_energy_distributed),
            "threat_encounters": float(self.threat_encounters),
            "threat_kills_male": float(self.threat_kills["predator_male"]),
            "threat_kills_female": float(self.threat_kills["predator_female"]),
            "threat_kills_alone": float(self.threat_kills_alone),
            "threat_repelled": float(self.threat_repelled),
            "ostracism_events": float(self.ostracism_events),
            "ostracized_now": float(sum(1 for a in self.predator_positions if self._is_ostracized(a))),
            "band_fissions": float(self.band_fissions),
            "band_fusions": float(self.band_fusions),
            "band_drift_outs": float(self.band_drift_outs),
            "mean_band_size": float(
                np.mean([len(m) for m in self._bands_now().values()]) if self.agent_band and self.predator_positions else 0.0
            ),
            "bands_alive": float(len({self.agent_band[a] for a in self.predator_positions if a in self.agent_band})),
            "female_gift_events": float(self.female_gift_events),
            "female_gift_energy_total": float(self.female_gift_energy_total),
            "diet_deaths_male_fruit": float(self.diet_deaths["predator_male"]["fruit"]),
            "diet_deaths_male_meat": float(self.diet_deaths["predator_male"]["meat"]),
            "diet_deaths_female_fruit": float(self.diet_deaths["predator_female"]["fruit"]),
            "diet_deaths_female_meat": float(self.diet_deaths["predator_female"]["meat"]),
            "parental_care_events": float(self.parental_care_events),
            "parental_care_energy_total": float(self.parental_care_energy_total),
        }

    def _resolve_hunting_attempt(
        self,
        agent,
        predator_position,
        caught_prey,
        success_prob,
        death_prob,
        observations,
        rewards,
        terminations,
        truncations,
    ):
        """Resolve a predator (either sex) attempting to hunt a co-located prey.

        Returns (outcome, reward_delta, energy_gained), outcome in
        {"success", "predator_dies", "failure"}.

        "success": prey is fully removed (same bookkeeping as a deterministic
          catch); the caller still finalizes the predator's own turn (it may
          also gather fruit this same step). energy_gained is the raw energy
          amount the predator's own agent_energies was just credited by --
          returned so the caller can apply male provisioning
          (_apply_male_gift) on it, since the prey's own energy is gone
          (deleted) by the time this returns.
        "predator_dies": the prey is left completely untouched -- it proceeds
          to its own turn normally later in the Step 3 loop. The predator is
          removed using the same bookkeeping as the starvation-removal branch
          above. The caller MUST `continue` after this outcome (skip fruit-
          gathering -- the agent no longer exists). energy_gained is 0.0.
        "failure": no side effects; caller proceeds to fruit-gathering as
          normal. energy_gained is 0.0.
        """
        sex = "predator_male" if "predator_male" in agent else "predator_female"
        self.hunting_attempts[sex] += 1
        roll = self.rng.random()
        if roll < success_prob:
            if self.verbose_engagement:
                print(f"[ENGAGE] {agent} caught {caught_prey} at {predator_position}!")
            self.agents_just_ate.add(agent)
            # Clamp at 0: a prey that starved this same step (energy <= 0 after Step 1) can still be
            # caught if the predator is processed before it in Step 3's loop; it must not take energy
            # from the predator nor pay a negative energy-proportional reward.
            energy_gained = max(self.agent_energies[caught_prey], 0.0) * self.prey_energy_yield
            self.agent_energies[agent] += energy_gained
            self.grid_world_state[1, *predator_position] = self.agent_energies[agent]

            observations[caught_prey] = self._get_observation(caught_prey)
            rewards[caught_prey] = self.penalty_prey_caught
            self.cumulative_rewards[caught_prey] += rewards[caught_prey]
            terminations[caught_prey] = True
            truncations[caught_prey] = False
            self.current_num_prey -= 1
            self.episode_deaths["prey"] += 1
            self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
            del self.agent_positions[caught_prey]
            del self.prey_positions[caught_prey]
            del self.agent_energies[caught_prey]
            self.hunting_successes[sex] += 1
            return (
                "success",
                self.reward_predator_catch_prey + self.reward_predator_per_energy * energy_gained,
                energy_gained,
            )

        if roll < success_prob + death_prob:
            if self.verbose_engagement:
                print(f"[ENGAGE] {agent} attacked {caught_prey} at {predator_position} and was killed!")
            observations[agent] = self._get_observation(agent)
            rewards[agent] = self.penalty_predator_death_in_combat
            self.cumulative_rewards[agent] += rewards[agent]
            terminations[agent] = True
            truncations[agent] = False
            if "predator_male" in agent:
                self.current_num_predator_male -= 1
                self.episode_deaths["predator_male"] += 1
            else:
                self.current_num_predator_female -= 1
                self.episode_deaths["predator_female"] += 1
            self.hunting_deaths[sex] += 1
            self.grid_world_state[1, *predator_position] = 0
            del self.predator_positions[agent]
            del self.agent_positions[agent]
            del self.agent_energies[agent]
            self.agent_fruit_store.pop(agent, None)
            self.agent_band.pop(agent, None)
            self.defense_opportunities.pop(agent, None)
            self.defense_exposures.pop(agent, None)
            self.ostracized_until.pop(agent, None)
            return "predator_dies", 0.0, 0.0

        return "failure", 0.0, 0.0

    def _apply_male_gift(self, agent, energy_gained):
        """Exclusive (pair-bonded), unidirectional male -> female provisioning:
        a predator_male donates male_gift_donation_rate of a successful
        hunt's energy gain to HIS RECORDED MATE ONLY (self.agent_mate),
        never to any other nearby female -- more biologically apt for the
        human pair-bonding this module studies than broadcasting to whoever
        happens to be nearby (contrast eco_evolutionary_nuptial_gift, which
        deliberately broadcasts to any nearby female, modeling a
        non-pair-bonded species). Mechanically executed (not a learned
        action) -- same credit-assignment rationale as
        eco_evolutionary_nuptial_gift's male_donation_rate (a donor's own
        reward stream never reflects a recipient's downstream fitness, so a
        learned "donate" action would face a real credit-assignment gap).
        Meant to offset predator_female's post-birth energy deficit: she
        pays the larger share of birth cost (predator_birth_cost_share_female)
        but her only reliable income (fruit) is weak, shared, and depleting,
        unlike the male's much higher hunting success rate.

        No-ops if he has no recorded mate yet (never successfully
        reproduced), if she's no longer alive, if she's already at <= 0
        energy this step (about to starve -- see the historical note below),
        or if she's outside predator_gift_range (Chebyshev distance) -- the
        gift is still a physical exchange, so proximity still matters even
        though the recipient is now a specific individual rather than
        anyone nearby.

        Historical note on the <= 0 energy guard: Step 3 processes agents in
        self.agents order, and on the very first step of an episode (before
        self.agents has ever been sorted) predator_male_* entries precede
        predator_female_* -- so a mate who already took a fatal Step-1
        homeostatic hit this step, but whose own starvation check hasn't run
        yet, would otherwise still be reachable here and receive a gift that
        pushes her energy back above zero, making her survive purely as an
        accident of iteration order. On every later step (self.agents is
        sorted at the end of step(), putting predator_female_* first) a
        starving mate is already removed before any male's turn runs, so
        this guard just makes that (already-starving-is-already-decided)
        behavior consistent across steps instead of order-dependent.
        """
        if energy_gained <= 0.0 or self.male_gift_donation_rate <= 0.0:
            return
        mate = self.agent_mate.get(agent)
        if (
            mate is None
            or mate not in self.agent_positions
            or self.agent_energies[mate] <= 0.0
            or self._diet_death_cause(mate) is not None  # already doomed: not rescued by an accident of turn order
        ):
            return
        position = self.agent_positions[agent]
        mate_position = self.agent_positions[mate]
        radius = self.predator_gift_range
        if max(abs(mate_position[0] - position[0]), abs(mate_position[1] - position[1])) > radius:
            return
        donation = self.male_gift_donation_rate * energy_gained
        self.agent_energies[agent] -= donation
        self.grid_world_state[1, *position] = self.agent_energies[agent]
        self.agent_energies[mate] += donation
        self.grid_world_state[1, *mate_position] = self.agent_energies[mate]
        self.mate_gift_events += 1
        self.mate_gift_energy_total += donation

    def _share_energy_with_offspring(self, agent, energy_gained, is_fruit=False):
        """Parental care: both parents share parent_offspring_share_rate of
        any successful forage (hunt or fruit -- called from every
        successful-forage branch of both predator sexes) with their own
        nearby living offspring (self.agent_parents), within
        predator_gift_range. Mechanically executed, like _apply_male_gift,
        for the same credit-assignment reasoning. Unlike _apply_male_gift
        (exclusive to one recorded mate), this splits evenly across
        however many of the forager's own children are currently nearby,
        since a parent can have multiple living offspring at once.

        Cuts off once an offspring has reproduced itself
        (self.has_reproduced): it's then a breeding adult, not a dependent
        juvenile, regardless of how close it still stands to its own
        parents. This is a reproduction-based independence cutoff, not an
        age-based one -- this module tracks no per-agent age at all, and
        "started its own family" is a cheaper, already-available proxy for
        "grown up" than adding age/weaning-duration bookkeeping would be.
        Distance still tapers care off naturally too: a grown offspring
        that simply wanders away (without yet reproducing) falls out of
        range on its own, no extra state needed for that part.

        Note on same-step ordering: this runs during Step 3, which precedes
        Step 5b (where has_reproduced gets updated). An offspring that
        reproduces for the first time later in THIS step can therefore
        still receive care earlier in this same step -- it only becomes
        ineligible starting next step. Deterministic, not a bug.
        """
        if energy_gained <= 0.0 or self.parent_offspring_share_rate <= 0.0:
            return
        position = self.agent_positions[agent]
        radius = self.predator_gift_range
        children = [
            other
            for other, pos in self.predator_positions.items()
            if agent in self.agent_parents.get(other, ())
            and other not in self.has_reproduced
            and self.agent_energies[other] > 0.0
            and self._diet_death_cause(other) is None  # a child already doomed by a store is not rescued
            and max(abs(pos[0] - position[0]), abs(pos[1] - position[1])) <= radius
        ]
        if not children:
            return
        donation_total = self.parent_offspring_share_rate * energy_gained
        share = donation_total / len(children)
        self.agent_energies[agent] -= donation_total
        if is_fruit:  # care keeps the food type: fruit comes out of her fruit store and goes into the child's
            self.agent_fruit_store[agent] -= donation_total
        self.grid_world_state[1, *position] = self.agent_energies[agent]
        for child in children:
            self.agent_energies[child] += share
            if is_fruit:
                self.agent_fruit_store[child] += share
            self.grid_world_state[1, *self.agent_positions[child]] = self.agent_energies[child]
        self.parental_care_events += 1
        self.parental_care_energy_total += donation_total

    def _engagement_order(self):
        """Agents processed in the engagement / reproduction loops: the learning agents, plus the scripted prey."""
        if not self.scripted_prey:
            return self.agents[:]
        return self.agents + [p for p in self._scripted_prey_ids if p in self.agent_positions]

    def _scripted_prey_action(self, prey):
        """Rule for a scripted prey: if a predator is within prey_flee_radius (Chebyshev), take the move that
        maximizes the distance to the nearest such predator; otherwise walk toward the nearest grass in view (stay put
        on grass), or move at random if none is visible. Ties are broken with the env rng."""
        x, y = self.agent_positions[prey]
        n = self.grid_size

        def destination(dx, dy):
            nx, ny = min(max(x + dx, 0), n - 1), min(max(y + dy, 0), n - 1)
            if (nx, ny) != (x, y) and self.grid_world_state[2, nx, ny] > 0:  # another prey there: the move is blocked
                return x, y
            return nx, ny

        threats = [
            p for p in self.predator_positions.values() if max(abs(p[0] - x), abs(p[1] - y)) <= self.prey_flee_radius
        ]
        moves = self.action_to_move_tuple
        if threats:
            score = {
                a: min(max(abs(px - dest[0]), abs(py - dest[1])) for px, py in threats)
                for a, dest in ((a, destination(*mv)) for a, mv in moves.items())
            }
        else:
            r = (self.prey_obs_range - 1) // 2
            x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, n), max(y - r, 0), min(y + r + 1, n)
            cells = np.argwhere(self.grid_world_state[3, x0:x1, y0:y1] > 0)
            if len(cells) == 0:
                return int(self.rng.integers(len(moves)))
            cells = cells + np.array([x0, y0])
            if (cells == np.array([x, y])).all(axis=1).any():
                return self.noop_action_id  # standing on grass: eat it
            score = {}
            for a, mv in moves.items():
                dest = destination(*mv)
                score[a] = -int(np.max(np.abs(cells - np.array(dest)), axis=1).min())
        best = max(score.values())
        choices = [a for a, v in score.items() if v == best]
        return int(choices[int(self.rng.integers(len(choices)))])

    # ------------------------------------------------------------------------------------------------ bands
    def _band_centers(self):
        """Band centres by farthest-point sampling (Chebyshev), from the env rng: spread out across the grid."""
        n = self.grid_size
        centers = [tuple(int(v) for v in self.rng.integers(0, n, size=2))]
        while len(centers) < self.num_bands:
            candidates = [tuple(int(v) for v in self.rng.integers(0, n, size=2)) for _ in range(200)]
            best = max(candidates, key=lambda c: min(max(abs(c[0] - o[0]), abs(c[1] - o[1])) for o in centers))
            centers.append(best)
        return centers

    def _build_band_layout(self, male_ids, female_ids):
        """Initial bands: per band `band_couples` founding couples (recorded mates) each with
        `band_children_per_couple` dependent children (alternating male, female), plus unpaired adult males/females.
        Members are placed on the free cells nearest the band centre. Returns positions, band ids, mates, parents."""
        males, females = iter(male_ids), iter(female_ids)
        positions, band, mates, parents = {}, {}, {}, {}
        occupied = set()
        for b, center in enumerate(self._band_centers()):
            members = []
            for _ in range(self.band_couples):
                m, f = next(males), next(females)
                members += [m, f]
                mates[m], mates[f] = f, m
                for j in range(self.band_children_per_couple):
                    kid = next(males) if j % 2 == 0 else next(females)
                    parents[kid] = (m, f)
                    members.append(kid)
            members += [next(males) for _ in range(self.band_singles_male)]
            members += [next(females) for _ in range(self.band_singles_female)]
            free = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in occupied
            ]
            jitter = {c: self.rng.random() for c in free}
            free.sort(key=lambda c: (max(abs(c[0] - center[0]), abs(c[1] - center[1])), jitter[c]))
            free = [c for c in free if max(abs(c[0] - center[0]), abs(c[1] - center[1])) <= self.band_spawn_radius]
            if len(free) < len(members):
                raise ValueError(
                    f"band {b} has {len(members)} members but only {len(free)} free cells within band_spawn_radius="
                    f"{self.band_spawn_radius} of its centre {center}; raise band_spawn_radius or shrink the band"
                )
            for a, cell in zip(members, free):
                positions[a] = cell
                band[a] = b
                occupied.add(cell)
        return {"positions": positions, "band": band, "mates": mates, "parents": parents}

    # ---------------------------------------------------------------------------------------------- band dynamics
    def _bands_now(self):
        """band id -> sorted list of viable predators in it (alive, positive energy, not doomed by an empty store)."""
        bands: Dict[int, List[str]] = {}
        viable = self._viable_predators()
        for a in sorted(self.predator_positions):
            if a not in viable:
                continue
            b = self.agent_band.get(a)
            if b is not None:
                bands.setdefault(b, []).append(a)
        return bands

    def _dependents_of(self, member):
        """Living, unreproduced children of `member`."""
        return [
            c for c, parents in self.agent_parents.items()
            if member in parents and c not in self.has_reproduced and c in self.predator_positions
        ]

    def _two_means(self, members):
        """Split members into two spatial clusters (2-means on positions, seeded with the two members farthest apart; ties by id)."""
        pts = {a: np.array(self.predator_positions[a], dtype=float) for a in members}
        best = None
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                d = float(np.abs(pts[a] - pts[b]).max())
                if best is None or d > best[0]:
                    best = (d, a, b)
        _, a0, b0 = best
        centers = [pts[a0].copy(), pts[b0].copy()]
        labels = {}
        for _ in range(8):
            labels = {a: int(np.linalg.norm(pts[a] - centers[1]) < np.linalg.norm(pts[a] - centers[0])) for a in members}
            for k in (0, 1):
                grp = [pts[a] for a in members if labels[a] == k]
                if grp:
                    centers[k] = np.mean(grp, axis=0)
        return labels

    def _update_bands(self):
        """Drift, then fission, then fusion (each optional; deterministic given positions; band ids are never reused). Membership is the
        start-of-phase snapshot of VIABLE predators, so results do not depend on iteration order.
        Drift: a member with no same-band member within band_share_range for band_drift_steps consecutive steps (the counter is tied to
        its current band and resets whenever the band changes or it is alone in it) moves: with band_drift_join it joins the band of the
        nearest viable predator of another band within band_share_range, if any (a singleton can do this too); otherwise it leaves for a
        one-member band. A drifting member has no band-mate within range, so its dependent children (if in range of the band) stay.
        Fission: a band larger than band_max_size splits into two spatial clusters (each at least band_min_split_size; repeated up to 5 passes
        until no band exceeds the cap); the cluster with the larger total energy keeps the id, the other gets a new id; dependent children
        follow their living mother's cluster, else their father's.
        Fusion: two bands whose centroids stay within band_fuse_distance for band_fuse_steps ELAPSED steps (each check adds
        band_check_interval) fuse when the union is at most 0.75 x band_max_size (hysteresis against an immediate re-split); the lower id
        survives. Bands created by fission in this call do not fuse in it."""
        interval = self.band_check_interval
        viable = self._viable_predators()
        # -- drift
        if self.band_drift_steps > 0:
            bands = self._bands_now()
            band_of = {a: b for b, m in bands.items() for a in m}
            moves = {}
            for band, members in bands.items():
                for a in members:
                    ax, ay = self.predator_positions[a]
                    near = any(
                        max(abs(self.predator_positions[o][0] - ax), abs(self.predator_positions[o][1] - ay)) <= self.band_share_range
                        for o in members if o != a
                    )
                    prev = self._band_out_steps.get(a)
                    if near or len(members) == 1 and not self.band_drift_join:
                        self._band_out_steps.pop(a, None)  # in range (or a lone member with nowhere to go): the counter restarts
                        continue
                    steps = (prev[1] if prev is not None and prev[0] == band else 0) + interval
                    self._band_out_steps[a] = (band, steps)
                    if steps < self.band_drift_steps:
                        continue
                    target = None
                    if self.band_drift_join:
                        best = None
                        for o in sorted(viable):
                            ob = band_of.get(o)
                            if o == a or ob is None or ob == band:
                                continue
                            d = max(abs(self.predator_positions[o][0] - ax), abs(self.predator_positions[o][1] - ay))
                            if d <= self.band_share_range and (best is None or d < best[0]):
                                best = (d, ob)
                        target = None if best is None else best[1]
                    if target is None and len(members) == 1:
                        continue  # already alone and nobody to join
                    moves[a] = target
            new_ids = {}
            for a, target in sorted(moves.items()):
                if target is None:
                    new_ids[a] = self._next_band_id
                    self._next_band_id += 1
                    self.band_drift_outs += 1
                else:
                    new_ids[a] = target
                    self.band_drift_outs += 1
            for a, new_id in new_ids.items():
                self.agent_band[a] = new_id
                self._band_out_steps.pop(a, None)
        for a in [a for a in self._band_out_steps if a not in self.predator_positions]:
            self._band_out_steps.pop(a)
        # -- fission (repeat until no band is above the cap, at most 5 passes)
        fresh = set()
        if self.band_fission:
            for _ in range(5):
                split_any = False
                for band, members in list(self._bands_now().items()):
                    if len(members) <= self.band_max_size:
                        continue
                    labels = self._two_means(members)
                    parts = {0: [a for a in members if labels[a] == 0], 1: [a for a in members if labels[a] == 1]}
                    if min(len(parts[0]), len(parts[1])) < self.band_min_split_size:
                        continue
                    keep = max((0, 1), key=lambda k: (sum(self.agent_energies[a] for a in parts[k]), -k))
                    leave = 1 - keep
                    new_id = self._next_band_id
                    self._next_band_id += 1
                    moving = set(parts[leave])
                    for a in members:  # dependent children follow their living mother's part, else their father's
                        if a in self.agent_parents and a not in self.has_reproduced:
                            father, mother = self.agent_parents[a]
                            for parent in (mother, father):
                                if parent in labels:
                                    (moving.add if labels[parent] == leave else moving.discard)(a)
                                    break
                    for a in moving:
                        self.agent_band[a] = new_id
                        self._band_out_steps.pop(a, None)
                    fresh.add(new_id)
                    self.band_fissions += 1
                    split_any = True
                if not split_any:
                    break
        # -- fusion
        if self.band_fusion:
            bands = self._bands_now()
            cents = {b: np.mean([self.predator_positions[a] for a in m], axis=0) for b, m in bands.items()}
            ids = sorted(bands)
            close = set()
            for i, b1 in enumerate(ids):
                for b2 in ids[i + 1 :]:
                    if float(np.abs(cents[b1] - cents[b2]).max()) <= self.band_fuse_distance:
                        close.add((b1, b2))
            self._band_contact = {p: self._band_contact.get(p, 0) + interval for p in close}
            for (b1, b2), steps in sorted(self._band_contact.items()):
                if steps < self.band_fuse_steps or b1 not in bands or b2 not in bands or b1 in fresh or b2 in fresh:
                    continue
                if len(bands[b1]) + len(bands[b2]) > 0.75 * self.band_max_size:
                    continue
                for a in bands[b2]:
                    self.agent_band[a] = b1
                    self._band_out_steps.pop(a, None)
                bands[b1] = bands[b1] + bands.pop(b2)
                self.band_fusions += 1
            self._band_contact = {p: n for p, n in self._band_contact.items() if p[0] in bands and p[1] in bands}

    # ---------------------------------------------------------------------------------------------- mammoths
    def _mammoth_party(self, attacker, viable):
        """The attacker plus, when bands are on, its viable band-mates within mammoth_party_radius of the mammoth it stands on; with
        bands off, every viable predator within the radius."""
        mx, my = self.agent_positions[attacker]
        band = self.agent_band.get(attacker)
        party = []
        for other in sorted(viable):
            ox, oy = self.predator_positions[other]
            if max(abs(ox - mx), abs(oy - my)) > self.mammoth_party_radius:
                continue
            if self.num_bands > 0 and band is not None and self.agent_band.get(other) != band:
                continue
            party.append(other)
        return party

    def _other_band_party_size(self, mammoth_pos, band, viable):
        """Size of the largest party of another band within mammoth_party_radius of the mammoth."""
        counts = {}
        mx, my = mammoth_pos
        for other in viable:
            b = self.agent_band.get(other)
            if b is None or b == band:
                continue
            ox, oy = self.predator_positions[other]
            if max(abs(ox - mx), abs(oy - my)) <= self.mammoth_party_radius:
                counts[b] = counts.get(b, 0) + 1
        return max(counts.values()) if counts else 0

    def _mammoths_act(self):
        """Respawn, wander and resolve hunts. A mammoth hunt is a BAND hunt: the predator standing on the mammoth's cell is the attacker;
        its party is that predator plus its band-mates within mammoth_party_radius (other bands do not count, share or die). One attempt per
        mammoth per step, resolved where the mammoth stands before it wanders; it is blocked if a strictly larger party of another band is present. Success probability and the per-member death
        probability on failure depend on the party size (1, 2, 3, 4+). On success the mammoth's energy is split equally among the party as
        meat (energy only, not the fruit store), each member is credited reward_predator_per_energy x its share, and the mammoth respawns
        after mammoth_respawn_steps at a random free cell. A member that dies has its energy set to -1 (removed in Step 3)."""
        n = self.grid_size
        moves = [m for m in self.action_to_move_tuple.values() if m != (0, 0)]
        # respawn
        for mid in sorted(self.mammoth_respawn_at, key=lambda m: int(m.split("_")[1])):
            if self.current_step >= self.mammoth_respawn_at[mid]:
                taken = set(self.agent_positions.values()) | set(self.mammoth_positions.values()) | set(self.threat_positions.values())
                free = [(x, y) for x in range(n) for y in range(n) if (x, y) not in taken]
                if free:
                    self.mammoth_positions[mid] = free[int(self.rng.integers(len(free)))]
                    del self.mammoth_respawn_at[mid]
        for mid in sorted(self.mammoth_positions, key=lambda m: int(m.split("_")[1])):
            viable = self._viable_predators()  # recomputed per mammoth: a member killed at an earlier mammoth is out
            pos = self.mammoth_positions[mid]
            attackers = [a for a in sorted(viable) if self.predator_positions[a] == pos]
            killed = False
            if attackers:  # a hunt is resolved where the mammoth stands NOW, before it can wander off
                attacker = attackers[0]
                party = self._mammoth_party(attacker, viable)
                size_idx = min(len(party), 4) - 1
                if self.num_bands > 0 and self._other_band_party_size(pos, self.agent_band.get(attacker), viable) > len(party):
                    self.mammoth_blocked += 1
                else:
                    self.mammoth_attempts[size_idx] += 1
                    if self.rng.random() < self.mammoth_success_by_party[size_idx]:
                        share = self.mammoth_energy / len(party)
                        for member in party:
                            self.agent_energies[member] += share
                            self.grid_world_state[1, *self.agent_positions[member]] = self.agent_energies[member]
                            self._pending_rewards[member] = (
                                self._pending_rewards.get(member, 0.0) + self.reward_predator_per_energy * share
                            )
                        self.mammoth_kills[size_idx] += 1
                        self.mammoth_energy_distributed += self.mammoth_energy
                        del self.mammoth_positions[mid]
                        self.mammoth_respawn_at[mid] = self.current_step + self.mammoth_respawn_steps
                        killed = True
                    else:
                        for member in party:
                            if self.rng.random() < self.mammoth_death_by_party[size_idx]:
                                self.agent_energies[member] = -1.0
                                self.mammoth_party_deaths[size_idx] += 1
            if killed:
                continue
            # wander (never onto a predator or another mammoth)
            if self.rng.random() < self.mammoth_move_prob:
                mx, my = self.mammoth_positions[mid]
                occupied = set(self.predator_positions.values()) | {p for t, p in self.mammoth_positions.items() if t != mid}
                cands = [
                    c
                    for c in ((min(max(mx + dx, 0), n - 1), min(max(my + dy, 0), n - 1)) for dx, dy in moves)
                    if c != (mx, my) and c not in occupied
                ]
                if cands:
                    self.mammoth_positions[mid] = cands[int(self.rng.integers(len(cands)))]

    # ---------------------------------------------------------------------------------------------- threats
    def _viable_predators(self):
        """Predators a threat can target or that can defend: alive, positive energy, and not already doomed by an empty store."""
        return {
            a for a in self.predator_positions if self.agent_energies[a] > 0.0 and self._diet_death_cause(a) is None
        }

    def _is_ostracized(self, agent):
        return self.band_ostracism and self.current_step < self.ostracized_until.get(agent, -1)

    def _defender_ids(self, target, viable=None, ostracized=None):
        """The living, in-range predators that count as target's defenders: its band-mates (threat_defense_by == 'band') or
        any predator ('any'); a currently-ostracized predator (band_ostracism) never counts, even if otherwise in range.
        ostracized: an explicit set of ids to treat as ostracized, frozen at the start of the calling phase (see
        _threats_act, which computes it once alongside viable so a reputation update mid-phase cannot change who defended
        an already-resolved target -- the same no-order-dependence guarantee _threats_act documents for itself). None (the
        default, for a direct or test call outside that phase) falls back to a live _is_ostracized check per predator."""
        viable = self._viable_predators() if viable is None else viable
        tx, ty = self.agent_positions[target]
        band = self.agent_band.get(target)
        result = []
        for other in viable:
            if other == target:
                continue
            if self.threat_defense_by == "band" and (band is None or self.agent_band.get(other) != band):
                continue
            if (other in ostracized) if ostracized is not None else self._is_ostracized(other):
                continue
            ox, oy = self.predator_positions[other]
            if max(abs(ox - tx), abs(oy - ty)) <= self.threat_defense_radius:
                result.append(other)
        return result

    def _threat_defenders(self, target, viable=None):
        """Number of viable predators (not the target) within threat_defense_radius of the target that defend it (see
        _defender_ids)."""
        return len(self._defender_ids(target, viable))

    def _update_reputation(self, defender_ids, threat_x, threat_y):
        """band_ostracism bookkeeping, called once per genuinely resolved attack (see _threats_act): each predator in
        defender_ids gets one recorded 'opportunity' (it was close enough to a threatened band-mate to be credited as a
        defender); if it was also within distance 1 of the threat itself -- genuinely at risk, not just close enough for
        free credit -- it gets one 'exposure' too. Once a predator has ostracism_min_opportunities recorded and its
        exposure ratio is below ostracism_exposure_threshold, it is ostracized for ostracism_duration steps and its counts
        reset (see _is_ostracized, _defender_ids, _apply_band_share)."""
        for other in defender_ids:
            self.defense_opportunities[other] = self.defense_opportunities.get(other, 0) + 1
            ox, oy = self.predator_positions[other]
            if max(abs(ox - threat_x), abs(oy - threat_y)) <= 1:
                self.defense_exposures[other] = self.defense_exposures.get(other, 0) + 1
            opportunities = self.defense_opportunities[other]
            if opportunities >= self.ostracism_min_opportunities:
                ratio = self.defense_exposures.get(other, 0) / opportunities
                if ratio < self.ostracism_exposure_threshold:
                    self.ostracized_until[other] = self.current_step + self.ostracism_duration
                    self.defense_opportunities[other] = 0
                    self.defense_exposures[other] = 0
                    self.ostracism_events += 1

    def _threats_act(self):
        """Threats act in numeric id order but all attack decisions use the state at the START of the phase, so a kill by one threat
        does not change who defends or is targeted for the next (no order dependence): each threat attacks the nearest adjacent
        viable predator (ties: lowest agent id), else chases the nearest sensed viable predator, else wanders. With
        threat_attack_all_adjacent (default False), a threat instead attacks every viable predator within distance 1 at once
        (same ordering), each resolved independently below -- so several band-mates standing next to one threat face the kill
        roll together and can be each other's defenders in that same step, rather than only the single nearest one being at
        risk. An attack on a target with n defenders: n >= threat_defenders_to_repel drives the threat off (moved to a free cell
        at least threat_flee_distance away, else the farthest free cell; counted as repelled only if it moved) and ends its
        turn -- it does not go on to attack any remaining adjacent targets this step, since it has left the cell; otherwise the
        target dies with probability threat_kill_prob * (1 - n / threat_defenders_to_repel). Kills are applied after all threats
        have acted (the target's energy is set to -1 so that Step 3 removes it; a target killed by several threats counts once).
        With threat_attack_all_adjacent, a single threat's turn can now both kill one target and then be repelled or survive a
        failed attack by another: its rest state is resolved once, after its whole target list, and a kill always wins (sated
        for threat_satiation_steps) over a repel or a failed attack (threat_cooldown_steps) that happened in the same turn.
        Threats may share a cell with prey, grass or fruit (they do not interact with them) but never with a predator or
        another threat. band_ostracism: who is ostracized is likewise frozen at the start of the phase (ostracized_at_start),
        so a predator newly ostracized partway through this call still defends any other target or threat resolved later in
        this same call -- it only stops counting as a defender from the next call (step) onward."""
        n = self.grid_size
        moves = [m for m in self.action_to_move_tuple.values() if m != (0, 0)]
        viable = self._viable_predators()
        ostracized_at_start = {a for a in viable if self._is_ostracized(a)} if self.band_ostracism else set()
        kills = {}
        for tid in sorted(self.threat_positions, key=lambda t: int(t.split("_")[1])):
            tx, ty = self.threat_positions[tid]
            resting = self.current_step < self.threat_rest_until.get(tid, 0)
            alive = sorted(
                (max(abs(px - tx), abs(py - ty)), a) for a, (px, py) in self.predator_positions.items() if a in viable
            )
            nearest = alive[0] if alive and not resting else None  # a resting threat neither attacks nor chases
            adjacent = [a for d, a in alive if d <= 1] if nearest is not None and nearest[0] <= 1 else []
            if adjacent:
                targets = adjacent if self.threat_attack_all_adjacent else adjacent[:1]
                killed_any = False
                stopped = False  # repelled, or attacked and failed to kill: eligible for the cooldown rest
                for target in targets:
                    self.threat_encounters += 1
                    defender_ids = self._defender_ids(target, viable, ostracized_at_start)
                    defenders = len(defender_ids)
                    if self.band_ostracism:
                        self._update_reputation(defender_ids, tx, ty)
                    if defenders >= self.threat_defenders_to_repel:
                        stopped = True
                        blocked = set(self.predator_positions.values()) | {p for t, p in self.threat_positions.items() if t != tid}
                        px, py = self.agent_positions[target]
                        cells = [
                            (max(abs(x - px), abs(y - py)), (x, y)) for x in range(n) for y in range(n) if (x, y) not in blocked
                        ]
                        far = [c for d, c in cells if d >= self.threat_flee_distance]
                        if not far and cells:
                            best = max(d for d, _ in cells)
                            far = [c for d, c in cells if d == best]
                        if far:
                            self.threat_positions[tid] = far[int(self.rng.integers(len(far)))]
                            self.threat_repelled += 1
                        break  # driven off: leaves before attacking any remaining adjacent targets this step
                    if self.rng.random() < self.threat_kill_prob * (1.0 - defenders / self.threat_defenders_to_repel):
                        kills[target] = min(kills.get(target, defenders), defenders)
                        killed_any = True
                    else:
                        stopped = True
                if killed_any:
                    self.threat_rest_until[tid] = self.current_step + self.threat_satiation_steps
                elif stopped:
                    self.threat_rest_until[tid] = self.current_step + self.threat_cooldown_steps
                continue
            occupied = set(self.predator_positions.values()) | {p for t, p in self.threat_positions.items() if t != tid}
            candidates = []
            for dx, dy in moves:
                cell = (min(max(tx + dx, 0), n - 1), min(max(ty + dy, 0), n - 1))
                if cell != (tx, ty) and cell not in occupied:
                    candidates.append(cell)
            if not candidates:
                continue
            if nearest is not None and nearest[0] <= self.threat_sense_radius:
                px, py = self.predator_positions[nearest[1]]
                best = min((px - c[0]) ** 2 + (py - c[1]) ** 2 for c in candidates)
                candidates = [c for c in candidates if (px - c[0]) ** 2 + (py - c[1]) ** 2 == best]
            self.threat_positions[tid] = candidates[int(self.rng.integers(len(candidates)))]
        for target, defenders in kills.items():
            self.agent_energies[target] = -1.0
            self.threat_kills["predator_male" if "predator_male" in target else "predator_female"] += 1
            if defenders == 0:
                self.threat_kills_alone += 1

    def _fill_band_compass(self, observation, agent, band, xp, yp):
        """Channels 8-11: constant planes pointing to the nearest same-band predator anywhere on the grid (see __init__)."""
        best = None
        for other, (ox, oy) in sorted(self.predator_positions.items()):  # sorted: ties go to the lowest agent id
            if (
                other == agent
                or self.agent_band.get(other) != band
                or self.agent_energies[other] <= 0.0
                or self._diet_death_cause(other) is not None  # not a dead or doomed member
            ):
                continue
            d = max(abs(ox - xp), abs(oy - yp))
            if best is None or d < best[0]:
                best = (d, ox - xp, oy - yp)
        if best is None:
            return
        d, dx, dy = best
        norm = float(np.hypot(dx, dy)) or 1.0
        observation[8] = 1.0
        observation[9] = (dx / norm + 1.0) / 2.0
        observation[10] = (dy / norm + 1.0) / 2.0
        observation[11] = d / self.grid_size

    def _kin_blocked(self, male, female):
        """Kin exclusion: no mating between parent and child, or between siblings (a shared parent)."""
        if not self.kin_exclusion:
            return False
        pm, pf = self.agent_parents.get(male, ()), self.agent_parents.get(female, ())
        kin = female in pm or male in pf or bool(set(pm) & set(pf))
        if kin:
            self.kin_blocked_checks += 1
        return kin

    def _record_pairing(self, male, female):
        """A within-band pairing is counted; a cross-band one is a marriage: one partner (marriage_rule) joins the
        other's band, taking their still-dependent children (unreproduced offspring) along."""
        bm, bf = self.agent_band[male], self.agent_band[female]
        if bm == bf:
            self.within_band_pairings += 1
            return
        self.marriages += 1
        mover, target = (female, bm) if self.marriage_rule == "female_joins_male" else (male, bf)
        self.agent_band[mover] = target
        for child, parents in self.agent_parents.items():
            if mover in parents and child not in self.has_reproduced and child in self.agent_band:
                self.agent_band[child] = target

    def _apply_band_share(self, agent, gained, is_fruit):
        """Band sharing (mechanical, like the gifts): a fraction band_share_rate of any forage is split equally among
        the forager's living band members within band_share_range. Meat stays meat and fruit stays fruit (stores are
        preserved). Members already doomed by a store or at <= 0 energy are not rescued (turn-order safe). A member
        currently ostracized (band_ostracism) is excluded from receiving, even if otherwise in range."""
        rate = self.band_fruit_share_rate if is_fruit else self.band_meat_share_rate
        if self.num_bands == 0 or rate <= 0.0 or gained <= 0.0:
            return
        band = self.agent_band.get(agent)
        if band is None:
            return
        x, y = self.agent_positions[agent]
        recipients = [
            other
            for other, (ox, oy) in self.predator_positions.items()
            if other != agent
            and self.agent_band.get(other) == band
            and self.agent_energies[other] > 0.0
            and self._diet_death_cause(other) is None
            and not self._is_ostracized(other)
            and max(abs(ox - x), abs(oy - y)) <= self.band_share_range
        ]
        if not recipients:
            return
        share = rate * gained / len(recipients)
        decay = self.band_share_distance_decay
        amounts = {}
        for other in recipients:
            ox, oy = self.predator_positions[other]
            d = max(abs(ox - x), abs(oy - y))
            amounts[other] = share * (1.0 - decay * d / (self.band_share_range + 1.0))
        total = sum(amounts.values())  # what is actually delivered (== rate * gained when decay is 0)
        self.agent_energies[agent] -= total
        if is_fruit:
            self.agent_fruit_store[agent] -= total
        self.grid_world_state[1, x, y] = self.agent_energies[agent]
        for other, amount in amounts.items():
            self.agent_energies[other] += amount
            if is_fruit:
                self.agent_fruit_store[other] += amount
            self.grid_world_state[1, *self.agent_positions[other]] = self.agent_energies[other]
        self.band_share_events += 1
        if is_fruit:
            self.band_share_fruit_total += total
        else:
            self.band_share_meat_total += total

    def _apply_female_gift(self, agent, fruit_gained):
        """Reciprocal counterpart of _apply_male_gift: a predator_female donates female_gift_donation_rate of
        the fruit she just ate to HER RECORDED MATE (fruit-derived energy: her fruit store and total energy fall
        by the donation, his rise). Same rules as the male gift (recorded mate alive, positive energy, within
        predator_gift_range), mechanically executed for the same credit-assignment reason."""
        if fruit_gained <= 0.0 or self.female_gift_donation_rate <= 0.0:
            return
        mate = self.agent_mate.get(agent)
        if (
            mate is None
            or mate not in self.agent_positions
            or self.agent_energies[mate] <= 0.0
            or self._diet_death_cause(mate) is not None  # already doomed: not rescued by an accident of turn order
        ):
            return
        position = self.agent_positions[agent]
        mate_position = self.agent_positions[mate]
        if max(abs(mate_position[0] - position[0]), abs(mate_position[1] - position[1])) > self.predator_gift_range:
            return
        donation = self.female_gift_donation_rate * fruit_gained
        self.agent_energies[agent] -= donation
        self.agent_fruit_store[agent] -= donation
        self.grid_world_state[1, *position] = self.agent_energies[agent]
        self.agent_energies[mate] += donation
        self.agent_fruit_store[mate] += donation
        self.grid_world_state[1, *mate_position] = self.agent_energies[mate]
        self.female_gift_events += 1
        self.female_gift_energy_total += donation

    def _debit(self, agent, amount):
        """Charge a predator a running/birth cost: total energy falls by `amount`, drawn from the meat and fruit
        stores in the fixed shares diet_meat_cost_share / (1 - diet_meat_cost_share)."""
        self.agent_energies[agent] -= amount
        self.agent_fruit_store[agent] -= amount * (1.0 - self.diet_meat_cost_share)

    def _meat_store(self, agent):
        return self.agent_energies[agent] - self.agent_fruit_store[agent]

    def _diet_death_cause(self, agent):
        """'fruit' or 'meat' if a predator's store is exhausted while diet_required (else None). Only meaningful
        for predators; total-energy starvation is handled separately by the caller."""
        if not self.diet_required or "predator" not in agent:
            return None
        if self.agent_fruit_store[agent] <= 0.0:
            return "fruit"
        if self._meat_store(agent) <= 0.0:
            return "meat"
        return None

    def _can_reproduce(self, agent):
        """Energy threshold, plus (when diet_required) a minimum in BOTH stores: an all-fruit or all-meat
        predator cannot breed however much total energy it holds."""
        if self.agent_energies[agent] < self.predator_creation_energy_threshold:
            return False
        if not self.diet_required:
            return True
        floor = self.predator_min_store_fraction_for_reproduction * self.predator_creation_energy_threshold
        # Each store must also stay above zero AFTER this parent pays its share of the birth cost (drawn from the
        # stores in the fixed shares), for the larger of the two possible offspring energies; otherwise a parent
        # could breed and die of a diet deficiency on the very next step.
        share = self.predator_birth_cost_share_female if "predator_female" in agent else self.predator_birth_cost_share_male
        cost = share * max(self.initial_energy_predator_male, self.initial_energy_predator_female)
        need_fruit = max(floor, cost * (1.0 - self.diet_meat_cost_share) + 1e-9)
        need_meat = max(floor, cost * self.diet_meat_cost_share + 1e-9)
        return self.agent_fruit_store[agent] >= need_fruit and self._meat_store(agent) >= need_meat

    def _get_movement_energy_cost(self, agent, action):
        if action == self.noop_action_id:
            return 0.0
        return self.move_energy_cost_per_step_predator if "predator" in agent else self.move_energy_cost_per_step_prey

    def _get_move(self, agent: AgentID, action: int) -> Tuple[int, int]:
        agent_type_nr = 1 if "predator" in agent else 2
        current_position = self.agent_positions[agent]
        move_vector = self.action_to_move_tuple[action]
        new_position = (current_position[0] + move_vector[0], current_position[1] + move_vector[1])
        new_position = tuple(np.clip(new_position, 0, self.grid_size - 1))
        if self.grid_world_state[agent_type_nr, *new_position] > 0:
            new_position = current_position

        return new_position

    def _get_observation(self, agent):
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, observation_range)
        observation = np.zeros(
            (self.num_obs_channels, observation_range, observation_range),
            dtype=np.float64,
        )
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]
        # Last channel: the fruit store of every predator in the window (own included), drawn at its cell. With the
        # predator channel (total energy) the meat store is derivable (E - F); nothing here reveals sex or identity.
        # Channels 6 and 7 (bands): a marker at every same-band / other-band predator in the window (the observer itself
        # is not marked); all zero when num_bands == 0 or the observer is prey.
        store_channel, same_channel, other_channel = 5, 6, 7
        observation[store_channel:].fill(0.0)
        my_band = self.agent_band.get(agent) if self.num_bands > 0 and "predator" in agent else None
        for other, (ox, oy) in self.predator_positions.items():
            rx, ry = ox - xp + observation_range // 2, oy - yp + observation_range // 2
            if 0 <= rx < observation_range and 0 <= ry < observation_range and other in self.agent_fruit_store:
                observation[store_channel, rx, ry] = max(self.agent_fruit_store[other], 0.0)
                if my_band is not None and other != agent and other in self.agent_band:
                    observation[same_channel if self.agent_band[other] == my_band else other_channel, rx, ry] = 1.0
        if self.band_compass and my_band is not None:
            self._fill_band_compass(observation, agent, my_band, xp, yp)
        if self.num_threats > 0 and "predator" in agent:
            threat_channel = 8 + (4 if self.band_compass else 0)
            for tx, ty in self.threat_positions.values():
                rx, ry = tx - xp + observation_range // 2, ty - yp + observation_range // 2
                if 0 <= rx < observation_range and 0 <= ry < observation_range:
                    observation[threat_channel, rx, ry] = 1.0
        if self.num_mammoths > 0 and "predator" in agent:
            mammoth_channel = 8 + (4 if self.band_compass else 0) + (1 if self.num_threats > 0 else 0)
            for mx, my in self.mammoth_positions.values():
                rx, ry = mx - xp + observation_range // 2, my - yp + observation_range // 2
                if 0 <= rx < observation_range and 0 <= ry < observation_range:
                    observation[mammoth_channel, rx, ry] = self.mammoth_energy

        return observation

    def _obs_clip(self, x, y, observation_range):
        observation_offset = (observation_range - 1) // 2
        xld, xhd = x - observation_offset, x + observation_offset
        yld, yhd = y - observation_offset, y + observation_offset
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(xhd, 0, self.grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(yhd, 0, self.grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -observation_offset, 0)), abs(np.clip(yld, -observation_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _get_agent_by_position(self) -> dict:
        return {position: agent for agent, position in self.agent_positions.items()}

    def _remove_agent(self, agent: AgentID):
        """Removes an agent from all tracking dictionaries."""
        del self.agent_positions[agent]
        del self.agent_energies[agent]
        self.agent_fruit_store.pop(agent, None)
        self.agent_band.pop(agent, None)
        self.defense_opportunities.pop(agent, None)
        self.defense_exposures.pop(agent, None)
        self.ostracized_until.pop(agent, None)

        if "predator_male" in agent:
            del self.predator_positions[agent]
            self.current_num_predator_male -= 1
        elif "predator_female" in agent:
            del self.predator_positions[agent]
            self.current_num_predator_female -= 1
        elif "prey" in agent:
            del self.prey_positions[agent]
            self.current_num_prey -= 1

    def _find_available_spawn_position(self, reference_position, occupied_positions):
        """
        Finds an available position for spawning a new agent.
        Tries to spawn near the reference position first before selecting a random free position.
        """
        x, y = reference_position
        potential_positions = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
        ]

        valid_positions = [pos for pos in potential_positions if pos not in occupied_positions]

        if valid_positions:
            return valid_positions[0]

        all_positions = {(i, j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(all_positions - occupied_positions)

        if free_positions:
            return free_positions[self.rng.integers(len(free_positions))]

        return None

    def get_state_snapshot(self):
        return {
            "current_step": self.current_step,
            "agent_positions": self.agent_positions.copy(),
            "agent_energies": self.agent_energies.copy(),
            "predator_positions": self.predator_positions.copy(),
            "prey_positions": self.prey_positions.copy(),
            "grass_positions": self.grass_positions.copy(),
            "grass_energies": self.grass_energies.copy(),
            "fruit_positions": self.fruit_positions.copy(),
            "fruit_energies": self.fruit_energies.copy(),
            "grid_world_state": self.grid_world_state.copy(),
            "agents": self.agents.copy(),
            "cumulative_rewards": self.cumulative_rewards.copy(),
            "agent_mate": self.agent_mate.copy(),
            "agent_fruit_store": self.agent_fruit_store.copy(),
            "scripted_prey_ids": self._scripted_prey_ids.copy(),
            "agent_band": self.agent_band.copy(),
            "rng_state": self.rng.bit_generator.state,
            "threat_positions": dict(self.threat_positions),
            "threat_rest_until": dict(self.threat_rest_until),
            "mammoth_positions": dict(self.mammoth_positions),
            "mammoth_respawn_at": dict(self.mammoth_respawn_at),
            "pending_rewards": dict(self._pending_rewards),
            "mammoth_counters": (
                list(self.mammoth_attempts), list(self.mammoth_kills), list(self.mammoth_party_deaths),
                self.mammoth_blocked, self.mammoth_energy_distributed,
            ),
            "threat_counters": (
                self.threat_encounters, dict(self.threat_kills), self.threat_kills_alone, self.threat_repelled,
            ),
            "ostracism_state": (
                dict(self.defense_opportunities), dict(self.defense_exposures), dict(self.ostracized_until),
                self.ostracism_events,
            ),
            "band_dynamics": (
                self._next_band_id, dict(self._band_contact), dict(self._band_out_steps),
                self.band_fissions, self.band_fusions, self.band_drift_outs,
            ),
            "band_counters": (
                self.band_share_events, self.band_share_meat_total, self.band_share_fruit_total,
                self.marriages, self.within_band_pairings, self.kin_blocked_checks,
            ),
            "female_gift_events": self.female_gift_events,
            "female_gift_energy_total": self.female_gift_energy_total,
            "diet_deaths": {sex: dict(v) for sex, v in self.diet_deaths.items()},
            "agent_parents": self.agent_parents.copy(),
            "has_reproduced": self.has_reproduced.copy(),
            "current_num_predator_male": self.current_num_predator_male,
            "current_num_predator_female": self.current_num_predator_female,
            "current_num_prey": self.current_num_prey,
            "agents_just_ate": self.agents_just_ate.copy(),
            "pending_removal": self._pending_removal.copy(),
            "next_predator_male_idx": self._next_predator_male_idx,
            "next_predator_female_idx": self._next_predator_female_idx,
            "next_prey_idx": self._next_prey_idx,
        }

    def restore_state_snapshot(self, snapshot):
        self.current_step = snapshot["current_step"]
        self.agent_positions = snapshot["agent_positions"].copy()
        self.agent_energies = snapshot["agent_energies"].copy()
        self.predator_positions = snapshot["predator_positions"].copy()
        self.prey_positions = snapshot["prey_positions"].copy()
        self.grass_positions = snapshot["grass_positions"].copy()
        self.grass_energies = snapshot["grass_energies"].copy()
        self.fruit_positions = snapshot["fruit_positions"].copy()
        self.fruit_energies = snapshot["fruit_energies"].copy()
        self.grid_world_state = snapshot["grid_world_state"].copy()
        self.agents = snapshot["agents"].copy()
        self.cumulative_rewards = snapshot["cumulative_rewards"].copy()
        self.agent_mate = snapshot["agent_mate"].copy()
        self.agent_fruit_store = snapshot["agent_fruit_store"].copy()
        self._scripted_prey_ids = snapshot["scripted_prey_ids"].copy()
        self.agent_band = snapshot["agent_band"].copy()
        (
            self._next_band_id, contact, out_steps, self.band_fissions, self.band_fusions, self.band_drift_outs,
        ) = snapshot["band_dynamics"]
        self._band_contact, self._band_out_steps = dict(contact), dict(out_steps)
        self.rng.bit_generator.state = snapshot["rng_state"]
        self.threat_positions = dict(snapshot["threat_positions"])
        self.threat_rest_until = dict(snapshot["threat_rest_until"])
        self.mammoth_positions = dict(snapshot["mammoth_positions"])
        self.mammoth_respawn_at = dict(snapshot["mammoth_respawn_at"])
        self._pending_rewards = dict(snapshot["pending_rewards"])
        (
            attempts, kills, deaths, self.mammoth_blocked, self.mammoth_energy_distributed,
        ) = snapshot["mammoth_counters"]
        self.mammoth_attempts, self.mammoth_kills, self.mammoth_party_deaths = list(attempts), list(kills), list(deaths)
        (self.threat_encounters, kills, self.threat_kills_alone, self.threat_repelled) = snapshot["threat_counters"]
        self.threat_kills = dict(kills)
        (opportunities, exposures, ostracized, self.ostracism_events) = snapshot["ostracism_state"]
        self.defense_opportunities, self.defense_exposures, self.ostracized_until = (
            dict(opportunities), dict(exposures), dict(ostracized),
        )
        (
            self.band_share_events, self.band_share_meat_total, self.band_share_fruit_total,
            self.marriages, self.within_band_pairings, self.kin_blocked_checks,
        ) = snapshot["band_counters"]
        self.female_gift_events = snapshot["female_gift_events"]
        self.female_gift_energy_total = snapshot["female_gift_energy_total"]
        self.diet_deaths = {sex: dict(v) for sex, v in snapshot["diet_deaths"].items()}
        self.agent_parents = snapshot["agent_parents"].copy()
        self.has_reproduced = snapshot["has_reproduced"].copy()
        self.current_num_predator_male = snapshot["current_num_predator_male"]
        self.current_num_predator_female = snapshot["current_num_predator_female"]
        self.current_num_prey = snapshot["current_num_prey"]
        self.agents_just_ate = snapshot["agents_just_ate"].copy()
        self._pending_removal = snapshot["pending_removal"].copy()
        self._next_predator_male_idx = snapshot["next_predator_male_idx"]
        self._next_predator_female_idx = snapshot["next_predator_female_idx"]
        self._next_prey_idx = snapshot["next_prey_idx"]
