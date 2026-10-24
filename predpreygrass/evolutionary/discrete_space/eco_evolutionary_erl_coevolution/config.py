"""Simulation parameters for eco_evolutionary_erl_coevolution.

`config_step0` is eco_evolutionary_erl_baldwin's config exactly as it stood
at commit 47534a0 -- the run that produced that module's §9 comparative
study (ERL beats E/L/F/B, p<0.00001, n=100/condition). Do not edit it:
tests/test_step0_reproduces_study.py pins it against §9's actual
extinction steps. `config_step1` layers the step-1 changes on top (see its
comment block below). `config_erl` is kept as an alias of `config_step0`.

Original (47534a0) docstring follows -- it describes World AL, matching Ackley &
Littman's World AL (1991) structure as closely as the paper specifies.

IMPORTANT LIMITATION, read before trusting any number below as "faithful":
the paper describes World AL's mechanics qualitatively ("minor damage",
"geometric growth up to a crowding limit", "sufficiently nourished",
"sufficiently damaged or hungry") and never publishes the actual numeric
constants they used for damage amounts, energy thresholds, or growth
probabilities. Only a handful of exact numbers appear in the text/figures at
all:
  - grid_size = 100 (100x100, non-toroidal)
  - min_plants = 50 (reseed floor)
  - carnivore_spawn_interval = 200 (a new carnivore every 200 steps)
  - carnivore_sense_range = 6, agent_sense_range = 4 (visual input range)
  - step limit = 1,000,000 (comparative-study ceiling)
Every other constant below (damage amounts, growth probabilities, energy
thresholds, wall density, tree birth/death rates) is *my own chosen value*,
picked to be reasonable given the qualitative description -- not recovered
from the paper, because the paper doesn't contain it. See README.md.
"""

config_step0 = {
    "seed": 41,
    "strategy": "ERL",  # one of ERL, E, L, F, B -- see world.py::ErlWorld docstring

    # --- World AL structure ---
    "grid_size": 100,  # paper: 100x100, non-toroidal
    "agent_sense_range": 4,  # paper: agents see 4 cells in each compass direction
    "carnivore_sense_range": 6,  # paper: carnivores see 6 cells
    "n_initial_agents": 60,  # not specified by paper; chosen to give a workable starting population at this grid size
    "n_initial_carnivores": 5,  # not specified; carnivores also spawn continuously (see below)
    "carnivore_spawn_interval": 200,  # paper: exact number, Figure 4
    "carnivore_immigration_until": None,  # NEW (not in 47534a0): step after which immigration stops; None = never
    "founder_weight_std": 0.5,

    # --- Walls (permanent, placed once at reset) ---
    "wall_interior_density": 0.02,  # fraction of non-border cells that are walls; not specified by paper
    "wall_damage": 5.0,  # health damage for walking into a wall; not specified

    # --- Trees (shelter from carnivores; infrequent birth/death) ---
    "tree_birth_prob": 0.0005,  # per empty-cell, per-step; not specified ("infrequent")
    "tree_death_prob": 0.0005,  # per existing tree, per-step; not specified
    # min_trees raised 20->100 in the 2026-08-10 retune: the paper's stated purpose
    # for trees is specifically shelter from carnivores ("provide shelter for
    # agents from carnivores but no food"), and a full-scale run at 20 gave agents
    # almost nowhere to flee to -- see RESULTS.md for the diagnosis (all 5
    # strategies died at statistically the same rate, no room for behavior to
    # matter). More trees is the paper-consistent lever: it's the *escape*
    # mechanism the paper describes, not a change to combat/damage numbers,
    # which the paper explicitly says should stay lethal ("the agent always loses").
    "min_trees": 100,  # reseed floor; not specified, retuned 2026-08-10 (was 20)

    # --- Plants (agents' food; geometric growth up to a crowding limit) ---
    "plant_growth_prob": 0.02,  # per empty-cell, per-step; not specified
    "plant_crowding_limit_frac": 0.3,  # max fraction of empty cells that can hold a plant; not specified
    "min_plants": 50,  # paper: exact number, Figure 4
    "plant_energy": 3.0,  # energy gained eating a plant ("eat all"); not specified

    # --- Agent (the single adaptive species) energetics ---
    "initial_energy_agent": 5.0,
    "initial_health_agent": 10.0,
    # max_energy_agent raised 15.0->30.0 and reproduction_energy_threshold_agent
    # 10.0->22.0 in the 2026-08-10 retune, root-caused via direct trace: at the
    # old values, agents reproduced after ~2 plant bites, and abundant plants
    # (plant_crowding_limit_frac=0.3 allows ~2,700 on a 100x100 grid) let the
    # founder population of 60 explode to 1,916 agents by step 120 -- a
    # population overshoot the environment can't sustain, which then fed an
    # equally explosive carnivore boom (6->739 by step 280) that wiped
    # everything out by step ~400, at statistically the same rate regardless
    # of `strategy`. This is the actual root cause diagnosed after the first
    # retune attempt (throttling carnivore reproduction alone) didn't help --
    # see RESULTS.md. Raising the agent reproduction bar directly targets the
    # trigger, not the downstream symptom.
    "max_energy_agent": 30.0,  # was 15.0
    "max_health_agent": 10.0,
    "basal_energy_cost_agent": 0.05,  # per step; not specified
    "move_energy_cost_agent": 0.02,  # per move; not specified
    "health_regen_agent": 0.05,  # passive recovery per step when not at max; paper says this happens, not the rate
    "reproduction_energy_threshold_agent": 22.0,  # was 10.0
    "reproduction_energy_cost_agent": 10.0,  # was 5.0
    "corpse_bite_energy": 2.0,  # energy per "eat some" bite off a corpse; not specified
    "corpse_total_energy": 6.0,  # total energy in a corpse before it's fully consumed; not specified
    "agent_attack_damage": 1.5,  # damage dealt by "Living agent -> Damage other" (agent-on-agent aggression); not specified
    "mate_search_radius": 3,

    # --- Carnivore (non-adaptive, hard-coded FSA; never affected by `strategy`) ---
    "initial_energy_carnivore": 8.0,
    "initial_health_carnivore": 15.0,
    "max_energy_carnivore": 20.0,
    "max_health_carnivore": 15.0,
    "basal_energy_cost_carnivore": 0.08,
    "move_energy_cost_carnivore": 0.03,
    "health_regen_carnivore": 0.05,
    "carnivore_attack_damage": 6.0,  # paper: "in a slugfest ... the agent always loses" -> must clearly exceed agent_attack_damage
    # carnivore_reproduction_energy_threshold/cost retuned 2026-08-10: at 14.0/7.0
    # against max_energy_carnivore=20.0, carnivores reproduced easily on top of
    # the paper's own fixed spawn-every-200-steps rate, creating a runaway
    # population feedback loop (more carnivores -> more agents killed -> more-fed
    # carnivores -> more carnivore reproduction) not described in the paper at
    # all -- diagnosed as the likely cause of every strategy dying at the same
    # rate in the first full-scale run (see RESULTS.md). Raised close to the
    # energy cap so the fixed 200-step spawn interval (the paper's actual,
    # published carnivore-growth mechanism) dominates population growth instead.
    "carnivore_reproduction_energy_threshold": 18.0,  # was 14.0
    "carnivore_reproduction_energy_cost": 10.0,  # was 7.0
    "carnivore_energy_conserving_birth": False,  # NEW (not in 47534a0): True = newborn gets exactly the parent's cost, not initial_energy_carnivore
    # --- Step 2: carnivore genome (NEW, not in 47534a0; "fsa" = the hand-coded rule) ---
    "carnivore_mode": "fsa",  # "fsa" | "genome" | "genome_neutral" -- see world.py docstring
    "carnivore_obs": "basic",  # "basic" (10) | "rich" (18: prey split living/sheltered/corpse) | "rich_memory" (22: + previous move)
    "carnivore_seed_pursuit_weight": 10.0,  # founder weight prey-signal_i -> action_i ("basic")
    "carnivore_seed_living_weight": 10.0,  # "rich" seed: all three prey channels equal = same behavior as basic
    "carnivore_seed_sheltered_weight": 10.0,
    "carnivore_seed_corpse_weight": 10.0,
    "carnivore_seed_prev_weight": 0.0,  # "rich_memory": previous move i -> action i; 0 = no built-in persistence
    "carnivore_seed_block_weight": -10.0,  # founder weight blocked_i -> action_i
    "carnivore_founder_weight_std": 1.0,  # per-founder variation on every weight/bias
    "carnivore_founder_prev_std": None,  # "rich_memory": if set, founder persistence weights ~ N(seed, this) instead
    # --- Step 3: learning carnivores ("erl") ---
    "carnivore_seed_eval_energy_weight": 5.0,  # founders' innate eval: + this * energy_norm (a guess; evolvable)
    "carnivore_lr_positive": 0.05,  # same as the prey's lr_positive / lr_negative
    "carnivore_lr_negative": 0.02,
    "carnivore_founder_eval_std": None,  # "erl": founder eval-network noise (None = carnivore_founder_weight_std; 0 = pure energy goal)
    "carnivore_reward_baseline": None,  # e.g. 0.01: subtract a running mean of reinforcement (None = off)
    "carnivore_trace_decay": None,  # e.g. 0.9: eligibility trace over recent moves (None = one-step, like the prey)
    "carnivore_mutation_rate": 0.05,  # same per-site rate as the prey
    "carnivore_mutation_std": 0.2,  # prey use 0.05 on ~0.5-scale weights; carnivore weights are ~10-scale
    "mixed_mutant_pursuit_weight": 10.0,  # "mixed" competition test: the mutant type's network
    "mixed_mutant_block_weight": -10.0,  # (defaults = identical to the resident, the neutral check)
    "mixed_mutant_living_weight": 10.0,  # "rich" mutant channels (defaults = identical to the resident)
    "mixed_mutant_sheltered_weight": 10.0,
    "mixed_mutant_corpse_weight": 10.0,
    "mixed_mutant_prev_weight": 0.0,
    "mixed_mutant_strategy": "network",  # or a hand-coded state-dependent variant: persist | sated_scavenger | wounded_scavenger
    "mixed_persist_prob": 0.9,
    "mixed_sated_energy_frac": 0.75,
    "mixed_wounded_health_frac": 0.5,
    "mixed_assign_step": None,  # None = types from the start; else split 50/50 at this step (after the opening)

    # --- Genome mutation (unchanged mechanism from earlier version) ---
    "mutation_rate": 0.05,
    "mutation_std": 0.05,

    # --- Local reinforcement learning (within-lifetime; live action network only) ---
    # NOTE: still a REINFORCE-style approximation of the paper's exact CRBP
    # algorithm (documented in networks.py), not rebuilt as part of this
    # world/mechanics pass -- see README.md.
    "lr_positive": 0.05,
    "lr_negative": 0.02,

    "max_population_cap": 2000,  # safety valve (pure-Python performance), not in the paper
}

config_erl = config_step0

# --- Step 1: carnivores regulated by prey supply, not by immigration ---
# In step 0 (the paper's setup, and §9), carnivore numbers are dominated by a
# fixed immigration of one new carnivore every 200 steps; their own
# reproduction threshold was deliberately raised close to their energy cap
# (47534a0's retune) so it would almost never fire. The longitudinal run
# ended at 878 agents vs. 5 carnivores -- carnivore numbers there don't track
# prey at all, so there is no predator-prey feedback to coevolve against.
#
# Step 1 keeps step 0 exactly for a 20k-step warm-up (immigration on, so
# naive prey learn and evolve under the same predation §9 had), then switches
# immigration OFF: from then on carnivore numbers rise and fall with prey only,
# through individual energy budgets (no census rule, no respawn). Story: an
# established prey population meets a carnivore population that must now live
# off it alone. Carnivore behavior stays the fixed hand-coded FSA -- the
# question is only sustainability (do both species coexist?).
#
# Chosen by calibration (README.md, sweeps 1-4, 2026-09-28): starting without
# immigration fails in ~90% of runs (opening boom-bust on naive prey); faster
# carnivore breeding only looked better through a free-energy birth artifact.
# This setting uses §9's own carnivore parameters (energy-lossy births) and
# coexisted in 6/6 ERL seeds over the 40k steps after immigration stopped.
AREA_SCALED_KEYS = (
    "n_initial_agents", "n_initial_carnivores", "min_plants", "min_trees", "max_population_cap",
)


def scale_world_area(cfg: dict, grid_size: int) -> dict:
    """Resize the world while holding every density fixed. Per-cell rates
    (plant growth, crowding limit, wall density, tree birth/death) scale on
    their own; the absolute counts in AREA_SCALED_KEYS scale with area, and
    the immigration interval shrinks so immigrants per area stay the same."""
    ratio = (grid_size / cfg["grid_size"]) ** 2
    out = dict(cfg, grid_size=grid_size)
    for key in AREA_SCALED_KEYS:
        out[key] = round(cfg[key] * ratio)
    if cfg["carnivore_spawn_interval"] > 0:
        out["carnivore_spawn_interval"] = round(cfg["carnivore_spawn_interval"] / ratio)
    return out


STEP1_OVERRIDES = {
    "carnivore_immigration_until": 20_000,  # step 0: None (immigration forever)
    # With no immigration left, carnivore extinction is permanent: the rest of
    # the run would be predator-free, which can't show coevolution. End the run
    # there and record it as the coexistence time.
    "end_on_carnivore_extinction": True,
}

# 150x150 at step-0 densities (decided 2026-09-28): on 100x100, carnivores dip
# to 2-14 at cycle lows and die out by chance in ~half the runs; at 150x150
# the lows stay at 18-54 and 19/19 ERL runs coexisted (README.md).
STEP1_GRID_SIZE = 150

config_step1 = scale_world_area({**config_step0, **STEP1_OVERRIDES}, STEP1_GRID_SIZE)


# --- Step 2: carnivores carry a genome (evolution, no learning) ---
# Seeded network + sexual reproduction (decided 2026-09-29); step2_neutral is the
# neutral-marker control (same inheritance, genome not expressed -- all carnivores
# act with the canonical seed network), so its genome change is pure drift.
config_step2 = {**config_step1, "carnivore_mode": "genome"}
config_step2_neutral = {**config_step1, "carnivore_mode": "genome_neutral"}
# Matched control (after the pilot): same starting competence and phenotypic
# variation as step2, but behavior is non-heritable (fresh seed+noise per birth).
config_step2_nonheritable = {**config_step1, "carnivore_mode": "genome_nonheritable"}

PRESETS = {
    "step0": config_step0,
    "step1": config_step1,
    "step2": config_step2,
    "step2_neutral": config_step2_neutral,
    "step2_nonheritable": config_step2_nonheritable,
    "mixed": {**config_step1, "carnivore_mode": "mixed"},
}

