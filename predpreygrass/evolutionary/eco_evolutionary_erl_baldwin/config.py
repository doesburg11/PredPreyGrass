"""Simulation parameters for eco_evolutionary_erl_baldwin, matching Ackley &
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

config_erl = {
    "seed": 41,
    "strategy": "ERL",  # one of ERL, E, L, F, B -- see world.py::ErlWorld docstring

    # --- World AL structure ---
    "grid_size": 100,  # paper: 100x100, non-toroidal
    "agent_sense_range": 4,  # paper: agents see 4 cells in each compass direction
    "carnivore_sense_range": 6,  # paper: carnivores see 6 cells
    "n_initial_agents": 60,  # not specified by paper; chosen to give a workable starting population at this grid size
    "n_initial_carnivores": 5,  # not specified; carnivores also spawn continuously (see below)
    "carnivore_spawn_interval": 200,  # paper: exact number, Figure 4
    "founder_weight_std": 0.5,

    # --- Walls (permanent, placed once at reset) ---
    "wall_interior_density": 0.02,  # fraction of non-border cells that are walls; not specified by paper
    "wall_damage": 5.0,  # health damage for walking into a wall; not specified

    # --- Trees (shelter from carnivores; infrequent birth/death) ---
    "tree_birth_prob": 0.0005,  # per empty-cell, per-step; not specified ("infrequent")
    "tree_death_prob": 0.0005,  # per existing tree, per-step; not specified
    "min_trees": 20,  # reseed floor; not specified, chosen by analogy to min_plants

    # --- Plants (agents' food; geometric growth up to a crowding limit) ---
    "plant_growth_prob": 0.02,  # per empty-cell, per-step; not specified
    "plant_crowding_limit_frac": 0.3,  # max fraction of empty cells that can hold a plant; not specified
    "min_plants": 50,  # paper: exact number, Figure 4
    "plant_energy": 3.0,  # energy gained eating a plant ("eat all"); not specified

    # --- Agent (the single adaptive species) energetics ---
    "initial_energy_agent": 5.0,
    "initial_health_agent": 10.0,
    "max_energy_agent": 15.0,
    "max_health_agent": 10.0,
    "basal_energy_cost_agent": 0.05,  # per step; not specified
    "move_energy_cost_agent": 0.02,  # per move; not specified
    "health_regen_agent": 0.05,  # passive recovery per step when not at max; paper says this happens, not the rate
    "reproduction_energy_threshold_agent": 10.0,  # "sufficiently nourished"; not specified
    "reproduction_energy_cost_agent": 5.0,  # not specified
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
    "carnivore_reproduction_energy_threshold": 14.0,  # "reproduce by eating enough agents"; not specified
    "carnivore_reproduction_energy_cost": 7.0,

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
