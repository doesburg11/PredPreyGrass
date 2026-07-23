from predpreygrass.evolutionary.eco_evolutionary_investment.config.config_env_eco_evolutionary import config_env as _base_config_env

# Trial 6 (population scaling): identical to the real experiment config in every
# respect except grid/population size, so any change in detectable genome-drift
# signal versus R7 is attributable to population scale (shrinking the neutral-drift
# noise floor), not some other config difference. See predpreygrass/evolutionary/RESULTS.md.
#
# Scaled to ~2x (down from an initial 4x attempt that cost ~2.75 days/seed and ran
# GPU memory to the edge -- see RESULTS.md Trial 6 for that history). Area scales
# ~1.96x (25x25 -> 35x35, closest clean grid_size to 2x area); initial population
# and grass scale 2x to match, keeping agent density ~unchanged from the base
# config (not a denser or sparser ecology, just more individuals). n_possible_*
# (the agent-ID pool ceiling, not a carrying-capacity target) is raised generously
# above the expected steady state.
config_env = {
    **_base_config_env,
    "grid_size": 35,
    "initial_num_grass": 200,
    "n_initial_active_predators": 12,
    "n_initial_active_prey": 16,
    "n_possible_predators": 400,
    "n_possible_prey": 2000,
}
