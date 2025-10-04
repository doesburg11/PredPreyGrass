from .config_env_base import config_env_base as _base


# Lineage-reward variant of the perimeter-four-gaps walls config.
# - Enables Tier-1 lineage reward accounting.
# - Disables direct reproduction rewards so lineage counts are used instead.
# - Optionally set lineage windows (global and/or per-species) here.

config_env = {
    **_base,
    # Use windowed, living-offspring lineage reward
    "lineage_reward_enabled": True,
    # Disable direct reproduction reward so only lineage contributes
    "reproduction_reward_enabled": False,
    # Choose window size (steps) globally or per species; tweak as needed
    "lineage_reward_window": 150,
    # You can override per species with:
    # "lineage_reward_window_predator": 150,
    # "lineage_reward_window_prey": 150,
    # ---- Enable sharing to surface helping metrics ----
    "share_enabled": True,
    "share_roles": ["prey"],
    # Keep radius small to encourage local help
    "share_radius": 1,
    # Make sharing feasible early in training
    "share_amount": 1.0,
    "share_efficiency": 0.8,
    "share_donor_min": 4.0,
    "share_donor_safe": 2.0,
    "share_cooldown": 2,
    "share_respect_los": True,
    "share_kin_only": True,
    # Observation helpers (disable action_mask to avoid Dict-obs encoder issues in RLlib new API)
    "action_mask_enabled": False,
    "include_kin_energy_channel": True,
    "kin_energy_respect_los": True,
}
