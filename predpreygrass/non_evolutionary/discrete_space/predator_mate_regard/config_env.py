"""Configuration for Iteration 21: compound-gift environment plus a mate-regard reward weight."""
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.config_env import config_env as BASE_CONFIG

config_env = {
    **BASE_CONFIG,
    "compound_gift": "on",
    "mate_regard_weight": 0.0,  # set per arm: 0.0 (W0) or 0.5 (W05)
}
