"""Forward pass and local reinforcement learning for Trial 13's prey genomes.

Re-exported unchanged from eco_evolutionary_erl_baldwin/networks.py: `evaluate`,
`action_probs`, `sample_action`, and `reinforce_update` are pure `obs @ weights`
matrix operations parameterized entirely by array shape (obs_dim, n_actions) --
nothing in them is specific to Trial 12's 7-channel/4-action setup, so they apply
verbatim here with obs_dim=8, n_actions=9 (see features.py, config.py).
"""

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import (  # noqa: F401
    action_probs,
    evaluate,
    reinforce_update,
    sample_action,
)
