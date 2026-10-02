"""Configuration for the linear SARSA(lambda) Baldwin variant.

The ecology and evolutionary parameters are inherited unchanged from the
validated ``eco_evolutionary_erl_baldwin`` baseline.  Only the within-life
learner is replaced.
"""

from copy import deepcopy

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import (
    config_erl as _baseline_config,
)


config_sarsa = deepcopy(_baseline_config)
config_sarsa.update(
    {
        # Q(s, a) = obs @ action_weights[:, a] + action_bias[a].  These are
        # alpha=0.05 was selected on development seeds 41-43, then evaluated
        # without retuning on held-out seeds 44-51 (see RESULTS.md).
        "sarsa_alpha": 0.05,
        "sarsa_gamma": 1.0,
        # lambda=0.9 failed the 3,000-step stability gate on 2/3 fresh seeds;
        # lambda=0 survived 3/3 with sampled Q maxima remaining finite (see RESULTS.md).
        "sarsa_lambda": 0.0,
        "sarsa_temperature": 1.0,
        # A fatal transition uses the final observable evaluator difference,
        # has zero bootstrap, and adds this explicit bonus once.  Zero retains
        # the inherited evaluator's semantics without an ad-hoc death penalty.
        "sarsa_terminal_bonus": 0.0,
    }
)

# Compatibility alias for callers patterned after the baseline module.
config_erl = config_sarsa
