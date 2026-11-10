"""Environment settings. Edit this dictionary before running train.py or
evolve.py.

Starts from environment.DEFAULT_CONFIG (deer_stag_hunt's world plus the
hunters' genes); change values here, not there. RESULTS.md section 2 (the
live stage) uses the touch rule; section 1's runs used the party rule and
keep it in their saved settings.
"""

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import DEFAULT_CONFIG

config_env = dict(DEFAULT_CONFIG, deer_catch_rule="touch")
