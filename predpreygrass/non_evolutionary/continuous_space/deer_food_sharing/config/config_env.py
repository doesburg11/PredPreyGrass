"""Environment settings. Edit this dictionary before running train.py.

Starts from environment.DEFAULT_CONFIG (deer_stag_hunt's defaults) and sets
this module's world (README.md, DESIGN.md): deer_stag_hunt's touch rule
(its RESULTS.md section 2) with adult catches shared, s = 0.5 within
R = 128. Change values here, not there.
"""

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import DEFAULT_CONFIG

config_env = dict(DEFAULT_CONFIG)
config_env.update({"deer_catch_rule": "touch", "adult_share_fraction": 0.5, "adult_share_radius": 128.0})
