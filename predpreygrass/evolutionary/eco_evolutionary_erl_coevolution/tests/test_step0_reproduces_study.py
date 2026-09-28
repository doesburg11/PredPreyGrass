"""Step 0 guard: config_step0 must reproduce eco_evolutionary_erl_baldwin's §9
comparative study (commit 47534a0) run-for-run, not just statistically.

Expected values are the actual agent-extinction steps from §9's run logs
(~/simulation_results/erl_results/erl_full_study_logs/<strategy>_seed<n>.log),
hard-coded so this test runs without those logs. One fast-extinct seed per
strategy (~5-10s each). If any of these drift, a change has altered the RNG
stream or the dynamics, and §9 is no longer this module's baseline -- the
longer-horizon check is `study.py analyze --compare-study`.
"""

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import config_step0
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world import ErlWorld

STUDY_EXTINCTION_STEPS = [
    ("ERL", 12, 1038),
    ("E", 6, 896),
    ("L", 3, 1232),
    ("F", 5, 890),
    ("B", 6, 673),
]


@pytest.mark.parametrize("strategy,seed,expected", STUDY_EXTINCTION_STEPS)
def test_step0_matches_study_extinction_step(strategy, seed, expected):
    cfg = dict(config_step0, strategy=strategy, seed=seed)
    world = ErlWorld(cfg, np.random.default_rng(seed))
    while world.current_step < expected + 100:
        world.step()
        if world.population_counts()["agent"] == 0:
            break
    assert world.population_counts()["agent"] == 0, f"still alive at {world.current_step}, §9 died at {expected}"
    assert world.current_step == expected
