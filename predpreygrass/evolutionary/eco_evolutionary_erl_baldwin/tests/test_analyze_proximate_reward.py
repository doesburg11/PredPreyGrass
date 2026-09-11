"""Tests for analyze_proximate_reward.py's data loading -- in particular that
combining a lineage_fitness.csv (real deaths only, see run_erl_simulation.py)
with a checkpoint's live agents (right-censored survivors) produces the
expected combined dataset, since lineage_fitness.csv deliberately never
contains survivor rows itself (see checkpoint.py / test_checkpoint.py).
"""

import csv

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.analyze_proximate_reward import _load
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.checkpoint import save_checkpoint
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import lineage_fieldnames, lineage_record, CsvLogger
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld, OBS_DIM


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _small_world_cfg(**overrides):
    base = dict(config_erl, grid_size=16, n_initial_agents=4, n_initial_carnivores=1, min_plants=4, min_trees=2)
    base.update(overrides)
    return base


def _write_lineage_csv(tmp_path, agents_offspring_lifespan):
    path = tmp_path / "lineage_fitness.csv"
    logger = CsvLogger(path, lineage_fieldnames(OBS_DIM))
    for offspring, lifespan in agents_offspring_lifespan:
        row = {
            "agent_id": 0, "generation": 0, "born_step": 0, "death_step": lifespan,
            "lifespan": lifespan, "offspring_count": offspring, "censored": False,
            "eval_bias": 0.0,
        }
        for i in range(OBS_DIM):
            row[f"eval_weight_{i}"] = float(i)
        logger.log(row)
    logger.close()
    return path


def test_load_from_csv_only(tmp_path):
    path = _write_lineage_csv(tmp_path, [(2, 10), (5, 20)])
    weights, offspring, lifespan, obs_dim, n_censored = _load([path], [])
    assert obs_dim == OBS_DIM
    assert n_censored == 0
    assert list(offspring) == [2.0, 5.0]
    assert list(lifespan) == [10.0, 20.0]


def test_load_combines_csv_and_checkpoint(tmp_path, rng):
    csv_path = _write_lineage_csv(tmp_path, [(3, 15)])

    world = ErlWorld(_small_world_cfg(), rng)
    for a in world.agents:
        a.offspring_count = 7
    checkpoint_path = tmp_path / "checkpoint_step_0.pkl"
    save_checkpoint(world, checkpoint_path)
    n_living = len([a for a in world.agents if a.alive])

    weights, offspring, lifespan, obs_dim, n_censored = _load([csv_path], [checkpoint_path])

    assert n_censored == n_living
    assert len(offspring) == 1 + n_living
    # The one real death from the CSV plus every living agent's offspring_count=7.
    assert sorted(offspring.tolist()) == sorted([3.0] + [7.0] * n_living)


def test_load_checkpoint_only(tmp_path, rng):
    world = ErlWorld(_small_world_cfg(), rng)
    checkpoint_path = tmp_path / "checkpoint_step_0.pkl"
    save_checkpoint(world, checkpoint_path)
    n_living = len([a for a in world.agents if a.alive])

    weights, offspring, lifespan, obs_dim, n_censored = _load([], [checkpoint_path])
    assert n_censored == n_living
    assert len(offspring) == n_living


def test_load_raises_on_obs_dim_mismatch_between_csv_and_checkpoint(tmp_path, rng):
    csv_path = _write_lineage_csv(tmp_path, [(1, 5)])  # OBS_DIM=7 columns

    world = ErlWorld(_small_world_cfg(strategy="S"), rng)  # obs_dim=8
    checkpoint_path = tmp_path / "checkpoint_step_0.pkl"
    save_checkpoint(world, checkpoint_path)

    with pytest.raises(ValueError):
        _load([csv_path], [checkpoint_path])
