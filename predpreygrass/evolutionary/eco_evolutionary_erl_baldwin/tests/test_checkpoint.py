"""Tests for checkpoint save/load (checkpoint.py) and the resumable-CsvLogger
support it depends on (metrics.CsvLogger's append mode). Covers exactly what
run_erl_simulation.py's --resume-from and eval_checkpoint.py rely on.
"""

import csv

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import CsvLogger, truncate_csv_after_step
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _small_world_cfg(**overrides):
    base = dict(
        config_erl,
        grid_size=16,
        n_initial_agents=6,
        n_initial_carnivores=1,
        min_plants=4,
        min_trees=2,
    )
    base.update(overrides)
    return base


def test_save_load_roundtrip_preserves_state(tmp_path, rng):
    world = ErlWorld(_small_world_cfg(), rng)
    for _ in range(30):
        world.step()

    path = tmp_path / "checkpoint_step_30.pkl"
    save_checkpoint(world, path)
    loaded = load_checkpoint(path)

    assert loaded.current_step == world.current_step
    assert loaded.strategy == world.strategy
    assert len(loaded.agents) == len(world.agents)
    assert [a.agent_id for a in loaded.agents] == [a.agent_id for a in world.agents]
    for a, b in zip(world.agents, loaded.agents):
        assert np.array_equal(a.genome.eval_weights, b.genome.eval_weights)
        assert a.offspring_count == b.offspring_count
        assert a.born_step == b.born_step
    assert np.array_equal(world.terrain, loaded.terrain)
    assert np.array_equal(world.plant, loaded.plant)


def test_load_checkpoint_preserves_occupant_agent_identity(tmp_path, rng):
    """world.occupant and world.agents must reference the SAME objects after a
    round-trip, not independent copies -- otherwise a step() after resume would
    read/write agent state through one dict and silently diverge from the other."""
    world = ErlWorld(_small_world_cfg(), rng)
    path = tmp_path / "checkpoint_step_0.pkl"
    save_checkpoint(world, path)
    loaded = load_checkpoint(path)

    for agent in loaded.agents:
        occupant = loaded.occupant.get((agent.row, agent.col))
        assert occupant is agent


def test_on_agent_death_is_not_pickled_and_original_world_keeps_its_callback(tmp_path, rng):
    world = ErlWorld(_small_world_cfg(n_initial_agents=1, n_initial_carnivores=0), rng)
    calls = []
    world.on_agent_death = lambda a, step: calls.append((a, step))

    path = tmp_path / "checkpoint_step_0.pkl"
    save_checkpoint(world, path)

    # save_checkpoint must not permanently strip the callback from the live world.
    assert world.on_agent_death is not None
    world._kill_agent(world.agents[0])
    assert len(calls) == 1

    loaded = load_checkpoint(path)
    assert loaded.on_agent_death is None


def test_loaded_world_can_continue_stepping(tmp_path, rng):
    world = ErlWorld(_small_world_cfg(), rng)
    for _ in range(20):
        world.step()

    path = tmp_path / "checkpoint_step_20.pkl"
    save_checkpoint(world, path)
    loaded = load_checkpoint(path)

    loaded.step()
    assert loaded.current_step == 21


def test_save_checkpoint_leaves_no_tmp_file(tmp_path, rng):
    world = ErlWorld(_small_world_cfg(), rng)
    path = tmp_path / "checkpoint_step_0.pkl"
    save_checkpoint(world, path)

    assert path.exists()
    assert not (tmp_path / "checkpoint_step_0.pkl.tmp").exists()


def test_latest_checkpoint_picks_highest_step(tmp_path, rng):
    world = ErlWorld(_small_world_cfg(), rng)
    for step in (100, 20_000, 3_000):
        save_checkpoint(world, tmp_path / f"checkpoint_step_{step}.pkl")

    found = latest_checkpoint(tmp_path)
    assert found == tmp_path / "checkpoint_step_20000.pkl"


def test_latest_checkpoint_none_when_missing_or_empty(tmp_path):
    assert latest_checkpoint(tmp_path / "does_not_exist") is None
    tmp_path.mkdir(exist_ok=True)
    assert latest_checkpoint(tmp_path) is None


def test_csv_logger_append_skips_header_and_keeps_prior_rows(tmp_path):
    path = tmp_path / "progress.csv"
    first = CsvLogger(path, ["step", "value"])
    first.log({"step": 1, "value": "a"})
    first.close()

    second = CsvLogger(path, ["step", "value"], append=True)
    second.log({"step": 2, "value": "b"})
    second.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"step": "1", "value": "a"}, {"step": "2", "value": "b"}]


def test_csv_logger_append_to_missing_file_writes_header(tmp_path):
    path = tmp_path / "progress.csv"
    logger = CsvLogger(path, ["step", "value"], append=True)
    logger.log({"step": 1, "value": "a"})
    logger.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"step": "1", "value": "a"}]


def test_csv_logger_append_to_empty_existing_file_writes_header(tmp_path):
    """A zero-byte file (e.g. created but never written to before a crash) must
    still get a header -- append=True only means 'don't overwrite prior rows',
    not 'never write a header'."""
    path = tmp_path / "progress.csv"
    path.touch()

    logger = CsvLogger(path, ["step", "value"], append=True)
    logger.log({"step": 1, "value": "a"})
    logger.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"step": "1", "value": "a"}]


def test_csv_logger_append_raises_on_header_mismatch(tmp_path):
    """Appending with a different schema than the file already has must fail
    loudly, not silently write misaligned rows into an existing file."""
    path = tmp_path / "progress.csv"
    first = CsvLogger(path, ["step", "value"])
    first.log({"step": 1, "value": "a"})
    first.close()

    with pytest.raises(ValueError):
        CsvLogger(path, ["step", "other_field"], append=True)


def test_truncate_csv_after_step_drops_rows_past_boundary(tmp_path):
    path = tmp_path / "progress.csv"
    logger = CsvLogger(path, ["step", "value"])
    for step in (100, 200, 300, 400):
        logger.log({"step": step, "value": step * 10})
    logger.close()

    truncate_csv_after_step(path, "step", max_step=200)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["step"] for r in rows] == ["100", "200"]


def test_truncate_csv_after_step_noop_when_file_missing(tmp_path):
    # Must not raise -- this is the normal case for a fresh (non-resumed) run.
    truncate_csv_after_step(tmp_path / "does_not_exist.csv", "step", max_step=100)
