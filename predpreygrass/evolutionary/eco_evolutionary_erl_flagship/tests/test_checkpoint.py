import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import PreyGenomeState, Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.genome import founder_genome


class _FakeEnv:
    """Exposes exactly what save_checkpoint needs: get_state_snapshot()."""

    def __init__(self):
        self.agent_positions = {"prey_0": (1, 2)}

    def get_state_snapshot(self):
        return {"agent_positions": dict(self.agent_positions), "current_step": 42}


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_checkpoint_round_trip(tmp_path, rng):
    env = _FakeEnv()
    cfg = {"seed": 7, "predator_checkpoint_dir": "/fake", "predator_deterministic": False}
    driver = Trial13Driver(env=env, predator_policy=None, cfg=cfg, rng=rng)
    genome = founder_genome(8, 9, rng)
    driver.registry = {
        "prey_0": PreyGenomeState(
            agent_id="prey_0", genome=genome,
            action_weights=genome.action_weights.copy(), action_bias=genome.action_bias.copy(),
            generation=2, born_step=5, offspring_count=3,
        )
    }
    driver.current_step = 99

    path = tmp_path / "checkpoint_step_99.pkl"
    save_checkpoint(driver, cfg, config_env={"grid_size": 25}, path=path)

    payload = load_checkpoint(path)
    assert payload["current_step"] == 99
    assert payload["cfg"] == cfg
    assert payload["config_env"] == {"grid_size": 25}
    assert payload["env_snapshot"]["current_step"] == 42
    assert payload["rng_state"] == rng.bit_generator.state

    restored = payload["registry"]["prey_0"]
    assert restored.offspring_count == 3
    assert restored.generation == 2
    np.testing.assert_array_equal(restored.genome.eval_weights, genome.eval_weights)


def test_checkpoint_write_is_atomic_no_leftover_tmp_file(tmp_path, rng):
    env = _FakeEnv()
    cfg = {"seed": 1, "predator_checkpoint_dir": "/fake", "predator_deterministic": False}
    driver = Trial13Driver(env=env, predator_policy=None, cfg=cfg, rng=rng)
    driver.current_step = 1

    path = tmp_path / "checkpoint_step_1.pkl"
    save_checkpoint(driver, cfg, config_env={}, path=path)

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_latest_checkpoint_picks_highest_step(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint_step_100.pkl").write_bytes(b"")
    (ckpt_dir / "checkpoint_step_5000.pkl").write_bytes(b"")
    (ckpt_dir / "checkpoint_step_250.pkl").write_bytes(b"")
    assert latest_checkpoint(ckpt_dir).name == "checkpoint_step_5000.pkl"


def test_latest_checkpoint_missing_dir_returns_none(tmp_path):
    assert latest_checkpoint(tmp_path / "nope") is None
