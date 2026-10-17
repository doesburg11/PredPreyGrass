"""Check scratch-directory isolation without starting Ray or training workers."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.impala import Impala
from ray import cloudpickle

from predpreygrass.training import LocalPPO, LocalAPPO, LocalImpala


@pytest.mark.parametrize("local,base", [(LocalPPO, PPO), (LocalAPPO, APPO), (LocalImpala, Impala)])
def test_standalone_workdirs_are_temporary_and_independent(monkeypatch, local, base):
    monkeypatch.setattr(base, "stop", lambda self: None)
    first, second = object.__new__(local), object.__new__(local)
    for algo in (first, second):
        algo._storage = None
        algo._setup_logdir()
    first_path, second_path = Path(first._logdir), Path(second._logdir)
    try:
        assert first_path != second_path
        assert first_path.is_dir() and second_path.is_dir()
        assert Path.home() / "ray_results" not in first_path.parents
        first.stop()
        assert not first_path.exists()
        assert second_path.is_dir()
    finally:
        first.stop()
        second.stop()
    assert not second_path.exists()


@pytest.mark.parametrize("chdir", ["0", "1"])
def test_tune_storage_is_preserved(monkeypatch, tmp_path, chdir):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAY_CHDIR_TO_TRIAL_DIR", chdir)
    algo = object.__new__(LocalPPO)
    algo._storage = SimpleNamespace(trial_working_directory=str(tmp_path / "trial"))
    algo._setup_logdir()
    assert algo._logdir == str(tmp_path / "trial")
    assert Path(algo._logdir).is_dir()
    assert not hasattr(algo, "_scratch_logdir")
    assert Path.cwd() == (tmp_path / "trial" if chdir == "1" else tmp_path)


def test_scratch_cleanup_even_if_stop_raises(monkeypatch):
    def fail(self):
        raise RuntimeError("worker shutdown failed")

    monkeypatch.setattr(PPO, "stop", fail)
    algo = object.__new__(LocalPPO)
    algo._storage = None
    algo._setup_logdir()
    path = Path(algo._logdir)
    with pytest.raises(RuntimeError, match="worker shutdown failed"):
        algo.stop()
    assert not path.exists()


@pytest.mark.parametrize("local", [LocalPPO, LocalAPPO, LocalImpala])
def test_algorithm_class_survives_config_serialization(local):
    config = local.get_default_config()
    config.algo_class = local
    # Tune receives the algorithm class separately from its config dictionary.
    restored_class, restored_dict = cloudpickle.loads(
        cloudpickle.dumps((config.algo_class, config.to_dict()))
    )
    assert restored_class is local
    assert isinstance(restored_dict, dict)
    # Direct build_algo() uses the config object, whose class must also survive.
    assert cloudpickle.loads(cloudpickle.dumps(config)).algo_class is local
