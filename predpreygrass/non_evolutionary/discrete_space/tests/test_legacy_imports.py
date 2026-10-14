"""Saved checkpoints must still resolve pre-relocation project module names."""
import importlib
import pickle
from pathlib import Path


def test_legacy_environment_class_pickle_round_trip():
    legacy = importlib.import_module(
        "predpreygrass.non_evolutionary.base_environment_discrete.predpreygrass_rllib_env"
    )
    current = importlib.import_module(
        "predpreygrass.non_evolutionary.discrete_space.base_environment_discrete.predpreygrass_rllib_env"
    )
    assert Path(legacy.__file__) == Path(current.__file__)
    assert pickle.loads(pickle.dumps(legacy.PredPreyGrass)) is legacy.PredPreyGrass
