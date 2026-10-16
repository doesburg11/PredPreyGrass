"""Checkpoints saved before the space grouping still resolve their classes."""
import importlib
import pickle
from pathlib import Path


def test_pre_move_environment_pickle():
    old_module = "predpreygrass.non_evolutionary.base_environment_continuous.environment"
    current_module = "predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.environment"
    legacy = importlib.import_module(old_module)
    current = importlib.import_module(current_module)
    assert Path(legacy.__file__) == Path(current.__file__)
    # Protocol 0 GLOBAL records the old module explicitly, as existing pickles do.
    saved_class = f"c{old_module}\nPredPreyGrassContinuous\n.".encode()
    assert pickle.loads(saved_class) is legacy.PredPreyGrassContinuous
