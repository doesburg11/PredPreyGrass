"""Pickles written before the 2026-10-24 discrete_space/continuous_space move still load."""

import io
import pickle

from predpreygrass.evolutionary import legacy_paths
from predpreygrass.evolutionary.discrete_space.eco_evolutionary_erl_baldwin.world import ErlWorld


def test_current_module_maps_moved_modules_only():
    old = "predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world"
    assert legacy_paths.current_module(old) == "predpreygrass.evolutionary.discrete_space.eco_evolutionary_erl_baldwin.world"
    assert legacy_paths.current_module(
        "predpreygrass.evolutionary.eco_evolutionary_continuous_traits.environment"
    ) == "predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment"
    for unchanged in (
        "predpreygrass.evolutionary.discrete_space.eco_evolutionary_erl_baldwin.world",
        "predpreygrass.evolutionary.model_selection",
        "numpy.core.multiarray",
    ):
        assert legacy_paths.current_module(unchanged) == unchanged


def test_old_layout_pickle_loads():
    # Protocol 0 stores module names as plain text lines, so the old path can be written in.
    new = pickle.dumps(ErlWorld, protocol=0)
    old = new.replace(
        b"predpreygrass.evolutionary.discrete_space.eco_evolutionary_erl_baldwin",
        b"predpreygrass.evolutionary.eco_evolutionary_erl_baldwin",
    )
    assert old != new
    assert legacy_paths.load(io.BytesIO(old)) is ErlWorld
    assert legacy_paths.load(io.BytesIO(new)) is ErlWorld
