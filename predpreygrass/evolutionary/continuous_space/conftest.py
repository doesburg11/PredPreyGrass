"""Skips for tests that need local training runs, which aren't in git (CI has
none).

eco_evolutionary_deer_stag_hunt's learn-mode settings tests call
train.load_settings(), which checks the world against deer_stag_hunt's local
run touch_1. The skip lives here, outside that module, because the module's
run_main_3.sh refuses to (re)start if the module's code or tests changed
since its pilot commit (RESULTS.md section 3.2).
"""

from pathlib import Path

import pytest

TOUCH_1 = (
    Path(__file__).resolve().parents[2]
    / "non_evolutionary/continuous_space/deer_stag_hunt/runs/touch_1/source_code/settings.json"
)
NEEDS_TOUCH_1 = {
    "eco_evolutionary_deer_stag_hunt/tests/test_section_3.py::test_learn_mode_is_the_default_and_valid",
    # without touch_1 the expected ValueError would come from the missing run, not the world
    "eco_evolutionary_deer_stag_hunt/tests/test_section_3.py::test_learn_mode_requires_deer_stag_hunts_touch_world",
}


def pytest_collection_modifyitems(config, items):
    if TOUCH_1.is_file():
        return
    skip = pytest.mark.skip(reason="deer_stag_hunt's touch_1 is not in this checkout")
    for item in items:
        for name in NEEDS_TOUCH_1:
            if item.nodeid.endswith(name):
                item.add_marker(skip)
