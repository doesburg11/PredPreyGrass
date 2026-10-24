"""Load pickles saved before the evolutionary modules moved into
discrete_space/ and continuous_space/ (2026-10-24).

Pickled objects record the module path of their classes, e.g.
predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world.ErlWorld; after
the move that module is predpreygrass.evolutionary.discrete_space.
eco_evolutionary_erl_baldwin.world. LegacyPathUnpickler rewrites such paths
when it looks up a class, so checkpoints from either layout load.
"""

import pickle

MOVED = {
    "eco_evolutionary_continuous_traits": "continuous_space",
    "eco_evolutionary_erl_baldwin": "discrete_space",
    "eco_evolutionary_erl_coevolution": "discrete_space",
    "eco_evolutionary_metabolic_rate_positive_control": "discrete_space",
    "stag_hunt_forward_view_nature_nurture": "discrete_space",
}
PREFIX = "predpreygrass.evolutionary."


def current_module(name):
    """The module path a pre-move module path now lives at (unchanged otherwise)."""
    if name.startswith(PREFIX):
        rest = name[len(PREFIX):]
        head = rest.split(".", 1)[0]
        if head in MOVED:
            return PREFIX + MOVED[head] + "." + rest
    return name


class LegacyPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(current_module(module), name)


def load(file):
    """pickle.load that also accepts pickles written under the old layout."""
    return LegacyPathUnpickler(file).load()
