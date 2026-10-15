"""Reproduction failures must not partially consume IDs or mutate the world."""
from copy import deepcopy
import pickle

import unittest

from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_defection.predpreygrass_rllib_env import PredPreyGrass as DefectionEnvironment

from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_forward_view.config.config_env_stag_hunt_forward_view import config_env
from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_forward_view.predpreygrass_rllib_env import PredPreyGrass


def _check_full_grid_reproduction_is_atomic(species, env_class: type[PredPreyGrass] | type[DefectionEnvironment] = PredPreyGrass, env_config=config_env):
    env = env_class(deepcopy(env_config))
    env.reset(seed=12)
    parent = next(a for a in env.agents if species in a)
    env.agent_energies[parent] = 100000.0
    env.wall_positions = {(x, y) for x in range(env.grid_size) for y in range(env.grid_size)}
    before = pickle.dumps(env.__dict__)
    getattr(env, f"_handle_{species}_reproduction")(parent)
    assert pickle.dumps(env.__dict__) == before


def _check_exhausted_id_pool_does_not_advance_rng(species, env_class: type[PredPreyGrass] | type[DefectionEnvironment] = PredPreyGrass, env_config=config_env):
    env = env_class(deepcopy(env_config))
    env.reset(seed=12)
    parent = next(a for a in env.agents if species in a)
    env.agent_energies[parent] = 100000.0
    env._available_id_pools[f"type_{parent.split('_')[1]}_{species}"].clear()
    # Force spawning to require the randomized fallback if it is attempted.
    x, y = env.agent_positions[parent]
    env.wall_positions.update({(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)})
    before = deepcopy(env.rng.bit_generator.state)
    getattr(env, f"_handle_{species}_reproduction")(parent)
    assert env.rng.bit_generator.state == before


class TestReproductionCapacity(unittest.TestCase):
    def test_full_grid_is_atomic(self):
        for species in ("predator", "prey"):
            with self.subTest(species=species):
                _check_full_grid_reproduction_is_atomic(species)

    def test_capacity_preserves_rng(self):
        for species in ("predator", "prey"):
            with self.subTest(species=species):
                _check_exhausted_id_pool_does_not_advance_rng(species)
