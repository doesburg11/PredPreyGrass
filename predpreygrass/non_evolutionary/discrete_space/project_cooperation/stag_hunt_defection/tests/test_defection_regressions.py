"""Defection shares the reproduction and optional-renderer edge cases."""
import unittest

from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_defection.config.config_env_stag_hunt_defection import config_env
from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_defection.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_defection.utils.pygame_grid_renderer_rllib_new import ViewerControlHelper
from predpreygrass.non_evolutionary.discrete_space.project_cooperation.stag_hunt_forward_view.tests.test_reproduction_capacity import (
    _check_full_grid_reproduction_is_atomic,
    _check_exhausted_id_pool_does_not_advance_rng,
)


class TestDefectionRegressions(unittest.TestCase):
    def test_full_grid_reproduction(self):
        for species in ("predator", "prey"):
            with self.subTest(species=species):
                _check_full_grid_reproduction_is_atomic(species, PredPreyGrass, config_env)

    def test_capacity_preserves_rng(self):
        for species in ("predator", "prey"):
            with self.subTest(species=species):
                _check_exhausted_id_pool_does_not_advance_rng(species, PredPreyGrass, config_env)

    def test_unconnected_slider_is_safe(self):
        control = ViewerControlHelper()
        control._update_fps_slider(10)
