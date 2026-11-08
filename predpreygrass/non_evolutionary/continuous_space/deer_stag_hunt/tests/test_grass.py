"""Logistic grass regrowth (grass.py) and the instant grass it replaced as the
default."""

import json

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment import DEFAULT_CONFIG, DeerStagHunt
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.grass import MATURE, Grass


class Prey:
    def __init__(self, x, y, id="deer_0"):
        self.position, self.alive, self.id = Vector(x, y), True, id


def grass(**kw):
    g = Grass(800, 800, **dict(dict(regrowth="logistic", overgrazing=False), **kw))
    g.reset(seed=0)
    return g


def test_default_is_logistic_and_instant_still_available():
    assert DEFAULT_CONFIG["grass_regrowth"] == "logistic"
    assert DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="instant")).grass.regrowth == "instant"
    with pytest.raises(ValueError):
        DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="linear"))
    with pytest.raises(ValueError):
        DeerStagHunt(dict(DEFAULT_CONFIG, grass_growth_rate=0))


def test_bare_grass_reaches_095_in_about_400_steps():
    g = grass(dispersal=False)
    patch = g.patches[0]
    patch.biomass = 0.0
    steps = 0
    while patch.biomass < 0.95:
        g.grow()
        steps += 1
    assert 380 <= steps <= 420


def test_growth_is_slow_when_short_fastest_at_half_slow_near_full():
    g = grass(dispersal=False)
    patch = g.patches[0]
    rates = {}
    for b in (0.0, 0.5, 0.97):
        patch.biomass = b
        g.grow()
        rates[b] = patch.biomass - b
    assert rates[0.5] > 10 * rates[0.0] and rates[0.5] > 5 * rates[0.97]


def test_keeping_a_patch_bare_does_not_pay_a_resting_deer():
    """A deer eating one patch every physics step gains less than its
    resting metabolic cost."""
    g = grass(count=1, cluster_count=1, dispersal=False)
    patch = g.patches[0]
    deer = Prey(patch.position.x, patch.position.y)
    g.advance([deer])  # eats the whole start patch
    eaten = sum(g.advance([deer]).get("deer_0", 0) for _ in range(2000))
    gain = eaten * DEFAULT_CONFIG["energy_grass_gain"] / 2000
    assert gain < DEFAULT_CONFIG["energy_deer_resting_metabolic_cost"]


def test_a_deer_eats_all_and_gains_that_share():
    g = grass(count=1, cluster_count=1, dispersal=False)
    patch = g.patches[0]
    patch.biomass = 0.4
    eaten = g.advance([Prey(patch.position.x, patch.position.y)])
    r, b0 = g.growth_rate, g.root_reserve
    assert eaten["deer_0"] == pytest.approx(0.4 + r * (0.4 + b0) * 0.6)  # grown first, then eaten
    assert patch.biomass == 0.0 and not g.edible(patch)


def test_eaten_patch_reseeds_next_to_a_mature_patch_of_its_cluster():
    g = grass(count=40, cluster_count=1, dispersal_distance=1.0)
    for p in g.patches:
        p.biomass = 0.1
    parent = g.patches[5]
    parent.biomass = 1.0
    eater = g.patches[0]
    eater.biomass = 0.3
    g.advance([Prey(eater.position.x, eater.position.y)])
    assert g.torus.distance(eater.position, parent.position) < 8
    assert eater.biomass == 0.0 and eater.regrow is None


def test_cluster_collapses_without_mature_patches():
    g = grass(count=10, cluster_count=1, overgrazing=True, dispersal=False)
    for p in g.patches:
        p.biomass = MATURE - 0.2
    g.advance([])
    assert g.total_collapses == 1
    assert all(p.biomass == 0.0 and p.regrow == "centre" for p in g.patches)
    for _ in range(g.overgrazing_delay):
        g.advance([])
    assert all(p.regrow is None for p in g.patches) and all(0 < p.biomass < 0.1 for p in g.patches)


def test_deer_and_predators_see_the_biomass_of_the_nearest_patch():
    env = DeerStagHunt(dict(DEFAULT_CONFIG))
    obs, _ = env.reset(seed=0)
    instant = DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="instant"))
    assert env.obs_size["deer"] == instant.obs_size["deer"] + 1
    assert env.obs_size["predator"] == instant.obs_size["predator"] + 1
    assert (instant.obs_size["predator"], instant.obs_size["deer"]) == (48, 45)
    deer = env.deer[0]
    patch = env.grass.patches[0]
    patch.biomass = 0.37
    deer.position = Vector(patch.position.x - 5, patch.position.y)
    deer.orientation = 0.0
    food = env.grass.observation(deer, deer.view_distance, deer.fov)
    assert food[0] == 1.0 and food[4] == pytest.approx(0.37)


def test_old_run_settings_without_grass_regrowth_load_as_instant(tmp_path):
    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.checkpoints import checkpoint_env_config

    old = {k: v for k, v in DEFAULT_CONFIG.items() if k not in ("deer_catch_rule", "grass_regrowth", "grass_growth_rate", "grass_root_reserve")}
    (tmp_path / "source_code").mkdir()
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": old, "config_ppo": {}}))
    (tmp_path / "checkpoint").mkdir()
    config = checkpoint_env_config(tmp_path / "checkpoint")
    assert config["grass_regrowth"] == "instant" and config["deer_catch_rule"] == "party"


def test_a_logistic_episode_runs_and_deer_graze():
    env = DeerStagHunt(dict(DEFAULT_CONFIG, max_time_steps=2000))
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, _, t, tr, _ = env.step({a: int(rng.integers(48)) for a in env.active})
        for a, o in obs.items():
            assert o.shape == env.observation_spaces[a].shape
        if t["__all__"] or tr["__all__"]:
            break
    assert env.grass.total_consumed > 0


def test_collapsed_cluster_recovers_rearms_and_can_collapse_again():
    g = grass(count=10, cluster_count=1, overgrazing=True, dispersal=False)
    for p in g.patches:
        p.biomass = 0.1
    g.advance([])
    assert g.collapsed[0] and g.total_collapses == 1
    for _ in range(g.overgrazing_delay + 400):
        g.advance([])
    assert not g.collapsed[0] and all(p.biomass >= MATURE for p in g.patches)
    for p in g.patches:
        p.biomass = 0.1
    g.advance([])
    assert g.total_collapses == 2


def test_biomass_is_the_food_entry_after_distance_for_both_species():
    env = DeerStagHunt(dict(DEFAULT_CONFIG))
    env.reset(seed=0)
    patch = env.grass.patches[0]
    patch.biomass = 0.37
    for animal, slots in ((env.deer[0], 3 + 3), (env.predators[0], 4 + 2)):
        animal.position = Vector(patch.position.x - 5, patch.position.y)
        animal.orientation = 0.0
        obs = env._observe(animal)
        food = 4 + 6 * slots
        assert obs[food] == 1.0 and obs[food + 4] == pytest.approx(0.37)


def test_a_real_old_checkpoint_loads_as_instant():
    from pathlib import Path

    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.checkpoints import checkpoint_env_config, load_modules

    checkpoint = Path(__file__).resolve().parents[1] / "runs" / "run_2" / "checkpoint"
    if not checkpoint.is_dir():
        pytest.skip("run_2 is not in this checkout")
    assert checkpoint_env_config(checkpoint)["grass_regrowth"] == "instant"
    assert set(load_modules(checkpoint)) == {"predator", "deer"}
