"""Logistic and grazed grass regrowth (grass.py), and instant grass, the
default."""

import json

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import DEFAULT_CONFIG, DeerStagHunt
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.grass import MATURE, Grass


class Prey:
    def __init__(self, x, y, id="deer_0"):
        self.position, self.alive, self.id = Vector(x, y), True, id


def grass(**kw):
    g = Grass(800, 800, **dict(dict(regrowth="logistic", overgrazing=False), **kw))
    g.reset(seed=0)
    return g


def test_default_is_instant_and_logistic_available():
    assert DEFAULT_CONFIG["grass_regrowth"] == "instant"
    assert DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="logistic")).grass.regrowth == "logistic"
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
    env = DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="logistic"))
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
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.checkpoints import checkpoint_env_config

    old = {k: v for k, v in DEFAULT_CONFIG.items() if k not in ("deer_catch_rule", "grass_regrowth", "grass_growth_rate", "grass_root_reserve")}
    (tmp_path / "source_code").mkdir()
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": old, "config_ppo": {}}))
    (tmp_path / "checkpoint").mkdir()
    config = checkpoint_env_config(tmp_path / "checkpoint")
    assert config["grass_regrowth"] == "instant" and config["deer_catch_rule"] == "party"


def test_a_logistic_episode_runs_and_deer_graze():
    env = DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="logistic", max_time_steps=2000))
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
    env = DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="logistic"))
    env.reset(seed=0)
    patch = env.grass.patches[0]
    patch.biomass = 0.37
    for animal, slots in ((env.deer[0], 3 + 3), (env.predators[0], 4 + 2)):
        animal.position = Vector(patch.position.x - 5, patch.position.y)
        animal.orientation = 0.0
        obs = env._observe(animal)
        food = 4 + 6 * slots
        assert obs[food] == 1.0 and obs[food + 4] == pytest.approx(0.37)


def test_a_real_deer_stag_hunt_touch_checkpoint_loads_as_instant_without_sharing():
    from pathlib import Path

    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.checkpoints import checkpoint_env_config, load_modules

    checkpoint = Path(__file__).resolve().parents[2] / "deer_stag_hunt" / "runs" / "touch_1" / "checkpoint"
    if not checkpoint.is_dir():
        pytest.skip("deer_stag_hunt's touch_1 is not in this checkout")
    config = checkpoint_env_config(checkpoint)
    assert config["grass_regrowth"] == "instant" and config["deer_catch_rule"] == "touch" and config["adult_share_fraction"] == 0.0
    assert set(load_modules(checkpoint)) == {"predator", "deer"}


# Grazed grass (RESULTS.md section 4): deer crop down to stubble.


def grazed(**kw):
    return grass(**dict(dict(regrowth="grazed"), **kw))


def test_a_deer_bites_at_most_bite_per_step_never_below_stubble():
    g = grazed(count=1, cluster_count=1, dispersal=False, bite=0.1, stubble=0.2)
    patch = g.patches[0]
    where = (patch.position.x, patch.position.y)
    patch.biomass = 1.0
    eaten = g.advance([Prey(*where)])
    assert eaten["deer_0"] == pytest.approx(0.1)
    assert patch.biomass == pytest.approx(0.9)
    patch.biomass = 0.25
    r, b0 = g.growth_rate, g.root_reserve
    grown = 0.25 + r * (0.25 + b0) * 0.75
    eaten = g.advance([Prey(*where)])
    assert eaten["deer_0"] == pytest.approx(grown - 0.2)  # only down to stubble
    assert patch.biomass == pytest.approx(0.2)
    assert not g.edible(patch)


def test_grazed_patch_keeps_its_place_and_regrows_from_stubble():
    g = grazed(count=1, cluster_count=1, bite=1.0, stubble=0.2)
    patch = g.patches[0]
    where = Vector(patch.position.x, patch.position.y)
    g.advance([Prey(where.x, where.y)])
    assert patch.biomass == pytest.approx(0.2) and patch.regrow is None
    assert patch.position.x == where.x and patch.position.y == where.y
    for _ in range(50):
        g.advance([])
    assert patch.biomass > 0.35  # fast: from stubble, not from bare


def test_stubble_patches_are_not_seen():
    g = grazed(count=1, cluster_count=1, dispersal=False, stubble=0.2)
    patch = g.patches[0]
    observer = Prey(patch.position.x - 10, patch.position.y)
    observer.velocity, observer.orientation = Vector(1, 0), 0.0
    patch.biomass = 0.2
    assert g.observation(observer, 200, 360)[0] == 0.0
    patch.biomass = 0.6
    seen = g.observation(observer, 200, 360)
    assert seen[0] == 1.0 and seen[4] == pytest.approx(0.6) and len(seen) == 5


def test_grazed_cluster_collapses_without_mature_patches_but_not_above_stubble_mature():
    g = grazed(count=10, cluster_count=1, overgrazing=True, dispersal=False, stubble=0.2)
    for p in g.patches:
        p.biomass = MATURE - 0.2
    g.advance([])
    assert g.total_collapses == 1 and all(p.biomass == 0.0 and p.regrow == "centre" for p in g.patches)
    g = grazed(count=10, cluster_count=1, overgrazing=True, dispersal=False, stubble=0.6, bite=1.0)
    for _ in range(100):
        g.advance([Prey(p.position.x, p.position.y, f"deer_{i}") for i, p in enumerate(g.patches)])
    assert g.total_collapses == 0 and all(p.biomass >= 0.6 for p in g.patches)


def test_grazed_settings_are_checked():
    for bad in (dict(bite=0), dict(bite=1.5), dict(stubble=1.0), dict(stubble=-0.1), dict(stubble=True)):
        with pytest.raises(ValueError):
            grazed(**bad)


def test_a_grazed_episode_runs_and_deer_gain_energy():
    env = DeerStagHunt(dict(DEFAULT_CONFIG, grass_regrowth="grazed", max_time_steps=2000))
    obs, _ = env.reset(seed=1)
    assert env.obs_size["deer"] == 46 and env.obs_size["predator"] == 49
    rng = np.random.default_rng(0)
    gained = 0.0
    for _ in range(200):
        obs, _, t, tr, infos = env.step({a: int(rng.integers(48)) for a in env.active})
        gained += sum(i.get("grass_eaten", 0) for i in infos.values())
        if t["__all__"] or tr["__all__"]:
            break
    assert env.grass.total_consumed > 0 and gained > 0


def test_two_deer_at_one_patch_take_one_bite_the_closer_one():
    g = grazed(count=1, cluster_count=1, dispersal=False, bite=0.1, stubble=0.2)
    patch = g.patches[0]
    patch.biomass = 1.0
    near, far = Prey(patch.position.x + 2, patch.position.y, "deer_1"), Prey(patch.position.x + 6, patch.position.y, "deer_0")
    eaten = g.advance([far, near])
    assert eaten == {"deer_1": pytest.approx(0.1)}
    assert patch.biomass == pytest.approx(0.9)


def test_a_patch_kept_short_dies_back_and_reseeds_next_to_a_mature_one():
    g = grazed(count=40, cluster_count=1, dispersal_distance=1.0, bite=1.0, stubble=0.2, dieback_delay=50)
    parent = g.patches[5]
    victim = g.patches[0]
    deer = Prey(victim.position.x, victim.position.y)
    start = (victim.position.x, victim.position.y)
    for step in range(60):
        g.advance([deer])  # cropped to stubble every step
        if victim.biomass == 0.0:  # died back (only then is it bare)
            break
    assert step == 50 and g.total_diebacks >= 1  # last mature at tick 1 (before its first bite)
    assert (victim.position.x, victim.position.y) != start and victim.regrow is None
    near_mature = [p for p in g.patches if p is not victim and p.biomass >= MATURE and g.torus.distance(p.position, victim.position) < 8]
    assert near_mature and parent.biomass >= MATURE


def test_light_grazing_does_not_kill():
    g = grazed(count=1, cluster_count=1, dispersal=False, bite=1.0, stubble=0.2, dieback_delay=100)
    patch = g.patches[0]
    where = (patch.position.x, patch.position.y)
    for step in range(1000):
        g.advance([Prey(*where)] if patch.biomass >= 0.6 else [])  # grazed only when tall again
    assert g.total_diebacks == 0


def test_dieback_delay_is_checked():
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError):
            grazed(dieback_delay=bad)


def test_a_collapsing_cluster_does_not_die_back_first():
    g = grazed(count=10, cluster_count=1, overgrazing=True, dieback_delay=5)
    for p in g.patches:
        p.biomass = 0.3
        p.short_since = -100  # timers long expired
    g.advance([])
    assert g.total_collapses == 1 and g.total_diebacks == 0
    assert all(p.regrow == "centre" for p in g.patches)


def test_growing_to_mature_then_bitten_below_resets_the_clock():
    g = grazed(count=1, cluster_count=1, dispersal=False, bite=0.02, stubble=0.2, dieback_delay=20)
    patch = g.patches[0]
    where = (patch.position.x, patch.position.y)
    patch.biomass = MATURE - 0.001
    for _ in range(200):  # grows just past MATURE, bitten just below it, every step
        patch.biomass = MATURE - 0.001
        g.advance([Prey(*where)])
        assert patch.biomass < MATURE
    assert g.total_diebacks == 0


def test_unavailable_time_does_not_count_toward_dieback():
    g = grazed(count=1, cluster_count=1, dispersal=False, overgrazing=False, dieback_delay=20)
    patch = g.patches[0]
    patch.biomass, patch.ready_at = 0.3, 50
    for _ in range(50):
        g.advance([])
    assert g.total_diebacks == 0 and patch.short_since == 49
