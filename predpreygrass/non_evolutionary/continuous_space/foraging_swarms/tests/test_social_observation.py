"""The observe_neighbors / observe_neighbor_movement settings (README,
"Conditions"): with the defaults this environment is
base_environment_continuous; switched off, they blank only the observer's
own-species slots (or only their velocities), and never change the
dynamics."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.environment import (
    PredPreyGrassContinuous as BaseEnvironment,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import (
    FIXES,
    SELF_SIZE,
    SLOT_SIZE,
    PredPreyGrassContinuous,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_swarms import (
    layout,
    rotate_conspecific_velocities,
)

FLAGS = (
    "predator_observe_neighbors",
    "prey_observe_neighbors",
    "predator_observe_neighbor_movement",
    "prey_observe_neighbor_movement",
)
# Larger populations than the milestone start, so slots are often filled.
CROWDED = {"predator_count": 12, "prey_count": 30, "predator_observe_count": 2}


def play(env, seed=3, decisions=40):
    """(observations, positions) per decision under seeded random actions."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    history = []
    for _ in range(decisions):
        history.append((obs, [(a.id, a.position.x, a.position.y) for a in env.predators + env.prey]))
        obs, _, terms, truncs, _ = env.step({agent: int(rng.integers(48)) for agent in env.active})
        if terms["__all__"] or truncs["__all__"]:
            break
    return history


def own_slot_columns(species, config):
    n_pred = config.get("predator_observe_count", 1)
    n_prey = config.get("prey_observe_count", 3)
    start = SELF_SIZE if species == "predator" else SELF_SIZE + SLOT_SIZE * n_pred
    return start, start + SLOT_SIZE * (n_pred if species == "predator" else n_prey)


@pytest.mark.parametrize("world", [{}, FIXES, CROWDED])
def test_defaults_reproduce_base_environment(world):
    ours = play(PredPreyGrassContinuous(dict(world)))
    base = play(BaseEnvironment(dict(world)))
    assert len(ours) == len(base)
    for (obs_a, pos_a), (obs_b, pos_b) in zip(ours, base):
        assert pos_a == pos_b
        assert obs_a.keys() == obs_b.keys()
        for agent in obs_a:
            np.testing.assert_array_equal(obs_a[agent], obs_b[agent])


@pytest.mark.parametrize("species", ["predator", "prey"])
@pytest.mark.parametrize("setting", ["observe_neighbors", "observe_neighbor_movement"])
def test_switch_changes_only_own_species_slots(species, setting):
    config = dict(CROWDED, **FIXES)
    full = play(PredPreyGrassContinuous(config))
    changed = play(PredPreyGrassContinuous(dict(config, **{f"{species}_{setting}": False})))
    lo, hi = own_slot_columns(species, config)
    blanked = 0
    for (obs_a, pos_a), (obs_b, pos_b) in zip(full, changed):
        assert pos_a == pos_b  # observation only: same dynamics
        for agent, a in obs_a.items():
            b = obs_b[agent]
            if not agent.startswith(species):
                np.testing.assert_array_equal(a, b)
                continue
            np.testing.assert_array_equal(a[:lo], b[:lo])
            np.testing.assert_array_equal(a[hi:], b[hi:])
            slots_a = a[lo:hi].reshape(-1, SLOT_SIZE)
            slots_b = b[lo:hi].reshape(-1, SLOT_SIZE)
            if setting == "observe_neighbors":
                assert not slots_b.any()
            else:
                np.testing.assert_array_equal(slots_a[:, :4], slots_b[:, :4])
                assert not slots_b[:, 4:].any()
            blanked += int(slots_a[:, 0].sum())
    assert blanked > 0  # the test saw conspecifics to blank


@pytest.mark.parametrize("flag", FLAGS)
def test_flags_must_be_booleans(flag):
    with pytest.raises(ValueError, match=flag):
        PredPreyGrassContinuous({flag: 1})


def test_probe_rotation_touches_only_own_species_velocities():
    env = PredPreyGrassContinuous(dict(CROWDED))
    lay = layout(env)
    rng = np.random.default_rng(0)
    batch = rng.uniform(-0.7, 0.7, size=(5, env.obs_size)).astype(np.float32)
    turned = rotate_conspecific_velocities(batch, "prey", lay, 90)
    columns = [lay["prey"] + SLOT_SIZE * k + c for k in range(lay["count"]["prey"]) for c in (4, 5)]
    others = [i for i in range(env.obs_size) if i not in columns]
    np.testing.assert_array_equal(turned[:, others], batch[:, others])
    vx, vy = columns[0], columns[1]
    np.testing.assert_allclose(turned[:, vx], -batch[:, vy], atol=1e-6)
    np.testing.assert_allclose(turned[:, vy], batch[:, vx], atol=1e-6)
    np.testing.assert_allclose(rotate_conspecific_velocities(batch, "prey", lay, 360), batch, atol=1e-6)
