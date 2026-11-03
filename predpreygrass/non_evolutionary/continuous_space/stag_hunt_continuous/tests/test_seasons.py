"""Tests of seasonal grass and the seasonal measures."""

import json
import math

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import checkpoint_env_config
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import DEFAULT_CONFIG, MOVES, PARTY_SIZES, StagHuntContinuous, species_of
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.grass import Grass
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.seasons import PHASES, cycle_phase, play_episode, summarise


def random_policy(seed):
    rng = np.random.default_rng(seed)

    def choose(obs):
        return {a: (np.array([rng.integers(MOVES), rng.integers(2)]) if species_of(a) == "predator" else int(rng.integers(MOVES))) for a in obs}

    return choose


def lone_patch(**seasons):
    grass = Grass(
        800, 800, count=1, respawn_delay=100, clustered=False, dispersal=False, overgrazing=False, **seasons
    )
    grass.reset(seed=1)
    return grass


class Grazer:
    alive = True
    id = "rabbit_0"

    def __init__(self, position):
        self.position = position


def regrowth_steps(grass, wait):
    """Physics steps until the lone patch regrows, when eaten after wait steps."""
    for _ in range(wait):
        grass.advance([])
    assert grass.advance([Grazer(grass.patches[0].position.copy())])
    steps = 0
    while not grass.available(grass.patches[0]):
        grass.advance([])
        steps += 1
    return steps


def test_growth_clock_averages_one_per_step_over_a_cycle():
    grass = lone_patch(season_period=400, season_amplitude=0.8)
    for _ in range(400):
        grass.advance([])
    assert grass.tick == pytest.approx(400)


def test_growth_rate_starts_into_the_good_season():
    grass = lone_patch(season_period=400, season_amplitude=0.5)
    grass.advance([])
    assert grass.tick == 1
    grass.steps = 100
    assert grass.growth_rate() == pytest.approx(1.5)
    grass.steps = 300
    assert grass.growth_rate() == pytest.approx(0.5)


def test_plain_growth_until_the_season_start():
    grass = lone_patch(season_period=400, season_amplitude=0.5, season_start=300)
    grass.steps = 299
    assert grass.growth_rate() == 1
    grass.steps = 300
    assert grass.growth_rate() == 1  # sin(0): the first seasonal step
    grass.steps = 301
    assert grass.growth_rate() > 1
    grass.steps = 400
    assert grass.growth_rate() == pytest.approx(1.5)
    grass.steps = 600
    assert grass.growth_rate() == pytest.approx(0.5)


def test_grass_eaten_in_the_lean_season_regrows_later():
    good = regrowth_steps(lone_patch(season_period=1000, season_amplitude=0.9), 150)
    lean = regrowth_steps(lone_patch(season_period=1000, season_amplitude=0.9), 650)
    plain = regrowth_steps(lone_patch(), 650)
    assert good < plain < lean
    assert plain == 100


def test_full_amplitude_stops_growth_at_the_trough_only():
    grass = lone_patch(season_period=400, season_amplitude=1)
    grass.steps = 300
    assert grass.growth_rate() == pytest.approx(0)
    grass.steps = 299
    assert grass.growth_rate() > 0


@pytest.mark.parametrize(
    "seasons",
    [
        {"season_period": -1},
        {"season_start": -1},
        {"season_start": 1.5},
        {"season_period": 1.5},
        {"season_amplitude": 0.5},
        {"season_period": 100, "season_amplitude": -0.1},
        {"season_period": 100, "season_amplitude": 1.1},
        {"season_period": 100, "season_amplitude": math.nan},
        {"season_period": 100, "season_amplitude": True},
    ],
)
def test_invalid_seasons(seasons):
    with pytest.raises(ValueError):
        lone_patch(**seasons)


def plays(config, seed=3):
    env = StagHuntContinuous(config)
    choose = random_policy(seed)
    obs, _ = env.reset(seed=seed)
    trace = []
    while True:
        obs, rewards, terms, truncs, _ = env.step(choose({a: obs[a] for a in sorted(obs) if a in env.active}))
        trace.append((env.time_step, {a: o.tolist() for a, o in obs.items()}, rewards, terms, truncs))
        if terms["__all__"] or truncs["__all__"]:
            return trace, env.episode_hunt_stats


def test_zero_amplitude_plays_as_no_seasons():
    plain = plays({"max_time_steps": 800})
    assert plain == plays({"max_time_steps": 800, "grass_season_period": 200, "grass_season_amplitude": 0.0, "steppe_season_period": 300})


def test_old_checkpoints_load_without_seasons(tmp_path):
    config = {key: value for key, value in DEFAULT_CONFIG.items() if "season" not in key}
    (tmp_path / "source_code").mkdir()
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": config}))
    loaded = checkpoint_env_config(tmp_path / "checkpoint")
    assert all(loaded[f"{layer}_season_{name}"] == 0 for layer in ("grass", "steppe") for name in ("period", "amplitude"))


def test_collapse_waits_on_the_growth_clock():
    def regrows(wait):
        grass = Grass(800, 800, count=1, respawn_delay=100, cluster_count=1, dispersal=False, overgrazing_delay=100, season_period=1000, season_amplitude=0.9)
        grass.reset(seed=1)
        for _ in range(wait):
            grass.advance([])
        grass.advance([Grazer(grass.patches[0].position.copy())])
        assert grass.total_collapses == 1
        steps = 0
        while not grass.available(grass.patches[0]):
            grass.advance([])
            steps += 1
        return steps

    assert regrows(150) < 100 < regrows(650)


def test_seasons_change_the_grass_layer_chosen_only():
    env = StagHuntContinuous({"grass_season_period": 400, "grass_season_amplitude": 0.5})
    assert env.food["grass"].season_amplitude == 0.5
    assert env.food["steppe"].season_amplitude == 0.0


@pytest.mark.parametrize("time_step, expected", [(0, (0, 0)), (249, (0, 7)), (250, (1, 0)), (500, (2, 0)), (312, (1, 1))])
def test_cycle_phase(time_step, expected):
    assert cycle_phase(time_step, 250) == expected
    assert 0 <= cycle_phase(time_step, 250)[1] < PHASES


def seasonal_world(steps=1600, period=400):
    return StagHuntContinuous({"predator_count": 12, "mammoth_count": 6, "max_time_steps": steps, "grass_season_period": period, "grass_season_amplitude": 0.8})


def test_cycle_phase_counts_from_the_season_start():
    assert cycle_phase(99, 80, 100)[0] == -1
    assert cycle_phase(100, 80, 100) == (0, 0)
    assert cycle_phase(180, 80, 100) == (1, 0)


def test_steps_before_the_season_start_stay_out_of_the_cycles():
    env = StagHuntContinuous({"predator_count": 12, "mammoth_count": 6, "max_time_steps": 160, "grass_season_period": 40, "grass_season_amplitude": 0.8, "grass_season_start": 80})
    row = play_episode(env, random_policy(5), 5)
    assert sorted(row["cycles"], key=int) == ["-2", "-1", "0", "1"]
    assert sum(phase["physics_steps"] for phase in row["cycles"]["-1"]) == 40
    assert row["start"] == 80
    pooled = summarise([row], skip_cycles=0)["phases"]
    assert sum(phase["physics_steps"] for phase in pooled) == 80


def test_counters_add_up_to_the_episode():
    env = seasonal_world()
    row = play_episode(env, random_policy(5), 5)
    totals = {key: sum(phase[key] for rows in row["cycles"].values() for phase in rows) for key in ("physics_steps", "catches", "rabbit_energy", "mammoth_energy", "kills")}
    stats = env.episode_hunt_stats
    assert totals["physics_steps"] == env.time_step == row["physics_steps"]
    assert totals["catches"] == stats["rabbits_caught"]
    assert totals["rabbit_energy"] == pytest.approx(stats["rabbit_energy"])
    assert totals["mammoth_energy"] == pytest.approx(stats["mammoth_energy"])
    assert totals["kills"] == sum(stats[f"kills_party_{size}"] for size in range(1, PARTY_SIZES + 1))
    assert row["grass_eaten"] == env.food["grass"].total_consumed
    assert (row["period"], row["amplitude"]) == (400, 0.8)


def test_physics_steps_split_exactly_across_phases_when_decisions_cross_them():
    # Phases of 10 steps, decisions of up to 8: most cross a boundary.
    env = seasonal_world(steps=160, period=80)
    row = play_episode(env, random_policy(5), 5)
    assert env.time_step == 160
    assert sorted(row["cycles"]) == ["0", "1"]
    assert all(phase["physics_steps"] == 10 for rows in row["cycles"].values() for phase in rows)


def test_populations_count_at_every_physics_step():
    env = seasonal_world(steps=800)
    counted = {"rabbit": 0, "hunter": 0, "grass": 0}
    measure = play_episode
    step = env._physics_step
    grass = env.food["grass"]

    def independent(moves):
        counted["rabbit"] += len(env.rabbits)
        counted["hunter"] += len(env.predators)
        counted["grass"] += sum(grass.edible(patch) for patch in grass.patches)
        return step(moves)

    env._physics_step = independent
    row = measure(env, random_policy(7), 7)
    totals = lambda key: sum(phase[key] for rows in row["cycles"].values() for phase in rows)
    assert totals("rabbit_steps") == counted["rabbit"]
    assert totals("hunter_steps") == counted["hunter"]
    assert totals("grass_steps") == counted["grass"]


def test_measure_leaves_the_environment_unwrapped():
    env = seasonal_world(steps=80)
    play_episode(env, random_policy(1), 1)
    assert "_physics_step" not in vars(env)


def test_summarise_leaves_out_early_cycles():
    def phase(steps, rabbits, rabbit_energy, mammoth_energy):
        counters = dict.fromkeys(("grass_steps", "hunter_steps", "mammoth_steps", "catches", "kills", "group_kills"), 0.0)
        return dict(counters, physics_steps=steps, rabbit_steps=rabbits * steps, rabbit_energy=rabbit_energy, mammoth_energy=mammoth_energy)

    empty = phase(0, 0, 0, 0)
    first = [phase(10, 100, 9, 1)] + [empty] * (PHASES - 1)
    second = [phase(10, 20, 1, 3)] + [empty] * (PHASES - 1)
    rows = [{"outcome": "all", "period": 400, "amplitude": 0.5, "start": 0, "grass_eaten": 30, "physics_steps": 20, "cycles": {"0": first, "1": second}}]
    assert summarise(rows, skip_cycles=0)["phases"][0]["rabbits"] == 60
    summary = summarise(rows, skip_cycles=1)
    assert summary["phases"][0]["rabbits"] == 20
    assert summary["phases"][0]["mammoth_share"] == 0.75
    assert summary["phases"][1]["rabbits"] is None
    assert (summary["period"], summary["amplitude"]) == (400, 0.5)
    assert summary["grass_eaten_per_1000_steps"] == 1500


@pytest.mark.parametrize("period", [0, PHASES - 1])
def test_seasons_measure_needs_a_seasonal_world(period):
    with pytest.raises(ValueError):
        play_episode(StagHuntContinuous({"max_time_steps": 80, "grass_season_period": period}), random_policy(1), 1)
