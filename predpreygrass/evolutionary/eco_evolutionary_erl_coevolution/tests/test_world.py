import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import (
    STEP1_OVERRIDES,
    config_step0,
    config_step1,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.study import run_one
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world import (
    Carnivore,
    ErlWorld,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _small_world_cfg(**overrides):
    """A small, fast World AL config for deterministic unit tests."""
    base = dict(
        config_step0,
        grid_size=12,
        n_initial_agents=1,
        n_initial_carnivores=0,
        min_plants=2,
        min_trees=1,
        carnivore_spawn_interval=0,
        mutation_rate=0.0,
    )
    base.update(overrides)
    return base


# --- carried over from erl_baldwin: the Darwinian-inheritance property ---


def test_offspring_genome_does_not_inherit_parents_learned_weights(rng):
    world = ErlWorld(_small_world_cfg(), rng)
    parent = world.agents[0]
    original = parent.genome.action_weights.copy()
    parent.action_weights += 50.0  # a "lifetime of learning"

    parent.energy = world.cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()

    child = world.agents[-1]
    assert np.array_equal(child.genome.action_weights, original)
    assert not np.array_equal(child.genome.action_weights, parent.action_weights)
    assert np.array_equal(child.action_weights, child.genome.action_weights)


def test_carnivores_have_no_genome_or_learning():
    fields = set(Carnivore.__dataclass_fields__)
    assert "genome" not in fields
    assert "action_weights" not in fields


def test_strategy_E_never_updates_live_action_network(rng):
    world = ErlWorld(_small_world_cfg(strategy="E", n_initial_agents=10), rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    for _ in range(30):
        world.step()
    for agent in world.agents:
        if agent.agent_id in before:
            assert np.array_equal(agent.action_weights, before[agent.agent_id])


@pytest.mark.parametrize("strategy", ["L", "F"])
def test_strategies_L_and_F_clone_genome_exactly(rng, strategy):
    world = ErlWorld(_small_world_cfg(strategy=strategy, mutation_rate=0.9), rng)
    parent = world.agents[0]
    parent.energy = world.cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()
    child = [a for a in world.agents if a.generation == 1][0]
    assert np.array_equal(child.genome.action_weights, parent.genome.action_weights)
    assert np.array_equal(child.genome.eval_weights, parent.genome.eval_weights)


def test_strategy_L_learns(rng):
    world = ErlWorld(_small_world_cfg(strategy="L", n_initial_agents=10), rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    for _ in range(30):
        world.step()
    changed = [
        not np.array_equal(a.action_weights, before[a.agent_id]) for a in world.agents if a.agent_id in before
    ]
    assert any(changed)


# --- step 1: carnivores regulated by prey, not immigration ---


def test_step1_preset_only_changes_the_documented_keys():
    changed = {k for k in config_step1 if config_step1[k] != config_step0.get(k)}
    assert changed <= set(STEP1_OVERRIDES)
    assert config_step1["carnivore_immigration_until"] == 20_000
    assert config_step1["carnivore_spawn_interval"] == config_step0["carnivore_spawn_interval"]


def test_immigration_off_spawns_no_carnivores(rng):
    world = ErlWorld(_small_world_cfg(carnivore_spawn_interval=0), rng)
    for _ in range(50):
        world.step()
    assert world.carnivore_immigrants == 0
    assert world.population_counts()["carnivore"] == 0


def test_immigration_on_counts_immigrants(rng):
    world = ErlWorld(_small_world_cfg(carnivore_spawn_interval=10), rng)
    for _ in range(50):
        world.step()
    assert world.carnivore_immigrants == 5


def test_carnivore_reproduction_counts_births_and_pays_cost(rng):
    world = ErlWorld(_small_world_cfg(n_initial_carnivores=1), rng)
    carnivore = world.carnivores[0]
    carnivore.energy = world.cfg["carnivore_reproduction_energy_threshold"]
    world._handle_carnivore_reproduction()
    assert world.carnivore_births == 1
    assert len(world.carnivores) == 2
    assert carnivore.energy == pytest.approx(
        world.cfg["carnivore_reproduction_energy_threshold"] - world.cfg["carnivore_reproduction_energy_cost"]
    )


def test_carnivore_kill_is_recorded_as_predation(rng):
    world = ErlWorld(_small_world_cfg(n_initial_carnivores=1), rng)
    agent, carnivore = world.agents[0], world.carnivores[0]
    agent.health = 1.0
    # Put the agent directly north of the carnivore on empty ground.
    del world.occupant[(agent.row, agent.col)]
    del world.occupant[(carnivore.row, carnivore.col)]
    carnivore.row, carnivore.col = 5, 5
    agent.row, agent.col = 4, 5
    for cell in [(5, 5), (4, 5)]:
        world.terrain[cell] = 0
        world.plant[cell] = False
        world.corpses.pop(cell, None)
    world.occupant[(5, 5)] = carnivore
    world.occupant[(4, 5)] = agent

    world._resolve_carnivore_action(carnivore, 0)  # action 0 = north

    assert not agent.alive
    assert world.deaths["agent"] == {"carnivore": 1}


def test_death_counts_account_for_every_death(rng):
    world = ErlWorld(dict(config_step0, grid_size=30, n_initial_agents=40, n_initial_carnivores=4), rng)
    born = len(world.agents)
    for _ in range(300):
        world.step()
        born += sum(1 for a in world.agents if a.born_step == world.current_step)
        if world.population_counts()["agent"] == 0:
            break
    dead = born - world.population_counts()["agent"]
    assert sum(world.deaths["agent"].values()) == dead
    assert set(world.deaths["agent"]) <= {"carnivore", "agent_attack", "starvation", "wounds", "tree_fall"}


def test_run_one_stops_at_carnivore_extinction_under_step1(tmp_path):
    job = {
        "tag": "t", "preset": "step1", "strategy": "ERL", "seed": 1, "steps": 5000,
        "sample_every": 100, "out_dir": str(tmp_path),
        # Carnivores that can never eat enough to survive: they must die out.
        "overrides": {"n_initial_carnivores": 2, "basal_energy_cost_carnivore": 5.0, "carnivore_immigration_until": 0},
    }
    result = run_one(job)
    assert result["end_reason"] == "carnivore_extinct"
    assert result["end_step"] == result["carnivore_extinction_step"]
    assert result["carnivore_immigrants"] == 0
    assert (tmp_path / "timeseries" / "t_ERL_seed1.csv").exists()


def test_run_one_rejects_unknown_override(tmp_path):
    job = {
        "tag": "t", "preset": "step1", "strategy": "ERL", "seed": 1, "steps": 10,
        "sample_every": 5, "out_dir": str(tmp_path), "overrides": {"not_a_key": 1},
    }
    with pytest.raises(KeyError):
        run_one(job)


def test_immigration_stops_after_configured_step(rng):
    world = ErlWorld(_small_world_cfg(carnivore_spawn_interval=10, carnivore_immigration_until=30), rng)
    for _ in range(100):
        world.step()
    assert world.carnivore_immigrants == 3  # steps 10, 20, 30
    assert not world.immigration_active()


def test_carnivore_extinction_during_warmup_is_not_final(tmp_path):
    job = {
        "tag": "t", "preset": "step1", "strategy": "ERL", "seed": 1, "steps": 600,
        "sample_every": 100, "out_dir": str(tmp_path),
        "overrides": {
            "n_initial_carnivores": 2, "basal_energy_cost_carnivore": 5.0,
            "carnivore_spawn_interval": 200, "carnivore_immigration_until": 400,
        },
    }
    result = run_one(job)
    # Starving carnivores die out repeatedly during warm-up; only the first
    # extinction after immigration stops (step 400) ends the run.
    assert result["end_reason"] == "carnivore_extinct"
    assert result["carnivore_extinction_step"] > 400
    assert result["carnivore_immigrants"] == 2


@pytest.mark.parametrize("conserving,expected_child_energy", [(False, None), (True, 6.0)])
def test_carnivore_birth_energy(rng, conserving, expected_child_energy):
    world = ErlWorld(_small_world_cfg(
        n_initial_carnivores=1, carnivore_reproduction_energy_threshold=12.0,
        carnivore_reproduction_energy_cost=6.0, carnivore_energy_conserving_birth=conserving,
    ), rng)
    parent = world.carnivores[0]
    parent.energy = 12.0
    world._handle_carnivore_reproduction()
    child = world.carnivores[-1]
    expected = world.cfg["initial_energy_carnivore"] if expected_child_energy is None else expected_child_energy
    assert child.energy == expected
    if conserving:
        assert parent.energy + child.energy == pytest.approx(12.0)


def test_failed_immigration_is_not_counted(rng):
    world = ErlWorld(_small_world_cfg(carnivore_spawn_interval=1), rng)
    world._random_empty_cell = lambda: None  # a full world: nowhere to spawn
    world.step()
    assert world.carnivore_immigrants == 0


def test_lethal_wound_takes_precedence_over_starvation(rng):
    world = ErlWorld(_small_world_cfg(), rng)
    agent = world.agents[0]
    agent.health, agent.energy = -1.0, 0.01  # both lethal after this turn's basal charge
    world._resolve_agent_action = lambda a, action: None
    world._step_agents()
    assert world.deaths["agent"] == {"wounds": 1}
