import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import (
    AREA_SCALED_KEYS,
    STEP1_OVERRIDES,
    config_step0,
    config_step1,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.networks import action_probs
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


def test_fsa_carnivores_have_no_genome_and_never_learn(rng):
    world = ErlWorld(_small_world_cfg(n_initial_carnivores=2), rng)
    assert all(c.genome is None for c in world.carnivores)
    assert "action_weights" not in set(Carnivore.__dataclass_fields__)  # no live, learnable copy


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
    assert changed <= set(STEP1_OVERRIDES) | set(AREA_SCALED_KEYS) | {"grid_size", "carnivore_spawn_interval"}
    assert config_step1["carnivore_immigration_until"] == 20_000


def test_step1_preset_matches_the_validated_150_grid_values():
    """The exact values the 19/19 coexistence test ran with (README.md)."""
    expected = dict(grid_size=150, n_initial_agents=135, n_initial_carnivores=11, min_plants=112,
                    min_trees=225, max_population_cap=4500, carnivore_spawn_interval=89)
    got = {k: config_step1[k] for k in expected}
    assert got == expected


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
    assert carnivore.kills == 1 and world.carnivore_kills == 1


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


# --- step 2: carnivore genome ---


def _genome_world(rng, mode="genome", **overrides):
    return ErlWorld(_small_world_cfg(**{"carnivore_mode": mode, "n_initial_carnivores": 1, **overrides}), rng)


def _clear_cell(world, cell):
    world.terrain[cell] = 0
    world.plant[cell] = False
    world.corpses.pop(cell, None)
    world.occupant.pop(cell, None)


def _place(world, entity, cell):
    world.occupant.pop((entity.row, entity.col), None)
    _clear_cell(world, cell)
    entity.row, entity.col = cell
    world.occupant[cell] = entity


def test_seeded_founder_pursues_visible_prey(rng):
    world = _genome_world(rng, carnivore_founder_weight_std=0.0)
    carnivore, agent = world.carnivores[0], world.agents[0]
    for cell in [(5, c) for c in range(2, 10)] + [(r, 5) for r in range(2, 10)]:
        _clear_cell(world, cell)
    _place(world, carnivore, (6, 5))
    _place(world, agent, (4, 5))  # two cells north
    obs = world._observe_carnivore(carnivore)
    assert obs[0] > 0.5 and obs[1:4].max() == 0.0
    probs = action_probs(obs, carnivore.genome.action_weights, carnivore.genome.action_bias)
    assert probs[0] > 0.99


def test_seeded_founder_avoids_blocked_cells(rng):
    world = _genome_world(rng, carnivore_founder_weight_std=0.0, n_initial_agents=0)
    carnivore = world.carnivores[0]
    _place(world, carnivore, (6, 6))
    for cell in [(5, 6), (7, 6), (6, 7)]:
        world.occupant.pop(cell, None)
        world.terrain[cell] = 1  # walls N, S, E -- only W is open
    world.terrain[(6, 5)] = 0
    obs = world._observe_carnivore(carnivore)
    assert list(obs[4:8]) == [1.0, 1.0, 1.0, 0.0]
    probs = action_probs(obs, carnivore.genome.action_weights, carnivore.genome.action_bias)
    assert probs[3] > 0.99


def test_genome_carnivore_child_inherits_parent_genome(rng):
    world = _genome_world(rng, carnivore_mutation_rate=0.0)
    parent = world.carnivores[0]
    parent.energy = world.cfg["carnivore_reproduction_energy_threshold"]
    world._handle_carnivore_reproduction()
    child = world.carnivores[-1]
    assert child is not parent
    assert np.array_equal(child.genome.action_weights, parent.genome.action_weights)
    assert child.genome is not parent.genome
    assert child.generation == 1 and parent.offspring_count == 1


def test_neutral_control_inherits_like_genome_mode_but_does_not_express_it(rng):
    """Same seed, same world: "genome" and "genome_neutral" must build identical
    founder genomes (inheritance/RNG parity); only the neutral mode acts with
    the canonical seed network instead of the carnivore's own genome."""
    real = _genome_world(np.random.default_rng(5), mode="genome")
    neutral = _genome_world(np.random.default_rng(5), mode="genome_neutral")
    assert np.array_equal(real.carnivores[0].genome.action_weights, neutral.carnivores[0].genome.action_weights)

    carnivore = neutral.carnivores[0]
    carnivore.genome.action_weights[:] = 0.0
    carnivore.genome.action_weights[:, 1] = 100.0  # its own genome would always pick action 1
    canonical = neutral._canonical_carnivore_genome
    obs = np.zeros(10)
    obs[0] = 1.0  # prey north
    probs = action_probs(obs, canonical.action_weights, canonical.action_bias)
    assert probs[0] > 0.99  # canonical pursues north, ignoring the carnivore's own genome
    picks = []
    for _ in range(30):
        neutral._observe_carnivore = lambda c: obs
        neutral._resolve_carnivore_action = lambda c, action: picks.append(action)
        neutral._step_carnivores()
    assert picks.count(0) >= 29


def test_neutral_and_real_child_genomes_are_built_identically(rng):
    real = _genome_world(np.random.default_rng(9), mode="genome", n_initial_carnivores=2, mate_search_radius=200)
    neutral = _genome_world(np.random.default_rng(9), mode="genome_neutral", n_initial_carnivores=2, mate_search_radius=200)
    a = real._carnivore_child_genome(real.carnivores[0])
    b = neutral._carnivore_child_genome(neutral.carnivores[0])
    assert np.array_equal(a.action_weights, b.action_weights)
    assert real.carnivores[1].offspring_count == neutral.carnivores[1].offspring_count == 1


def test_step2_run_writes_carnivore_lineage(tmp_path):
    job = {
        "tag": "t", "preset": "step2", "strategy": "ERL", "seed": 1, "steps": 300,
        "sample_every": 100, "out_dir": str(tmp_path), "overrides": {},
    }
    result = run_one(job)
    assert result["carnivore_steps"] > 0
    lineage = (tmp_path / "carnivore_lineage" / "t_ERL_seed1.csv").read_text().splitlines()
    header = lineage[0].split(",")
    assert header[:7] == ["carnivore_id", "generation", "born_step", "death_step", "censored",
                          "offspring_count", "kills"]
    ids = [row.split(",")[0] for row in lineage[1:]]
    assert len(ids) == len(set(ids)) == world_carnivores_ever(result)  # exactly one row per carnivore
    ts = (tmp_path / "timeseries" / "t_ERL_seed1.csv").read_text().splitlines()
    assert ts[0].startswith("step,agent_count,carnivore_count,carnivore_kills")


def world_carnivores_ever(result):
    # founders + immigrants + births; every one gets exactly one lineage row
    from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import PRESETS
    return PRESETS["step2"]["n_initial_carnivores"] + result["carnivore_immigrants"] + result["carnivore_births"]


def test_fsa_skip_sheltered_ignores_agents_in_trees(rng):
    """The plain rule always heads for a visible sheltered agent (and wastes the
    move: carnivores can't enter trees); the probe variant treats it as absent."""
    picks = {}
    for mode in ("fsa", "fsa_skip_sheltered"):
        world = ErlWorld(_small_world_cfg(carnivore_mode=mode, n_initial_carnivores=1), np.random.default_rng(0))
        carnivore, agent = world.carnivores[0], world.agents[0]
        for cell in [(r, 5) for r in range(2, 10)] + [(6, c) for c in range(2, 10)]:
            _clear_cell(world, cell)
        _place(world, carnivore, (6, 5))
        _place(world, agent, (4, 5))
        world.terrain[(4, 5)] = 2  # the agent's cell is a tree
        agent.in_tree = True
        picks[mode] = [world._carnivore_fsa_action(carnivore) for _ in range(20)]
    assert set(picks["fsa"]) == {0}
    assert set(picks["fsa_skip_sheltered"]) != {0}


def test_nonheritable_control_expresses_fresh_noise_not_the_parents(rng):
    world = _genome_world(rng, mode="genome_nonheritable", carnivore_mutation_rate=0.0)
    parent = world.carnivores[0]
    assert parent.phenotype is parent.genome  # founders express their own genome
    parent.energy = world.cfg["carnivore_reproduction_energy_threshold"]
    world._handle_carnivore_reproduction()
    child = world.carnivores[-1]
    assert np.array_equal(child.genome.action_weights, parent.genome.action_weights)  # marker inherited
    assert not np.array_equal(child.phenotype.action_weights, parent.genome.action_weights)  # behavior fresh
    idx = np.arange(4)
    assert np.allclose(child.phenotype.action_weights[idx, idx], 10.0, atol=5.0)  # still seed-centered


def test_mixed_mode_types_alternate_inherit_and_count(rng):
    world = ErlWorld(_small_world_cfg(carnivore_mode="mixed", n_initial_carnivores=4,
                                      mixed_mutant_pursuit_weight=30.0), rng)
    assert [c.ctype for c in world.carnivores] == [0, 1, 0, 1]
    resident, mutant = world._mixed_type_genomes
    assert resident.action_weights[0, 0] == 10.0 and mutant.action_weights[0, 0] == 30.0
    parent = world.carnivores[1]
    parent.energy = world.cfg["carnivore_reproduction_energy_threshold"]
    world._handle_carnivore_reproduction()
    assert world.carnivores[-1].ctype == 1 and world.type_births == [0, 1]
    world.step()
    assert sum(world.type_steps) == world.carnivore_steps
