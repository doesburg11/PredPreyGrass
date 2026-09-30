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


def test_rich_seed_behaves_exactly_like_basic_seed():
    """Same world state, every living carnivore: the rich seed network (all prey
    channels +10) must give the same action probabilities as the basic seed."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import PRESETS
    basic = ErlWorld(dict(PRESETS["step2_neutral"], seed=3, strategy="ERL"), np.random.default_rng(3))
    for _ in range(300):
        basic.step()
    rich = ErlWorld.__new__(ErlWorld)
    rich.__dict__.update(basic.__dict__)
    rich.cfg = dict(basic.cfg, carnivore_obs="rich")
    rich.carn_layout = __import__(
        "predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world", fromlist=["x"]
    ).CARN_OBS_LAYOUTS["rich"]
    rich.carn_obs_dim = 18
    rich._canonical_cache = None
    b_net, r_net = basic._canonical_carnivore_genome, rich._canonical_carnivore_genome
    checked = 0
    for c in basic.carnivores:
        pb = action_probs(basic._observe_carnivore(c), b_net.action_weights, b_net.action_bias)
        pr = action_probs(rich._observe_carnivore(c), r_net.action_weights, r_net.action_bias)
        assert np.allclose(pb, pr)
        checked += 1
    assert checked > 10


def test_rich_observation_channels(rng):
    world = ErlWorld(_small_world_cfg(carnivore_mode="mixed", carnivore_obs="rich",
                                      n_initial_carnivores=1, n_initial_agents=2), rng)
    carnivore, living, sheltered = world.carnivores[0], world.agents[0], world.agents[1]
    for cell in [(r, 5) for r in range(1, 11)] + [(6, c) for c in range(1, 11)]:
        _clear_cell(world, cell)
    _place(world, carnivore, (6, 5))
    _place(world, living, (4, 5))  # north, 2 cells
    _place(world, sheltered, (6, 8))  # east, 3 cells, in a tree
    world.terrain[(6, 8)] = 2
    sheltered.in_tree = True
    world.corpses[(8, 5)] = __import__(
        "predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world", fromlist=["x"]
    ).Corpse(kind="agent", energy=3.0)  # south, 2 cells
    obs = world._observe_carnivore(carnivore)
    assert obs.shape == (18,)
    assert obs[0] > 0 and obs[4 + 0] == 0 and obs[8 + 0] == 0  # N: living
    assert obs[4 + 2] > 0 and obs[2] == 0  # E: sheltered
    assert obs[8 + 1] > 0 and obs[1] == 0  # S: corpse


def _mixed_rich_world(strategy, **overrides):
    world = ErlWorld(_small_world_cfg(**{"carnivore_mode": "mixed", "carnivore_obs": "rich", "n_initial_carnivores": 2,
                                         "mixed_mutant_strategy": strategy, **overrides}), np.random.default_rng(0))
    return world, world.carnivores[1]  # id 1 = mutant


def test_persist_mutant_repeats_last_move_when_nothing_seen():
    world, mutant = _mixed_rich_world("persist", mixed_persist_prob=1.0)
    obs = np.zeros(18)
    world._observe_carnivore = lambda c: obs
    mutant.last_action = 2
    assert all(world._state_strategy_action(mutant) == 2 for _ in range(20))
    obs[12 + 2] = 1.0  # east now blocked -> falls back to the seed network
    assert world._state_strategy_action(mutant) != 2


def test_sated_scavenger_ignores_living_prey_only_when_sated():
    world, mutant = _mixed_rich_world("sated_scavenger")
    obs = np.zeros(18)
    obs[0] = 1.0  # living prey north
    obs[8 + 1] = 1.0  # carcass south, same distance
    world._observe_carnivore = lambda c: obs
    obs[16] = 0.9  # sated
    assert [world._state_strategy_action(mutant) for _ in range(30)].count(1) >= 29
    obs[16] = 0.3  # hungry: north and south tie under the seed
    picks = [world._state_strategy_action(mutant) for _ in range(60)]
    assert picks.count(0) > 10 and picks.count(1) > 10


def test_network_mutant_path_unchanged_by_state_strategies():
    """Default strategy 'network' must not touch the state-strategy code path."""
    world, mutant = _mixed_rich_world("network")
    world._state_strategy_action = lambda c: (_ for _ in ()).throw(AssertionError("must not be called"))
    world.step()


def test_mixed_assign_step_splits_living_carnivores_after_warmup():
    world = ErlWorld(_small_world_cfg(carnivore_mode="mixed", n_initial_carnivores=6, grid_size=20,
                                      mixed_assign_step=5), np.random.default_rng(0))
    assert all(c.ctype == 0 for c in world.carnivores)
    for _ in range(5):
        world.step()
    living = sorted((c for c in world.carnivores if c.alive), key=lambda c: c.carnivore_id)
    assert [c.ctype for c in living] == [i % 2 for i in range(len(living))]
    before = sum(world.type_deaths)
    victim = living[0]
    world._kill_carnivore(victim, "starvation")
    assert sum(world.type_deaths) == before + 1


def test_rich_memory_encodes_previous_move_and_seed_ignores_it():
    world = ErlWorld(_small_world_cfg(carnivore_mode="genome_neutral", carnivore_obs="rich_memory",
                                      n_initial_carnivores=1), np.random.default_rng(0))
    carnivore = world.carnivores[0]
    obs0 = world._observe_carnivore(carnivore)
    assert obs0.shape == (22,) and obs0[16:20].sum() == 0  # no previous move yet
    carnivore.last_action = 3
    obs = world._observe_carnivore(carnivore)
    assert list(obs[16:20]) == [0, 0, 0, 1]
    seed = world._canonical_carnivore_genome
    assert np.all(seed.action_weights[16:20] == 0)  # no built-in persistence
    assert np.allclose(action_probs(obs, seed.action_weights, seed.action_bias),
                       action_probs(obs0, seed.action_weights, seed.action_bias))


def test_rich_memory_network_can_express_persistence():
    world = ErlWorld(_small_world_cfg(carnivore_mode="genome_neutral", carnivore_obs="rich_memory",
                                      carnivore_seed_prev_weight=3.3, n_initial_carnivores=1), np.random.default_rng(0))
    seed = world._canonical_carnivore_genome
    obs = np.zeros(22)
    obs[16 + 2] = 1.0  # moved east last step, nothing visible, nothing blocked
    assert action_probs(obs, seed.action_weights, seed.action_bias)[2] > 0.85
    obs[0] = 1.0  # prey north: pursuit (+10) dominates the persistence weight
    assert action_probs(obs, seed.action_weights, seed.action_bias)[0] > 0.99


def test_founder_prev_std_widens_only_persistence_weights():
    world = ErlWorld(_small_world_cfg(carnivore_mode="genome", carnivore_obs="rich_memory",
                                      carnivore_founder_prev_std=3.0, n_initial_carnivores=0), np.random.default_rng(0))
    genomes = [world._founder_carnivore_genome() for _ in range(400)]
    idx = np.arange(4)
    diag = np.array([g.action_weights[16 + idx, idx] for g in genomes]).ravel()
    offdiag = np.array([g.action_weights[16 + idx, (idx + 1) % 4] for g in genomes]).ravel()
    assert 2.6 < diag.std() < 3.4 and abs(diag.mean()) < 0.3
    assert 0.8 < offdiag.std() < 1.2


def _erl_carn_world(**overrides):
    cfg = {"carnivore_mode": "erl", "carnivore_obs": "rich_memory", "n_initial_carnivores": 1,
           "carnivore_mutation_rate": 0.0, **overrides}
    return ErlWorld(_small_world_cfg(**cfg), np.random.default_rng(0))


def test_erl_carnivore_founder_eval_is_energy_seeded():
    world = _erl_carn_world(carnivore_founder_weight_std=0.0)
    genome = world.carnivores[0].genome
    assert genome.eval_weights[-2] == 5.0 and np.all(genome.eval_weights[:-2] == 0.0)


def test_erl_carnivore_learns_live_but_genome_is_untouched():
    world = _erl_carn_world(n_initial_agents=8, grid_size=20)
    carnivore = world.carnivores[0]
    genome_before = carnivore.genome.action_weights.copy()
    for _ in range(60):
        world.step()
        if not carnivore.alive:
            break
    assert np.array_equal(carnivore.genome.action_weights, genome_before)
    assert not np.array_equal(carnivore.live_weights, genome_before)  # learning happened


def test_erl_carnivore_offspring_inherit_genome_not_learned_weights():
    world = _erl_carn_world()
    parent = world.carnivores[0]
    parent.live_weights += 50.0  # a "lifetime of learning"
    parent.energy = world.cfg["carnivore_reproduction_energy_threshold"]
    world._handle_carnivore_reproduction()
    child = world.carnivores[-1]
    assert np.array_equal(child.genome.action_weights, parent.genome.action_weights)
    assert np.array_equal(child.live_weights, child.genome.action_weights)
    assert not np.array_equal(child.live_weights, parent.live_weights)


def test_erl_carnivore_reinforces_previous_move_with_eval_change():
    """Controlled two steps: step 1 has no history (no update); step 2 must
    reinforce step 1's (obs, action) by e(obs2) - e(obs1)."""
    import predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world as wmod
    world = _erl_carn_world()
    carnivore = world.carnivores[0]
    obs1, obs2 = np.zeros(22), np.zeros(22)
    obs1[20], obs2[20] = 0.4, 0.9  # energy_norm rises
    seen = iter([obs1, obs2])
    world._observe_carnivore = lambda c: next(seen)
    calls = []
    original = wmod.reinforce_update
    wmod.reinforce_update = lambda w, b, o, a, r, lp, ln: calls.append((o.copy(), a, r))
    try:
        a1 = world._erl_carnivore_action(carnivore)
        assert calls == []
        world._erl_carnivore_action(carnivore)
    finally:
        wmod.reinforce_update = original
    (o, a, r), = calls
    e = lambda obs: float(obs @ carnivore.genome.eval_weights + carnivore.genome.eval_bias)
    assert np.array_equal(o, obs1) and a == a1 and np.isclose(r, e(obs2) - e(obs1))


def test_erl_offspring_with_mate_ignore_both_parents_learned_state():
    world = _erl_carn_world(n_initial_carnivores=2, mate_search_radius=200)
    parent, mate = world.carnivores
    for c in (parent, mate):
        c.live_weights += 50.0
        c.live_bias += 50.0
        c.prev_obs, c.prev_eval, c.last_action = np.ones(22), 3.0, 1
    parent.energy = world.cfg["carnivore_reproduction_energy_threshold"]
    world._handle_carnivore_reproduction()
    child = world.carnivores[-1]
    assert np.all(np.abs(child.live_weights) < 30) and np.all(np.abs(child.live_bias) < 30)
    assert np.array_equal(child.live_weights, child.genome.action_weights)
    assert np.array_equal(child.live_bias, child.genome.action_bias)
    assert child.prev_obs is None and child.prev_eval is None and child.last_action is None


def test_reward_baseline_subtracts_running_mean():
    import predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world as wmod
    world = _erl_carn_world(carnivore_reward_baseline=0.5)
    carnivore = world.carnivores[0]
    obs = [np.zeros(22) for _ in range(3)]
    obs[1][20], obs[2][20] = 0.2, 0.4  # eval rises by 5*0.2 = 1.0 twice (plus the noise-free... std 1 noise below)
    seen = iter(obs)
    world._observe_carnivore = lambda c: next(seen)
    got = []
    original = wmod.reinforce_update
    wmod.reinforce_update = lambda w, b, o, a, r, lp, ln: got.append(r)
    try:
        for _ in range(3):
            world._erl_carnivore_action(carnivore)
    finally:
        wmod.reinforce_update = original
    e = lambda o: float(o @ carnivore.genome.eval_weights + carnivore.genome.eval_bias)
    r1, r2 = e(obs[1]) - e(obs[0]), e(obs[2]) - e(obs[1])
    assert np.isclose(got[0], r1)  # baseline starts at 0
    assert np.isclose(got[1], r2 - 0.5 * r1)  # baseline after one update = 0.5 * r1


def test_trace_update_accumulates_decayed_gradients():
    world = _erl_carn_world(carnivore_trace_decay=0.5)
    carnivore = world.carnivores[0]
    obs = np.zeros(22)
    obs[0] = 1.0
    carnivore.prev_obs, carnivore.last_action = obs, 0
    before = carnivore.live_weights.copy()
    world._trace_update(carnivore, 0.0, 0.5)  # zero reinforcement: trace builds, no weight change
    g1 = carnivore.trace_w.copy()
    assert np.array_equal(carnivore.live_weights, before) and g1[0, 0] > 0
    world._trace_update(carnivore, 1.0, 0.5)
    assert np.allclose(carnivore.trace_w, 0.5 * g1 + g1, atol=1e-9)  # same obs/action/probs twice
    assert carnivore.live_weights[0, 0] > before[0, 0]


def test_founder_eval_std_zero_gives_pure_energy_goal():
    world = _erl_carn_world(carnivore_founder_eval_std=0.0)
    genome = world.carnivores[0].genome
    assert genome.eval_weights[-2] == 5.0 and np.all(genome.eval_weights[:-2] == 0.0) and genome.eval_bias == 0.0
    assert np.any(genome.action_weights != world._seed_weights("carnivore_seed"))  # action noise untouched
