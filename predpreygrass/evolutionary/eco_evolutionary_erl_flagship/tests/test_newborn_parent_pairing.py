import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import PreyGenomeState, Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.genome import founder_genome


class _FakeEnv:
    """Exposes exactly what _handle_reproduction needs: agents, agent_positions,
    reproduction_reward_prey -- no real PredPreyGrass/RLlib needed."""

    def __init__(self, agents, agent_positions, reproduction_reward_prey=10.0):
        self.agents = agents
        self.agent_positions = agent_positions
        self.reproduction_reward_prey = reproduction_reward_prey


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _cfg():
    return {"founder_weight_std": 0.5, "mutation_rate": 0.05, "mutation_std": 0.05}


def _state(agent_id, genome, generation=0, born_step=0):
    return PreyGenomeState(
        agent_id=agent_id, genome=genome,
        action_weights=genome.action_weights.copy(), action_bias=genome.action_bias.copy(),
        generation=generation, born_step=born_step,
    )


def test_single_reproduction_registers_newborn_with_mutated_genome(rng):
    positions = {"prey_0": (5, 5), "prey_1": (5, 6)}  # prey_1: fresh id, adjacent to prey_0
    env = _FakeEnv(agents=["prey_0", "prey_1"], agent_positions=positions)
    driver = Trial13Driver(env=env, predator_policy=None, cfg=_cfg(), rng=rng)
    parent_genome = founder_genome(8, 9, rng)
    driver.registry = {"prey_0": _state("prey_0", parent_genome)}
    driver.current_step = 10

    driver._handle_reproduction(rewards={"prey_0": 10.0})

    assert driver.registry["prey_0"].offspring_count == 1
    assert "prey_1" in driver.registry
    child = driver.registry["prey_1"]
    assert child.generation == 1
    assert child.born_step == 10
    # Mutation-only inheritance: child stays close to the parent's genome (small
    # Gaussian perturbation), not an arbitrary/unrelated genome.
    assert np.abs(child.genome.eval_weights - parent_genome.eval_weights).max() < 1.0


def test_newborn_matched_to_nearest_parent_among_multiple(rng):
    positions = {
        "prey_0": (0, 0), "prey_1": (10, 10),
        "prey_2": (0, 1),  # adjacent to prey_0
        "prey_3": (10, 11),  # adjacent to prey_1
    }
    env = _FakeEnv(agents=list(positions), agent_positions=positions)
    driver = Trial13Driver(env=env, predator_policy=None, cfg=_cfg(), rng=rng)
    g0 = founder_genome(8, 9, rng)
    g1 = founder_genome(8, 9, rng)
    driver.registry = {"prey_0": _state("prey_0", g0), "prey_1": _state("prey_1", g1)}

    driver._handle_reproduction(rewards={"prey_0": 10.0, "prey_1": 10.0})

    assert driver.registry["prey_2"].generation == 1
    assert driver.registry["prey_3"].generation == 1
    # prey_2 (adjacent to prey_0) should carry a mutated copy of g0, not g1 --
    # and vice versa for prey_3/g1.
    assert np.abs(driver.registry["prey_2"].genome.eval_weights - g0.eval_weights).max() < 1.0
    assert np.abs(driver.registry["prey_3"].genome.eval_weights - g1.eval_weights).max() < 1.0


def test_no_reproduction_reward_leaves_registry_unchanged(rng):
    positions = {"prey_0": (0, 0)}
    env = _FakeEnv(agents=["prey_0"], agent_positions=positions)
    driver = Trial13Driver(env=env, predator_policy=None, cfg=_cfg(), rng=rng)
    g0 = founder_genome(8, 9, rng)
    driver.registry = {"prey_0": _state("prey_0", g0)}

    driver._handle_reproduction(rewards={"prey_0": 0.0})

    assert driver.registry["prey_0"].offspring_count == 0
    assert len(driver.registry) == 1


def test_empty_rewards_is_a_noop(rng):
    env = _FakeEnv(agents=["prey_0"], agent_positions={"prey_0": (0, 0)})
    driver = Trial13Driver(env=env, predator_policy=None, cfg=_cfg(), rng=rng)
    g0 = founder_genome(8, 9, rng)
    driver.registry = {"prey_0": _state("prey_0", g0)}

    driver._handle_reproduction(rewards={})

    assert len(driver.registry) == 1
