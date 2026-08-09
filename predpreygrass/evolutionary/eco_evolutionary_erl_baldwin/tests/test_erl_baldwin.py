import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import (
    crossover,
    founder_genome,
    mutate,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import (
    action_probs,
    evaluate,
    reinforce_update,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld, N_ACTIONS, OBS_DIM


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_founder_genome_shapes(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    assert g.eval_weights.shape == (OBS_DIM,)
    assert g.action_weights.shape == (OBS_DIM, N_ACTIONS)
    assert g.action_bias.shape == (N_ACTIONS,)
    assert isinstance(g.eval_bias, float)


def test_genome_copy_is_independent(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    c = g.copy()
    c.action_weights[0, 0] += 100.0
    assert g.action_weights[0, 0] != c.action_weights[0, 0]


def test_mutate_changes_some_but_not_all_sites(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, init_std=1.0)
    m = mutate(g, rng, rate=0.5, std=0.1)
    diffs = m.action_weights != g.action_weights
    assert diffs.any(), "mutation with rate=0.5 should change at least some sites"
    assert not diffs.all(), "mutation with rate=0.5 should not change every site"


def test_mutate_zero_rate_is_noop(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    m = mutate(g, rng, rate=0.0, std=1.0)
    assert np.array_equal(m.action_weights, g.action_weights)
    assert np.array_equal(m.eval_weights, g.eval_weights)


def test_crossover_mixes_both_parents(rng):
    a = founder_genome(OBS_DIM, N_ACTIONS, rng)
    b = founder_genome(OBS_DIM, N_ACTIONS, rng)
    child = crossover(a, b, rng)
    from_a = np.isclose(child.action_weights, a.action_weights)
    from_b = np.isclose(child.action_weights, b.action_weights)
    assert (from_a | from_b).all()
    assert from_a.any() and from_b.any(), "crossover should draw sites from both parents"


def test_action_probs_sum_to_one(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    obs = rng.uniform(0, 1, size=OBS_DIM)
    probs = action_probs(obs, g.action_weights, g.action_bias)
    assert probs.shape == (N_ACTIONS,)
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()


def test_positive_reinforcement_increases_taken_action_probability(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, init_std=0.01)  # near-uniform start
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    action_weights = g.action_weights.copy()
    action_bias = g.action_bias.copy()
    before = action_probs(obs, action_weights, action_bias)
    taken = 2
    reinforce_update(action_weights, action_bias, obs, taken, reinforcement=1.0, lr_positive=0.5, lr_negative=0.5)
    after = action_probs(obs, action_weights, action_bias)
    assert after[taken] > before[taken]


def test_negative_reinforcement_decreases_taken_action_probability(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, init_std=0.01)
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    action_weights = g.action_weights.copy()
    action_bias = g.action_bias.copy()
    before = action_probs(obs, action_weights, action_bias)
    taken = 2
    reinforce_update(action_weights, action_bias, obs, taken, reinforcement=-1.0, lr_positive=0.5, lr_negative=0.5)
    after = action_probs(obs, action_weights, action_bias)
    assert after[taken] < before[taken]


def test_evaluate_is_linear_scalar(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    obs = np.zeros(OBS_DIM)
    assert evaluate(obs, g.eval_weights, g.eval_bias) == pytest.approx(g.eval_bias)


# --- The Darwinian-not-Lamarckian invariant ---


def test_offspring_genome_does_not_inherit_parents_learned_weights(rng):
    """The critical correctness property: an offspring's genome must come from the
    parent's GENOME record, never from the parent's LIVE (post-learning) action
    network. Simulates a parent that has "learned" during its life (live weights
    diverged from its genome) and asserts the offspring's genome matches the
    parent's original genome, not the learned live weights.
    """
    cfg = dict(config_erl, n_initial_prey=1, n_initial_predators=1, mutation_rate=0.0)
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]

    original_genome_action_weights = parent.genome.action_weights.copy()

    # Simulate a lifetime of learning: the live action network drifts away from
    # the genome-specified initial weights, but the genome record itself must
    # stay untouched by this.
    parent.action_weights += 50.0
    assert not np.array_equal(parent.action_weights, parent.genome.action_weights)
    assert np.array_equal(parent.genome.action_weights, original_genome_action_weights)

    # Force reproduction directly (bypass energy threshold for a deterministic test).
    parent.energy = cfg["reproduction_energy_threshold_prey"] + 1
    world.rng = np.random.default_rng(1)  # avoid crossover finding a mate (none exists)
    world._handle_reproduction()

    assert len(world.agents) == 3  # original prey + original predator + new prey child
    child = world.agents[-1]
    assert child.species == "prey"
    # mutation_rate=0.0, no mate available -> child genome == parent's GENOME record exactly
    assert np.array_equal(child.genome.action_weights, original_genome_action_weights)
    # and NOT equal to the parent's learned live weights
    assert not np.array_equal(child.genome.action_weights, parent.action_weights)
    # child's own live network starts from its (inherited) genome, not the parent's live state
    assert np.array_equal(child.action_weights, child.genome.action_weights)


def test_world_smoke_runs_without_crashing(rng):
    world = ErlWorld(dict(config_erl), rng)
    for _ in range(200):
        world.step()
        counts = world.population_counts()
        if counts["predator"] == 0 or counts["prey"] == 0:
            break
    # No assertion on survival -- 200 steps is too short to expect stability,
    # this only checks the mechanics don't crash.


def test_genome_stats_nan_when_species_extinct(rng):
    world = ErlWorld(dict(config_erl, n_initial_prey=0, n_initial_predators=1), rng)
    stats = world.genome_stats()
    assert np.isnan(stats["prey_eval_weight_absmean"])
    assert not np.isnan(stats["predator_eval_weight_absmean"])
