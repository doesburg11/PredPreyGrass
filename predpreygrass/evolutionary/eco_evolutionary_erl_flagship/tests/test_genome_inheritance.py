import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.genome import founder_genome, mutate


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_founder_genome_shapes(rng):
    genome = founder_genome(obs_dim=8, n_actions=9, rng=rng)
    assert genome.eval_weights.shape == (8,)
    assert genome.action_weights.shape == (8, 9)
    assert genome.action_bias.shape == (9,)


def test_founder_genome_fixed_eval_weights(rng):
    fixed = np.arange(8, dtype=float)
    genome = founder_genome(obs_dim=8, n_actions=9, rng=rng, fixed_eval_weights=fixed)
    np.testing.assert_array_equal(genome.eval_weights, fixed)


def test_mutate_perturbs_at_least_one_site_with_high_rate(rng):
    genome = founder_genome(obs_dim=8, n_actions=9, rng=rng)
    child = mutate(genome, rng, rate=1.0, std=0.1)  # rate=1.0: every site mutates
    assert not np.array_equal(genome.eval_weights, child.eval_weights)
    assert not np.array_equal(genome.action_weights, child.action_weights)


def test_mutate_zero_rate_leaves_genome_unchanged(rng):
    genome = founder_genome(obs_dim=8, n_actions=9, rng=rng)
    child = mutate(genome, rng, rate=0.0, std=0.1)
    np.testing.assert_array_equal(genome.eval_weights, child.eval_weights)
    np.testing.assert_array_equal(genome.action_weights, child.action_weights)


def test_mutate_returns_a_copy_not_same_object(rng):
    genome = founder_genome(obs_dim=8, n_actions=9, rng=rng)
    child = mutate(genome, rng, rate=0.5, std=0.1)
    assert child is not genome
    assert child.eval_weights is not genome.eval_weights


def test_offspring_genome_does_not_inherit_parents_learned_weights(rng):
    """The genome-level `action_weights` (what mutate/inheritance operates on) must
    stay independent of a parent's LIVE, learned action_weights (the ones
    reinforce_update modifies during life, tracked separately in
    driver.PreyGenomeState.action_weights) -- Darwinian, not Lamarckian, exactly as
    in eco_evolutionary_erl_baldwin."""
    genome = founder_genome(obs_dim=8, n_actions=9, rng=rng)
    original_genome_action_weights = genome.action_weights.copy()

    # Simulate a lifetime of learning on a LIVE copy, as driver.py does.
    live_action_weights = genome.action_weights.copy()
    live_action_weights += 100.0  # drastic "learning"

    # The genome record itself must be untouched by that.
    np.testing.assert_array_equal(genome.action_weights, original_genome_action_weights)

    child = mutate(genome, rng, rate=0.05, std=0.05)
    # Child's action_weights should be close to the ORIGINAL genome record (plus
    # small mutation noise), nowhere near the drastically "learned" live copy.
    assert np.abs(child.action_weights - live_action_weights).min() > 50.0
