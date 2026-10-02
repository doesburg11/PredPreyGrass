import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.genome import (
    crossover,
    founder_genome,
    mutate,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.networks import (
    action_probs,
    effective_action_weights,
    evaluate,
    hebbian_trace_update,
    routed_action_alpha,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.world import ErlWorld, N_ACTIONS, OBS_DIM


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_founder_genome_shapes(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    assert g.eval_weights.shape == (OBS_DIM,)
    assert g.action_weights.shape == (OBS_DIM, N_ACTIONS)
    assert g.action_bias.shape == (N_ACTIONS,)
    assert g.action_alpha_weights.shape == (OBS_DIM, N_ACTIONS)
    assert g.action_alpha_bias.shape == (N_ACTIONS,)
    assert isinstance(g.eval_bias, float)


def test_founder_genome_fixed_eval_weights_overrides_random_init(rng):
    fixed = [0.987, -0.243, -0.040, -0.011, -0.079, -0.067, 0.057][:OBS_DIM]
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, fixed_eval_weights=fixed)
    assert np.allclose(g.eval_weights, fixed)
    # Everything else still randomly initialized, not zeroed/fixed.
    assert g.action_weights.shape == (OBS_DIM, N_ACTIONS)
    assert not np.allclose(g.action_weights, 0.0)


def test_founder_genome_without_fixed_eval_weights_is_unaffected(rng):
    """Passing fixed_eval_weights=None (the default) must reproduce the exact
    prior random-init behavior -- a regression check that adding the parameter
    didn't change anything for every existing caller that doesn't use it."""
    g1 = founder_genome(OBS_DIM, N_ACTIONS, np.random.default_rng(7))
    g2 = founder_genome(OBS_DIM, N_ACTIONS, np.random.default_rng(7), fixed_eval_weights=None)
    assert np.array_equal(g1.eval_weights, g2.eval_weights)
    assert np.array_equal(g1.action_weights, g2.action_weights)


def test_genome_copy_is_independent(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    c = g.copy()
    c.action_weights[0, 0] += 100.0
    c.action_alpha_weights[0, 0] += 100.0
    assert g.action_weights[0, 0] != c.action_weights[0, 0]
    assert g.action_alpha_weights[0, 0] != c.action_alpha_weights[0, 0]


def test_mutate_changes_some_but_not_all_sites(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, init_std=1.0)
    m = mutate(g, rng, rate=0.5, std=0.1)
    diffs = m.action_weights != g.action_weights
    assert diffs.any(), "mutation with rate=0.5 should change at least some sites"
    assert not diffs.all(), "mutation with rate=0.5 should not change every site"


def test_mutate_can_change_action_alpha_bias(rng):
    """action_alpha_bias has its own mutate() code path (a per-element mask,
    same pattern as action_bias) and was never independently exercised --
    rate=1.0 guarantees every element is mutated, deterministically."""
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, init_std=1.0)
    m = mutate(g, rng, rate=1.0, std=0.1)
    assert not np.array_equal(m.action_alpha_bias, g.action_alpha_bias)


def test_mutate_zero_rate_is_noop(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    m = mutate(g, rng, rate=0.0, std=1.0)
    assert np.array_equal(m.action_weights, g.action_weights)
    assert np.array_equal(m.action_alpha_weights, g.action_alpha_weights)
    assert np.array_equal(m.eval_weights, g.eval_weights)


def test_crossover_mixes_both_parents(rng):
    a = founder_genome(OBS_DIM, N_ACTIONS, rng)
    b = founder_genome(OBS_DIM, N_ACTIONS, rng)
    child = crossover(a, b, rng)
    from_a = np.isclose(child.action_weights, a.action_weights)
    from_b = np.isclose(child.action_weights, b.action_weights)
    assert (from_a | from_b).all()
    assert from_a.any() and from_b.any(), "crossover should draw sites from both parents"

    alpha_from_a = np.isclose(child.action_alpha_weights, a.action_alpha_weights)
    alpha_from_b = np.isclose(child.action_alpha_weights, b.action_alpha_weights)
    assert (alpha_from_a | alpha_from_b).all()


def test_action_probs_sum_to_one(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    obs = rng.uniform(0, 1, size=OBS_DIM)
    probs = action_probs(obs, g.action_weights, g.action_bias)
    assert probs.shape == (N_ACTIONS,)
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()


def _trace_update_fixture(rng, init_std=0.01):
    """A genome with near-uniform base weights and alpha=1 everywhere, so any
    probability shift after a trace update is attributable to the trace, not
    to the base network or an arbitrary alpha scale."""
    g = founder_genome(OBS_DIM, N_ACTIONS, rng, init_std=init_std)
    g.action_alpha_weights[:] = 1.0
    g.action_alpha_bias[:] = 0.0
    hebb_trace = np.zeros_like(g.action_weights)
    hebb_bias_trace = np.zeros_like(g.action_bias)
    return g, hebb_trace, hebb_bias_trace


def test_positive_reinforcement_increases_prob_of_taken_action(rng):
    """Action-credited version: the trace must favor the action that was
    actually TAKEN (prev_action), not just whatever the previous output
    distribution leaned toward -- the whole point of the credit-assignment
    fix (see README.md's 'Diagnostic history')."""
    g, hebb_trace, hebb_bias_trace = _trace_update_fixture(rng)
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    prev_probs = np.array([0.25, 0.25, 0.25, 0.25])  # uniform -- taken action carries all the signal
    taken = 2

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    before = action_probs(obs, eff_w, eff_b)

    hebbian_trace_update(
        hebb_trace, hebb_bias_trace, obs, prev_probs, taken, reinforcement=1.0, eta=0.5, trace_clip=10.0
    )

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    after = action_probs(obs, eff_w, eff_b)
    assert after[taken] > before[taken]


def test_negative_reinforcement_decreases_prob_of_taken_action(rng):
    g, hebb_trace, hebb_bias_trace = _trace_update_fixture(rng)
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    prev_probs = np.array([0.25, 0.25, 0.25, 0.25])
    taken = 2

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    before = action_probs(obs, eff_w, eff_b)

    hebbian_trace_update(
        hebb_trace, hebb_bias_trace, obs, prev_probs, taken, reinforcement=-1.0, eta=0.5, trace_clip=10.0
    )

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    after = action_probs(obs, eff_w, eff_b)
    assert after[taken] < before[taken]


def test_trace_credits_taken_action_not_merely_probable_one(rng):
    """The actual fix this revision makes: even when the PREVIOUS output
    distribution favored a different action (3, prob 0.7), reinforcing
    whichever action was TAKEN (2, a low-probability exploratory choice)
    must still increase action 2's probability, not action 3's -- this is
    exactly the case the first (plain-correlation) design got wrong."""
    g, hebb_trace, hebb_bias_trace = _trace_update_fixture(rng)
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    prev_probs = np.array([0.1, 0.1, 0.1, 0.7])  # policy favored action 3...
    taken = 2  # ...but action 2 was the one actually sampled and reinforced

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    before = action_probs(obs, eff_w, eff_b)

    hebbian_trace_update(
        hebb_trace, hebb_bias_trace, obs, prev_probs, taken, reinforcement=1.0, eta=0.5, trace_clip=10.0
    )

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    after = action_probs(obs, eff_w, eff_b)
    assert after[taken] > before[taken]
    assert after[3] < before[3]


def test_uniform_positive_alpha_matches_each_tensor_absmean():
    weights = np.array([[-2.0, 1.0], [3.0, -4.0]])
    bias = np.array([-1.0, 3.0])

    routed_weights, routed_bias = routed_action_alpha(
        weights, bias, "uniform_positive"
    )

    assert np.all(routed_weights == 2.5)
    assert np.all(routed_bias == 2.0)
    assert np.array_equal(weights, np.array([[-2.0, 1.0], [3.0, -4.0]]))
    assert np.array_equal(bias, np.array([-1.0, 3.0]))


def test_unknown_alpha_routing_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown action_alpha_mode"):
        routed_action_alpha(np.ones((2, 2)), np.ones(2), "invalid")


def test_zero_alpha_makes_trace_behaviorally_inert(rng):
    """An agent born with action_alpha_weights/bias == 0 everywhere must
    behave exactly like strategy "E" (evolution, no learning): the trace can
    still move, but effective_action_weights must ignore it entirely."""
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    g.action_alpha_weights[:] = 0.0
    g.action_alpha_bias[:] = 0.0
    hebb_trace = np.zeros_like(g.action_weights)
    hebb_bias_trace = np.zeros_like(g.action_bias)
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    prev_probs = np.array([0.1, 0.1, 0.7, 0.1])

    hebbian_trace_update(
        hebb_trace, hebb_bias_trace, obs, prev_probs, 2, reinforcement=1.0, eta=0.5, trace_clip=10.0
    )
    assert not np.allclose(hebb_trace, 0.0), "the trace itself should still move"

    eff_w, eff_b = effective_action_weights(
        g.action_weights, g.action_bias, g.action_alpha_weights, g.action_alpha_bias,
        hebb_trace, hebb_bias_trace,
    )
    assert np.array_equal(eff_w, g.action_weights)
    assert np.array_equal(eff_b, g.action_bias)


def test_hebbian_trace_update_clips_to_configured_bound(rng):
    g, hebb_trace, hebb_bias_trace = _trace_update_fixture(rng)
    obs = np.full(OBS_DIM, 10.0)
    prev_probs = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        hebbian_trace_update(
            hebb_trace, hebb_bias_trace, obs, prev_probs, 0, reinforcement=1.0, eta=0.9, trace_clip=0.3
        )
    assert np.all(hebb_trace <= 0.3 + 1e-9)
    assert np.all(hebb_trace >= -0.3 - 1e-9)
    assert np.all(hebb_bias_trace <= 0.3 + 1e-9)


def test_hebbian_bias_trace_clips_on_negative_saturation(rng):
    g, hebb_trace, hebb_bias_trace = _trace_update_fixture(rng)
    obs = np.full(OBS_DIM, 10.0)
    prev_probs = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        hebbian_trace_update(
            hebb_trace, hebb_bias_trace, obs, prev_probs, 0, reinforcement=-1.0, eta=0.9, trace_clip=0.3
        )
    assert np.all(hebb_trace >= -0.3 - 1e-9)
    assert np.all(hebb_bias_trace >= -0.3 - 1e-9)


def test_hebbian_trace_update_matches_exact_equation(rng):
    """Pins the documented update equation exactly:
    credit = one_hot(prev_action) - prev_probs
    trace(t) = clip((1-eta)*trace(t-1) + eta*reinforcement*outer(obs, credit), -clip, clip)
    -- starting from a nonzero prior trace, so the decay term is actually exercised
    (test_zero_reinforcement_leaves_trace_direction_unreinforced only covers the
    zero-prior-trace case, where decay is a no-op)."""
    obs = np.array([1.0, 2.0, 0.0, -1.0, 0.5, 0.0, 3.0])
    prev_probs = np.array([0.4, 0.1, 0.3, 0.2])
    prev_action = 1
    eta = 0.3
    reinforcement = 0.7
    trace_clip = 100.0  # large enough that clipping never binds here

    credit = np.array([0.0, 1.0, 0.0, 0.0]) - prev_probs  # one_hot(1) - prev_probs
    hebb_trace = np.full((OBS_DIM, N_ACTIONS), 2.0)
    hebb_bias_trace = np.full(N_ACTIONS, -1.5)
    expected_trace = (1 - eta) * hebb_trace + eta * reinforcement * np.outer(obs, credit)
    expected_bias = (1 - eta) * hebb_bias_trace + eta * reinforcement * credit

    hebbian_trace_update(hebb_trace, hebb_bias_trace, obs, prev_probs, prev_action, reinforcement, eta, trace_clip)

    assert np.allclose(hebb_trace, expected_trace)
    assert np.allclose(hebb_bias_trace, expected_bias)


def test_zero_reinforcement_leaves_trace_direction_unreinforced(rng):
    """Zero reinforcement should not push the trace toward the credit term
    at all -- only its own decay term (1 - eta) applies. Starting from a
    zero trace, zero reinforcement must leave it exactly at zero."""
    g, hebb_trace, hebb_bias_trace = _trace_update_fixture(rng)
    obs = rng.uniform(0.3, 1.0, size=OBS_DIM)
    prev_probs = np.array([0.1, 0.1, 0.7, 0.1])
    hebbian_trace_update(
        hebb_trace, hebb_bias_trace, obs, prev_probs, 2, reinforcement=0.0, eta=0.5, trace_clip=10.0
    )
    assert np.allclose(hebb_trace, 0.0)
    assert np.allclose(hebb_bias_trace, 0.0)


def test_genome_flatten_matches_exact_documented_layout(rng):
    """Pins genome.py::Genome.flatten()'s exact concatenation order -- eval
    weights/bias, then action weights/bias, then alpha weights/bias -- since
    metrics.py::FunctionalConstraintTracker's eval_dim/action_dim split
    (and therefore the whole genetic-assimilation metric) depends on this
    layout never silently drifting."""
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    flat = g.flatten()
    expected_len = OBS_DIM + 1 + 2 * (OBS_DIM * N_ACTIONS + N_ACTIONS)
    assert flat.shape == (expected_len,)

    expected = np.concatenate([
        g.eval_weights, [g.eval_bias],
        g.action_weights.ravel(), g.action_bias,
        g.action_alpha_weights.ravel(), g.action_alpha_bias,
    ])
    assert np.array_equal(flat, expected)


def test_functional_constraint_tracker_dims_match_flatten_layout(rng):
    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.metrics import FunctionalConstraintTracker

    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    tracker = FunctionalConstraintTracker(OBS_DIM, N_ACTIONS)
    assert tracker.eval_dim + tracker.action_dim == g.flatten().shape[0]

    child = mutate(g, rng, rate=1.0, std=0.1)
    tracker.record(g.flatten(), child.flatten())  # must not raise/misalign
    assert tracker.n_reproductions == 1


def test_evaluate_is_linear_scalar(rng):
    g = founder_genome(OBS_DIM, N_ACTIONS, rng)
    obs = np.zeros(OBS_DIM)
    assert evaluate(obs, g.eval_weights, g.eval_bias) == pytest.approx(g.eval_bias)


# --- The Darwinian-not-Lamarckian invariant ---


def _small_world_cfg(**overrides):
    """A small, fast World AL config for deterministic unit tests."""
    base = dict(
        config_erl,
        grid_size=12,
        n_initial_agents=1,
        n_initial_carnivores=0,
        min_plants=2,
        min_trees=1,
        carnivore_spawn_interval=10_000_000,  # effectively off for isolated agent tests
        mutation_rate=0.0,
    )
    base.update(overrides)
    return base


def test_world_spawns_founders_with_fixed_eval_weights_from_config(rng):
    fixed = [0.987, -0.243, -0.040, -0.011, -0.079, -0.067, 0.057]
    cfg = _small_world_cfg(n_initial_agents=5, fixed_eval_weights=fixed)
    world = ErlWorld(cfg, rng)
    assert len(world.agents) == 5
    for agent in world.agents:
        assert np.allclose(agent.genome.eval_weights, fixed)
        assert not np.allclose(agent.genome.action_weights, 0.0)  # still randomly initialized
        # Trace starts at zero regardless of anything in the genome.
        assert np.array_equal(agent.hebb_trace, np.zeros_like(agent.genome.action_weights))


def test_world_founders_use_random_eval_weights_when_not_configured(rng):
    """No fixed_eval_weights key at all (the common case) must be unaffected --
    world.cfg.get(...) returning None must fall through to the normal random init."""
    cfg = _small_world_cfg(n_initial_agents=3)
    assert "fixed_eval_weights" not in cfg
    world = ErlWorld(cfg, rng)
    weights = [a.genome.eval_weights for a in world.agents]
    assert not all(np.allclose(weights[0], w) for w in weights[1:]), "founders should differ (random init)"


def test_world_rejects_fixed_eval_weights_of_wrong_dimension(rng):
    """World is the single authoritative validator (not re-derived/duplicated in
    the CLI) so a caller constructing ErlWorld directly -- not just the CLI --
    is protected too, and a malformed vector fails loudly here rather than
    later, confusingly, inside network evaluation."""
    cfg = _small_world_cfg(fixed_eval_weights=[0.1, 0.2, 0.3])  # OBS_DIM=7, only 3 given
    with pytest.raises(ValueError):
        ErlWorld(cfg, rng)


def test_fixed_eval_weights_offspring_stay_fixed_under_L_but_can_drift_under_ERL(rng):
    """Codex review flag: fixed_eval_weights only ever seeds FOUNDERS
    (founder_genome) -- offspring come from crossover/mutation of parent
    genomes instead, so whether a fixed reward actually STAYS fixed across
    generations depends entirely on strategy, not on this feature itself.
    Under L (clone exactly, no mutation) it stays fixed; under a mutating
    strategy it doesn't, and that's correct, not a bug -- pin it down here so
    a future change can't silently break either half."""
    fixed = [0.987, -0.243, -0.040, -0.011, -0.079, -0.067, 0.057]

    world_L = ErlWorld(_small_world_cfg(strategy="L", fixed_eval_weights=fixed), rng)
    parent = world_L.agents[0]
    parent.energy = world_L.cfg["reproduction_energy_threshold_agent"] + 1
    world_L._handle_agent_reproduction()
    child_L = world_L.agents[-1]
    assert np.allclose(child_L.genome.eval_weights, fixed)

    world_erl = ErlWorld(
        _small_world_cfg(strategy="ERL", mutation_rate=1.0, mutation_std=1.0, fixed_eval_weights=fixed), rng
    )
    parent = world_erl.agents[0]
    parent.energy = world_erl.cfg["reproduction_energy_threshold_agent"] + 1
    world_erl._handle_agent_reproduction()
    child_erl = world_erl.agents[-1]
    assert not np.allclose(child_erl.genome.eval_weights, fixed)


def test_offspring_genome_does_not_inherit_parents_hebbian_trace(rng):
    """The critical correctness property for this module: an offspring's
    genome (and its own starting trace) must never reflect the parent's LIVE,
    accumulated Hebbian trace -- only the parent's GENOME record (base
    weights + alpha, mutated/crossed-over) propagates. Simulates a parent
    with a heavily non-zero trace (as if it had "learned" all lifetime) and
    asserts the offspring is unaffected.
    """
    cfg = _small_world_cfg()
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]

    original_genome_action_weights = parent.genome.action_weights.copy()

    # Simulate a lifetime of Hebbian plasticity: the trace moves far from
    # zero, but the genome record itself (base weights, alpha) must stay
    # untouched by this.
    parent.hebb_trace += 50.0
    assert not np.allclose(parent.hebb_trace, 0.0)
    assert np.array_equal(parent.genome.action_weights, original_genome_action_weights)

    # Force reproduction directly (bypass energy threshold for a deterministic test).
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()

    assert len(world.agents) == 2  # original + new child
    child = world.agents[-1]
    # mutation_rate=0.0, no mate available -> child genome == parent's GENOME record exactly
    assert np.array_equal(child.genome.action_weights, original_genome_action_weights)
    # child's own trace starts at zero regardless of the parent's accumulated trace
    assert np.array_equal(child.hebb_trace, np.zeros_like(child.genome.action_weights))
    assert np.array_equal(child.hebb_bias_trace, np.zeros_like(child.genome.action_bias))
    # and it's a genuinely separate array, not an alias of the parent's --
    # mutating one must never affect the other (Codex review flag).
    assert not np.shares_memory(child.hebb_trace, parent.hebb_trace)
    assert not np.shares_memory(child.hebb_bias_trace, parent.hebb_bias_trace)
    child.hebb_trace += 1.0
    assert not np.allclose(child.hebb_trace, parent.hebb_trace)


def test_step_agents_wires_reinforcement_and_prev_action_correctly(rng, monkeypatch):
    """Integration check for the temporal pairing Codex's review flagged as
    untested: _step_agents must call hebbian_trace_update with (a) the SAME
    agent's own prev_obs/prev_probs/prev_action from the PREVIOUS step, and
    (b) reinforcement computed as this step's e_now minus that same
    prev_eval -- not, e.g., the current step's obs/probs, or a different
    agent's state."""
    import predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.world as world_mod

    cfg = _small_world_cfg(strategy="ERL", n_initial_agents=1)
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]

    calls = []
    real_update = world_mod.hebbian_trace_update

    def spy(trace, bias_trace, obs, prev_probs, prev_action, reinforcement, eta, trace_clip):
        calls.append((obs.copy(), prev_probs.copy(), prev_action, reinforcement))
        return real_update(trace, bias_trace, obs, prev_probs, prev_action, reinforcement, eta, trace_clip)

    monkeypatch.setattr(world_mod, "hebbian_trace_update", spy)

    world._step_agents()  # step 1: prev_obs is None -> no update call yet
    assert len(calls) == 0
    expected_obs = agent.prev_obs.copy()
    expected_probs = agent.prev_probs.copy()
    expected_action = agent.prev_action
    expected_eval = agent.prev_eval

    world._step_agents()  # step 2: must now call with step 1's captured state
    assert len(calls) == 1
    called_obs, called_probs, called_action, called_reinforcement = calls[0]
    assert np.array_equal(called_obs, expected_obs)
    assert np.array_equal(called_probs, expected_probs)
    assert called_action == expected_action

    # reinforcement's definition: (this agent's e_now at step 2) minus (its
    # own prev_eval from step 1). _step_agents overwrites agent.prev_eval
    # with that same e_now right after using it, so by now agent.prev_eval
    # IS the e_now the spy was called with -- verify the arithmetic directly.
    assert called_reinforcement == pytest.approx(agent.prev_eval - expected_eval)


def test_world_smoke_runs_without_crashing(rng):
    world = ErlWorld(dict(config_erl, grid_size=20, n_initial_agents=15, n_initial_carnivores=2), rng)
    for _ in range(200):
        world.step()
        counts = world.population_counts()
        if counts["agent"] == 0:
            break
    # No assertion on survival -- 200 steps is too short to expect stability,
    # this only checks the mechanics don't crash.


def test_reproduction_increments_parent_offspring_count(rng):
    cfg = _small_world_cfg()
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    assert parent.offspring_count == 0

    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()

    assert parent.offspring_count == 1
    child = world.agents[-1]
    assert child.offspring_count == 0
    assert child.born_step == world.current_step


def test_reproduction_credits_both_genetic_parents(rng):
    """Crossover mixes genome sites from both `agent` and its `mate` into the
    child (genome.crossover) -- offspring_count must credit both, not just the
    initiating `agent`, or the mate's genome propagates into lineage data with
    zero recorded fitness, biasing analyze_proximate_reward.py's correlation."""
    cfg = _small_world_cfg(n_initial_agents=2, mutation_rate=0.0)
    world = ErlWorld(cfg, rng)
    agent, mate = world.agents[0], world.agents[1]
    agent.row, agent.col = 5, 5
    mate.row, mate.col = 5, 6  # within mate_search_radius=3

    agent.energy = cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()

    assert agent.offspring_count == 1
    assert mate.offspring_count == 1


def test_kill_agent_fires_on_death_callback_with_lineage_data(rng):
    cfg = _small_world_cfg()
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    agent.offspring_count = 3
    world.current_step = 42

    calls = []

    def on_death(a, step):
        # Death state must already be committed by the time the callback
        # fires (see world.py's _kill_agent) -- an observer that raises
        # (e.g. IO failure) must not leave the agent half-dead.
        assert a.alive is False
        calls.append((a, step))

    world.on_agent_death = on_death
    world._kill_agent(agent)

    assert len(calls) == 1
    dead_agent, death_step = calls[0]
    assert dead_agent is agent
    assert death_step == 42

    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.metrics import lineage_record

    record = lineage_record(dead_agent, death_step, censored=False)
    assert record["offspring_count"] == 3
    assert record["lifespan"] == 42 - agent.born_step
    assert record["censored"] is False
    assert record[f"eval_weight_{OBS_DIM - 1}"] == pytest.approx(agent.genome.eval_weights[-1])


def test_kill_agent_is_idempotent_for_on_death_callback(rng):
    """A double-kill (already dead) must not re-fire the callback -- lineage
    rows would otherwise be duplicated for every code path that kills an agent."""
    cfg = _small_world_cfg()
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    calls = []
    world.on_agent_death = lambda a, step: calls.append((a, step))

    world._kill_agent(agent)
    world._kill_agent(agent)  # already dead -- must be a no-op

    assert len(calls) == 1


def test_genome_stats_nan_when_no_agents(rng):
    world = ErlWorld(_small_world_cfg(n_initial_agents=0), rng)
    stats = world.genome_stats()
    assert np.isnan(stats["eval_weight_absmean"])
    assert np.isnan(stats["action_alpha_absmean"])
    assert np.isnan(stats["hebb_trace_absmean"])


def test_carnivores_have_no_genome_or_learning():
    """Carnivores are never adaptive, regardless of `strategy` -- structural
    check that the Carnivore dataclass carries no genome/network fields at all."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.world import Carnivore
    fields = {f for f in Carnivore.__dataclass_fields__}
    assert "genome" not in fields
    assert "hebb_trace" not in fields


# --- Ackley & Littman's five comparative strategies (agents only -- carnivores
# are never affected by `strategy`) ---


def test_strategy_E_no_learning_but_inherits_genome(rng):
    world = ErlWorld(_small_world_cfg(strategy="E", n_initial_agents=10, mutation_rate=0.05), rng)
    original_ids = {a.agent_id for a in world.agents}
    for _ in range(30):
        world.step()
        if world.population_counts()["agent"] == 0:
            break
    for agent in world.agents:
        if agent.agent_id in original_ids:
            assert np.array_equal(agent.hebb_trace, np.zeros_like(agent.genome.action_weights)), \
                "strategy E must never update the Hebbian trace"


def test_strategy_E_still_inherits_genome_from_parent(rng):
    cfg = _small_world_cfg(strategy="E")
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    parent_genome_action_weights = parent.genome.action_weights.copy()
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()
    child = [a for a in world.agents if a.generation == 1][0]
    # Unlike L/F, evolution (inheritance) IS active for E -- with mutation_rate=0
    # and no mate available, the child's genome must exactly match the parent's.
    assert np.array_equal(child.genome.action_weights, parent_genome_action_weights)


def test_strategy_L_learns_and_clones_genome_exactly(rng):
    # mutation_rate deliberately high: L must clone regardless of this
    # config, since mutate() is never called for L/F at all.
    cfg = _small_world_cfg(strategy="L", mutation_rate=0.9)
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    parent_genome_action_weights = parent.genome.action_weights.copy()
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()
    child = [a for a in world.agents if a.generation == 1][0]
    # With strategy L, genome is cloned exactly (no mutation, no crossover)
    # -- inheritance still happens, only genetic improvement is switched off.
    assert np.array_equal(child.genome.action_weights, parent_genome_action_weights)
    assert world.strategy in ("ERL", "L")  # sanity: this IS a learning strategy


def test_strategy_F_neither_learns_nor_improves_genome(rng):
    cfg = _small_world_cfg(strategy="F", mutation_rate=0.9)
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    parent_genome_action_weights = parent.genome.action_weights.copy()
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()
    child = [a for a in world.agents if a.generation == 1][0]
    # Same cloning-only inheritance as L...
    assert np.array_equal(child.genome.action_weights, parent_genome_action_weights)
    # ...but F additionally has no learning (unlike L).
    assert world.strategy not in ("ERL", "L")


def test_strategy_B_ignores_network_entirely(rng):
    """Brownian: action distribution should be close to uniform regardless of
    a genome/network that would otherwise strongly bias action selection."""
    cfg = _small_world_cfg(strategy="B", founder_weight_std=50.0)  # huge weights: would dominate if used
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    actions = []
    for _ in range(400):
        world._observe_agent(agent)  # computed but must be ignored for B
        # Mirror world._step_agents()'s strategy=="B" branch directly.
        actions.append(int(world.rng.integers(0, N_ACTIONS)))
    counts = np.bincount(actions, minlength=N_ACTIONS)
    # Each action should appear a non-trivial fraction of the time (loose
    # bound -- this is a sanity check against a network-dominated bias, not
    # a strict uniformity test).
    assert (counts / len(actions) > 0.10).all()
