"""Tests for MuServer and distributed single-island env mode."""

from __future__ import annotations

import numpy as np
from predpreygrass.malthusian_rl.utils.mu_server import _MuServerCore


# ---------------------------------------------------------------------------
# _MuServerCore unit tests (pure-Python, no Ray required)
# ---------------------------------------------------------------------------


def test_mu_server_initial_mu_is_uniform():
    srv = _MuServerCore(num_species=2, num_islands=4, alpha=0.0, eta=0.0)
    mu = srv.get_mu()
    for s in range(2):
        assert np.allclose(mu[s], [0.25, 0.25, 0.25, 0.25])


def test_mu_server_register_worker_assigns_sequential_indices():
    srv = _MuServerCore(num_species=1, num_islands=3, alpha=0.0, eta=0.0)
    assert srv.register_worker() == 0
    assert srv.register_worker() == 1
    assert srv.register_worker() == 2
    assert srv.register_worker() is None  # all slots claimed


def test_mu_server_report_phi_returns_none_until_all_islands_report():
    srv = _MuServerCore(num_species=1, num_islands=3, alpha=0.0, eta=0.0)
    result0 = srv.report_phi(0, {0: 1.0})
    assert result0 is None
    result1 = srv.report_phi(1, {0: 2.0})
    assert result1 is None
    result2 = srv.report_phi(2, {0: 3.0})
    assert result2 is not None  # eco-step complete


def test_mu_server_eco_step_increments_on_completion():
    srv = _MuServerCore(num_species=1, num_islands=2, alpha=0.0, eta=0.0)
    assert srv.get_eco_step() == 0
    srv.report_phi(0, {0: 1.0})
    assert srv.get_eco_step() == 0
    srv.report_phi(1, {0: 1.0})
    assert srv.get_eco_step() == 1


def test_mu_server_duplicate_island_report_is_idempotent():
    srv = _MuServerCore(num_species=1, num_islands=2, alpha=0.0, eta=0.0)
    srv.report_phi(0, {0: 10.0})
    srv.report_phi(0, {0: 5.0})  # overwrite island 0
    result = srv.report_phi(1, {0: 1.0})
    assert result is not None
    # Island 0's phi is the overwritten 5.0, not 10.0


def test_mu_server_phi_buffer_cleared_after_eco_step():
    srv = _MuServerCore(num_species=1, num_islands=2, alpha=0.0, eta=0.0)
    srv.report_phi(0, {0: 5.0})
    srv.report_phi(1, {0: 1.0})
    # Second eco-step: islands can report again from scratch
    assert srv.report_phi(0, {0: 3.0}) is None
    result = srv.report_phi(1, {0: 3.0})
    assert result is not None
    assert srv.get_eco_step() == 2


def test_mu_server_update_concentrates_mu_on_high_phi_island():
    srv = _MuServerCore(num_species=1, num_islands=2, alpha=1.0, eta=0.0)
    # Island 0 gets much higher phi than island 1
    srv.report_phi(0, {0: 10.0})
    new_mu = srv.report_phi(1, {0: 0.0})
    assert new_mu is not None
    # After update, island 0 should have higher mu
    assert new_mu[0][0] > new_mu[0][1]


def test_mu_server_entropy_coeff_prevents_collapse_to_zero():
    srv = _MuServerCore(num_species=1, num_islands=3, alpha=1.0, eta=100.0)
    # Moderate phi advantage for island 0; high eta keeps entropy high
    srv.report_phi(0, {0: 3.0})
    srv.report_phi(1, {0: 0.0})
    new_mu = srv.report_phi(2, {0: 0.0})
    assert new_mu is not None
    # With eta=100 the entropy term dominates; islands 1 and 2 stay > 0
    for i in range(3):
        assert new_mu[0][i] > 0.0


def test_mu_server_mu_sums_to_one_after_update():
    srv = _MuServerCore(num_species=2, num_islands=5, alpha=0.01, eta=0.1)
    for i in range(5):
        srv.report_phi(i, {0: float(i), 1: float(4 - i)})
    srv.report_phi(4, {0: 4.0, 1: 0.0})
    # report_phi(4, ...) for i=4 is the 5th call — eco-step fires
    # (The loop already reported island 4 once above; this is a fresh eco-step)
    # Let's redo cleanly:
    srv2 = _MuServerCore(num_species=2, num_islands=5, alpha=0.01, eta=0.1)
    result = None
    for i in range(5):
        result = srv2.report_phi(i, {0: float(i), 1: float(4 - i)})
    assert result is not None
    for s in range(2):
        assert abs(sum(result[s]) - 1.0) < 1e-9, f"mu for species {s} does not sum to 1"


# ---------------------------------------------------------------------------
# Distributed single-island env tests (using _MuServerCore directly, no Ray)
# ---------------------------------------------------------------------------


def test_distributed_env_claims_island_slot_from_server():
    from predpreygrass.malthusian_rl.article_tasks import ArticleAllelopathyEnv

    srv = _MuServerCore(num_species=1, num_islands=3, alpha=0.0, eta=0.0)
    srv.register_worker = srv.register_worker  # ensure it's callable

    # Monkey-patch a fake mu_server with .register_worker.remote() interface
    class _FakeRemote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    class _FakeMuServer:
        def __init__(self, core):
            self._core = core
            self.register_worker = _FakeRemote(core.register_worker)
            self.get_mu = _FakeRemote(core.get_mu)
            self.report_phi = _FakeRemote(core.report_phi)

    import ray as _ray

    # Patch ray.get to be identity (since _FakeRemote.remote() already returns the value)
    _original_ray_get = _ray.get

    def _fake_ray_get(ref, *args, **kwargs):
        return ref  # remote() already returns the value directly

    _ray.get = _fake_ray_get
    try:
        fake_srv = _FakeMuServer(srv)
        env0 = ArticleAllelopathyEnv({
            "num_species": 1,
            "total_individuals": 30,
            "num_islands": 3,
            "episode_horizon": 2,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
            "mu_server": fake_srv,
        })
        assert env0._global_island_index == 0
        assert env0.num_islands == 1

        env1 = ArticleAllelopathyEnv({
            "num_species": 1,
            "total_individuals": 30,
            "num_islands": 3,
            "episode_horizon": 2,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
            "mu_server": fake_srv,
        })
        assert env1._global_island_index == 1

        # Third env gets island 2
        env2 = ArticleAllelopathyEnv({
            "num_species": 1,
            "total_individuals": 30,
            "num_islands": 3,
            "episode_horizon": 2,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
            "mu_server": fake_srv,
        })
        assert env2._global_island_index == 2

        # Fourth env: no slot left → falls back to local all-island mode
        env3 = ArticleAllelopathyEnv({
            "num_species": 1,
            "total_individuals": 30,
            "num_islands": 3,
            "episode_horizon": 2,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
            "mu_server": fake_srv,
        })
        assert env3._global_island_index is None
        assert env3.num_islands == 3  # local all-island mode
    finally:
        _ray.get = _original_ray_get


def test_distributed_env_reports_phi_to_server_at_episode_end():
    """After one episode, the env must call report_phi on the server."""
    from predpreygrass.malthusian_rl.article_tasks import ArticleAllelopathyEnv

    srv = _MuServerCore(num_species=1, num_islands=1, alpha=0.0, eta=0.0)

    reported = {}

    class _FakeRemote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *a, **kw):
            return self._fn(*a, **kw)

    class _TrackingServer:
        def __init__(self, core):
            self._core = core
            self.register_worker = _FakeRemote(core.register_worker)
            self.get_mu = _FakeRemote(core.get_mu)

            def _tracking_report_phi(island, phi_by_species):
                reported["island"] = island
                reported["phi"] = dict(phi_by_species)
                return core.report_phi(island, phi_by_species)

            self.report_phi = _FakeRemote(_tracking_report_phi)

    import ray as _ray
    _orig = _ray.get

    def _passthrough(ref, *a, **kw):
        return ref

    _ray.get = _passthrough
    try:
        fake_srv = _TrackingServer(srv)
        env = ArticleAllelopathyEnv({
            "num_species": 1,
            "total_individuals": 2,
            "num_islands": 1,
            "episode_horizon": 2,
            "height": 3,
            "width": 3,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
            "mu_server": fake_srv,
        })
        assert env._is_distributed_single_island

        obs, _ = env.reset(seed=0)
        dones = {}
        while not all(dones.get(a, False) for a in env.agents) and env.current_step < 2:
            actions = {a: 0 for a in env.agents}
            obs, rew, terms, truncs, info = env.step(actions)
            dones = {**terms, **truncs}

        # The episode ended; report_phi should have been called
        assert "island" in reported
        assert reported["island"] == 0  # global island index
    finally:
        _ray.get = _orig
