"""
Shared Malthusian mu-state server for distributed island training.

Paper: Leibo et al. (2019) Section 2.4 —
  "The island simulation and the species neural network updates were implemented
   as separate processes, potentially running on different machines. Islands
   produce trajectories and send them to a circular queue on the species update
   process."

With num_env_runners=NI (one RLlib env runner per island), each runner holds
one ArticleAllelopathyEnv / ArticleClamityEnv instance that handles a single
island.  The shared mu distribution is owned by this actor; every runner
reports its episode phi here, and the ecological update fires once all NI
reports for the current eco-step have arrived.
"""

from __future__ import annotations

import numpy as np

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAY_AVAILABLE = False


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


class _MuServerCore:
    """Pure-Python mu server — used directly in tests and as the Ray Actor body."""

    def __init__(
        self,
        num_species: int,
        num_islands: int,
        alpha: float,
        eta: float,
    ) -> None:
        self.num_species = num_species
        self.num_islands = num_islands
        self.alpha = alpha
        self.eta = eta
        self._logits: dict[int, np.ndarray] = {
            s: np.zeros(num_islands, dtype=np.float64)
            for s in range(num_species)
        }
        self._mu: dict[int, np.ndarray] = {
            s: np.ones(num_islands, dtype=np.float64) / num_islands
            for s in range(num_species)
        }
        self._phi_buffer: dict[tuple[int, int], float] = {}
        self._pending: set[int] = set(range(num_islands))
        self.eco_step: int = 0
        self._next_island: int = 0

    # ---------------------------------------------------------------------- API

    def register_worker(self) -> int | None:
        """Claim the next available island index for a new env runner.

        Returns None if all NI islands are already claimed — the caller
        (e.g. RLlib's local evaluator env) should fall back to local mode.
        """
        if self._next_island >= self.num_islands:
            return None
        idx = self._next_island
        self._next_island += 1
        return idx

    def get_mu(self) -> dict[int, list[float]]:
        """Current mu as {species_int: [prob_island_0, …, prob_island_NI-1]}."""
        return {s: self._mu[s].tolist() for s in range(self.num_species)}

    def get_eco_step(self) -> int:
        return self.eco_step

    def report_phi(
        self,
        island: int,
        phi_by_species: dict[int, float],
    ) -> dict[int, list[float]] | None:
        """Report end-of-episode phi for one island.

        Returns the updated mu dict once ALL NI islands have reported for this
        eco-step (triggering the ecological update), otherwise returns None.
        The caller may proceed immediately; it does not need to block.
        """
        for s in range(self.num_species):
            self._phi_buffer[(island, s)] = phi_by_species.get(s, 0.0)
        self._pending.discard(island)
        if self._pending:
            return None
        self._update_mu()
        self._pending = set(range(self.num_islands))
        self._phi_buffer = {}
        self.eco_step += 1
        return self.get_mu()

    # ----------------------------------------------------------------- internal

    def _update_mu(self) -> None:
        for s in range(self.num_species):
            phi_vec = np.array(
                [self._phi_buffer.get((i, s), 0.0) for i in range(self.num_islands)],
                dtype=np.float64,
            )
            mu = self._mu[s]
            centered = phi_vec - float(np.sum(mu * phi_vec))
            entropy_grad = -(np.log(np.maximum(mu, 1e-12)) + 1.0)
            self._logits[s] = (
                self._logits[s] + self.alpha * (centered + self.eta * entropy_grad)
            )
            self._mu[s] = _softmax(self._logits[s])


if _RAY_AVAILABLE:
    @ray.remote
    class MuServer:
        """Ray Actor: MuServer for distributed multi-runner island training."""

        def __init__(
            self,
            num_species: int,
            num_islands: int,
            alpha: float,
            eta: float,
        ) -> None:
            self._core = _MuServerCore(num_species, num_islands, alpha, eta)

        def register_worker(self) -> int | None:
            return self._core.register_worker()

        def get_mu(self) -> dict[int, list[float]]:
            return self._core.get_mu()

        def get_eco_step(self) -> int:
            return self._core.get_eco_step()

        def report_phi(
            self,
            island: int,
            phi_by_species: dict[int, float],
        ) -> dict[int, list[float]] | None:
            return self._core.report_phi(island, phi_by_species)

    def make_mu_server(
        num_species: int,
        num_islands: int,
        alpha: float,
        eta: float,
    ) -> "MuServer":
        """Create and return a MuServer Ray Actor."""
        return MuServer.remote(  # type: ignore[attr-defined]
            num_species, num_islands, alpha, eta
        )

else:  # pragma: no cover
    MuServer = _MuServerCore  # type: ignore[misc,assignment]

    def make_mu_server(
        num_species: int,
        num_islands: int,
        alpha: float,
        eta: float,
    ) -> _MuServerCore:
        return _MuServerCore(num_species, num_islands, alpha, eta)
