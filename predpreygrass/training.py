"""RLlib algorithms with disposable working directories outside Tune-managed runs."""
from tempfile import TemporaryDirectory

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.impala import Impala


class _TemporaryLogdirMixin:
    """Keep Tune storage intact; replace standalone fallback with scratch space."""

    def _setup_logdir(self):
        if self._storage:
            return super()._setup_logdir()
        # Ray's default creates an otherwise unused timestamped directory in
        # ~/ray_results. Standalone construction needs only a scratch directory.
        self._scratch_logdir = TemporaryDirectory(prefix="predpreygrass-ppo-")
        self._logdir = self._scratch_logdir.name

    def stop(self):
        try:
            return super().stop()
        finally:
            scratch = getattr(self, "_scratch_logdir", None)
            if scratch is not None:
                scratch.cleanup()


class LocalPPO(_TemporaryLogdirMixin, PPO):
    """PPO without persistent standalone scratch directories."""


class LocalAPPO(_TemporaryLogdirMixin, APPO):
    """APPO without persistent standalone scratch directories."""


class LocalImpala(_TemporaryLogdirMixin, Impala):
    """IMPALA without persistent standalone scratch directories."""
