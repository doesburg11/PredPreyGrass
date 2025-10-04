from ray.rllib.callbacks.callbacks import RLlibCallback
from predpreygrass.rllib.kin_selection.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.kin_selection.utils.helping_metrics_callback import HelpingMetricsCallback


class CombinedCallbacks(RLlibCallback):
    """Compose EpisodeReturn and HelpingMetricsCallback into a single RLlibCallback."""

    def __init__(self):
        super().__init__()
        self._a = EpisodeReturn()
        self._b = HelpingMetricsCallback()

    # Delegate lifecycle methods to both callbacks when present
    def on_episode_start(self, **kwargs):
        for cb in (self._a, self._b):
            if hasattr(cb, "on_episode_start"):
                try:
                    cb.on_episode_start(**kwargs)
                except Exception as e:
                    # Avoid crashing workers if a sub-callback fails
                    print(f"[CombinedCallbacks] on_episode_start error in {type(cb).__name__}: {e}")

    def on_episode_step(self, **kwargs):
        for cb in (self._a, self._b):
            if hasattr(cb, "on_episode_step"):
                try:
                    cb.on_episode_step(**kwargs)
                except Exception as e:
                    print(f"[CombinedCallbacks] on_episode_step error in {type(cb).__name__}: {e}")

    def on_episode_end(self, **kwargs):
        for cb in (self._a, self._b):
            if hasattr(cb, "on_episode_end"):
                try:
                    cb.on_episode_end(**kwargs)
                except Exception as e:
                    print(f"[CombinedCallbacks] on_episode_end error in {type(cb).__name__}: {e}")

    def on_train_result(self, **kwargs):
        for cb in (self._a, self._b):
            if hasattr(cb, "on_train_result"):
                try:
                    cb.on_train_result(**kwargs)
                except Exception as e:
                    print(f"[CombinedCallbacks] on_train_result error in {type(cb).__name__}: {e}")
