
from stable_baselines3.common.callbacks import BaseCallback

class SampleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_length = 0
        self.episode_lengths = []

    def _on_step(self) -> bool:
        self.current_episode_length += 1
        # If the episode is done, log the episode length and reset the counter
        if "done" in self.locals and self.locals["done"]:
            # print("done")
            self.episode_lengths.append(self.current_episode_length)
            self.logger.record("train/episode_length", self.current_episode_length)
            self.current_episode_length = 0
        return True  # Continue training

