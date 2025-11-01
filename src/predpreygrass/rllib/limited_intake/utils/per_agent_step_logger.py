import csv
import os
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class PerAgentStepLogger(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.log_path = None
        self.header_written = False
        self._step_records_buffer = {}  # episode.id_ -> list of step records

    def on_episode_start(self, *args, **kwargs):
        episode = kwargs.get("episode") if "episode" in kwargs else args[3]
        if self.log_path is None:
            log_dir = os.path.join("logs", "rllib", "per_agent_step")
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, "per_agent_step_log.csv")
            self.header_written = os.path.exists(self.log_path)
        self._step_records_buffer[episode.id_] = []

    def on_episode_step(self, *args, **kwargs):
        # Defensive: handle missing positional arguments gracefully
        episode = kwargs.get("episode") if "episode" in kwargs else (args[2] if len(args) > 2 else None)
        base_env = kwargs.get("base_env") if "base_env" in kwargs else (args[1] if len(args) > 1 else None)
        if episode is None:
            return
        env = base_env.get_unwrapped()[0] if base_env is not None else None
        step = getattr(episode, "length", None)
        recs = self._step_records_buffer.setdefault(episode.id_, [])
        for agent_id in getattr(episode, "_agent_reward_history", {}):
            reward = episode._agent_reward_history[agent_id][-1] if episode._agent_reward_history[agent_id] else 0.0
            terminated = env.terminations.get(agent_id, False) if env is not None else False
            recs.append({
                "episode": getattr(episode, "episode_id", None),
                "step": step,
                "agent_id": agent_id,
                "reward": reward,
                "terminated": terminated,
            })

    def on_episode_end(self, *args, **kwargs):
        episode = kwargs.get("episode") if "episode" in kwargs else args[2]
        recs = self._step_records_buffer.pop(episode.id_, [])
        if not self.header_written:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "step", "agent_id", "reward", "terminated"])
                writer.writeheader()
            self.header_written = True
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "step", "agent_id", "reward", "terminated"])
            for rec in recs:
                writer.writerow(rec)
