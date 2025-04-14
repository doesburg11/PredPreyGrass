import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from datetime import datetime
import os
import json

# ---- Import your environment setup ----
from predpreygrass.rllib.v4_gpu.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v4_gpu.config_env import config_env

# ---- Fast test config ----
train_batch_size = 100
sgd_minibatch_size = 50
num_env_runners = 1
num_envs_per_env_runner = 1
num_cpus_per_env_runner = 1
rollout_fragment_length = 100
num_gpus_per_learner = 0  # No GPU for fast test

# ---- Metadata writer ----
def save_run_metadata_to_path(config_env: dict, ppo_config: dict, model_arch: dict, path: str):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_env": config_env,
        "ppo_config": ppo_config,
        "model_architecture": model_arch,
        "note": "Auto-saved metadata from RLlib callback",
    }

    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, "run_metadata.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[✔] Metadata saved to: {filepath}")

# ---- Callback that saves metadata ----
class MetadataTestCallback(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        log_dir = getattr(algorithm, "log_dir", None)
        if log_dir:
            save_run_metadata_to_path(
                config_env=config_env,
                ppo_config={
                    "train_batch_size": train_batch_size,
                    "sgd_minibatch_size": sgd_minibatch_size,
                    "gamma": 0.99,
                    "lr": 0.0003,
                    "rollout_fragment_length": rollout_fragment_length,
                    "num_env_runners": num_env_runners,
                    "num_envs_per_env_runner": num_envs_per_env_runner,
                    "num_cpus_per_env_runner": num_cpus_per_env_runner,
                    "num_gpus_per_learner": num_gpus_per_learner,
                },
                model_arch={
                    "conv_filters": [
                        [16, [3, 3], 1],
                        [32, [3, 3], 1],
                        [64, [3, 3], 1],
                    ],
                    "fcnet_hiddens": [256, 256],
                },
                path=log_dir,
            )

# ---- Register env ----
def env_creator(config):
    return PredPreyGrass(config or config_env)

# ---- Run test ----
if __name__ == "__main__":
    ray.init()
    register_env("PredPreyGrass", env_creator)

    config = (
        PPOConfig()
        .environment("PredPreyGrass", env_config=config_env)
        .framework("torch")
        .callbacks(MetadataTestCallback)
        .training(
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            lr=0.0003,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            rollout_fragment_length=rollout_fragment_length,
            num_cpus_per_env_runner=num_cpus_per_env_runner
        )
        # .learners(num_gpus_per_learner=num_gpus_per_learner)  # not needed if 0 GPU
    )

    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=train.RunConfig(
            stop={"training_iteration": 1},  # ✅ single iteration test
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            ),
        )
    )

    tuner.fit()
    ray.shutdown()
