"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
This implements MultiRLModuleSpec explicitly to define the policies for predators
and prey separately.

Checkpoints and a run_config.json snapshot of the env/PPO config are saved under
~/simulation_results/ray_results/<experiment_name>/ for provenance -- see
--seed and --name below, and drive_conditioned_environment/tune_ppo_drive_conditioned_environment.py
for the counterpart run this script is meant to be compared against.
"""
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.base_environment.config_env import config_env
from predpreygrass.global_config import RAY_RESULTS_DIR

#  external libraries
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import AgentID, EpisodeType, PolicyID
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base RNG seed for this run (for multi-seed replication). Tags the "
             "experiment name so runs don't collide.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override the auto-generated experiment name (used as the Tune "
             "RunConfig name and the ray_results subdirectory).",
    )
    parser.add_argument(
        "--max-iters", type=int, default=1000,
        help="Training-iteration stop condition.",
    )
    return parser.parse_args()


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0

    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger: Optional[MetricsLogger] = None,
        env=None,
        env_index: int = 0,
        **kwargs,
    ):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey,
        plus the ecology metrics (episode length, births/deaths, extinction,
        population) needed to compare this baseline against
        drive_conditioned_environment -- see that module's README for why.
        """
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()

        # Initialize reward tracking
        predator_total_reward = 0.0
        prey_total_reward = 0.0
        predator_count = 0
        prey_count = 0

        # Retrieve rewards. env_steps=False is required here: with the default
        # env_steps=True, indices are shared env-step positions applied to every
        # agent's own reward buffer, which raises IndexError for any agent whose
        # lifetime (buffer length) is shorter than the episode itself -- the norm
        # here, since agents die/spawn continuously. env_steps=False indexes each
        # agent's rewards by its own per-agent timestep instead.
        rewards = episode.get_rewards(env_steps=False)  # Dictionary of {agent_id: list_of_rewards}

        for agent_id, reward_list in rewards.items():
            total_reward = sum(reward_list)  # Sum all rewards for the episode

            if "predator" in agent_id:
                predator_total_reward += total_reward
                predator_count += 1
            elif "prey" in agent_id:
                prey_total_reward += total_reward
                prey_count += 1

        # Compute average rewards (avoid division by zero)
        predator_avg_reward = predator_total_reward / predator_count if predator_count > 0 else 0
        prey_avg_reward = prey_total_reward / prey_count if prey_count > 0 else 0

        # Print episode logs
        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")
        print(f"  - Predators: Total Reward = {predator_total_reward:.2f}, Avg Reward = {predator_avg_reward:.2f}")
        print(f"  - Prey: Total Reward = {prey_total_reward:.2f}, Avg Reward = {prey_avg_reward:.2f}")

        resolved_env = self._resolve_env(env=env, env_index=env_index, **kwargs)
        build_metrics = getattr(resolved_env, "_build_episode_training_metrics", None)
        if metrics_logger is not None and callable(build_metrics):
            for metric_name, metric_value in build_metrics().items():
                # reduce="mean": the default (EMA, coeff=0.01) would smooth
                # these over ~100 episodes, badly lagging population/extinction
                # swings; a per-iteration mean is what a baseline-vs-drive-
                # conditioned comparison actually needs to read off cleanly.
                metrics_logger.log_value(f"ecology/{metric_name}", float(metric_value), reduce="mean")

    @staticmethod
    def _resolve_env(env=None, env_index: int = 0, **kwargs) -> Any:
        """Return the underlying PredPreyGrass env from RLlib's (possibly
        vectorized) wrapper shapes, so _build_episode_training_metrics can be
        called regardless of num_envs_per_env_runner."""

        def looks_like_metrics_env(obj) -> bool:
            return obj is not None and hasattr(obj, "_build_episode_training_metrics")

        def safe_index(value) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        def unwrap(candidate, index: int):
            current = candidate
            seen = set()
            for _ in range(10):
                if current is None:
                    return None
                if id(current) in seen:
                    return None
                seen.add(id(current))

                if looks_like_metrics_env(current):
                    return current

                if isinstance(current, (list, tuple)):
                    if not current:
                        return None
                    current = current[index] if 0 <= index < len(current) else current[0]
                    continue

                unwrapped = getattr(current, "unwrapped", None)
                if unwrapped is not None and unwrapped is not current:
                    current = unwrapped
                    continue

                if hasattr(current, "get_sub_environments"):
                    try:
                        sub_envs = current.get_sub_environments() or []
                    except Exception:
                        sub_envs = []
                    if sub_envs:
                        current = sub_envs[index] if 0 <= index < len(sub_envs) else sub_envs[0]
                        continue

                for attr in ("envs", "_envs"):
                    sub_envs = getattr(current, attr, None)
                    if isinstance(sub_envs, (list, tuple)) and sub_envs:
                        current = sub_envs[index] if 0 <= index < len(sub_envs) else sub_envs[0]
                        break
                else:
                    sub_envs = None
                if sub_envs is not None:
                    continue

                for attr in ("env", "_env", "vector_env", "_vector_env", "base_env"):
                    inner = getattr(current, attr, None)
                    if inner is not None and inner is not current:
                        current = inner
                        break
                else:
                    return None

            return None

        index = safe_index(env_index)
        for candidate in (
            env,
            kwargs.get("env_runner"),
            getattr(kwargs.get("env_runner"), "env", None),
            kwargs.get("worker"),
            getattr(kwargs.get("worker"), "env", None),
            kwargs.get("base_env"),
        ):
            resolved = unwrap(candidate, index)
            if resolved is not None:
                return resolved
        return None


def env_creator(config):
    return PredPreyGrass({**config_env, **(config or {})})


def policy_mapping_fn(agent_id: AgentID, episode: EpisodeType) -> PolicyID:
    agent_id_str = str(agent_id)
    if "predator" in agent_id_str:
        return "predator_policy"
    elif "prey" in agent_id_str:
        return "prey_policy"
    raise ValueError(f"No policy mapping defined for agent id: {agent_id!r}")


if __name__ == "__main__":
    args = parse_args()

    env_config = dict(config_env)

    register_env("PredPreyGrass", env_creator)
    ray.shutdown()
    ray.init(
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    ray_results_path = Path(RAY_RESULTS_DIR).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_tag = f"_SEED{args.seed}" if args.seed is not None else ""
    experiment_name = args.name or f"PPO_BASE_ENVIRONMENT{seed_tag}_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump({"config_env": env_config, "seed": args.seed}, f, indent=4)

    sample_env = env_creator(env_config)  # Create a single instance
    # Observation/action spaces for the sample policies
    if sample_env is None:
        raise RuntimeError("Failed to create sample environment")
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    if observation_spaces is None or action_spaces is None:
        raise RuntimeError("Sample environment did not initialize observation/action spaces")

    obs_space_pred = observation_spaces["predator_0"]
    act_space_pred = action_spaces["predator_0"]
    obs_space_prey = observation_spaces["prey_0"]
    act_space_prey = action_spaces["prey_0"]

    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred,
                action_space=act_space_pred,
                inference_only=False,
                model_config={
                    "conv_filters": [
                        [16, [3, 3], 1],
                        [32, [3, 3], 1],
                        [64, [3, 3], 1],
                    ],
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
                catalog_class=None,
            ),
            "prey_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_prey,
                action_space=act_space_prey,
                inference_only=False,
                model_config={
                    "conv_filters": [
                        [16, [3, 3], 1],
                        [32, [3, 3], 1],
                        [64, [3, 3], 1],
                    ],
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
                catalog_class=None,
            ),
        }
    )

    use_gpu = torch.cuda.is_available()
    num_gpus_per_learner = 1 if use_gpu else 0
    num_env_runners = 8 if use_gpu else 6
    num_envs_per_env_runner = 3 if use_gpu else 1
    num_cpus_per_env_runner = 3 if use_gpu else 1
    num_cpus_for_main_process = 4 if use_gpu else 1

    print(
        f"Starting new training experiment: {experiment_name}. "
        f"num_gpus_per_learner={num_gpus_per_learner}, "
        f"num_env_runners={num_env_runners}, "
        f"num_envs_per_env_runner={num_envs_per_env_runner}, "
        f"num_cpus_per_env_runner={num_cpus_per_env_runner}, "
        f"num_cpus_for_main_process={num_cpus_for_main_process}, "
        f"seed={args.seed}"
    )

    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=env_config)
        .framework("torch")
        .multi_agent(
            # This ensures that each policy is trained on the right observation/action space.
            policies={
                "predator_policy": (
                    None,
                    obs_space_pred,
                    act_space_pred,
                    {},
                ),
                "prey_policy": (None, obs_space_prey, act_space_prey, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .learners(
            num_gpus_per_learner=num_gpus_per_learner,
            num_learners=1,
        )
        .training(
            train_batch_size_per_learner=1024,
            minibatch_size=128,
            num_epochs=30,
            gamma=0.99,
            lr=0.0003,
            entropy_coeff=0.0,
            vf_loss_coeff=1.0,
            clip_param=0.3,
            kl_coeff=0.2,
            kl_target=0.01,
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .resources(num_cpus_for_main_process=num_cpus_for_main_process)
        .callbacks(EpisodeReturn)
    )
    if args.seed is not None:
        ppo = ppo.debugging(seed=args.seed)

    del sample_env  # to avoid any stray references

    tuner = Tuner(
        ppo.algo_class,
        param_space=ppo.to_dict(),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": args.max_iters},
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=10,
                checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end
            ),
        ),
    )
    # Run the Tuner and capture the results.
    results = tuner.fit()
    ray.shutdown()
