"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
This implements MultiRLModuleSpec explicitly to define the policies for predators
and prey separately.
"""
from predpreygrass.non_evolutionary.base_environment_seasonal.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.base_environment_seasonal.config_env import config_env

#  external libraries
import argparse
from datetime import datetime
from typing import Optional

import ray
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
        "--season-high", type=float, default=None,
        help="Override config_env['season_high_multiplier'] for this run. Tags the experiment name.",
    )
    parser.add_argument(
        "--season-low", type=float, default=None,
        help="Override config_env['season_low_multiplier'] for this run. Tags the experiment name.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=None,
        help="Override the training_iteration stop condition for this run.",
    )
    return parser.parse_args()


def _resolve_env(env=None, env_index: int = 0, **kwargs) -> Optional[PredPreyGrass]:
    """Unwrap RLlib's vector/wrapper env shapes down to the raw PredPreyGrass instance."""

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

            if isinstance(current, PredPreyGrass):
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
    ):
        resolved = unwrap(candidate, index)
        if resolved is not None:
            return resolved
    return None


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
        plus births and end-of-episode population/grass metrics for the
        seasonal-multiplier sweep (see run_season_multiplier_sweep.sh).
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

        # Births and end-of-episode population/grass metrics, read directly off
        # the env's own running counters (_next_predator_idx/_next_prey_idx count
        # up from n_initial_active_* on every reproduction event, so the delta is
        # exactly this episode's birth count). Defensive: this is diagnostic-only
        # logging and must never crash a sampling worker.
        if metrics_logger is not None:
            try:
                resolved_env = _resolve_env(env=env, env_index=env_index, **kwargs)
                if resolved_env is not None:
                    predator_births = resolved_env._next_predator_idx - resolved_env.n_initial_active_predator
                    prey_births = resolved_env._next_prey_idx - resolved_env.n_initial_active_prey
                    metrics_logger.log_value("predator_births", float(predator_births))
                    metrics_logger.log_value("prey_births", float(prey_births))
                    metrics_logger.log_value("predator_count_end", float(resolved_env.current_num_predators))
                    metrics_logger.log_value("prey_count_end", float(resolved_env.current_num_prey))
                    metrics_logger.log_value("grass_count_end", float(resolved_env.current_num_grass))
                    if resolved_env.grass_energies:
                        grass_energy_mean = sum(resolved_env.grass_energies.values()) / len(resolved_env.grass_energies)
                        metrics_logger.log_value("grass_energy_mean_end", float(grass_energy_mean))
            except Exception as e:
                print(f"Episode {self.num_episodes}: [births/population metrics failed, skipping: {type(e).__name__}: {e}]")


def env_creator(config):
    return PredPreyGrass(config or config_env)


def policy_mapping_fn(agent_id: AgentID, episode: EpisodeType) -> PolicyID:
    agent_id_str = str(agent_id)
    if "predator" in agent_id_str:
        return "predator_policy"
    elif "prey" in agent_id_str:
        return "prey_policy"
    raise ValueError(f"No policy mapping defined for agent id: {agent_id!r}")


if __name__ == "__main__":
    args = parse_args()
    if args.season_high is not None:
        config_env = dict(config_env, season_high_multiplier=args.season_high)
    if args.season_low is not None:
        config_env = dict(config_env, season_low_multiplier=args.season_low)

    register_env("PredPreyGrass", env_creator)
    ray.shutdown()
    ray.init(
        log_to_driver=True,
        ignore_reinit_error=True,
    )
    sample_env = env_creator({})  # Create a single instance
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

    print("Starting new training experiment.")

    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
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
            num_gpus_per_learner=1,
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
            num_env_runners=10,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .resources(num_cpus_for_main_process=1)
        .callbacks(EpisodeReturn)
    )

    max_iters = args.max_iters if args.max_iters is not None else 1000
    high_tag = f"_HIGH{config_env['season_high_multiplier']}" if args.season_high is not None else ""
    low_tag = f"_LOW{config_env['season_low_multiplier']}" if args.season_low is not None else ""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"BASE_ENV_SEASONAL{high_tag}{low_tag}_{timestamp}"

    tuner = Tuner(
        ppo.algo_class,
        param_space=ppo.to_dict(),
        run_config=RunConfig(
            name=experiment_name,
            stop={"training_iteration": max_iters},
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
