"""
This script trains the predator_sexual_reproduction environment with PPO using
Ray RLlib's new API stack. Predators are split into two sexed policies:
predator_male (hunts prey AND gathers fruit) and predator_female (gathers
fruit only). Prey is unchanged from base_environment_step_energy: it only
eats grass and reproduces asexually. Reproduction is sexual for predators: a
predator_male and predator_female must each independently clear
predator_creation_energy_threshold AND be within mate_search_radius of each
other -- see predpreygrass_rllib_env.py and config_env.py for the mechanism.

Checkpoints and a run_config.json snapshot of the env/PPO config are saved
under ~/simulation_results/ray_results/<experiment_name>/ for provenance.
"""
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env
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
    parser.add_argument(
        "--gpu-fraction", type=float, default=None,
        help="Override num_gpus_per_learner (e.g. 0.25 to let 4 runs share one GPU "
             "concurrently). Default: 1 if a GPU is available, else 0.",
    )
    parser.add_argument(
        "--num-env-runners", type=int, default=None,
        help="Override num_env_runners. Default: 8 if a GPU is available, else 6.",
    )
    parser.add_argument(
        "--num-cpus-per-env-runner", type=int, default=None,
        help="Override num_cpus_per_env_runner. Default: 3 if a GPU is available, else 1.",
    )
    parser.add_argument(
        "--num-cpus-main", type=int, default=None,
        help="Override num_cpus_for_main_process. Default: 4 if a GPU is available, else 1.",
    )
    parser.add_argument(
        "--mate-search-radius", type=int, default=None,
        help="Override mate_search_radius (Chebyshev distance for predator "
             "mate-finding). Default: config_env.py's shipped value.",
    )
    parser.add_argument(
        "--male-success-prob", type=float, default=None,
        help="Override prey_vs_predator_male_success_prob. Useful for a positive-"
             "control run (push odds to an extreme to confirm the risk-driven "
             "specialization mechanism CAN produce a visible signal before "
             "trusting a null result at the real settings).",
    )
    parser.add_argument(
        "--male-death-prob", type=float, default=None,
        help="Override prey_vs_predator_male_death_prob.",
    )
    parser.add_argument(
        "--female-success-prob", type=float, default=None,
        help="Override prey_vs_predator_female_success_prob.",
    )
    parser.add_argument(
        "--female-death-prob", type=float, default=None,
        help="Override prey_vs_predator_female_death_prob.",
    )
    parser.add_argument(
        "--reward-catch-prey", type=float, default=None,
        help="Override reward_predator_catch_prey (foraging-shaping capability "
             "check: can predators learn to steer at all given a dense signal?).",
    )
    parser.add_argument(
        "--reward-gather-fruit", type=float, default=None,
        help="Override reward_predator_gather_fruit.",
    )
    parser.add_argument(
        "--penalty-combat-death", type=float, default=None,
        help="Magnitude (>= 0) of the penalty a predator receives when it dies in a "
             "failed hunt. Stored as penalty_predator_death_in_combat = -value, since "
             "the env uses that key directly as the dying predator's reward. Default: "
             "config_env.py's shipped value (0.0, i.e. no penalty).",
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=128,
        help="PPO minibatch_size (default 128, the value all earlier runs used).",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=30,
        help="PPO num_epochs (default 30, the value all earlier runs used).",
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
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()

        predator_male_total_reward = 0.0
        predator_female_total_reward = 0.0
        prey_total_reward = 0.0
        predator_male_count = 0
        predator_female_count = 0
        prey_count = 0

        rewards = episode.get_rewards(env_steps=False)

        for agent_id, reward_list in rewards.items():
            total_reward = sum(reward_list)
            if "predator_male" in agent_id:
                predator_male_total_reward += total_reward
                predator_male_count += 1
            elif "predator_female" in agent_id:
                predator_female_total_reward += total_reward
                predator_female_count += 1
            elif "prey" in agent_id:
                prey_total_reward += total_reward
                prey_count += 1

        predator_male_avg = predator_male_total_reward / predator_male_count if predator_male_count > 0 else 0
        predator_female_avg = predator_female_total_reward / predator_female_count if predator_female_count > 0 else 0
        prey_avg = prey_total_reward / prey_count if prey_count > 0 else 0

        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")
        print(f"  - Predator males:   Total={predator_male_total_reward:.2f}, Avg={predator_male_avg:.2f}")
        print(f"  - Predator females: Total={predator_female_total_reward:.2f}, Avg={predator_female_avg:.2f}")
        print(f"  - Prey:             Total={prey_total_reward:.2f}, Avg={prey_avg:.2f}")

        resolved_env = self._resolve_env(env=env, env_index=env_index, **kwargs)
        build_metrics = getattr(resolved_env, "_build_episode_training_metrics", None)
        if metrics_logger is not None and callable(build_metrics):
            for metric_name, metric_value in build_metrics().items():
                metrics_logger.log_value(f"ecology/{metric_name}", float(metric_value), reduce="mean")

    @staticmethod
    def _resolve_env(env=None, env_index: int = 0, **kwargs) -> Any:
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
    if "predator_male" in agent_id_str:
        return "predator_male_policy"
    if "predator_female" in agent_id_str:
        return "predator_female_policy"
    elif "prey" in agent_id_str:
        return "prey_policy"
    raise ValueError(f"No policy mapping defined for agent id: {agent_id!r}")


if __name__ == "__main__":
    args = parse_args()

    env_config = dict(config_env)
    if args.mate_search_radius is not None:
        env_config["mate_search_radius"] = args.mate_search_radius
    if args.male_success_prob is not None:
        env_config["prey_vs_predator_male_success_prob"] = args.male_success_prob
    if args.male_death_prob is not None:
        env_config["prey_vs_predator_male_death_prob"] = args.male_death_prob
    if args.female_success_prob is not None:
        env_config["prey_vs_predator_female_success_prob"] = args.female_success_prob
    if args.female_death_prob is not None:
        env_config["prey_vs_predator_female_death_prob"] = args.female_death_prob

    if args.reward_catch_prey is not None:
        env_config["reward_predator_catch_prey"] = args.reward_catch_prey
    if args.reward_gather_fruit is not None:
        env_config["reward_predator_gather_fruit"] = args.reward_gather_fruit
    if args.penalty_combat_death is not None:
        if args.penalty_combat_death < 0:
            raise SystemExit("--penalty-combat-death must be >= 0 (it is a magnitude, stored as a negative reward)")
        env_config["penalty_predator_death_in_combat"] = -args.penalty_combat_death

    register_env("PredPreyGrass", env_creator)
    ray.shutdown()
    ray.init(
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    ray_results_path = Path(RAY_RESULTS_DIR).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_tag = f"_SEED{args.seed}" if args.seed is not None else ""
    experiment_name = args.name or f"PPO_PREDATOR_SEXUAL_REPRODUCTION{seed_tag}_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump({"config_env": env_config, "seed": args.seed}, f, indent=4)

    sample_env = env_creator(env_config)
    if sample_env is None:
        raise RuntimeError("Failed to create sample environment")
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    if observation_spaces is None or action_spaces is None:
        raise RuntimeError("Sample environment did not initialize observation/action spaces")

    obs_space_pred_male = observation_spaces["predator_male_0"]
    act_space_pred_male = action_spaces["predator_male_0"]
    obs_space_pred_female = observation_spaces["predator_female_0"]
    act_space_pred_female = action_spaces["predator_female_0"]
    obs_space_prey = observation_spaces["prey_0"]
    act_space_prey = action_spaces["prey_0"]

    conv_filters = [
        [16, [3, 3], 1],
        [32, [3, 3], 1],
        [64, [3, 3], 1],
    ]
    model_config = {
        "conv_filters": conv_filters,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }

    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_male_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred_male,
                action_space=act_space_pred_male,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
            "predator_female_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred_female,
                action_space=act_space_pred_female,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
            "prey_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_prey,
                action_space=act_space_prey,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
        }
    )

    use_gpu = torch.cuda.is_available()
    num_gpus_per_learner = args.gpu_fraction if args.gpu_fraction is not None else (1 if use_gpu else 0)
    num_env_runners = args.num_env_runners if args.num_env_runners is not None else (8 if use_gpu else 6)
    num_envs_per_env_runner = 3 if use_gpu else 1
    num_cpus_per_env_runner = args.num_cpus_per_env_runner if args.num_cpus_per_env_runner is not None else (3 if use_gpu else 1)
    num_cpus_for_main_process = args.num_cpus_main if args.num_cpus_main is not None else (4 if use_gpu else 1)

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
            policies={
                "predator_male_policy": (None, obs_space_pred_male, act_space_pred_male, {}),
                "predator_female_policy": (None, obs_space_pred_female, act_space_pred_female, {}),
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
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
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

    del sample_env

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
                checkpoint_at_end=True,
            ),
        ),
    )
    results = tuner.fit()
    ray.shutdown()
