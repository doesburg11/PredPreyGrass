"""
Manual PPO training loop version of train_rllib_ppo_multiagentenv.py

- No Tune used → full control of training loop
- Safe checkpointing via map_location="cpu"
- Compatible with future curriculum learning / open-ended learning
"""
from predpreygrass.rllib.v2_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_0.config.config_env_train import config_env
from predpreygrass.utils.episode_return_callback import EpisodeReturn

# External libraries
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
import os
from datetime import datetime
from pathlib import Path
import json


# --- Helper functions ---


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v7_modular.config.config_ppo_gpu import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v7_modular.config.config_ppo_cpu import config_ppo
    elif num_cpus == 2:
        from predpreygrass.rllib.v7_modular.config.config_ppo_colab import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config or config_env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    speed = parts[1]
    role = parts[2]
    return f"speed_{speed}_{role}"


def build_module_spec(obs_space, act_space):
    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=obs_space,
        action_space=act_space,
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
    )


# --- Main training loop ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results_manual/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Save config metadata
    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    print(f"Saved config to: {experiment_path/'run_config.json'}")

    # Build MultiRLModuleSpec
    sample_env = env_creator(config=config_env)
    sample_agents = ["speed_1_predator_0", "speed_2_predator_0", "speed_1_prey_0", "speed_2_prey_0"]
    module_specs = {}
    for sample_agent in sample_agents:
        policy = policy_mapping_fn(sample_agent)
        module_specs[policy] = build_module_spec(
            sample_env.observation_spaces[sample_agent],
            sample_env.action_spaces[sample_agent],
        )
    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    # Build PPO algorithm
    ppo_algo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=config_ppo["train_batch_size"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
            num_learners=config_ppo["num_learners"],
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"],
        )
        .callbacks(EpisodeReturn)
        .build()
    )

    # Manual training loop
    max_iters = 1000
    checkpoint_every = 10

    for iter in range(max_iters):
        print(f"\n=== Training iteration {iter + 1}/{max_iters} ===")
        result = ppo_algo.train()

        print(
            f"Iteration {iter + 1}: "
            f"Env steps sampled={result['num_env_steps_sampled_lifetime']}, "
            f"Mean episode return={result['env_runners/episode_return_mean']:.2f}, "
            f"Mean episode length={result['env_runners/episode_len_mean']:.2f}"
        )

        # Save checkpoint manually every N iterations
        if (iter + 1) % checkpoint_every == 0 or (iter + 1) == max_iters:
            checkpoint_path = experiment_path / f"checkpoint_iter_{iter + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo_algo.save_to_path(str(checkpoint_path), map_location="cpu")
            print(f"Saved checkpoint to {checkpoint_path}")

    ray.shutdown()
