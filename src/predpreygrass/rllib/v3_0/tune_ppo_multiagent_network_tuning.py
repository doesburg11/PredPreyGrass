"""
  Training script for network tuning
"""

from predpreygrass.rllib.v3_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_0.config.config_env_train_v1_0_network_tuning import config_env
from predpreygrass.rllib.v3_0.utils.episode_return_callback import EpisodeReturn

import ray
from ray import train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from ray.tune import Tuner

import os
from datetime import datetime
from pathlib import Path
import json


def custom_logger_creator(config):
    def logger_creator_func(config_):
        from ray.tune.logger import UnifiedLogger

        logdir = str(experiment_path)
        print(f"Redirecting RLlib logging to {logdir}")
        return UnifiedLogger(config_, logdir, loggers=None)

    return logger_creator_func


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v3_0.config.config_ppo_gpu_default import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v3_0.config.config_ppo_cpu_network_tuning import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config or config_env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type = parts[1]
    role = parts[2]
    return f"type_{type}_{role}"


def build_module_spec(obs_space, act_space, policy_name: str = None):
    # obs_space is a Box with shape (C, H, W)
    #   C = number of channels (layers of information: mask, predators, prey, grass → usually 4)
    #   H = height of the square observation window (e.g. 7 for predators, 9 for prey)
    #   W = width of the square observation window (equal to H here)
    C, H, W = obs_space.shape

    # We assert the window is square and odd-sized.
    # Odd size is important because the agent is always centered in its window.
    assert H == W and H % 2 == 1, "Expected odd square obs windows (e.g., 7x7, 9x9)."

    # Receptive field math:
    # Each 3x3 stride-1 conv layer expands the receptive field by +2.
    # General formula: RF = 1 + 2 * L
    # To cover the full observation window (size H), we solve:
    #   H = 1 + 2L  →  L = (H - 1) // 2
    # Example: H=7 → L=3 conv layers (RF=7), H=9 → L=4 conv layers (RF=9).
    L = (H - 1) // 2

    # Channel schedule:
    # We start small (16 filters), then increase (32, 64).
    # If more layers are needed (e.g., prey with 9x9), we keep adding 64-channel layers.
    base_channels = [16, 32, 64]
    if L <= len(base_channels):
        channels = base_channels[:L]
    else:
        channels = base_channels + [64] * (L - len(base_channels))

    # Assemble conv_filters list for RLlib (format: [num_filters, [kernel, kernel], stride])
    conv_filters = [[c, [3, 3], 1] for c in channels]

    # Adjust the fully-connected (FC) hidden sizes based on the action space.
    # If the agent has many actions (e.g., prey with 25 moves), we give it a wider first FC layer
    # so it has more capacity to rank actions effectively.
    num_actions = act_space.n if hasattr(act_space, "n") else None
    if num_actions is not None and num_actions > 20:
        fcnet_hiddens = [384, 256]
        head_note = "wide"
    else:
        fcnet_hiddens = [256, 256]
        head_note = "standard"

    # ---- Debug/trace log (once per policy) ----
    # Example: [MODEL] type_1_prey → obs CxHxW=4x9x9, L=4 (RF=9), conv=[16,32,64,64], actions=25, head=wide
    if policy_name is not None:
        rf = 1 + 2 * L
        conv_str = ",".join(str(c) for c in channels)
        print(
            f"[MODEL] {policy_name} → obs CxHxW={C}x{H}x{W}, "
            f"L={L} (RF={rf}), conv=[{conv_str}], "
            f"actions={num_actions}, head={head_note}"
        )


    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=obs_space,
        action_space=act_space,
        inference_only=False,
        model_config={
            "conv_filters": conv_filters,
            "fcnet_hiddens": fcnet_hiddens,
            "fcnet_activation": "relu",
        },
    )


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    sample_env = env_creator(config=config_env)
    module_specs = {}
    for agent_id in sample_env.observation_spaces:
        policy = policy_mapping_fn(agent_id)
        if policy not in module_specs:
            module_specs[policy] = build_module_spec(
                sample_env.observation_spaces[agent_id],
                sample_env.action_spaces[agent_id],
                policy_name=policy,
            )
    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    # Build config dictionary for Tune
    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=config_ppo["train_batch_size"],
            minibatch_size=config_ppo["minibatch_size"],
            num_epochs=config_ppo["num_epochs"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
            lambda_=config_ppo["lambda_"],
            entropy_coeff=config_ppo["entropy_coeff"],
            vf_loss_coeff=config_ppo["vf_loss_coeff"],
            clip_param=config_ppo["clip_param"],
            kl_coeff=config_ppo["kl_coeff"],
            kl_target=config_ppo["kl_target"],
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
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config.to_dict(),
        run_config=train.RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": max_iters},
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
        ),
    )

    result = tuner.fit()
    ray.shutdown()
