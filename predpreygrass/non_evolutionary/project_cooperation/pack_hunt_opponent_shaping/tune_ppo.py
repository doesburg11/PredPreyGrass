"""Naive-learner baseline for pack_hunt_opponent_shaping: independent PPO.

This is condition 1 of the module's README (Section 8): each predator has its
own separate policy (not weight-shared), trained by plain RLlib PPO with no
opponent-awareness at all -- exactly Foerster et al. (2018)'s "Naive Learner"
role (treat the other agents as a fixed part of the environment), just PPO
instead of REINFORCE. See the README for why that substitution is fine for
this baseline but doesn't carry over to condition 2 (pairwise opponent
shaping): PPO's clipped, multi-epoch surrogate objective is exactly the kind
of "build a surrogate loss and differentiate it twice" shortcut that silently
drops the terms an opponent-shaping correction needs, so condition 2 will need
its own training loop, not a modification of this one.

Expected baseline outcome, per the README: low engagement/catch rate, since
nothing here rewards a predator for anticipating whether its podmates will
reciprocate.
"""

import os
from datetime import datetime
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.registry import register_env

from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.config.config_env_pack_hunt_opponent_shaping import (
    config_env,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.config.config_ppo_pack_hunt_opponent_shaping import (
    config_ppo,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.predpreygrass_rllib_env import (
    PackHuntEnv,
)


def env_creator(config):
    return PackHuntEnv(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    # One independent policy per predator -- no weight sharing, so each
    # predator can specialize as an engager or a scrounger, which is the
    # whole point of the dilemma this environment is testing.
    return agent_id


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PackHunt", env_creator)
    env_config = {**config_env, "seed": None}

    ray_results_dir = os.getenv(
        "PACK_HUNT_RAY_RESULTS",
        str(Path(__file__).resolve().parent / "ray_results"),
    )
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PACK_HUNT_NAIVE_PPO_{timestamp}"

    sample_env = env_creator(config=env_config)
    sample_env.reset(seed=None)
    policies = {
        agent_id: (None, sample_env.observation_spaces[agent_id], sample_env.action_spaces[agent_id], {})
        for agent_id in sample_env.agents
    }
    del sample_env

    ppo_config = (
        PPOConfig()
        .environment(env="PackHunt", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
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
    )

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": config_ppo["max_iters"]},
            checkpoint_config=CheckpointConfig(
                num_to_keep=10,
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )

    result = tuner.fit()
    ray.shutdown()
