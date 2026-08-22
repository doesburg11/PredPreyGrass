"""
Trains the type_1-only Red Queen environment (config_env_eval.py) with PPO using
Ray RLlib's new API stack, producing checkpoints suitable for both
evaluate_red_queen_freeze_type_1_only.py and evaluate_red_queen_freeze_multi_seed.py.

config_env_eval.py is used deliberately, not config_env_train.py: it disables
type_2 predators/prey entirely (population 0), so exactly two policies ever
get created -- "type_1_predator" and "type_1_prey" -- matching what both
freeze-test evaluation scripts load. Training with config_env_train.py's
type_2-enabled setting instead would produce a checkpoint with extra
type_2_* policies neither evaluation script expects.

Checkpoints use RLlib Tune's standard naming (checkpoint_000000,
checkpoint_000010, ...). When pointing evaluate_red_queen_freeze_multi_seed.py
at a run produced by this script, pass:
    --checkpoint-dir-template "checkpoint_{iter:06d}"
"""
import json
import os
from datetime import datetime
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.registry import register_env

from predpreygrass.non_evolutionary.red_queen.config.config_env_eval import config_env
from predpreygrass.non_evolutionary.red_queen.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.red_queen.utils.episode_return_callback import EpisodeReturn
from predpreygrass.non_evolutionary.red_queen.utils.networks import build_multi_module_spec
from predpreygrass.global_config import RAY_RESULTS_DIR


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.non_evolutionary.red_queen.config.config_ppo_gpu_default import config_ppo
    elif num_cpus == 8:
        from predpreygrass.non_evolutionary.red_queen.config.config_ppo_cpu import config_ppo
    else:
        # Default to CPU config for other CPU counts to keep training usable across machines.
        from predpreygrass.non_evolutionary.red_queen.config.config_ppo_cpu import config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config or config_env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    """'type_1_predator_3' -> 'type_1_predator' (one policy per species/type combo)."""
    parts = agent_id.split("_")
    return f"type_{parts[1]}_{parts[2]}"


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_path = Path(RAY_RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_RED_QUEEN_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump({"config_env": config_env, "config_ppo": config_ppo}, f, indent=4)

    sample_env = env_creator(config=config_env)

    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]
    del sample_env

    print(f"Policies for this run: {sorted(obs_by_policy.keys())}")
    assert set(obs_by_policy.keys()) == {"type_1_predator", "type_1_prey"}, (
        "config_env_eval.py should disable type_2 entirely -- if you changed the "
        "env config, the resulting checkpoint won't match what the freeze-test "
        "evaluation scripts expect (they only load type_1_predator/type_1_prey)."
    )

    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)
    policies = {pid: (None, obs_by_policy[pid], act_by_policy[pid], {}) for pid in obs_by_policy}

    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
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
        .resources(num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"])
        .callbacks(EpisodeReturn)
    )

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": config_ppo["max_iters"]},
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )

    result = tuner.fit()
    ray.shutdown()
