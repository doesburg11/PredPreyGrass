import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env

from predpreygrass.rllib.v2_7.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_7.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.v2_7.config.config_env_train_v1_0 import config_env

import os


def env_creator(env_config):
    return PredPreyGrass(env_config or config_env)


register_env("PredPreyGrass", env_creator)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    return f"type_{parts[1]}_{parts[2]}"


def train_fn(config):
    sample_env = env_creator(config_env)

    module_specs = {}
    for agent_id in sample_env.observation_spaces:
        pid = policy_mapping_fn(agent_id)
        if pid not in module_specs:
            module_specs[pid] = RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=sample_env.observation_spaces[agent_id],
                action_space=sample_env.action_spaces[agent_id],
                model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
            )

    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space,
                            {"entropy_coeff": config["entropy_coeff"] if "prey" in pid else 0.0})
                      for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            gamma=0.99,
            lr=config["lr"],
            train_batch_size_per_learner=config["train_batch_size_per_learner"],
            minibatch_size=config["minibatch_size"],
            num_epochs=config["num_epochs"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(num_learners=1, num_gpus_per_learner=1)
        .env_runners(num_env_runners=4, num_envs_per_env_runner=3)
        .resources(num_cpus_for_main_process=4)
        .callbacks(EpisodeReturn)
        .build()
    )

    for i in range(100):  # 100 iters max
        result = ppo_config.train()
        tune.report(
            prey_reward=result["env_runners/agent_episode_returns_mean/type_1_prey_0"],
            predator_reward=result["env_runners/agent_episode_returns_mean/type_1_predator_0"],
            reward_mean=result["env_runners/episode_return_mean"],
        )

    ppo_config.cleanup()


search_space = {
    "lr": tune.uniform(1e-5, 5e-4),
    "entropy_coeff": tune.uniform(0.0, 0.05),
    "train_batch_size_per_learner": tune.choice([512, 1024, 2048]),
    "minibatch_size": tune.choice([128, 256]),
    "num_epochs": tune.choice([1, 3, 6]),
}

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=10,
    metric="predator_reward",
    mode="max",
    hyperparam_mutations={
        "lr": tune.uniform(1e-5, 5e-4),
        "entropy_coeff": tune.uniform(0.0, 0.05),
        "num_epochs": [1, 2, 4, 6],
        "train_batch_size_per_learner": [512, 1024, 2048],
        "minibatch_size": [64, 128, 256],
    },
)

tuner = tune.Tuner(
    train_fn,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=pbt,
        num_samples=4,
        metric="predator_reward",
        mode="max",
    ),
    run_config=train.RunConfig(
        name="PBT_PPO_PredPreyGrass",
        stop={"training_iteration": 100},
        storage_path=os.path.expanduser("~/ray_results"),
    )
)

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True)
    tuner.fit()
