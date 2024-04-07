
from environments.predpreygrass_env_actions import PredPreyGrassEnv
from config.config_rllib import configuration

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner  
from ray.rllib.utils.pre_checks.env import  check_env
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

import ray
from ray import train, tune

import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning) 

check_env(PredPreyGrassEnv(configuration))
print("Environment checked")


def env_creator(configuration):
    return PredPreyGrassEnv(configuration)  # return an env instance

register_env("pred_prey_grass", env_creator)

policy1 = PolicySpec()
policy2 = PolicySpec()

policies = { 
    "policy1": policy1,
    "policy2": policy2
}

def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    if agent_id.startswith("predator_"):
        return "policy1"
    if agent_id.startswith("prey_"):
        return "policy2"
    else:
        raise ValueError(f"Unexpected agent ID: {agent_id}")

config = (
    PPOConfig()
    .environment(env="pred_prey_grass")
    .experimental(_enable_new_api_stack=True)
    .rollouts(env_runner_cls=MultiAgentEnvRunner)
    .resources(
        num_learner_workers=1,
        num_gpus_per_learner_worker=0,
        num_cpus_for_local_worker=5,
    )    .framework("torch")
    .rollouts(
        create_env_on_local_worker=True,
        batch_mode="complete_episodes", #"truncate_episodes",
        num_rollout_workers=1,
        rollout_fragment_length= "auto",
    )
    .debugging(seed=0,log_level="ERROR")
    .training(model={
        "uses_new_env_runners": True,
        "fcnet_hiddens" : [64, 64], 
        "_disable_preprocessor_api": False,
        "conv_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [512, [1, 1], 1]] # Copilot
        },
        lr=tune.grid_search([0.0001, 0.00005, 0.00001])
        #lr=0.00001,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
    .evaluation(
        evaluation_num_workers=1,
        evaluation_interval=10,
        enable_async_evaluation=True
        )

)


if __name__ == "__main__":
    
    ray.init(num_cpus=8)

    tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={
            #"episode_reward_mean": 5,
            "time_total_s": 40000, # Stop a trial after it's run for more than "time_total_s" seconds.
            "training_iteration": 10000,

            },
    ),
    param_space=config,
    )

    tuner.fit()
    
    ray.shutdown()





  
    
