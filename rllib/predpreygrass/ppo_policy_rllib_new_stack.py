
from environments.predpreygrass_env_actions import PredPreyGrassEnv
from config.config_rllib import configuration

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO

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
    .environment(
        env="pred_prey_grass",
        render_env=True
        )
    .experimental(_enable_new_api_stack=True)
    .rollouts(
        env_runner_cls=MultiAgentEnvRunner,
        num_envs_per_worker=2, 
        num_rollout_workers=1
    )
    .resources(
        num_learner_workers=1,
        num_gpus_per_learner_worker=0,
        num_cpus_for_local_worker=6,
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
        #lr=tune.grid_search([0.001, 0.0001])
        lr=0.00001
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
    .evaluation(
        evaluation_num_workers=1,
        evaluation_interval=1,
        #enable_async_evaluation=True,
        evaluation_config=PPOConfig.overrides(
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            render_env=True,
        ),
    )
)


if __name__ == "__main__":
    
    
    algo = config.build()

    #trainer = PPO(config=config, env="pred_prey_grass")    
    
    print("Training started")
    for i in range(1000):
        results = algo.train()
        print(f"R={results['episode_reward_mean']}")

        #print(pretty_print(results))
        """
        if i % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}") 
        """  
    
    """
    env = PredPreyGrassEnv(configuration=configuration)
    obs, _ = env.reset()

    terminated = env.terminateds["__all__"]
    truncated = env.truncateds["__all__"]
    stop_loop = terminated or truncated
    
                
    while not stop_loop:
        actions = {
            "predator_0": algo.compute_single_action(obs["predator_0"], policy_id="policy1"),
            "predator_1": algo.compute_single_action(obs["predator_1"], policy_id="policy1"),
            "predator_2": algo.compute_single_action(obs["predator_2"], policy_id="policy1"),
            "prey_3": algo.compute_single_action(obs["prey_3"], policy_id="policy2"),
            "prey_4": algo.compute_single_action(obs["prey_4"], policy_id="policy2"),
            "prey_5": algo.compute_single_action(obs["prey_5"], policy_id="policy2")    
        }

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        env.render()
        
    time.sleep(3)
    env.close()
    
    """
    algo.stop()

