#NUMBER III
#define policies in dict with PolicySpec
#add width and height parameters to the environment

from environments.predpreygrass_env import PredPreyGrassEnv
from config.config_rllib import configuration

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.pre_checks.env import  check_env
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
import warnings
import time
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning) 

check_env(PredPreyGrassEnv(configuration))

def env_creator(configuration):
    return PredPreyGrassEnv(configuration)  # return an env instance

register_env("pred_prey_grass", env_creator)

policy1 = PolicySpec()
policy2 = PolicySpec()

policies = { 
    "policy1": policy1,
    "policy2": policy2
}

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id.startswith("predator_"):
        return "policy1"
    if agent_id.startswith("prey_"):
        return "policy2"
    else:
        raise ValueError(f"Unexpected agent ID: {agent_id}")

config = (
    PPOConfig()
    .environment(env="pred_prey_grass")
    .framework("torch")
    .rollouts(
        create_env_on_local_worker=True,
        batch_mode="complete_episodes", #"truncate_episodes",
        num_rollout_workers=0,
        rollout_fragment_length= "auto",
    )
    .debugging(seed=0,log_level="ERROR")
    .training(model={
        "fcnet_hiddens" : [64, 64], 
        "_disable_preprocessor_api": False,
        "conv_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [512, [1, 1], 1]] # Copilot
        }
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
    #.build()
)


if __name__ == "__main__":

    algo = config.build()

    env = PredPreyGrassEnv(configuration=configuration)
    obs, _ = env.reset()
    stop_loop = False



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
        
        terminated = env.terminateds["__all__"]
        truncated = truncateds["__all__"]
        stop_loop = terminated or truncated
        env.render()
        
    time.sleep(3)
    env.close()
    algo.stop()

    import ray
    from ray import train, tune

    ray.init()

    config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

    tuner = tune.Tuner(
        "PPO",
        run_config=train.RunConfig(
            stop={"episode_reward_mean": 150},
        ),
        param_space=config,
    )

    tuner.fit()
