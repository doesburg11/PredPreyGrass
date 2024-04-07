
from environments.predpreygrass_env_actions import PredPreyGrassEnv
from config.config_rllib import configuration

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner  
from ray.rllib.utils.pre_checks.env import  check_env
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import get_trainable_cls, register_env

from ray.tune.logger import pretty_print

import ray
from ray import train, tune

import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning) 

check_env(PredPreyGrassEnv(configuration))
print("Environment checked")

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
    check_compute_single_action,
)

parser = add_rllib_example_script_args(
    default_iters=10,
    default_timesteps=1000000,
    default_reward=0.0,

)




register_env("pred_prey_grass", lambda _: PredPreyGrassEnv(configuration))


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

base_config = (
    #PPOConfig()
    get_trainable_cls("PPO")
    .get_default_config()
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
        num_envs_per_worker=1, 
    )
    .debugging(seed=0,log_level="ERROR")
    .training(model={
        "uses_new_env_runners": True,
        "fcnet_hiddens" : [64, 64], 
        "_disable_preprocessor_api": False,
        "conv_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [512, [1, 1], 1]] # Copilot
        },
        #lr=tune.grid_search([0.0001, 0.00005, 0.00001])
        lr=0.00001,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn
    )
    .evaluation(
        evaluation_num_workers=1,
        evaluation_interval=1,
        enable_async_evaluation=True,
        evaluation_config=PPOConfig.overrides(
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            render_env=True,
        ),
    )

)


if __name__ == "__main__":


    args = parser.parse_args()
    print(args)

    assert (
        args.num_agents > 0
    ), "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"



    results = run_rllib_example_script_experiment(base_config, args)

    #https://github.com/ray-project/ray/blob/master/rllib/examples/inference_and_serving/policy_inference_after_training.py
    print("Training completed. Restoring new Algorithm for action inference.")
    # Get the last checkpoint from the above training run.
    checkpoint = results.get_best_result(
        metric="episode_reward_mean",
        mode="max",
        filter_nan_and_inf=False
    ).checkpoint
    
    # Create new Algorithm and restore its state from the last checkpoint.
    algo = Algorithm.from_checkpoint(checkpoint)

    env = PredPreyGrassEnv(configuration=configuration)
    obs, _ = env.reset()

    check_compute_single_action(algo)


    actions = {
        "predator_0": algo.compute_single_action(obs["predator_0"], policy_id="policy1"),
        "predator_1": algo.compute_single_action(obs["predator_1"], policy_id="policy1"),
        "predator_2": algo.compute_single_action(obs["predator_2"], policy_id="policy1"),
        "prey_3": algo.compute_single_action(obs["prey_3"], policy_id="policy2"),
        "prey_4": algo.compute_single_action(obs["prey_4"], policy_id="policy2"),
        "prey_5": algo.compute_single_action(obs["prey_5"], policy_id="policy2")    
    }


    #check_compute_single_action(algo)


    """
    ray.init(num_cpus=8)
    
    tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={
            #"episode_reward_mean": 5,
            #"time_total_s": 60, # Stop a trial after it's run for more than "time_total_s" seconds.
            "training_iteration": 10,

            },
    ),
    param_space=config,
    )

    results = tuner.fit()

    for i in range(len(results)):   
        result = results[i]
        if not result.error:
                print(f"Trial finishes successfully with metrics"
                f"{result.metrics}.")
        else:
                print(f"Trial failed with error {result.error}.")

    #https://github.com/ray-project/ray/blob/master/rllib/examples/inference_and_serving/policy_inference_after_training.py
    print("Training completed. Restoring new Algorithm for action inference.")
    # Get the last checkpoint from the above training run.
    checkpoint = results.get_best_result(
        metric="episode_reward_mean",
        mode="max",
        filter_nan_and_inf=False
    ).checkpoint
    
    # Create new Algorithm and restore its state from the last checkpoint.
    algo = Algorithm.from_checkpoint(checkpoint)

    env = PredPreyGrassEnv(configuration=configuration)
    obs, _ = env.reset()

    num_episodes = 0
    episode_reward = 0.0


    terminated = env.terminateds["__all__"]
    truncated = env.truncateds["__all__"]
    stop_loop = terminated or truncated

    while num_episodes < 10:
        # Compute an action (`a`).


        actions = {
            "predator_0": algo.compute_single_action(obs["predator_0"], policy_id="policy1"),
            "predator_1": algo.compute_single_action(obs["predator_1"], policy_id="policy1"),
            "predator_2": algo.compute_single_action(obs["predator_2"], policy_id="policy1"),
            "prey_3": algo.compute_single_action(obs["prey_3"], policy_id="policy2"),
            "prey_4": algo.compute_single_action(obs["prey_4"], policy_id="policy2"),
            "prey_5": algo.compute_single_action(obs["prey_5"], policy_id="policy2")    
        }

        # Send the computed actions to the env.
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        env.render()
        num_episodes += 1
        
    time.sleep(3)

    env.close()

    algo.stop()
    ray.shutdown()
    """





  
    
