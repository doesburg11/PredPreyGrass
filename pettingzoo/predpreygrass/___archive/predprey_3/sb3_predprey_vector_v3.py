"""Uses Stable-Baselines3 to train agents in the Pursuit environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.predpreygrass.predprey_3 import predprey_v3

def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
       
    # Train a single model to play as each agent in a parallel environment
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=8, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)
    model.save(saved_directory_and_model_file_name)
    print("saved path: ",saved_directory_and_model_file_name)
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    """
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )
    """
    model = PPO.load(loaded_policy)
    rewards = {agent: 0 for agent in env.possible_agents}
  
    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)
        rewards = {agent: 0 for agent in env.possible_agents}
        n_cycles = 0
        for agent in env.agent_iter():
            if agent=="pursuer_0":
                n_cycles += 1
            obs, reward, termination, truncation, info = env.last()
            for agent in env.agents:
                rewards[agent] += env.rewards[agent]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=False)[0]
                # :param deterministic: Whether or not to return deterministic actions.
            env.step(act)
        N = len(rewards.values())
        avg_reward = sum(rewards.values()) / N
        variance_rewards = 0
        for _agent in rewards:
            variance_rewards += pow(rewards[_agent]-avg_reward,2)
        variance_rewards = 1/(N-1)*variance_rewards
        std_rewards = pow(variance_rewards,0.5)
        print(f"Cycles = {n_cycles}", f"Avg = {round(avg_reward,1)}", f"Std = {round(std_rewards,1)}",end=" ")
        """
        for agent in rewards:
            print(agent[8:],"=",round(rewards[agent],1),end=", ")
        """
        print()
    env.close()
    # end evaluation

if __name__ == "__main__":
    env_fn = predprey_v3

    train_model = True  # True evaluates latest policy, False evaluates at predefined loaded policy
    eval_model = True
    eval_and_watch_model = True
    training_steps_string = "1_000_000"
    training_steps = int(training_steps_string)
    # loaded_policy = "./trained_models/pursuit/conclusions/1/pursuit_2023-10-24 11:04/pursuit_steps_5_000_000.zip"
    #loaded_policy = "./trained_models/predprey/predprey_2023-11-02 18:08/predprey_steps_1_000_000"
    loaded_policy = "./trained_models/predprey/conclusions/freeze_evaders/steps_1_000_000/predprey_steps_1_000_000"
    #loaded_policy = "./trained_models/predprey/conclusions/moore/time_step_vatiation/50_000_000/predprey_steps_50_000_000"
    env_kwargs = dict(
        max_cycles=100000, 
        x_size=16, 
        y_size=16, 
        shared_reward=False, 
        n_evaders=30,
        n_pursuers=8,
        obs_range_pursuers=7, 
        obs_range_predators=5, 
        obs_range_evaders=0,    
        n_catch=1,
        freeze_evaders=True, 
        tag_reward=0.01,
        catch_reward=5.0, 
        urgency_reward=-0.1, 
        surround=False, 
        moore_neighborhood_evaders=False,
        moore_neighborhood_pursuers=False,
        moore_neighborhood_predators=True
        )
    """
    max_cycles: After max_cycles steps all agents will return done
    x_size, y_size: Size of environment world space
    shared_reward: Whether the rewards should be distributed among all agents
    n_evaders: Number of evaders
    n_pursuers: Number of pursuers
    obs_range: Size of the box around the agent that the agent observes.
    n_catch: Number pursuers required around an evader to be considered caught
    freeze_evaders: Toggles if evaders can move or not
    tag_reward: Reward for ‘tagging’, or being single evader.
    term_pursuit: Reward added when a pursuer or pursuers catch an evader
    urgency_reward: Reward to agent added in each step
    surround: Toggles whether evader is removed when surrounded, or when n_catch pursuers are on top of evader
    constraint_window: Size of box (from center, in proportional units) which agents can randomly spawn into the environment world. 
    Default is 1.0, which means they can spawn anywhere on the map. A value of 0 means all agents spawn in the center.    
    moore_neighborhood: Toggles if pursuers and evaders can move to a moore neighborhood,
    False is a Von Neumann neighborhood
    """
    if train_model:
        #print("obs range: ", end="")
        #print(env_kwargs["obs_range"])

        # Save the trained model in specified directory
        start_time = str(time.strftime('%Y-%m-%d %H:%M'))
        environment_name = "predprey"
        file_name = f"{environment_name}_steps_{training_steps_string}"
        directory = "./trained_models/predprey/"+f"{environment_name}_{start_time}"
        os.makedirs(directory, exist_ok=True)
        saved_directory_and_model_file_name = os.path.join(directory, file_name)

        #save parameters to file
        saved_directory_and_parameter_file_name = os.path.join(directory, "parameters.txt")
        file = open(saved_directory_and_parameter_file_name, "w")
        file.write("parameters:\n")
        file.write("=========================\n")
        for item in env_kwargs:
            file.write(str(item)+" = "+str(env_kwargs[item])+"\n")
        file.write("=========================\n")
        file.close()

        train(env_fn, steps=training_steps, seed=0, **env_kwargs)
        # load latest policy
        loaded_policy = max(
            glob.glob(os.path.join(directory,f"{environment_name}*.zip")), key=os.path.getctime
            )
            
    if eval_model:
        # Evaluate games 
        eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    if eval_and_watch_model:
        # Evaluate and watch games
        eval(env_fn, num_games=1, render_mode="human", **env_kwargs)
