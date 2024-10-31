# discretionary libraries
from predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.envs._so_predpreygrass_v0.config.config_predpreygrass import (
    env_kwargs,
    training_steps_string,
)
# external libraries
import os
from os.path import dirname as up
from stable_baselines3 import PPO

def evaluate_trained_model(
        env_fn, 
        model_path, 
        watch_grid=False
    ):
    """Evaluates a trained PPO model in a PettingZoo environment, loaded from file."""
    model = PPO.load(model_path)

    render_mode = "human" if watch_grid else None
    env = env_fn.parallel_env(**env_kwargs, render_mode=render_mode)

    observations  = env.reset()[0]
    done = {agent: False for agent in env.agents}

    while not all(done.values()):
        actions = {}
        for agent in env.agents:
            if not done[agent]:
                action = model.predict(observations[agent], deterministic=True)[0]
                actions[agent] = action

        observations, rewards  = env.step(actions)[:2]
        for agent in rewards:
            if rewards[agent] > 0.0:
                print(f"{agent}, reward: {rewards[agent]}")


if __name__ == "__main__":
    env_fn = predpreygrass_parallel_v0 #.env(render_mode="human", **env_kwargs)
    environment_name = str(env_fn.parallel_env.metadata['name'])
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    evaluation_directory = os.path.dirname(os.path.abspath(__file__))
    destination_source_code_dir = up(up(up(up(__file__))))  # up 4 levels in directory tree
    output_directory = destination_source_code_dir +"/output/"
    loaded_policy = output_directory + model_file_name
    # input from so_config_predpreygrass.py
    watch_grid_model = env_kwargs["watch_grid_model"]
    num_episodes = env_kwargs["num_episodes"] 
    training_steps = int(training_steps_string)

    render_mode = "human" if watch_grid_model else None

    # Call the eval method to perform the evaluation
    evaluate_trained_model(
        env_fn,
        loaded_policy,
        True
    )    