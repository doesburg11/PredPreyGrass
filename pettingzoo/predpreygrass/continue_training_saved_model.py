"""
To log both the previous model's training and the continued training in TensorBoard, 
you can use the same log directory for both training sessions. 
Stable Baselines3's `learn` method will automatically continue logging to the same directory 
without overwriting previous logs.

The `tb_log_name` parameter is used to specify the name of the run for TensorBoard logging. 
The `reset_num_timesteps` parameter is set to `False` to ensure that the timestep count 
continues from where it left off, rather than resetting to zero.

After/during training, you can review the logs in TensorBoard

"""
# Continue training for X steps and log the results to TensorBoard
import environments.predpreygrass_available_energy_transfer as predpreygrass
from config.config_pettingzoo import env_kwargs, training_steps_string

import os

import supersuit as ss
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import parallel_wrapper_fn

continued_training_steps_string = "350_000"

continued_training_steps = int(continued_training_steps_string)
continued_model_file_name = "predpreygrass_steps_" + continued_training_steps_string
model_file_name = "predprey_steps_" + training_steps_string
script_directory = os.path.dirname(os.path.abspath(__file__))
loaded_policy = script_directory + "/output/" + model_file_name
continued_policy = script_directory + "/output/continued_" + continued_model_file_name
print("loaded_policy:", loaded_policy)
print()


# Load the trained model
model = PPO.load(loaded_policy)


env_fn = predpreygrass
parallel_env = parallel_wrapper_fn(env_fn.raw_env)

# Train a single model to play as each agent in a parallel environment
raw_parallel_env = parallel_env(**env_kwargs)
raw_parallel_env.reset()

print(f"Continue training on {str(raw_parallel_env.metadata['name'])}.")

raw_parallel_env = ss.pettingzoo_env_to_vec_env_v1(raw_parallel_env)
raw_parallel_env = ss.concat_vec_envs_v1(
    raw_parallel_env, 8, num_cpus=8, base_class="stable_baselines3"
)


# Set the environment
model.set_env(raw_parallel_env)


# Continue training with TensorBoard logging
model.learn(
    total_timesteps=continued_training_steps,
    tb_log_name="PPO",
    reset_num_timesteps=False,
    progress_bar=True,
)
model.save(continued_policy)
raw_parallel_env.close()
