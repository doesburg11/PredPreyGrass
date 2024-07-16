"""
-This file trains a multi agent reinforcement model in a parallel environment. 
-After traing evaluation can be done using the AEC API.
-The source code and the trained model are saved in a separate 
directory, for reuse and analysis. 
-The algorithm used is PPO from stable_baselines3. 
"""

import environments.predpreygrass_available_energy_transfer as predpreygrass

from config.config_pettingzoo import (
    env_kwargs,
    training_steps_string,
    local_output_directory,
)

import os
import time
import sys
import shutil

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils.conversions import parallel_wrapper_fn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger


class SampleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_length = 0
        self.episode_lengths = []

    def _on_step(self) -> bool:
        self.current_episode_length += 1
        # If the episode is done, log the episode length and reset the counter
        if "done" in self.locals and self.locals["done"]:
            #print("done")
            self.episode_lengths.append(self.current_episode_length)
            self.logger.record("train/episode_length", self.current_episode_length)
            self.current_episode_length = 0
        return True  # Continue training


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    parallel_env = parallel_wrapper_fn(env_fn.raw_env)

    # Train a single model to play as each agent in a parallel environment
    raw_parallel_env = parallel_env(render_mode=None, **env_kwargs)
    raw_parallel_env.reset(seed=seed)

    print(f"Starting training on {str(raw_parallel_env.metadata['name'])}.")
    if parameter_variation:
        print(
            "Parameter variation " + parameter_variation_parameter_string + ": ",
            env_kwargs[parameter_variation_parameter_string],
        )
    # create parallel environments by concatenating multiple copies of the base environment
    #
    num_vec_envs_concatenated = 8
    raw_parallel_env = ss.pettingzoo_env_to_vec_env_v1(raw_parallel_env)
    raw_parallel_env = ss.concat_vec_envs_v1(
        raw_parallel_env,
        num_vec_envs_concatenated,
        num_cpus=8,
        base_class="stable_baselines3",
    )

    #untuned
    """
    model = PPO(
        MlpPolicy,
        raw_parallel_env,
        verbose=0,  # 0 for no output, 1 for info messages, 2 for debug messages, 3 deafult
        batch_size=256,
        tensorboard_log=output_directory + "/ppo_predprey_tensorboard/",
    )
    """
    #tuned with optuna
    model = PPO(
        MlpPolicy,
        raw_parallel_env,
        verbose=0,  # 0 for no output, 1 for info messages, 2 for debug messages, 3 deafult
        n_steps=2048,
        batch_size=256,
        gamma=0.94,
        learning_rate=0.0015,
        ent_coef=0.0135,
        clip_range=0.17,
        tensorboard_log=output_directory + "/ppo_predprey_tensorboard/",
    )



    sample_logger_callback = SampleLoggerCallback()

    # model.learn(total_timesteps=steps, progress_bar=True)
    model.learn(
        total_timesteps=steps, progress_bar=True, callback=sample_logger_callback
    )
    model.save(saved_directory_and_model_file_name)
    print("saved path: ", saved_directory_and_model_file_name)
    print("Model has been saved.")
    print(f"Finished training on {str(raw_parallel_env.unwrapped.metadata['name'])}.")

    raw_parallel_env.close()


if __name__ == "__main__":
    environment_name = "predpreygrass"
    env_fn = predpreygrass
    training_steps = int(training_steps_string)
    parameter_variation = False
    parameter_variation_parameter_string = "prey_creation_energy_threshold"
    if parameter_variation:
        parameter_variation_scenarios = [8, 10, 12, 14, 16]
    else:
        parameter_variation_scenarios = [
            env_kwargs[parameter_variation_parameter_string]
        ]  # default value, must be iterable
    # output file name
    # start_time = str(time.strftime('%Y-%m-%d_%H:%M'))
    start_time = str(time.strftime("%Y-%m-%d_%H:%M:%S"))  # add seconds
    file_name = f"{environment_name}_steps_{training_steps_string}"

    for parameter_variation_parameter in parameter_variation_scenarios:
        if parameter_variation:
            env_kwargs[
                parameter_variation_parameter_string
            ] = parameter_variation_parameter
            # define the destination directory for the source code
            destination_directory_source_code = (
                local_output_directory
                + parameter_variation_parameter_string
                + "/"
                + str(parameter_variation_parameter)
            )
            output_directory = destination_directory_source_code + "/output/"
            loaded_policy = output_directory + file_name
        else:
            # define the destination directory for the source code
            destination_directory_source_code = os.path.join(
                local_output_directory, start_time
            )
            output_directory = destination_directory_source_code + "/output/"
            loaded_policy = output_directory + file_name

        # save the source code locally
        python_file_name = os.path.basename(sys.argv[0])
        python_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        file_names_in_directory = os.listdir(python_directory)
        # create the destination directory for the source code
        os.makedirs(destination_directory_source_code, exist_ok=True)

        # Copy all files and directories in the current directory to the local directory
        # for safekeeping experiment scenarios
        for item_name in file_names_in_directory:
            source_item = os.path.join(python_directory, item_name)
            destination_item = os.path.join(
                destination_directory_source_code, item_name
            )

            if os.path.isfile(source_item):
                shutil.copy2(source_item, destination_item)
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item)

        if parameter_variation:
            # overwrite config file locally
            # Start of the code string
            code = "local_output_directory = '{}'\n".format(local_output_directory)
            code += "training_steps_string = '{}'\n".format(training_steps_string)
            code += "env_kwargs = dict(\n"
            # Add each item from env_kwargs to the code string
            for key, value in env_kwargs.items():
                code += f"    {key}={value},\n"

            # Close the dict in the code string
            code += ")\n"
            config_file_name = "config_pettingzoo.py"
            config_file_directory = destination_directory_source_code + "/config/"

            with open(config_file_directory + config_file_name, "w") as config_file:
                config_file.write(code)
            config_file.close()
        # Create the output directory
        os.makedirs(output_directory, exist_ok=True)
        saved_directory_and_model_file_name = os.path.join(output_directory, file_name)

        # save parameters to file
        saved_directory_and_parameter_file_name = os.path.join(
            output_directory, "train_parameters.txt"
        )
        file = open(saved_directory_and_parameter_file_name, "w")
        file.write("model: PredPreyGrass\n")
        file.write("parameters:\n")
        file.write("training steps: " + training_steps_string + "\n")
        file.write("------------------------\n")
        for item in env_kwargs:
            file.write(str(item) + " = " + str(env_kwargs[item]) + "\n")
        file.write("------------------------\n")
        start_training_time = time.time()
        train(env_fn, steps=training_steps, seed=0, **env_kwargs)
        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        if training_time < 3600:  # seconds
            file.write(
                "training time (min)= " + str(round(training_time / 60, 1)) + "\n"
            )
        else:
            file.write(
                "training time (hours)= " + str(round(training_time / 3600, 1)) + "\n"
            )
        file.close()
