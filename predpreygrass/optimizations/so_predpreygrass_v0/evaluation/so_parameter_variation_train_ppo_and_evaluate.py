"""
This file trains a multi agent  reinforcement model in a parallel 
environment. Evaluation is done using the AEC API. After training, 
the source code and the trained model is saved in a separate 
directory, for reuse and analysis. 
The algorithm used is PPO from stable_baselines3. 
The environment used is predpreygrass
"""
# discretionary libraries
from predpreygrass.envs import so_predpreygrass_v0
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
    local_output_root,
)
from predpreygrass.optimizations.so_predpreygrass_v0.training.utils.trainer import (
    Trainer,
)
from predpreygrass.optimizations.so_predpreygrass_v0.training.utils.config_saver import ConfigSaver
from predpreygrass.optimizations.so_predpreygrass_v0.evaluation.utils.evaluator import Evaluator

# external libraries
import os
import time
import shutil
from os.path import dirname as up

if __name__ == "__main__":
    env_fn = so_predpreygrass_v0
    environment_name = str(env_fn.raw_env.metadata['name'])
    training_steps = int(training_steps_string)
    # create model file name for saving
    time_stamp_string = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    # parameter variation scenarios
    parameter_variation_parameter_string = "reproduction_reward_prey"
    parameter_variation_scenarios = [10] 
    if len(parameter_variation_scenarios) > 1:
        destination_root_dir = (
            local_output_root + "parameter_variation/"
            + parameter_variation_parameter_string + "_"
            + time_stamp_string
        )
    else:
        destination_root_dir = (
            local_output_root + time_stamp_string
        )
    # Training
    for parameter_variation_parameter in parameter_variation_scenarios:
        # if only one parameter variation scenario: do not 
        # create parameter variation subdirectory
        if len(parameter_variation_scenarios) > 1:
            print(
                "Parameter variation " + parameter_variation_parameter_string + ": ",
                str(parameter_variation_parameter),
            )
            destination_source_code_dir = (
                destination_root_dir
                + "/" + str(parameter_variation_parameter)
            )
        else:
            destination_source_code_dir = destination_root_dir
        env_kwargs[parameter_variation_parameter_string] = parameter_variation_parameter
        # define the destination directory for the source code
        destination_output_dir = destination_source_code_dir + "/output/"
        loaded_policy = destination_output_dir + model_file_name
        # copy the original source code to the local directory
        source_code_dir = up(up(up(up(__file__))))  # up 4 levels in directory tree
        # copy the project code to the local directory
        shutil.copytree(source_code_dir, destination_source_code_dir)
        os.makedirs(destination_output_dir, exist_ok=True)
        destination_training_file = os.path.join(
            destination_output_dir, "training.txt"
        )
        # Create an instance of the ConfigSaver class
        config_saver = ConfigSaver(
            destination_training_file, 
            environment_name, 
            training_steps_string, 
            env_kwargs, 
            local_output_root, 
            destination_source_code_dir
        )
        # Call the save method to save the configuration
        config_saver.save()
        # train the model
        trainer = Trainer(
            env_fn,
            destination_output_dir,
            model_file_name,
            steps=training_steps,
            seed=0,
            **env_kwargs,
        )
        start_training_time = time.time()
        trainer.train()
        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        # append training time to training_file
        training_file = open(destination_training_file, "a")
        if training_time < 3600:
            training_file.write(
                "training time (min)= " + str(round(training_time / 60, 1)) + "\n"
            )
        else:
            training_file.write(
                "training time (hours)= " + str(round(training_time / 3600, 1)) + "\n"
            )
        training_file.close()
    # Evaluation
    for parameter_variation_parameter in parameter_variation_scenarios:
        if len(parameter_variation_scenarios) > 1:
            destination_source_code_dir = (
                destination_root_dir
                + "/" + str(parameter_variation_parameter)
            )
        else:
            destination_source_code_dir = destination_root_dir
        env_kwargs[parameter_variation_parameter_string] = parameter_variation_parameter
        destination_output_dir = destination_source_code_dir + "/output/"
        loaded_policy = destination_output_dir + model_file_name
        # input from so_config_predpreygrass.py
        watch_grid_model = env_kwargs["watch_grid_model"]
        num_episodes = env_kwargs["num_episodes"]
        training_steps = int(training_steps_string)
        render_mode = "human" if watch_grid_model else None
                # Create an instance of the Evaluator class
        evaluator = Evaluator(
            env_fn,
            destination_output_dir,
            loaded_policy,
            destination_root_dir,
            render_mode,
            training_steps_string,
            destination_source_code_dir,
            **env_kwargs
        )
        # Call the eval method to perform the evaluation
        evaluator.eval()

