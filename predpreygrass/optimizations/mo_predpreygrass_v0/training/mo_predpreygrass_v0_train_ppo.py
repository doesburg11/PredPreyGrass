"""
-This file trains a mutlti object multi agent reinforcement model in a parallel environment. 
-After traing evaluation can be done using the AEC API.
-The source code and the trained model are saved in a separate 
directory, for reuse and analysis. 
-The algorithm used is PPO from stable_baselines3. 
"""
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
    local_output_directory,
)
# environment loop is parallel wrapped
env_kwargs["is_parallel_wrapped"] = True

from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.trainer import Trainer

from momaland.utils.conversions import mo_aec_to_parallel
from momaland.utils.parallel_wrappers import LinearizeReward

env = mo_predpreygrass_v0.env(**env_kwargs)

weights = {}

# Define the number of predators and prey
num_predators = env_kwargs["n_possible_predator"]
num_prey = env_kwargs["n_possible_prey"]

# Populate the weights dictionary for predators
for i in range(num_predators):
    weights[f"predator_{i}"] = [0.5, 0.5]

# Populate the weights dictionary for prey
for i in range(num_prey):
    weights[f"prey_{i + num_predators}"] = [0.5, 0.5]



parallel_env = mo_aec_to_parallel(env)

parallel_env = LinearizeReward(parallel_env, weights)


from os.path import dirname as up
import os
import shutil
import time

if __name__ == "__main__":
    env_fn = parallel_env
    environment_name = "mo_predpreygrass_v0"
    training_steps = int(training_steps_string)
    # create model file name for saving
    time_stamp_string = str(time.strftime("%Y-%m-%d_%H:%M:%S"))  
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    # create a local directory to save the project code and output results
    destination_directory_source_code = os.path.join(
        local_output_directory, time_stamp_string
    )

    project_directory = up(up(up(up(__file__)))) # up 4 levels in directory tree
    # copy the project code to the local directory
    shutil.copytree(project_directory, destination_directory_source_code)
    # Create the output directory
    output_directory = destination_directory_source_code + "/output/"
    os.makedirs(output_directory, exist_ok=True)

    # save environment configuration to file
    saved_directory_and_env_config_file_name = os.path.join(
        output_directory, "env_config.txt"
    )
    # write environment configuration to file
    file = open(saved_directory_and_env_config_file_name, "w")
    file.write("environment: " + environment_name + "\n")
    file.write("learning algorithm: PPO \n")
    file.write("training steps: " + training_steps_string + "\n")
    file.write("------------------------\n")
    for item in env_kwargs:
        file.write(str(item) + " = " + str(env_kwargs[item]) + "\n")
    file.write("------------------------\n")

    # train the model
    trainer = Trainer(
        env_fn, 
        output_directory,
        model_file_name, 
        steps=training_steps, 
        seed=0, 
        **env_kwargs
    )
    start_training_time = time.time()
    trainer.train()
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    if training_time < 3600:  # (seconds)
        file.write(
            "training time = " + str(round(training_time / 60, 1)) + " minutes \n"
        )
    else:
        file.write(
            "training time = " + str(round(training_time / 3600, 1)) + " hours \n"
        )
    file.close()

