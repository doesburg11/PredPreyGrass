"""
-This file trains a multi-object multi-agent reinforcement model in a parallel environment. 
-After traing evaluation can be done using the AEC API.
-The source code and the trained model are saved in a separate 
directory, for reuse and analysis. 
-The algorithm used is PPO from stable_baselines3. 
"""
# discretionary libraries
from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
    local_output_root,
)
env_kwargs["is_parallel_wrapped"] = True # environment loop is parallel wrapped
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.trainer import (
    Trainer
)
from utils.linearization_weights_constructor import (
    construct_linearalized_weights
)


# external libraries
from momaland.utils.conversions import mo_aec_to_parallel
from momaland.utils.parallel_wrappers import LinearizeReward
from os.path import dirname as up
import os
import time
import shutil


# Define the number of predators and prey
num_predators = env_kwargs["n_possible_predator"]
num_prey = env_kwargs["n_possible_prey"]
# Construct the weights
weights = construct_linearalized_weights(num_predators, num_prey)

env = mo_predpreygrass_v0.env(**env_kwargs)
parallel_env = mo_aec_to_parallel(env)
parallel_env = LinearizeReward(parallel_env, weights)


if __name__ == "__main__":
    env_fn = parallel_env
    environment_name = "mo_predpreygrass_v0"
    training_steps = int(training_steps_string)
    # create model file name for saving
    time_stamp_string = str(time.strftime("%Y-%m-%d_%H:%M:%S"))  
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    # create a local directory to save the project code and output results
    destination_source_code_dir = os.path.join(
        local_output_root, time_stamp_string
    )
    source_code_dir = up(up(up(up(__file__)))) # up 4 levels in directory tree
    # copy the project code to the local directory
    shutil.copytree(source_code_dir, destination_source_code_dir)
    # Create the output directory
    destination_output_dir = destination_source_code_dir + "/output/"
    os.makedirs(destination_output_dir, exist_ok=True)

    # save environment configuration to file
    destination_training_file = os.path.join(
        destination_output_dir, "training.txt"
    )
    # write environment configuration to file
    file = open(destination_training_file, "w")
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
    if training_time < 3600:  # (seconds)
        file.write(
            "training time = " + str(round(training_time / 60, 1)) + " minutes \n"
        )
    else:
        file.write(
            "training time = " + str(round(training_time / 3600, 1)) + " hours \n"
        )
    file.close()

