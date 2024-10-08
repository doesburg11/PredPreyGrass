"""
-This file trains a multi agent reinforcement model in a parallel environment. 
-After traing evaluation can be done using the AEC API.
-The source code and the trained model are saved in a separate 
directory, for reuse and analysis. 
-The algorithm used is PPO from stable_baselines3. 
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
from predpreygrass.optimizations.so_predpreygrass_v0.training.utils.config_saver import (
    ConfigSaver
)
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
    if training_time < 3600:  # (seconds)
        training_file.write(
            "training time = " + str(round(training_time / 60, 1)) + " minutes \n"
        )
    else:
        training_file.write(
            "training time = " + str(round(training_time / 3600, 1)) + " hours \n"
        )
    training_file.close()
