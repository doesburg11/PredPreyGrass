from predpreygrass.pettingzoo.envs import predpreygrass_aec_v0
from predpreygrass.pettingzoo.config.config_predpreygrass import env_kwargs, local_output_root, training_steps_string
from predpreygrass.pettingzoo.train.utils.trainer import Trainer
from predpreygrass.pettingzoo.train.utils.config_saver import ConfigSaver
from predpreygrass.pettingzoo.eval.utils.evaluator import Evaluator

import os
import time
import shutil
from os.path import dirname as up

if __name__ == "__main__":
    parameter_variation_parameter_string = "energy_gain_per_step_predator"
    parameter_variation_scenarios = [-0.19, -0.18]

    time_stamp_string = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
    env_fn = predpreygrass_aec_v0
    environment_name = str(env_fn.raw_env.metadata["name"])
    training_steps = int(training_steps_string)
    # Create model file name for saving
    model_file_name = f"{environment_name}_steps_{training_steps_string}"

    # Parameter variation scenarios
    if len(parameter_variation_scenarios) > 1:
        destination_root_dir = (
            local_output_root + "/parameter_variation/" + parameter_variation_parameter_string + "_" + time_stamp_string
        )
    else:
        destination_root_dir = local_output_root + "/" + time_stamp_string

    # Training
    for parameter_variation_parameter in parameter_variation_scenarios:
        # Define subdirectory for parameter variation
        if len(parameter_variation_scenarios) > 1:
            print("-----------------------------------------------------------------------------")
            print(
                "Parameter variation: " + parameter_variation_parameter_string + ": ",
                str(parameter_variation_parameter),
            )
            print("-----------------------------------------------------------------------------")
            destination_source_code_dir = destination_root_dir + "/" + str(parameter_variation_parameter)
        else:
            destination_source_code_dir = destination_root_dir

        env_kwargs[parameter_variation_parameter_string] = parameter_variation_parameter

        # Define the destination directory for output
        destination_output_dir = destination_source_code_dir + "/output/"
        loaded_policy = destination_output_dir + model_file_name

        # Copy the original source code to the local directory
        source_code_dir = up(up(__file__))  # Up 2 levels in directory tree
        shutil.copytree(source_code_dir, destination_source_code_dir)
        os.makedirs(destination_output_dir, exist_ok=True)

        destination_training_file = os.path.join(destination_output_dir, "training.txt")

        # Save configuration
        config_saver = ConfigSaver(
            destination_training_file,
            environment_name,
            training_steps_string,
            env_kwargs,
            local_output_root,
            destination_source_code_dir,
            time_stamp_string,
        )
        config_saver.save()

        # Train the model
        trainer = Trainer(
            env_fn,
            destination_output_dir,
            model_file_name,
            steps=training_steps,
            seed=0,
            **env_kwargs,
        )
        start_training_time = time.time()
        # Train with wrapped AEC environment
        trainer.train(is_wrapped=True)
        end_training_time = time.time()
        training_time = end_training_time - start_training_time

        # Log training time
        with open(destination_training_file, "a") as training_file:
            if training_time < 3600:
                training_file.write("training time (min)= " + str(round(training_time / 60, 1)) + "\n")
            else:
                training_file.write("training time (hours)= " + str(round(training_time / 3600, 1)) + "\n")

    # Evaluation
    for parameter_variation_parameter in parameter_variation_scenarios:
        if len(parameter_variation_scenarios) > 1:
            destination_source_code_dir = destination_root_dir + "/" + str(parameter_variation_parameter)
        else:
            destination_source_code_dir = destination_root_dir

        output_directory = destination_source_code_dir + "/output/"
        env_kwargs[parameter_variation_parameter_string] = parameter_variation_parameter
        destination_output_dir = destination_source_code_dir + "/output/"
        loaded_policy = destination_output_dir + model_file_name

        # Extract evaluation settings
        watch_grid_model = env_kwargs["watch_grid_model"]
        render_mode = "human" if (watch_grid_model and len(parameter_variation_scenarios) == 1) else None

        # Evaluate the model
        evaluator = Evaluator(
            env_fn,
            environment_name,
            output_directory,  # destination_output_dir,
            loaded_policy,
            destination_source_code_dir,  # destination_root_dir,
            render_mode,
            destination_source_code_dir,  # destination_source_code_dir,
            training_steps_string,
            **env_kwargs,
        )
        evaluator.parallel_wrapped_aec_env_training_aec_evaluation()
