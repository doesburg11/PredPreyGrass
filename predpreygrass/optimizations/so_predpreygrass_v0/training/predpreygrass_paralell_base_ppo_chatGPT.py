# discretionary libraries
from predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.envs._so_predpreygrass_v0.config.config_predpreygrass import (
    env_kwargs,
    training_steps_string,
    local_output_root,
)
from predpreygrass.optimizations.so_predpreygrass_v0.training.utils.trainer_par import (
    Trainer,
)
from predpreygrass.optimizations.so_predpreygrass_v0.training.utils.config_saver import (
    ConfigSaver
)

# external libraries
import argparse
import logging
import os
import shutil
import time
from os.path import dirname as up


def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Parses command line arguments for flexibility."""
    parser = argparse.ArgumentParser(description="Train a multi-agent reinforcement learning model.")
    parser.add_argument('--training_steps', type=int, default=int(training_steps_string), help='Number of training steps')
    parser.add_argument('--output_root', type=str, default=local_output_root, help='Root directory to save output files')
    return parser.parse_args()


def create_directories(output_root, time_stamp_string):
    """Creates necessary directories and copies source code to the output directory."""
    destination_source_code_dir = os.path.join(output_root, time_stamp_string)
    destination_output_dir = os.path.join(destination_source_code_dir, "output/")

    try:
        source_code_dir = up(up(up(up(__file__))))  # up 4 levels in directory tree
        shutil.copytree(source_code_dir, destination_source_code_dir)
        os.makedirs(destination_output_dir, exist_ok=True)
    except shutil.Error as e:
        logging.error(f"Error copying project code: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

    return destination_source_code_dir, destination_output_dir


def save_training_configuration(destination_output_dir, environment_name, training_steps_string, env_kwargs, output_root, destination_source_code_dir):
    """Saves the training configuration to a file."""
    destination_training_file = os.path.join(destination_output_dir, "training.txt")
    config_saver = ConfigSaver(
        destination_training_file,
        environment_name,
        training_steps_string,
        env_kwargs,
        output_root,
        destination_source_code_dir
    )
    config_saver.save()
    return destination_training_file


def train_model(env_fn, destination_output_dir, model_file_name, training_steps, env_kwargs, training_file_path):
    """Trains the model and logs the training time."""
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

    # Append training time to training file
    with open(training_file_path, "a") as training_file:
        if training_time < 3600:  # seconds
            training_file.write(f"training time = {round(training_time / 60, 1)} minutes\n")
        else:
            training_file.write(f"training time = {round(training_time / 3600, 1)} hours\n")


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()

    try:
        env_fn = predpreygrass_parallel_v0
        environment_name = str(env_fn.parallel_env.metadata['name'])
        training_steps = args.training_steps

        # Create model file name for saving
        time_stamp_string = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
        model_file_name = f"{environment_name}_steps_{training_steps}"

        # Create directories and copy source code
        destination_source_code_dir, destination_output_dir = create_directories(args.output_root, time_stamp_string)

        # Save environment configuration
        training_file_path = save_training_configuration(
            destination_output_dir,
            environment_name,
            str(training_steps),
            env_kwargs,
            args.output_root,
            destination_source_code_dir
        )

        # Train the model
        train_model(env_fn, destination_output_dir, model_file_name, training_steps, env_kwargs, training_file_path)

    except Exception as e:
        logging.error(f"An error occurred during the training process: {e}")
        raise
