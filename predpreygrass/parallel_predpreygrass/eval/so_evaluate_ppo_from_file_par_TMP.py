# discretionary libraries
from predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.envs._so_predpreygrass_v0.config.config_predpreygrass import (
    env_kwargs,
    training_steps_string,
)
from predpreygrass.optimizations.so_predpreygrass_v0.evaluation.utils.evaluator_par_TMP import Evaluator

# external libraries
import os
from os.path import dirname as up


if __name__ == "__main__":
    env_fn = predpreygrass_parallel_v0 
    is_aec_evaluated = False
    environment_name = str(env_fn.parallel_env.metadata['name'])
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    evaluation_directory = os.path.dirname(os.path.abspath(__file__))
    destination_source_code_dir = up(up(up(up(__file__))))  # up 4 levels in directory tree
    output_directory = destination_source_code_dir + "/output/"
    loaded_policy = output_directory + model_file_name
    watch_grid_model = env_kwargs["watch_grid_model"]
    num_episodes = env_kwargs["num_episodes"] 
    training_steps = int(training_steps_string)
    render_mode = "human" if watch_grid_model else None
    # Create an instance of the Evaluator class
    evaluator = Evaluator(
        env_fn,
        output_directory, 
        loaded_policy,
        destination_source_code_dir, 
        render_mode,
        training_steps_string,
        destination_source_code_dir, 
        **env_kwargs
    )

    # Call the eval method to perform the evaluation
    if is_aec_evaluated:
        evaluator.aec_evaluation()
    else:
        evaluator.parallel_evaluation()
