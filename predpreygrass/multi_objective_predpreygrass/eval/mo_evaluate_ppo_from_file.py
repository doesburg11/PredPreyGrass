"""
- to evaluate a ppo trained and saved model
- evaluation can be done with or without watching the grid
"""
"""
instructions:
- navigate with linux mint file-explorer to your local directory 
  (definedd in "env/_predpreygraas_v0/config/config_predpreygrass.py").
- note that the entire files/directories/configurations of the project 
  are copied and reproduced to the defined local directory
- go to the directory with the appropriate time stamp of training
- right mouseclick "mo_evaluate_from_file.py" and:
- select "Open with"
- select "Visual Studio Code" (or default VS Code for .py files)
- select "Run" (in taskbar of Visual Studio Code)
- select "Run without debugging"
"""
# discretionary libraries
from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
)
from predpreygrass.optimizations.mo_predpreygrass_v0.evaluation.utils.evaluator import Evaluator

# external libraries
import os
from os.path import dirname as up


if __name__ == "__main__":
    env_fn = mo_predpreygrass_v0
    environment_name = str(env_fn.raw_env.metadata['name'])
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    evaluation_directory = os.path.dirname(os.path.abspath(__file__))
    destination_source_code_dir = up(up(up(up(__file__))))  # up 4 levels in directory tree
    output_directory = destination_source_code_dir +"/output/"
    loaded_policy = output_directory + model_file_name
    # input from so_config_predpreygrass.py
    watch_grid_model = env_kwargs["watch_grid_model"]
    eval_model_only = not watch_grid_model
    num_episodes = env_kwargs["num_episodes"] 
    training_steps = int(training_steps_string)

    render_mode = "human" if watch_grid_model else None

    # Create an instance of the Evaluator class
    evaluator = Evaluator(
        env_fn,
        output_directory, # destination_output_dir,
        loaded_policy,
        destination_source_code_dir, # destination_root_dir,
        render_mode,
        training_steps_string,
        destination_source_code_dir, # destination_source_code_dir,
        **env_kwargs
    )
    # Call the eval method to perform the evaluation
    evaluator.eval()
