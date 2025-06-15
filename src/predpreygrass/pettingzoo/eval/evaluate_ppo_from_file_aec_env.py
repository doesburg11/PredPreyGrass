"""
Instructions:for evaluating a ppo trained model from a file:
- evaluation can be done with or without watching the grid
- navigate with linux mint file-explorer to your local directory as defined in:
  `src/predpreygrass/global_config.py`
- note that the entire file/directory structure of the project is copied
  during training to the defined local directory
- go to the directory with the appropriate time stamp of training and go
  down the directory tree:
  predpreygrass/pettingzoo/eval/evaluate_ppo_from_file.py
- right mouseclick "evaluate_ppo_from_file_aec_env.py" and:
- select "Open with"
- select "Visual Studio Code"
- select "Run without debugging"
- results can be found in: /[time_stamp]/output/
"""
# external libraries
import os
import sys
from os.path import dirname as up

from predpreygrass.pettingzoo.envs import predpreygrass_aec_v0
from predpreygrass.pettingzoo.config.config_predpreygrass import env_kwargs, training_steps_string
from predpreygrass.pettingzoo.eval.utils.evaluator import Evaluator

# Add the parent directory to sys.path
sys.path.append(up(up(__file__)))


if __name__ == "__main__":
    watch_grid_model = env_kwargs["watch_grid_model"]
    num_episodes = env_kwargs["num_episodes"]
    env_fn = predpreygrass_aec_v0
    environment_name = str(env_fn.raw_env.metadata["name"])
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    evaluation_directory = os.path.dirname(os.path.abspath(__file__))
    print("-----------------------------------------------------------------------------")
    print("Evaluation_directory: ", evaluation_directory)
    destination_source_code_dir = up(up(__file__))  # up 2 levels in directory tree
    print("Destination_source_code_dir: ", destination_source_code_dir)
    print("-----------------------------------------------------------------------------")
    output_directory = destination_source_code_dir + "/output/"
    loaded_policy = output_directory + model_file_name
    render_mode = "human" if watch_grid_model else None

    # Create an instance of the Evaluator class
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

    # Call the eval method to perform the evaluation
    evaluator.parallel_wrapped_aec_env_training_aec_evaluation()
