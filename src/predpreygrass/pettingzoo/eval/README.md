## Centralized training and decentralized execution

During training, the Predators and Prey agents share a single PPO network. The training process has access to the observations, actions, and rewards of all agents, enabling a single coordinated policy optimization process. During evaluation, each agent receives a unique slice of the state as input, tailored to its role (e.g., predators observe prey positions; prey focus on predator positions and grass), enabling decentralized execution.

## evaluate_ppo_from_file_aec_env.py

The `eval` function evaluates the trained model. Notably, it uses the AEC (Agent Environment Cycle) API during evaluation, which differs from the parallel API used during training. This requires handling individual steps and actions for each agent sequentially within each environment cycle.

## Instructions

To evaluate a ppo trained and saved model:
- Evaluation can be done with or without watching the grid
instructions set in config_file ```predpreygrass/pettingzoo/config/config_predpreygrass.py```
- Navigate with linux mint file-explorer to your ```local_output_root``` defined in config_file
- go to its subdirectory with the appropriate time stamp of training.
- note that the entire file/directory structure of the project is copied into ```local_output_root/[time_stamp]/``` for reuse and
  analysis and go down the directory tree:
  ```local_output_root/[time_stamp]/eval/evaluate_ppo_from_file_aec_env.py```
- right mouseclick evaluate_ppo_rom_file.py and:
- select "Open with"
- select "Visual Studio Code"
- select "Run without debugging"
- results are saved in: ```local_output_root/[time_stamp]/output/```
