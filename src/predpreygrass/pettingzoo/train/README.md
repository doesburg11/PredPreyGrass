## Centralized Training

The Predators and Prey agents share a single PPO network during training. The training process has access to the observations, actions, and rewards of all agents, enabling a single coordinated policy optimization process.

## Stable Baselines3 PPO in multi-agent setting
The training process utilizes the PPO algorithm from the external Stable Baselines3 library (SB3), which is designed primarily for single-agent environments, but can also be implemented in multi-agent setting. This is done by wrapping the multi-agent environment to appear as a single-agent. This involves:

- Creating a wrapper that aggregates observations from all agents into a single observation space.
- Combining the actions and rewards from all agents into single action and reward vectors that SB3 can process.

The SuperSuit library is used to convert the PettingZoo environment into a format compatible with SB3. With both `train_ppo_parallel_wrapped_aec_env.py` and `train_ppo_unwrapped_parallel_env.py` this conversion is done using the SuperSuit `pettingzoo_env_to_vec_env_v1` function which translates the PettingZoo environment into a vectorized environment. This makes the multi-agent environment appear as a single, though high-dimensional, environment to the SB3 model. The environments are further processed with `concat_vec_envs_v1`, which concatenates several copies of the environment into a single environment to enable more efficient training using multiple CPU cores. The number of copies can be adjusted accordingly. In this case 8 copies are concatenated to utilize 8 CPU cores. This is done in a parallel manner where the environment expects all agents to act simultaneously. In short, although the environment is multi-agent, the training loop treats it as a high-dimensional single-agent environment due to the transformations applied. This means the model learns to handle inputs and outputs for all agents simultaneously.


## PPO parallized training

The PPO training is performed on a parallized version of the Predator-Prey-Grass environment in two different fashions:

1. AEC environment wrapped to a parallel environment ('wrapped'): `train_ppo_parallel_wrapped_aex_env.py`
2. Direct parallel environment implementation ('unwrapped'): `train_ppo_unwarpped_parallel_env.py`

The reasons for training in parallel are for efficiency and compatibility reasons:

**Efficiency**: Parallel environments allow simultaneous execution of all agents’ actions, making training faster and more scalable for environments with many agents.
**Alignment with PPO**: The PPO implementations of SB3 is designed for parallel environments, which expects simultaneous actions for all agents at each time step.


## train_ppo_parallel_wrapped_aec_env.py

The function `train_parallel_wrapped_aec_env` from the `Trainer` class wraps the AEC environment to a parallel environment and uses the PPO algorithm from SB3 with an MLP policy to train a single model that handles actions and observations for all agents in the environment. This parallel training expects all agents to act simultaneously in the environment. In short, although the environment is multi-agent, the training loop treats it as a high-dimensional single-agent environment due to the transformations applied. This means the model learns to handle inputs and outputs for all agents simultaneously.
