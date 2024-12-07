### Stablebaseline3 PPO in multi-agent setting

Stable Baselines3 (SB3) is designed primarily for single-agent environments, but can also be implemented in multi-agent setting. This is done by wrapping the multi-agent environment to appear as single-agent. This involves:

- Creating a wrapper that aggregates observations from all agents into a single observation space.
- Combining the actions and rewards from all agents into single action and reward vectors that SB3 can process.

The SuperSuit library is used to convert the PettingZoo environment into a format compatible with Stable Baselines3. With both `train_ppo_parallel_wrapped_aec_env.py` and `train_ppo_unwrapped_parallel_env.py` this conversion is done using the `ss.pettingzoo_env_to_vec_env_v1` function which translates the PettingZoo environment into a vectorized environment. This makes the multi-agent environment appear as a single, though high-dimensional, environment to the SB3 model. The environments are further processed with `ss.concat_vec_envs_v1`, which concatenates several copies of the environment into a single environment to enable more efficient training using multiple CPU cores. The number of copies can be adjusted accordingly. In this case 8 copies are concatenated to utilize 8 CPU cores. 

### train_ppo_parallel_wrapped_aec_env.py

The function `train_parallel_wrapped_aec_env` from the `Trainer` class wraps the AEC environment to a parallel environment and uses the PPO algorithm from Stable Baselines3 with an MLP policy to train a single model that handles actions and observations for all agents in the environment. This parallel training expects all agents to act simultaneously in the environment. In short, although the environment is multi-agent, the training loop treats it as a high-dimensional single-agent environment due to the transformations applied. This means the model learns to handle inputs and outputs for all agents simultaneously.
 

### train_ppo_unwrapped_parallel_env.py

The function `train_ppo_unwrapped_parallel_env` from the `Trainer` class utilizes the raw parallel environment and uses the PPO algorithm from Stable Baselines3 with an MLP policy to train a single model that handles actions and observations for all agents in the environment similarilly as the AEC environment which is wrapped to a parallel environment in `train_ppo_parallel_wrapped_aec_env.py`.