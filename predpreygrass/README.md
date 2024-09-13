### Stablebaseline3 PPO in multi-agent setting

Stable Baselines3 (SB3) is designed primarily for single-agent environments, but can also be implemented in multi-agent setting. This is done by wrapping the multi-agent environment to appear as single-agent. This involves:

- Creating a wrapper that aggregates observations from all agents into a single observation space.
- Combining the actions and rewards from all agents into single action and reward vectors that SB3 can process.

The SuperSuit library is used to convert the PettingZoo environment into a format compatible with Stable Baselines3. 

### train_ppo.py

In the file `train_ppo.py` the conversion is done using the `ss.pettingzoo_env_to_vec_env_v1` function which translates the PettingZoo environment into a vectorized environment. This makes the multi-agent environment appear as a single, though high-dimensional, environment to the SB3 model. The environments are further processed with `ss.concat_vec_envs_v1`, which concatenates several copies of the environment into a single environment to enable more efficient training using multiple CPU cores. The number of copies can be adjusted accordingly. In this case 8 copies are concatenated to utilize 8 CPU cores. With for instance [`config_predpreygrass.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/config/config_predpreygrass.py) this creates an action and reward vector of length 336 (= 18 possible Predators + 24 possible Prey times 8 copies). These vectors can be optionally displayed during training with `SampleLoggerCallback`.

The function `train` sets up the environment and uses the PPO algorithm from Stable Baselines3 with an MLP policy to train a single model that handles actions and observations for all agents in the environment. This is done in a parallel manner where the environment expects all agents to act simultaneously. In short, although the environment is multi-agent, the training loop treats it as a high-dimensional single-agent environment due to the transformations applied. This means the model learns to handle inputs and outputs for all agents simultaneously.

In the file, training over multiple environment parameters can be utilized, by setting `parameter_variation` to `True` and defining the parameter and scenarios. Training results and the configuration are saved to local files from where the `evaluate_from_file.py` can be run. 

### evaluate_from_file.py

The `eval` function evaluates the trained model. Notably, it uses the AEC (Agent Environment Cycle) API during evaluation, which differs from the parallel API used during training. This requires handling individual steps and actions for each agent sequentially within each environment cycle.
