## Getting started with the PettingZoo/SB3 framework

<p align="center">
    <img src="../../../assets/images/readme/predpreygrass.png" width="700" height="80"/> 
</p>

### Visualize a random policy with the PettingZoo/SB3 solution
In Visual Studio Code run:
```src/predpreygrass/pettingzoo/eval/evaluate_random_policy.py```
</br>
<p align="center">
    <img src="../../../assets/images/gifs/predpreygrass_random.gif" width="1000" height="200"/>
</p>

Or in a a terminal:

-   ```bash 
    python src/predpreygrass/pettingzoo/eval/evaluate_random_policy.py 
    ```
### Training and visualize trained model using PPO from stable baselines3

Adjust parameters accordingly in:

[```src/predpreygrass/pettingzoo/config/config_predpreygrass.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/config/config_predpreygrass.py)

In the Visual Studio GUI Code run:

[```src/predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py)

Or in a a terminal:

-   ```bash 
    python predpreygrass/pettingzoo/config/config_predpreygrass.py 
    ```


To evaluate and visualize after training follow instructions in:

[```src/predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py)

Batch training and evaluating in one go:

[```src/predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py)


### How SB3 PPO is used in the Predator-Prey-Grass Multi-Agent Setting

#### 1. PettingZoo AEC to Parallel Conversion
- The environment is initially implemented as an **Agent-Environment-Cycle (AEC) environment** using **PettingZoo** ([`predpregrass_aec.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/envs/predpreygrass_aec.py) which inherits from [`predpregrass_base.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/envs/predpreygrass_base.py)).
- It is wrapped and converted into a **Parallel Environment** using `aec_to_parallel()` inside [`trainer.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/train/utils/trainer.py).
- This conversion enables multiple agents to take actions simultaneously rather than sequentially.

#### 2. Treating Multi-Agent Learning as a Single-Agent Problem
- SB3 PPO expects a **single-agent Gymnasium-style environment**.
- The converted parallel environment **stacks observations and actions for all agents**, making it appear as a single large observation-action space.
- PPO then treats the multi-agent problem as a **centralized learning problem**, where all agents share one policy.

#### 3. Performance Optimization with Vectorized Environments
- The environment is further wrapped using **SuperSuit**:
  ```python
  env = ss.pettingzoo_env_to_vec_env_v1(env)
  env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cores, base_class="stable_baselines3")
  ```
- This enables running multiple instances of the environment in parallel, significantly improving training efficiency.
- The training process treats the multi-agent setup as a **single centralized policy**, where PPO learns from the collective experiences of all agents.
