[![PettingZoo version dependency](https://img.shields.io/badge/PettingZoo-v1.24.3-blue)]()
[![Stable Baselines3](https://img.shields.io/github/v/release/DLR-RM/stable-baselines3?label=Stable-Baselines3)](https://github.com/DLR-RM/stable-baselines3/releases)
## Legacy framework: PettingZoo & Stable Baselines3 framework

### Centralized training, decentralized evaluation
The MARL environment [`predpregrass_base.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/envs/predpreygrass_base.py) is implemented using **PettingZoo**, and the agents are trained using **Stable-Baselines3 (SB3) PPO**. Essentially this solution demonstrates how SB3 can be adapted for MARL using parallel environments and **centralized training**. Rewards (stepping, eating, dying and reproducing) are aggregated and can be adjusted in the [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/config/config_predpreygrass.py) file. Stable Baseline3 is originally designed for single-agent training. This means that in this solution, training utilizes only one unified network for Predators as well Prey. See further below how SB3 PPO is used in this centralilzed trained Predator-Prey-Grass multi-agent setting.

<p align="center">
    <img src="../../../assets/images/readme/predpreygrass.png" width="700" height="80"/>
</p>


<p align="center">
    <b>Random policy Predator-Prey-Grass PettingZoo environment</b></p>
    <img align="center" src="../../../assets/images/gifs/predpreygrass_random.gif" width="1000" height="200"/>
</p>

#### Random policy with the PettingZoo framework
- [`src/predpreygrass/pettingzoo/eval/evaluate_random_policy.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/eval/evaluate_random_policy.py)

#### Training model using PPO from stable baselines3
- [```src/predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py)


#### Configuration environment parameters
- [`src/predpreygrass/pettingzoo/config/config_predpreygrass.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/config/config_predpreygrass.py)


#### Evaluate and visualize trained model
- [```src/predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py)

#### Batch training and evaluating in one go:
- [```src/predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py)


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

### Centralized training and decentralized evaluation

<p align="center">
    <b>Predator-Prey-Grass PettingZoo environment centralized trained using SB3's PPO</b></p>
    <img src="../../../assets/images/gifs/predpreygrass.gif" width="1000" height="200"/>
</p>

## Emergent Behaviors
Training the single objective environment [`predpregrass_base.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/envs/predpreygrass_base.py) with the SB3 PPO algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. In the above displayed MARL example, rewards for learning agents are solely obtained by reproduction. So all other reward options are set to zero in the [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/pettingzoo/config/config_predpreygrass.py). Despite this relativily sparse reward structure, maximizing these rewards results in elaborate emerging behaviors such as:
- Predators hunting Prey
- Prey finding and eating grass
- Predators hovering around grass to catch Prey
- Prey trying to escape Predators

Moreover, these learning behaviors lead to more complex emergent dynamics at the ecosystem level. The trained agents are displaying a classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time:

<p align="center">
    <img src="../../../assets/images/readme/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>
