[![Python 3.11.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![PettingZoo version dependency](https://img.shields.io/badge/PettingZoo-v1.24.3-blue)]()
[![Stable Baselines3](https://img.shields.io/github/v/release/DLR-RM/stable-baselines3?label=Stable-Baselines3)](https://github.com/DLR-RM/stable-baselines3/releases)
[![RLlib](https://img.shields.io/badge/RLlib-v2.43.0-blue)](https://docs.ray.io/en/latest/rllib/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doesburg11/PredPreyGrass/blob/main/predpreygrass.ipynb)

# Artificial Life and Intelligence
### Multi-Agent Reinforcement Learning and Artificial Selection

<p align="center">
    <img src="./assets/images/gifs/rlllib_evaluation_250.gif" width="300" height="300"/>
</p>

## Overview
This project explores emergent behaviors in a multi-agent dynamic ecosystem of predators, prey, and regenerating grass. At its core lies a grid-world simulation where agents are not just *trained*—they are *born*, *age*, *reproduce*, and even *mutate*.

We combine **multi-agent reinforcement learning** (MARL) with **evolutionary dynamics** to investigate the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via MARL algorithms). Agents differ by speed, vision, energy metabolism, and decision policies—offering ground for open-ended adaptation.

This environment doesn't just support pre-programmed behavior—it gives rise to **emergent population dynamics** through mutation, inheritance, and selection.

## Key Features

- **Nature vs. Nurture**: Agents inherit speed-based traits genetically, but refine behavior through learned policy optimization.
- **Energy-Based Life Cycle**: Movement, hunting, and grazing consume energy—agents must balance survival, reproduction, and exploration.
- **Multi-Policy Training**: Each agent type (e.g., speed-1 predator, speed-2 prey) is (decentralized) trained via its own policy module.
- **Gridworld Ecology**: Agents observe their local neighborhood with species-specific ranges; prey seek grass, predators hunt prey.
- **Procedural Regeneration**: Grass regrows over time; life and death shape a shifting ecological landscape.
- **Mutation and Selection**: When agents reproduce, they may randomly mutate (e.g., switching speed class). This introduces a natural selection (or more precise: *artificial* selectio*) pressure shaping the agent population over time.
- **Visual Diagnostics**: Integrated tools for population charts, evolution visualizers, and grid-based renderings.

## Starting point: MARL applied to a Predator-Prey-Grass environment

Displayed on top is a Predator-Prey-Grass gridworld deploying a multi-agent environment with dynamic deletion and spawning of partially observant agents. Learning agents Predators (red) and Prey (blue) both sequentially expend energy moving around, and replenish it by eating. Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. The agents obtain all the energy from the eaten resource. Predators die of starvation when their energy is run out, Prey die either of starvation or when being eaten by a Predator. Both learning agents asexually reproduce when energy levels exceed a certain threshold (by eating). In the base configuration, newly created agents are placed at random over the entire gridworld. Learning agents learn to move based on their partial observations of the environment.

### Centralized versus decentralized training
The described environment and training concept is implemented in **centralized training** as well as **decentralized training** utilizing two separate framework solutions: on the one hand PettingZoo in combination with StableBaseline3 for centralized training and on the other hand the RLlib framework for decentralized training.

<table align="center" width="100%">
  <tr>
    <th width="30%" align="left">Framework</th>
    <th width="70%" align="left">Solution</th>
  </tr>
  <tr>
    <td align="center">
      <a href="https://pettingzoo.farama.org/">
        <img src="./assets/images/icons/pettingzoo.png" alt="PettingZoo" height="40">
      </a><br>
      <a href="https://pettingzoo.farama.org/">PettingZoo Environment</a>
    </td>
    <td align="left">
      Single network for all agents (centralized learning) utilizing <a href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html">
        <strong>Stable-Baselines3 PPO Algorithm</strong>
      </a> applied to a customized PettingZoo multi-agent environment (<a href="https://pettingzoo.farama.org/api/aec/"><strong>AECEnv</strong></a>).
  </tr>
  <tr>
    <td align="center">
      <a href="https://docs.ray.io/en/master/rllib/index.html">
        <img src="./assets/images/icons/rllib.png" alt="RLlib" height="40">
      </a><br>
      <a href="https://docs.ray.io/en/master/rllib/index.html">RLlib (New API Stack) multi-agent environment </a>
    </td>
    <td align="left">
      Dual network for Predator and Prey seperately (decentralized learning) utilizing 
      <a href="https://docs.ray.io/en/master/rllib/rllib-algorithms.html#ppo">
        <strong>native RLlib PPO Solution</strong>
      </a>
      applied to the RLlib new API stack multi-agent environment (<a href="https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html"><strong>MultiAgentEnv</strong></a>).
    </td>
  </tr>
</table>

### Centralized training: Pred-Prey-Grass MARL with PettingZoo/SB3 PPO 
</br>

<p align="center">
    <img src="./assets/images/readme/predpreygrass.png" width="700" height="80"/> 
    
</p>

</br>

<br>
<p align="center">
    <img src="./assets/images/gifs/predpreygrass.gif" width="1000" height="200"/>
</p>

### Configuration of centralized training
The MARL environment [`predpregrass_base.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/envs/predpreygrass_base.py) is implemented using **PettingZoo**, and the agents are trained using **Stable-Baselines3 (SB3) PPO**. Essentially this solution demonstrates how SB3 can be adapted for MARL using parallel environments and centralized training. Rewards (stepping, eating, dying and reproducing) are aggregated and can be adjusted in the [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/config/config_predpreygrass.py) file. Basically, Stable Baseline3 is originally designed for single-agent training. This means in this solution, training utilizes only one unified network for Predators as well Prey. 

### How SB3 PPO is used in the Predator-Prey-Grass Multi-Agent Setting

#### 1. PettingZoo AEC to Parallel Conversion
- The environment is initially implemented as an **Agent-Environment-Cycle (AEC) environment** using **PettingZoo** ([`predpregrass_aec.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/envs/predpreygrass_aec.py) which inherits from [`predpregrass_base.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/envs/predpreygrass_base.py)).
- It is wrapped and converted into a **Parallel Environment** using `aec_to_parallel()` inside [`trainer.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/train/utils/trainer.py).
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


### Decentralized tarining: Pred-Prey-Grass MARL with RLlib new API stack 


<p align="center">
    <img src="./assets/images/gifs/rlllib_evaluation_250.gif" width="300" height="300"/>
</p>

### Configuration of decentralized training
Obviously, using only one network has its limitations as Predators and Prey lack true specialization in their training. The RLlib new API stack framework is able to circumvent this limitation, albeit at the cost of considerable more compute time. The environment dynamics of the RLlib environment ([`predpregrass_rllib_env.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/rllib/predpreygrass_rllib_env.py)) are largely the same as in the PettingZoo environment. However, newly spawned agents are placed in the vicinity of the parent, rather than randomly spawned in the entire gridworld. The implementation under-the-hood of the setup is somewhat different, utilizing more array lists to store agent data rather than implementing a seperate agent class. This is largely a result of experimentation with compute time of the `step` function. Similarly as in the PettingZoo environment, rewards can be adjusted in a seperate [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/rllib/config_env.py) file. 

Training is applied in accordance with the RLlib new API stack protocol. The training configuration is more out-of-the-box than the PettingZoo/SB3 solution, but nevertheless is much more applicable to MARL in general and especially decentralized training.

<p align="center">
    <img src="./assets/images/readme/multi_agent_setup.png" width="400" height="150"/>
</p>

A key difference of the decentralized training solution with the centralized training solution is that the concurrent agents become part of the environment rather than being part of a combined "super" agent. Since, the environment of the centralized training solution consists only of static grass objects, the environment complexity of the decentralized training solution is dramatically increased. This is probably one of the reasons that training time of the RLlib solution is a multiple of the PettingZoo/SB3 solution. This is however a hypothesis and is subject to future investigation.  

### Emergent Behaviors
Training the single objective environment [`predpregrass_base.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/envs/predpreygrass_base.py) with the SB3 PPO algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. In the above displayed MARL example, rewards for learning agents are solely obtained by reproduction. So all other reward options are set to zero in the [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/config/config_predpreygrass.py). Despite this relativily sparse reward structure, maximizing these rewards results in elaborate emerging behaviors such as: 
- Predators hunting Prey 
- Prey finding and eating grass 
- Predators hovering around grass to catch Prey 
- Prey trying to escape Predators

Moreover, these learning behaviors lead to more complex emergent dynamics at the ecosystem level. The trained agents are displaying a classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time:

<p align="center">
    <img src="./assets/images/readme/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>

More emergent behavior and findings are described [on our website](https://www.behaviorpatterns.info/predator-prey-grass-project/).

### Introducing mutation with reproducing agents

The environment described above, wether centralized or decentralized trained, esentially optimizes policies for a fixed policy task for both predator an prey agent groups throughout the entire episode of the environment. To introduce variability in an agent policy (and therefore some element of open-endedness) we introduce for both predator and prey a low-speed and high-speed agent variant. A low-speed agent can move within its [Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood), but a high-speed agent can move within its extended Moore neighborhood (with range *r*=2). Consequently, the high-speed agent can move faster across the gridworld, as depicted below.

<p align="center">
    <img src="./assets/images/readme/high-low-speed-agent.png" width="450" height="270"/>
    <p align="center"><b>Action spaces of low- and high-speed agents</b></p>
</p>


The environment setup is changed too make mutations possible with the reproduction of a agents. When reproduction occurs, there is a small change (say 2.5%) of mutating from a low-speed agent to a high-speed agent (or vice versa). When all 4 agents (low-speed-predator, high-speed-predator, low-speed-prey, high-speed-prey) are decentralized trained, it appears that 



 it appears that when we start with only low-speed-predators and low-speed-prey in our evaluation, eventually the population of low-speed agents is utlimately replaced by high-speed agents for predators as well as prey.

This crowding out of low-speed agents occurrs **without any manual reward shaping** or explicit encouragement. High-speed agents—once introduced via mutation—are more successful at acquiring energy and reproducing. As a result, they overtake the population at some point during the evaluation.

This is a clear example of **natural selection** within an artificial system:  
- **Variation**: Introduced by random mutation of inherited traits (speed class).  
- **Inheritance**: Agents retain behavior linked to their speed class via pre-trained policies.  
- **Differential Fitness**: Faster agents outperform slower ones under the same environmental constraints.  
- **Selection**: Traits that increase survival and reproduction become dominant.

### Co-Evolution and the Red Queen Effect

The mutual shift of both **prey and predator populations toward high-speed variants** reflects also a classic **Red Queen dynamic**: each species evolves not to get ahead absolutely, but also to keep up with the other. Faster prey escape better, which in turn favors faster predators. This escalating cycle is a hallmark of **co-evolutionary arms races**—where the relative advantage remains constant, but the baseline performance is continually ratcheted upward.

This ecosystem, therefore, is not only an instance of artificial selection—it’s also a model of **evolution in motion**, where fitness is relative, and adaptation is key.

Notably, agents in our system lack direct access to each other’s heritable traits such as speed class. Observations are limited to localized energy maps for predators, prey, and grass, with no explicit encoding of whether an observed agent is fast or slow. Despite this, we observe a clear evolutionary shift toward higher-speed phenotypes in both predator and prey populations. This shift occurs even when high-speed variants are initially absent and must arise through rare mutations, suggesting that selection is driven not by trait recognition but by differential survival and reproductive success. Faster agents outperform their slower counterparts in the competitive landscape created by evolving opponents, leading to a mutual escalation in speed. This dynamic constitutes an implicit form of co-evolution consistent with the Red Queen hypothesis: species must continuously adapt, not to gain an absolute advantage, but merely to maintain relative fitness in a co-adaptive system.

The Red Queen hypothesis (Van Valen, 1973) posits that organisms must continuously adapt and evolve not necessarily to gain a reproductive advantage, but to keep pace with the evolution of interacting species within a changing environment. It is named after the Red Queen’s remark in Through the Looking-Glass by Lewis Carroll: “It takes all the running you can do, to stay in the same place.” In ecological and evolutionary systems, this concept often manifests as co-evolutionary arms races—for instance, between predators and prey—where reciprocal selective pressures drive ongoing adaptation without stable equilibria.


## Installation

**Editor used:** Visual Studio Code 1.99.0 on Linux Mint 22.0 Cinnamon

1. Clone the repository: 
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
   - Choose environment: Conda 
   - Choose interpreter: Python 3.11.11 or higher
   - Open a new terminal
   - ```bash
     pip install -e .
     ```
 3. Install the following requirements:  
    -   ```bash 
        pip install pettingzoo==1.24.3
 
        ```
    -   ```bash 
        pip install stable-baselines3[extra]==2.5.0
 
        ```
    -   ```bash
        conda install -y -c conda-forge gcc=12.1.0
        ```    
    -   ```bash 
        pip install supersuit==3.9.3 
        ```
    -   ```bash 
        pip install ray[rllib]==2.43.0
        ```
    -   ```bash 
        pip install tensorboard==2.18.0 
        ```
    
## Getting started with the PettinfZoo/SB3 solution

### Visualize a random policy with the PettingZoo/SB3 solution
In Visual Studio Code run:
```predpreygrass/pettingzoo/eval/evaluate_random_policy.py```
</br>
<p align="center">
    <img src="./assets/images/gifs/predpreygrass_random.gif" width="1000" height="200"/>
</p>


### Training and visualize trained model using PPO from stable baselines3

Adjust parameters accordingly in:

[```predpreygrass/pettingzoo/config/config_predpreygrass.py```](hhttps://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/config/config_predpreygrass.py)

In Visual Studio Code run:

[```predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py```](hhttps://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/train/train_sb3_ppo_parallel_wrapped_aec_env.py)

To evaluate and visualize after training follow instructions in:

[```predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/eval/evaluate_ppo_from_file_aec_env.py)

Batch training and evaluating in one go:

[```predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py```](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/pettingzoo/eval/parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py)

## References

- [Terry, J and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and others. Pettingzoo: Gym for multi-agent reinforcement learning. 2021-2024](https://pettingzoo.farama.org/)    
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)



