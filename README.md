[![Python 3.11.7](https://img.shields.io/badge/python-3.11.7-blue.svg)](https://www.python.org/downloads/release/python-3117/)
[![PettingZoo version dependency](https://img.shields.io/badge/PettingZoo-v1.24.3-blue)]()
[![MOMAland version dependency](https://img.shields.io/badge/MOMAland-v0.1.1-blue)]()



</br>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/readme/mo_predpreygrass.png" width="700" height="80"/> 
</p>
</br>

## Predator-Prey-Grass multi-objective multi-agent reinforcement learning (MOMARL)
Predator-Prey-Grass gridworld deploying multi-objective and multi-agent environments with dynamic deletion and spawning of partially observant agents. Utilizing Farama's [PettingZoo](https://pettingzoo.farama.org/) and [Momaland](https://momaland.farama.org/).

</br>
</br>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="1000" height="200"/>
</p>

## The environments
[predpregrass_base.py](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/envs/base_env/predpreygrass_base.py): A (single-objective) multi-agent reinforcement learning (MARL) environment, [trained](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/train/train_ppo.py) and [evaluated](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/eval/evaluate_ppo_from_file.py) using [Proximal Policy Optimization (PPO)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). Learning agents Predators (red) and Prey (blue) both expend energy moving around, and replenish it by eating. Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. In the base case for simplicity, the agents obtain all the energy from the eaten Prey or Grass. Predators die of starvation when their energy is zero, Prey die either of starvation or when being eaten by a Predator. The agents asexually reproduce when energy levels of learning agents rise above a certain treshold by eating. Learning agents, learn to execute movement actions based on their partial observations (transparent red and blue squares respectively) of the environment to maximize cumulative reward. The single objective rewards (stepping, eating, dying and reproducing) are naively summed and can be adjusted in the [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/config/config_predpreygrass.py) file. 


## Emergent Behaviors
Training the single objective environment [predpregrass_base.py](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/envs/base_env/predpreygrass_base.py) with the PPO algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. In the above displayed MARL example, rewards for learning agents are solely obtained by reproduction. So all other reward options are set to zero in the [environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/config/config_predpreygrass.py). Despite these relative sparse reward structure, maximizing these rewards results in elaborate emerging behaviors such as: 
- Predators hunting Prey 
- Prey finding and eating grass 
- Predators hovering around grass to catch Prey 
- Prey trying to escape Predators

Moreover, these learning behaviors lead to more complex emergent dynamics at the ecosystem level. The trained agents are displaying a classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time:

<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/readme/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>

More emergent behavior and findings are described [on our website](https://www.behaviorpatterns.info/predator-prey-grass-project/).


## Installation

**Editor used:** Visual Studio Code 1.93.1 on Linux Mint 21.3 Cinnamon

1. Clone the repository: 
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
   - Choose environment: Conda 
   - Choose interpreter: Python 3.11.7
   - Open a new terminal
   - ```bash
     pip install -e . 
     ```
 3. Install the following requirements:  
    -   ```bash 
        pip install supersuit==3.9.3 
        ```
    -   ```bash 
        pip install tensorboard==2.18.0 
        ```
    -   ```bash 
        pip install stable-baselines3[extra] 
        ```
    - ```bash
      conda install -c conda-forge gcc=12.1.0
         ```
    
## Getting started

### Visualize a random policy
In Visual Studio Code run:
```predpreygrass/single_objective/eval/evaluate_random_policy.py```
</br>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass_random.gif" width="1000" height="200"/>
</p>


### Training and visualize trained model using PPO from stable baselines3

Adjust parameters accordingly in:

```predpreygrass/single_objective/config/config_predpreygrass.py```

In Visual Studio Code run:

```predpreygrass/single_objective/train/train_ppo.py```

To evaluate and visualize after training follow instructions in:

```predpreygrass/single_objective/eval/evaluate_ppo_from_file.py```

[UNDER (RE)CONSTRUCTION] Batch training and evaluating in one go:

```predpreygrass/single_objective/eval/_parameter_variation_train_ppo_and_evaluate.py```

## References

- [Terry, J and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and others. Pettingzoo: Gym for multi-agent reinforcement learning. 2021-2024](https://pettingzoo.farama.org/)    
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [MOMAland: A Set of Benchmarks for Multi-Objective Multi-Agent Reinforcement Learning (MOMARL). Florian Felten and Umut Ucak and Hicham Azmani and Gao Peng and Willem Röpke and Hendrik Baier and Patrick Mannion and Diederik M. Roijers and Jordan K. Terry and El-Ghazali Talbi and Grégoire Danoy and Ann Nowé and Roxana Rădulescu](https://momaland.farama.org/)
- [Multi-Objective Multi-Agent Decision Making: A Utility-based Analysis and Survey](https://arxiv.org/abs/1909.02964)
- [A Practical Guide to Multi-Objective Reinforcement Learning and Planning](https://arxiv.org/abs/2103.09568)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)



