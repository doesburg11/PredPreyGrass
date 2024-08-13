[![Python 3.11.7](https://img.shields.io/badge/python-3.11.7-blue.svg)](https://www.python.org/downloads/release/python-3117/)
[![PettingZoo version dependency](https://img.shields.io/badge/PettingZoo-v1.24.3-blue)]()
[![API test](https://img.shields.io/badge/API_test-passed-brightgreen)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Wiki](https://img.shields.io/badge/Wiki-background-green)]()
</br>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/readme/predpreygrass.png" width="700" height="80"/> 
</p>
</br>

## The environment
A multi-agent reinforcement learning (MARL) environment, trained using Proximal Policy Optimization (PPO). Learning agents Predators (red) and Prey (blue) both expend energy moving around, and replenish it by eating. Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. In the base case for simplicity, the agents obtain 100% of the energy from the eaten Prey or Grass. However, in the 'real world' this is much less because the [ecological efficiency](https://en.wikipedia.org/wiki/Ecological_efficiency) is only around 10% in most cases. Predators die of starvation when their energy is zero, Prey die either of starvation or when being eaten by a Predator. The agents asexually reproduce when energy levels of learning agents rise above a certain treshold by eating. Learning agents, learn to execute movement actions based on their partial observations (transparent red and blue squares respectively) of the environment to maximize cumulative reward.
</br>
</br>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="1000" height="200"/>
</p>


## Emergent Behaviors
This algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. In the above example, rewards for learning agents are only obtained by reproduction. However, maximizing these rewards results in emerging behaviors such as: 1) Predators hunting Prey, 2) Predators hovering around grass to catch Prey and 3) Prey trying to escape Predators. These emerging behaviors lead to more complex dynamics at the ecosystem level. The trained agents are displaying a classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time. This learned outcome is not obtained with a random policy:

<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/readme/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>

More emergent behavior and findings are described [on our website](https://www.behaviorpatterns.info/predator-prey-grass-project/).


## Installation

**Editor used:** Visual Studio Code 1.88.1 on Linux Mint 21.3 Cinnamon

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
   - Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If encountering "ERROR: Failed building wheel for box2d-py," run:
   ```bash
   conda install swig
   ```
   and
   ```bash
   pip install box2d box2d-kengz
   ```
4. Alternatively, a workaround is to copy Box2d files from [assets/box2d](https://github.com/doesburg11/PredPreyGrass/tree/main/assets/box2d) to the site-packages directory.
5. If facing "libGL error: failed to load driver: swrast," execute:
    ```bash
    conda install -c conda-forge gcc=12.1.0
    
## Getting started

### Visualize a random policy
In Visual Studio Code run:
```pettingzoo/predpreygrass/random_policy.py```
</br>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass_random.gif" width="1000" height="200"/>
</p>


### Training and visualize trained model using PPO from stable baselines3
Adjust parameters accordingly in:
```pettingzoo/predpreygrass/config/config_pettingzoo.py```
In Visual Studio Code run:
```pettingzoo/predpreygrass/train_sb3_vector_ppo_parallel.py```
To evaluate and visualize after training follow instructions in:
```pettingzoo/predpreygrass/evaluate_from_file.py```


## Configuration of the PredPreyGrass environment
[The benchmark configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/pettingzoo/predpreygrass/config/config_pettingzoo.py) used in the gif-video above.

## References

- [Terry, J and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and others. Pettingzoo: Gym for multi-agent reinforcement learning. 2021-2024](https://pettingzoo.farama.org/){:target="_blank"} The utlimate go-to for multi-agent reinforcement learning deployment.  
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers){:target="_blank"}


