[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### A Predator-Prey-Grass multiagent learning environment


<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/predatorprey.png" width="200" height="100"/> ### A Predator-Prey-Grass multiagent learning environment

</p>




<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="700" height="300"/>
</p>


### Explanation of the agents, the environment and the learning algorithm

A multi-agent reinforcement learning environment trained using Proximal Policy Optimization (PPO) is employed. Learning agents Predators (red) and Prey (blue) both expend energy moving around, and replenish it by eating. Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. Predators die of starvation when their energy is zero, Prey die either of starvation or when being eaten by a Predator. The agents asexually reproduce when energy levels of learning agents rise above a certain treshold by eating. This simulation represents a predator-prey-grass ecosystem within a multi-agent reinforcement learning framework. Learning agents, learn to execute movement actions based on their partially observations of the environment to maximize cumulative reward. The environment is a bounded grid world and the agents move within a Von Neumann neighborhood.

### Emergent Behavior
This algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. Each agent (Predator, Prey, Grass) follows simple rules based on its current state, but the interactions between agents can lead to more complex dynamics at the ecosystem level. The trained agents are displaying a classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time. This learned outcome is not obtained with a random policy:

<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>

More emergent behavior and findings are described in the [config directory](https://github.com/doesburg11/PredPreyGrass/tree/main/pettingzoo/predpreygrass/config).


### Installation Instructions

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
    
### Visualize a random policy
In Visual Studio Code run:
```pettingzoo/predpreygrass/random_policy.py```

### Training and visualize trained model using PPO from stable baselines3
Adjust parameters accordingly in:
```pettingzoo/predpreygrass/config/config_pettingzoo.py```
In Visual Studio Code run:
```pettingzoo/predpreygrass/train_sb3_vector_ppo_parallel.py```
To evaluate and visualize after training follow instructions in:
```pettingzoo/predpreygrass/evaluate_from_file.py```


### Configuration of the PredPreyGrass environment
[The benchmark configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/pettingzoo/predpreygrass/config/config_pettingzoo_benchmark_1.py) used in the gif-video above.


