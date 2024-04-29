
### A Predator-Prey-Grass multiagent learning environment

<p align="center">The benchmark configuration (without offspring creation)</p>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="700" height="300"/>
</p>


### Explanation of the agents, the environment and the learning algorithm

A multi-agent reinforcement learning environment trained using Proximal Policy Optimization (PPO) is employed. Predators (red) and Prey (blue) both expend energy moving around, and replenish it by eating. Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. This simulation represents a predator-prey-grass ecosystem within a multi-agent reinforcement learning framework. Agents,  Predators and Prey, learn to execute movement actions based on their partially observable environment to maximize cumulative reward. The environment is a bounded grid world and the agents move within a Von Neumann neighborhood.


High-level breakdown of the algorithm's ```step``` function:

1. **Predator Actions**: If the agent is a Predator and it's alive, it checks if the Predator has positive energy. If it does, the Predator moves and the model state is updated. If the predator lands on a cell with prey, it selects the prey to eat and to be removed at the end of a cycle (AEC). Otherwise, if the Predator has no positive energy left, it is being selected to become inactive at the end of a cycle. 

2. **Prey Actions**: If the agent is a prey and it's alive, it checks if the prey has positive energy. If it does, the prey moves and the model state is updated. If the prey lands on a cell with grass it selects the grass to eat and to be removed ath the end of a cycle. If the prey has no energy left, it is being selected to become inactive at the end of a cycle.

3. **End of Cycle Actions**: If it's the last step in the PettingZoo cycle (AEC), the function removes agents that have starved to death or have been eaten, and updates the rewards for the remaining agents. It also increments the number of cycles. If the energy of an agent (Predator or Prey) has reached a certain replication-treshold it reproduces a new agent at a random empty spot in the grid environment and the parent transfers a part of its energy to the child.

This algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. Each agent (Predator, Prey, Grass) follows simple rules based on its current state, but the interactions between agents can lead to more complex dynamics at the ecosystem level.

### Emerging Behavior
The trained agents are displaying a classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time. This result cannot be obtained with a random policy in the same setting:

<p align="center">The population dynamics of PPO trained Predators and Prey</p>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>

More emergening behavior and findings are described in the [wiki](https://github.com/doesburg11/PredPreyGrass/wiki/Emergent-behavior-in-the-Predator%E2%80%90Prey%E2%80%90Grass-environment).


### Installation Instructions

**Editor used:** Visual Studio Code 1.88.1

1. Clone the repository: 
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
3. Choose environment: Conda 
4. Choose interpreter: Python 3.11.7
5. Open a new terminal
6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
7. If encountering "ERROR: Failed building wheel for box2d-py," run:
   ```bash
   conda install swig
   ```
   and
   ```bash
   pip install box2d box2d-kengz
   ```
8. Alternatively, copy Box2d files from 'assets/box2d' (https://github.com/doesburg11/PredPreyGrass/tree/main/assets/box2d) to the site-packages directory.
9. If facing "libGL error: failed to load driver: swrast," execute:
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

### The AEC Environment Architecture

Due to unexpected behavior when agents terminate during a simulation in PettingZoo AEC (https://github.com/Farama-Foundation/PettingZoo/issues/713), we modified the architecture. The 'AECEnv.agents' array remains unchanged after agent death or creation. The removal of agents is managed by 'PredPrey.predator_instance_list' and `PredPreyGrass.[predator/prey]_instance_list`. The active status of agents is furthermore tracked by the boolean attribute `alive` of the agents. Optionally, a number of agents have this attribute `alive` set to `False` at `reset`, which gives room for creation of agents during run time. If so, the agents are (re)added to `PredPreyGrass.[predator/prey]_instance_list`.

This architecture provides an alternative to the unexpected behavior of individual agents terminating during simulation in the standard PettingZoo API and circumvents the PPO-algorithm's requirement of an unchanged number of agents during training. In that sense it is comparable to SuperSuit's "Black Death" wrapper.

### Optionalities of the PredPreyGrass AEC Environment
The benchmark configuration used in the gif-video :
- `max_cycles=10000`
- `x_grid_size=16`
- `y_grid_size=16`
- `n_predator=6`
- `n_prey=8`
- `n_grass=30`
- `max_observation_range=7` (must be odd)
- `obs_range_predator=5` (must be odd)  
- `obs_range_prey=7` (must be odd)
- `action_range=3` (must be odd)
- `energy_loss_per_step_predator=-0.1`
- `energy_loss_per_step_prey=-0.05`
- `initial_energy_predator=5.0`
- `initial_energy_prey=5.0`
- `catch_grass_reward=3.0` (for prey)
- `catch_prey_reward=5.0` (for predator)
- `pixel_scale=40`

This implementation supports different observation ranges per agent: If `obs_range < max_observation_range`, the 'outer layers' of observations are set to zero.



@readme{PredPreyGrass,
  Title={A Predator, Prey, Grass Multiagent Learning Environment},
  Author={Van Doesburg, P.},
  Year={2024}
}
```
