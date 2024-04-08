
### A Predator, Prey, Grass multiagent learning environment
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="700" height="300"/>
</p>

### Explanation of the agents, the environment and the learning algorithm

A multi-agent reinforcement learning environment trained using Proximal Policy Optimization (PPO) is employed. Predators (red) and Prey (blue) both expend energy moving around, and replenish it by eating. Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. This simulation represents a predator-prey-grass ecosystem within a multi-agent reinforcement learning framework. Agents,  Predators and Prey, learn to execute movement actions based on their partially observable environment to maximize cumulative reward. The environment is a bounded grid world and the agents move within a Von Neumann neighborhood.

The model has been tested and demonstrates:
- Bounded Grid environment
- Three agent types: Predator, Predy and Gras
- Two learning agent types: Predator and Prey; leaning two move in a Von Neumann neighborhood
- Learned behavior of Predators and Prey to avoid being eaten or starving to death
- Dynamically removing agents from the grid when eaten (Prey and Gras) or starving to death (Predator and Prey)
- Restricted to one Predator agent per cell
- Restricted to one Prey agent per cell
- Restricted to one Grass agent per cell


High-level breakdown of the algorithm's ```step``` function:

1. **Predator Actions**: If the agent is a predator and it's alive, it checks if the predator has energy. If it does, the predator moves and the model state is updated. If the predator lands on a cell with prey, it selects the prey to eat and to be removed at the end of the cycle (AEC). If the predator has no energy left, it is being selected to become inactive at the end of the cycle.

2. **Prey Actions**: If the agent is a prey and it's alive, it checks if the prey has energy. If it does, the prey moves and the model state is updated. If the prey lands on a cell with grass it selects the grass to eat and to be removed ath the end of the cycle. If the prey has no energy left, it is being selected to become inactive at the end of the cycle (AEC).

3. **End of Cycle Actions**: If it's the last step in the PettingZoo cycle (AEC), the function removes agents that have starved to death or have been eaten, and updates the rewards for the remaining agents. It also increments the number of cycles.

This algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. Each agent (predator, prey, grass) follows simple rules based on its current state, but the interactions between agents can lead to more complex dynamics at the ecosystem level.

### Installation Instructions


**Editor used:** Visual Studio Code 1.88.0

1. Clone the repository: 
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
3. Choose environment: Conda 
4. Choose interpreter: Python 3.11.5
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
```pettingzoo/predpreygrass/random_policy_aec_fixed_rewars.py```

### Training and visualize trained model using PPO from stable baselines3
Adjust parameters accordingly in:
```pettingzoo/predpreygrass/config/config_pettingzoo_fixed_rewards.py```
In Visual Studio Code run:
```pettingzoo/predpreygrass/train_sb3_vector_ppo_parallel_fixed_rewards.py```
To evaluate and visualize after training follow instructions in:
```pettingzoo/predpreygrass/evaluate_from_file.py```

### PettingZoo Modification

The PredPreyGrass environment is a significant modification of PettingZoo's (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
1. Added an additional 'predator' agent that can die of starvation.
2. Converted the Pursuer agent to a 'prey' agent, susceptible to being eaten by a predator.
3. Transformed the Evaders agent into a 'grass' agent, permanently 'frozen' and immovable, consumable by prey.

Similar to PettingZoo Pursuit, grass agents are excluded from the 'AECEnv.agents' array for computational efficiency.

### The AEC Environment Architecture

Due to unexpected behavior when agents terminate during a simulation in PettingZoo AEC (https://github.com/Farama-Foundation/PettingZoo/issues/713), we modified the architecture. The 'AECEnv.agents' array remains unchanged after agent death. The removal of agents is managed by 'PredPrey.predator_instance_list' and 'PredPrey.prey_instance_list.' The active status of agents is furthermore tracked by the boolean attribute ```alive``` of the agents.

This architecture provides an alternative to the unexpected behavior of individual agents terminating during simulation in the standard PettingZoo API and circumvents the PPO-algorithm's requirement of an unchanged number of agents during training.

### Optionalities of the PredPreyGrass AEC Environment
The configuration used in the gif-video:
- `max_cycles=10000`
- `x_grid_size=16`
- `y_grid_size=16`
- `n_predator=6`
- `n_prey=8`
- `n_grass=30`
- `max_observation_range=7` (must be odd)
- `obs_range_predator=3` (must be odd)  
- `obs_range_prey=7` (must be odd)
- `action_range=3` (must be odd)
- `energy_loss_per_step_predator=-0.1`
- `energy_loss_per_step_prey=-0.05`
- `initial_energy_predator=10.0`
- `initial_energy_prey=10.0`
- `catch_grass_reward=3.0` (for prey)
- `catch_prey_reward=5.0` (for predator)
- `pixel_scale=40`

This implementation supports different observation ranges per agent: If `obs_range < max_observation_range`, the 'outer layers' of observations are set to zero.

### Emergent Behavior

In this configuration, predators, after training, tend to hover around grass agents to capture prey. However, this strategy is less frequent when 'energy_loss_per_step_predator' becomes more negative, incentivizing predators to abandon the 'wait-and-see' approach.

Since the prey and predators strictly move within a Von Neumann neighborhood (left/right/up/down/stay) rather than a Moore neighborhood, it is theoretically possible for a given prey to consistently outmaneuver a given predator, ensuring it is never caught. This under two conditions: 1) the agents move in a turn-based fashion and 2) no other predators are involved in catching that specific prey. This is because a prey, even when at risk of being caught in the immediate next turn by a predator, can always make a single step towards a position where the threatening predator needs to move at least two times to catch it.

However, this prey behavior goes unnoticed in the above display because the simulation is trained in a parallel environment where all agents decide simultaneously, rather than in a turn-based fashion. To recap: the simulation is parallel trained but evaluated in a turn based fashion (as displayed above). 

```
@readme{PredPreyGrass,
  Title={A Predator, Prey, Grass Multiagent Learning Environment},
  Author={Van Doesburg, P.},
  Year={2024}
}
```
