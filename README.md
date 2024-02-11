
### A Predator, Prey, Grass multiagent learning environment
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="700" height="300"/>
</p>

### Explanation of the algorithm

 For questions or suggestions, gmail: vandoesburgpeter1

A multi-agent reinforcement learning environment trained with Proximal Policy Optimization. In the simulation blue agents (prey) learn to consume green agents (grass), while red agents (predators) learn to capture prey; a simulation for a predator-prey-grass ecosystem used in a multi-agent reinforcement learning context. Agents (predators and prey) learn to take actions (like moving, eating) based on their current state to maximize cumulative reward.

High-level breakdown of the algorithm's ```step``` function:

1. **Predator Actions**: If the agent is a predator and it's alive, it checks if the predator has energy. If it does, the predator moves and the model state is updated. If the predator lands on a cell with prey, it selects the prey to eat and to be removed at the end of the cycle (AEC). If the predator has no energy left, it is being selected to become inactive at the end of the cycle.

2. **Prey Actions**: If the agent is a prey and it's alive, it checks if the prey has energy. If it does, the prey moves and the model state is updated. If the prey lands on a cell with grass it selects the grass to eat and to be removed ath the end of the cycle. If the prey has no energy left, it is being selected to become inactive at the end of the cycle (AEC).

3. **End of Cycle Actions**: If it's the last step in the PettingZoo cycle (AEC), the function removes agents that have starved to death or have been eaten, and updates the rewards for the remaining agents. It also increments the number of cycles.

This algorithm is an example of how elaborate behaviors can emerge from simple rules in agent-based models. Each agent (predator, prey, grass) follows simple rules based on its current state, but the interactions between agents can lead to more complex dynamics at the ecosystem level.

### Installation Instructions


**Editor used:** Visual Studio Code 1.85.1

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
    ```
### Training and visualize trained model
In Visual Studio Code run:
```pettingzoo/predprey/predpreygrass_v0.py```
Adjust parameters accordingly in:
```pettingzoo/predprey/predpreygrass_v0/parameters.py```

### PettingZoo Modification

The PredPreyGrass environment is a significant modification of PettingZoo's (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
1. Added an additional 'predator' agent that can die of starvation.
2. Converted the Pursuer agent to a 'prey' agent, susceptible to being eaten by a predator.
3. Transformed the Evaders agent into a 'grass' agent, permanently 'frozen' and immovable, consumable by prey.

Similar to PettingZoo Pursuit, grass agents are excluded from the 'AECEnv.agents' array for computational efficiency.

### The AEC Environment Architecture

Due to unexpected behavior when agents terminate during a simulation in PettingZoo AEC (https://github.com/Farama-Foundation/PettingZoo/issues/713), we modified the architecture. The 'AECEnv.agents' array remains unchanged after agent death. The removal of agents is managed by 'PredPrey.predator_instance_list' and 'PredPrey.prey_instance_list.' The alive status of agents is furthermore tracked by the boolean attribute ```alive``` of the agents.

This architecture provides an alternative to the unexpected behavior of individual agents terminating during simulation in the standard PettingZoo API and circumvents the PPO-algorithm's requirement of an unchanged number of agents during training.

### Optionalities of the PredPreyGrass Environment
The configuration used in the gif-video:
- `max_cycles=10000`
- `x_grid_size=16`
- `y_grid_size=16`
- `n_predator=4`
- `n_prey=6`
- `n_grass=30`
- `max_observation_range=7` (must be odd)
- `obs_range_predator=3` (must be odd)  
- `obs_range_prey=7` (must be odd)
- `action_range=3` (must be odd)
- `moore_neighborhood_actions=False`
- `energy_loss_per_step_predator=-0.1`
- `energy_loss_per_step_prey=-0.05`
- `initial_energy_predator=10.0`
- `initial_energy_prey=10.0`
- `catch_grass_reward=2.0` (for prey)
- `catch_prey_reward=5.0` (for predator)
- `pixel_scale=40`

This implementation supports different observation ranges per agent: If `obs_range < max_observation_range`, the 'outer layers' of observations are set to zero.

### Emergent Behavior

In this configuration, predators, after training, tend to hover around grass agents to capture prey. However, this strategy is less frequent when 'energy_loss_per_step_predator' becomes more negative, incentivizing predators to abandon the 'wait-and-see' approach.

```
@readme{PredPreyGrass,
  Title={A Predator, Prey, Grass Multiagent Learning Environment},
  Author={Van Doesburg, P.},
  Year={2024}
}
```
