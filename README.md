
### A Predator, Prey, Grass multiagent learning environment
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="300" height="300"/>
</p>


Blue agents (prey) learn to consume green agents (grass), while red agents (predators) learn to capture prey.

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
   in the VS Code terminal.
8. If the issue persists, explore alternative solutions online.
9. Alternatively, copy Box2d files from 'assets/box2d' (https://github.com/doesburg11/PredPreyGrass/tree/main/assets/box2d) to the site-packages directory.
10. If facing "libGL error: failed to load driver: swrast," execute:
    ```bash
    conda install -c conda-forge gcc=12.1.0
    ```

### PettingZoo Modification

The PredPreyGrass environment is a significant modification of PettingZoo's (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
1. Added an additional 'predator' agent that can die of starvation.
2. Converted the Pursuer agent to a 'prey' agent, susceptible to being eaten by a predator.
3. Transformed the Evaders agent into a 'grass' agent, permanently 'frozen' and immovable, consumable by prey.

Similar to PettingZoo Pursuit, grass agents are excluded from the 'AECEnv.agents' array for computational efficiency.

### The AEC Environment Architecture

Due to unexpected behavior when agents terminate during a simulation in PettingZoo AEC (https://github.com/Farama-Foundation/PettingZoo/issues/713), we modified the architecture. The 'AECEnv.agents' array remains unchanged after agent death. The removal of agents is managed by 'PredPrey.predator_instance_list' and 'PredPrey.prey_instance_list.' The alive status of agents is furthermore tracked by 'PredPrey.predator_alive_dict' and 'PredPrey.prey_alive_dict.'

This architecture provides an alternative to the unexpected behavior of individual agents terminating during simulation in the standard PettingZoo API and circumvents the PPO-algorithm's requirement of an unchanged number of agents during training.

### Optionalities of the PredPreyGrass Environment
- `max_cycles=10000`
- `x_grid_size=16`
- `y_grid_size=16`
- `n_predator=4`
- `n_prey=8`
- `n_grass=30`
- `max_observation_range=7` (must be odd)
- `obs_range_predator=5` (must be odd)  
- `obs_range_prey=7` (must be odd)
- `action_range=3` (must be odd)
- `moore_neighborhood_actions=False`
- `energy_loss_per_step_predator=-0.4`
- `energy_loss_per_step_prey=-0.1`
- `initial_energy_predator=14.0`
- `initial_energy_prey=8.0`
- `catch_grass_reward=5.0` (for prey)
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