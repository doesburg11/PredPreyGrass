
### A Predator, Prey, Grass multiagent learning environment
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="300" height="300"/>
</p>

 Prey agents (blue) learn to eat grass agents (green). Predators (red) learn to capture prey.

### Installation instructions

Editor used: Visual Studio Code 1.85.1

1. ```git clone https://github.com/doesburg11/PredPreyGrass.git```
2. ctrl+shift+p, type and choose: "Python: Create Environment..."
3. Choose environment: Conda 
4. Choose interpreter: Python 3.11.5
5. Open New Terminal
5. ```pip install -r requirements.txt```
7. IF "ERROR: Failed building wheel for box2d-py" DO: ```conda install swig``` in VS Code terminal
8. If that does not work try first to find other solutions online.
9. Ultimately one can copy and past the Box2d files from the 'assets/box2d' directory (https://github.com/doesburg11/PredPreyGrass/tree/main/assets/box2d) into the site-packages directory. Not very elegant but it might work.
10. IF: "libGL error: failed to load driver: swrast" DO: ```conda install -c conda-forge gcc=12.1.0```

### PettingZoo modification

 The PredPreyGrass envrionment has been substantially modified from PettingZoo's (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
 1. The envrionment is added with an addtional 'predator' agent, which can die of starvation. 
 2. The Pursuer agent has been converted to a 'prey' agent and can be eaten by a predator agent.
 3. The Evaders agent has been converted to a 'grass' agent and are permanently 'freezed' and are unmovable. The gras agent can be eaten by a prey agent.

 Similar to the PettingZoo Pursuit environment, grass agents are left out of the 'AECEnv.agents' array. Including them results into signifcant loss of computing efficency without obvious advantages, hence the original Pursuit design has been kept in this respect.

 ### The AEC environment architecture
Since the creation and particulary the termination of agents during a simulation leads to unexpected behavior during a PettingZoo AEC (https://github.com/Farama-Foundation/PettingZoo/issues/713), we have modified the architecture of the original PettingZoo in that respect. The PettingZoo 'AECEnv.agents', the array [predator_0, predator_1,..,predator_n, prey_n+1,..,prey_n+m], remains unchanged after death of agents during simulation. Therefore, in PettingZoo's API terminology 'AECEnv.agents' remains equal to 'AECEnv.possible_agents' during training as well as evaluation.

The handling of dying predators and prey is effected by the removal of agents in the 'PredPrey.predator_instance_list' and the 'PredPrey.prey_instance_list' respectively. Wether or not a predator or prey is allive can be additionaly checked by the 'PredPrey.predator_not_alive_dict' and the 'PredPrey.prey_not_alive_dict. Agents not being alive have 'observations' and 'rewards' only existing of zeros, somewhat resembling SuperSuit's 'Black Death' wrapper.

This architecture does not only give an alternative to the unexpected behavior of individual agents terminating during simulation in the standard PettingZoo API. It does also circumvents the restriction of the PPO-algorithm, which requires an unchanged number of agents during training.

 ### Optionalities of the PredPreyGrass environment
        max_cycles=100000, 
        x_grid_size=16, 
        y_grid_size=16, 
        n_predator=4,
        n_prey=4,
        n_grass=30,
        max_observation_range=7, # influences number of calculations; make as small as possible
        obs_range_predator=3,   
        obs_range_prey=7, # must be odd
        action_range=3, # must be odd
        moore_neighborhood_actions=False,
        energy_loss_per_step_predator = -0.4,
        energy_loss_per_step_prey = -0.1,     
        pixel_scale=40
        catch_grass_reward = 5.0 # for prey
        catch_prey_reward = 5.0 # for predator

*this implementation facilitates different observations ranges per agent:
If obs_range < max_observation_range then 'outer layers' of the observations are set to zero.

### Emergent behavior
With this configuration predators, after training, try to hoover around the grass agents in order to capture prey. However, this strategy is less frequent when the 'energy_loss_per_step_predator'gets more negative and predators are incentivized to abonden the 'wait-and-see' approach.

 
```
@readme{PredPreyGrass,
  Title = {A Predator, Prey, Grass multiagent learning environment},
  Author = {Van Doesburg, P.},
  year={2024}
}
```


