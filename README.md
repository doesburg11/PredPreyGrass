
### Predator, Prey, Grass PettingZoo environment
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="300" height="300"/>
</p>

 Prey agents (blue) try to eat grass agents (green). Predators (red) try to capture prey.
 The PredPreyGrass envrionment has been modified from PettingZoo's (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
 1. The envrionment is added with an addtional Predator learning agent. 
 2. The Pursuers have been converted to a Prey learning agent,
 3. Evaders have been converted to grass and are permanently 'freezed' and do not move.

 Similar to the PettingZoo Pursuit environment, grass agents are left out of the 'self.agents' array. Including them results into signifcant loss of computing efficency without obvious advantages, hence the original Pursuit design has been kept in this respect.

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
With this configuration predators try to hoover around the grass agents in order to capture prey. Prey try to flee predators despite that this is not explicitly defined in the reward structure.

### Learning algorithm 
The Multi Agent Reinforcement Learning algorithm to control the PredPreyGrass environment is PPO from stable baselines3.





