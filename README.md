
### Predator, Prey, Grass PettingZoo environment
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass.gif" width="300" height="300"/>
</p>

 
 The PredPreyGrass envrionment has been initially modified from PettingZoo's the (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
 1. The envrionment is added with an addtional Predator learning agent. 
 2. The Pursuers have been converted to a Prey learning agent,
 3. Evaders have been converted to grass and are permanently 'freezed' and do not move.

 Similar to the PettingZoo Pursuit environment, grass agents are left out of the 'self.agents' array. Including them results into signifcant loss of computing efficency without obvious advantages, hence the original Pursuit design has been kept in this respect.

 ### Optionalities of the PredPreyGrass environment
    render_mode="human", 
    max_cycles=10000, 
    x_grid_size=16, 
    y_grid_size=16, 
    n_predator=4,
    n_prey=8,
    n_grass=30,
    max_observation_range=7,     
    obs_range_predator=3, # must be odd and not greater than 'max_observation_range'*  
    obs_range_prey=7, # must be odd
    action_range=7, # must be odd
    moore_neighborhood_actions=False,
    pixel_scale=40

*this implementation facilitates different observations ranges per agent:
If obs_range < max_observation_range then 'outer layers' of the observations are set to zero.

### The reward structure
homeostatic_energy_per_aec_cycle = -0.1 # for both predator and prey
catch_grass_reward = 5.0 # for prey
catch_prey_reward = 5.0 # for predator

### Learning algorithm 
The Multi Agent Reinforcement Learning algorithm to control the PredPreyGrass environment is PPO from stable baselines3.





