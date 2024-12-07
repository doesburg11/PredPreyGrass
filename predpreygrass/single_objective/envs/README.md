## Environments

[**predpregrass_aec_v0.py**](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/envs/predpreygrass_aec_v0.py): 
A (single-objective) multi-agent reinforcement learning (MARL) environment, 
[trained and evaluated](https://github.com/doesburg11/PredPreyGrass/tree/main/predpreygrass/optimizations/so_predpreygrass_v0) 
using [Proximal Policy Optimization (PPO)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). 
Learning agents Predators (red) and Prey (blue) both expend energy moving around, and replenish it by eating. 
Prey eat Grass (green), and Predators eat Prey if they end up on the same grid cell. In the base case for simplicity, 
the agents obtain all the energy from the eaten Prey or Grass. Predators die of starvation when their energy is zero, 
Prey die either of starvation or when being eaten by a Predator. The agents asexually reproduce when energy levels of 
learning agents rise above a certain treshold by eating. Learning agents, learn to execute movement actions based on 
their partial observations (transparent red and blue squares respectively) of the environment to maximize cumulative reward. 
The single objective rewards (stepping, eating, dying and reproducing) are naively summed and can be adjusted in the 
[environment configuration](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/envs/_so_predpreygrass_v0/config/so_config_predpreygrass.py) file. 


[**predpregrass_parallel_v0.py**](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/single_objective/envs/predpreygrass_parallel_v0.py):
The unwrapped parallel version of the multi-agent reinforcement learning (MARL) environment.

