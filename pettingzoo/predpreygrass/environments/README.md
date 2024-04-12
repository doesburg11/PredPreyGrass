### Environments
In order of development
1. `predpreygrass_fixed_rewards.py`:
Learning agents (Predators and Prey) receive fixed pre determined rewards for caputuring food.

2. `predatorpreygrass_energy_rewards.py`:
Learning agents of observe energy level of possiblefood agents in their observation range and receive reward depending on the accumulated energy  of the food agent (and the food agents dies). [note 2024-04-10: this environment is not able to learn very well with the StableBaseline3 PPO algorithm]

3. `predatorpreygrass_create_agents.py`: Same as `1.`. Additionaly: `possible_agents` >= `intial_agents` created to give room for future creation of agents during run time. Intially created but inactive agents at first have:
- attribute `energy` = 0,
- attribute is_alive = False
- are not observable
- are "parked" at a position outside the grid: [-1,-1] to not "stand in the way" of peer agents (remember that peer agents cannot occupy the same cell in the grid). 

    No actual learning agents are created yet so the environment name is a bit misleading.

4. `predator_regrowth_grass.py`: Same as `3.` but `grass` agents regrow after a certain predefined number of steps a the same spot.

