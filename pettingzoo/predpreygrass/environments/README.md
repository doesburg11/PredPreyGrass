### Environments
In order of development
1. `predpreygrass_fixed_rewards.py`:
Learning agents (Predators and Prey) receive fixed pre determined rewards for capturing food.

2. `predatorpreygrass_energy_rewards.py`:
Learning 'alive' agents observe energy level of possible food agents in their observation range and, by capturing, receive a reward depending on the accumulated energy of the caught food agent (the caught food agent dies thereafter). [note 2024-04-10: this environment is not able to learn very well with the StableBaseline3 PPO algorithm]

3. `predatorpreygrass_create_prey.py`: Same as `1.` but `grass` agents regrow after a certain predefined number of steps a the same spot. Additionaly: `n_possible_prey` >= `n_initial_active_prey` prey agents are created at `reset`.
4. Additionally at `reset`, with `n_possible_prey` minus `n_initial_active_prey` prey agents the `energy` is set to zero.
5. This result in removing the inactive prey agents to be removed from the prey_instance_list at the end of the first cycle
6. The inactive prey results can be 'created' during run time. Summarized, intially created but inactive prey at the end of the first cycle have:
- attribute `energy` = 0,
- attribute is_alive = False
- are not observable for active learning agents (Predator and Prey)
- are removed from the prey__instance_list
- are 'out of the game' and are not vizualised

