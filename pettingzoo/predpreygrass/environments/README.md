### Environments
`predatorpreygrass_create_prey.py`:
Learning agents (Predators and Prey) receive fixed pre determined rewards:
1. Penalty per time step 
2. Reward for capturing food 
3. Penalty for dying (either by being caught or by starving to death)

- Grass agents optionally regrow after a number of steps a the same spot. 
- Predators can be removed or optionally created.
- Prey can be removed or optionally created.

The removal or creation of Predators or Prey is handeld by the `is_alive` boolean of the agents.
At `reset`,`n_possible_predator` and `n_possible_prey` are initialized. However, a portion of agents is intialized at `is_alive` = `False`, this will give room for future creation of agents during runtime. Conversely, removal of agents during runtime is handled by setting the attribute `is_alive` from `True` to `False`.

Summarized, intially created but inactive Predator and Prey agents at the end of the first cycle:
- have attribute `energy` = 0,
- have attribute `is_alive` = False
- are not observable for active learning agents (Predator and Prey)
- are removed from the `predator_instance_list` or `prey_instance_list`
- are 'out of the game' and are not vizualised

