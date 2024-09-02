### Environments
`predatorpreygrass_fixed_energy_transfer.py`:
Learning agents (Predators and Prey) receive fixed pre determined energy and rewards optionally by eating, moving, dying and reproduction.
1. Penalty per time step 
2. Reward for capturing food 
3. Penalty for dying (either by being caught or by starving to death)

- Grass agents optionally regrow after a number of steps a the same spot. 
- Predators and Prey can be removed or optionally created; by reproduction the parent agent tranfers energy to the child agent and receives a positive reward

The removal or creation of Predators or Prey is handeled by the `is_active` boolean of the agents.
At `reset`,`n_possible_predator` and `n_possible_prey` are initialized. However, a portion of agents is intialized at `is_active` = `False`, this will give room for creation of agents during runtime. Conversely, removal of agents during runtime is handled by setting the attribute `is_active` from `True` to `False`. The rewards and observations of non active agents are zeros, comparable with SuperSuit's Black Death wrapper.

`predatorpreygrass.py`: The default environment. A generalization of 'predatorpreygrass_fixed_energy_transfer.py'. Rather than receiving fixed predeterimined energy quantities, agents exchange energy depending on the energy available from the eaten agents. The observation channels consists of the energy levels of agents, rather than just one's or zero's in 'predatorpreygrass_fixed_energy_transfer.py'. The purpose of this generalization is to gauge wether agents maken energy-efficiency choices. So, if they go after a resource which has higher energetic value than antother within their observation range.

