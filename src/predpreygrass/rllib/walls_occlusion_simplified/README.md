# Walls & Occlusion Efficiency in PredPreyGrass

## Goals:
- Make Walls & Occlusion environment more efficient
- Remove redunant code
- Make it "cooperation-search-ready"


## Potential setup
- Masking for stepping into self-species occupied cell?
- Remove all non-reproduction rewards (to get copilot better understand the confif_env)?
- Tidy up?
  - delete all unnecesarry configs?

## TODO
- Move costs prey and predator
- No move: zero costs


## Later on
- Introduce **max_eating**. So a predator/prey cannot eat a prey/grass in one step => predator decides to eat more or leave for other in the next step? This could be an efficient way to increase the action_space over multiple steps. Predator "stays" to eat carcass further in next step or it "leaves" to "share" energy with other predators.=> is already implemented with "max_energy_gain_per_prey".
  ("raw_gain = min(self.agent_energies[caught_prey], self.config.get("max_energy_gain_per_prey", float("inf"))))



- Adjustments: 
  - Killing prey at a higher energy cost than scavenging for predators?
  - Leave (prey) carcass on grid, as an unmovable energy source for predators.
  - Carcass depletes?
  - Ammount of eating different from killers (more?) than to scavengers (less?)
  - introduce chance of not killing a prey, but nevertheless loose enrgy for attempting?
  - Grass needs probably not much adjusted if it only gets eated partially


