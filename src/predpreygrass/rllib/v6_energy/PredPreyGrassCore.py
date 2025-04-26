# Step 1: Process energy depletion due to time steps and update age

# Precompute agent types to avoid repeated string checks
agent_types = {agent: ("predator" if "predator" in agent else "prey") for agent in action_dict}

if self.verbose_movement:
    sep_line = "-" * 110  # Cache verbose separator once

for agent, action in action_dict.items():
    agent_type = agent_types[agent]

    if self.verbose_movement:
        print(sep_line)
        print(f"[DEPLETE] {agent} energy: {self.agent_energies[agent]} -> ", end="")

    if agent_type == "predator":
        self.agent_energies[agent] -= self.energy_loss_per_step_predator
        self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
    else:  # prey
        self.agent_energies[agent] -= self.energy_loss_per_step_prey
        self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

    if self.verbose_movement:
        print(f"{self.agent_energies[agent]}")
        print(sep_line)

    # Update age
    internal_id = self.agent_internal_ids.get(agent)
    if internal_id is not None:
        self.agent_ages[internal_id] += 1

# Process grass energy regeneration
energy_gain = self.energy_gain_per_step_grass
energy_cap = self.initial_energy_grass

for grass, pos in self.grass_positions.items():
    new_energy = min(self.grass_energies[grass] + energy_gain, energy_cap)
    self.grass_energies[grass] = new_energy
    self.grid_world_state[3, *pos] = new_energy
