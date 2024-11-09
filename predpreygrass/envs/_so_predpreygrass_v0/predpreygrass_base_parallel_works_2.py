import numpy as np

class PredPreyGrass:
    # ... other methods ...

    def step(self, actions: Dict[str, int]) -> None:
        """
        Step function to update the environment state by applying actions for each agent
        and handling interactions such as eating, energy changes, and reproduction.
        """
        # Convert agent instances and relevant properties to NumPy arrays for vectorization
        predator_positions = np.array([instance.position for instance in self.active_agent_instance_list_type[self.predator_type_nr]])
        prey_positions = np.array([instance.position for instance in self.active_agent_instance_list_type[self.prey_type_nr]])
        grass_positions = np.array([instance.position for instance in self.active_agent_instance_list_type[self.grass_type_nr]])

        predator_energies = np.array([instance.energy for instance in self.active_agent_instance_list_type[self.predator_type_nr]])
        prey_energies = np.array([instance.energy for instance in self.active_agent_instance_list_type[self.prey_type_nr]])
        grass_energies = np.array([instance.energy for instance in self.active_agent_instance_list_type[self.grass_type_nr]])

        # Step 1: Apply energy gain per step
        predator_energies += self.energy_gain_per_step_predator
        prey_energies += self.energy_gain_per_step_prey
        grass_energies += self.energy_gain_per_step_grass

        # Step 2: Movement Actions - Update positions
        for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr]:
            self._apply_agent_action(predator_instance, actions[predator_instance.agent_name])
        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
            self._apply_agent_action(prey_instance, actions[prey_instance.agent_name])

        # Update positions in state arrays
        predator_positions = np.array([instance.position for instance in self.active_agent_instance_list_type[self.predator_type_nr]])
        prey_positions = np.array([instance.position for instance in self.active_agent_instance_list_type[self.prey_type_nr]])

        # Step 3: Handle engagements (e.g., eating, death, reproduction)
        # Vectorize predator-prey engagements
        for idx, predator_position in enumerate(predator_positions):
            matching_prey_indices = np.where((prey_positions == predator_position).all(axis=1))[0]
            if len(matching_prey_indices) > 0:
                prey_idx = matching_prey_indices[0]
                predator_energies[idx] += prey_energies[prey_idx] + self.catch_prey_energy
                prey_energies[prey_idx] = 0  # Mark prey as "eaten"
                self.rewards[self.active_agent_instance_list_type[self.predator_type_nr][idx].agent_name] += self.catch_reward_prey

        # Vectorize prey-grass engagements
        for idx, prey_position in enumerate(prey_positions):
            matching_grass_indices = np.where((grass_positions == prey_position).all(axis=1))[0]
            if len(matching_grass_indices) > 0:
                grass_idx = matching_grass_indices[0]
                prey_energies[idx] += grass_energies[grass_idx] + self.catch_grass_energy
                grass_energies[grass_idx] = 0  # Mark grass as "eaten"
                self.rewards[self.active_agent_instance_list_type[self.prey_type_nr][idx].agent_name] += self.catch_reward_grass

        # Step 4: Deactivate agents if energy drops below zero
        self._vectorize_deactivate(predator_energies, self.predator_type_nr)
        self._vectorize_deactivate(prey_energies, self.prey_type_nr)
        self._vectorize_deactivate(grass_energies, self.grass_type_nr)

        # Update agent energy levels
        for idx, instance in enumerate(self.active_agent_instance_list_type[self.predator_type_nr]):
            instance.energy = predator_energies[idx]
        for idx, instance in enumerate(self.active_agent_instance_list_type[self.prey_type_nr]):
            instance.energy = prey_energies[idx]
        for idx, instance in enumerate(self.active_agent_instance_list_type[self.grass_type_nr]):
            instance.energy = grass_energies[idx]

        # Step 5: Record population metrics and update cycles
        self._record_population_metrics()
        self.n_cycles += 1

    def _vectorize_deactivate(self, energies: np.ndarray, agent_type_nr: int) -> None:
        """
        Deactivate agents if their energy drops below zero.
        
        Args:
            energies (np.ndarray): Array of energies for the agents.
            agent_type_nr (int): The agent type number (e.g., predator, prey, grass).
        """
        indices_to_deactivate = np.where(energies <= 0)[0]
        for idx in indices_to_deactivate:
            instance = self.active_agent_instance_list_type[agent_type_nr][idx]
            self._deactivate_agent(instance)
            self.rewards[instance.agent_name] += self.death_reward_predator if agent_type_nr == self.predator_type_nr else self.death_reward_prey

