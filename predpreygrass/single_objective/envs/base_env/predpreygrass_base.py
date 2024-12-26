# discretionary libraries
from predpreygrass.single_objective.utils.env import PredPreyGrassSuperBaseEnv

# external libraries
from typing import Dict


class PredPreyGrassBaseEnv(PredPreyGrassSuperBaseEnv):
    """
    Base environment class for shared logic in PredPreyGrass environments.
    """
    def process_agent(self, agent_instance, agent_type_nr, target_type_nr, energy_gain_threshold, catch_reward, reproduction_reward, death_reward):
        if agent_instance.energy > 0:
            x_new, y_new = agent_instance.position
            target_instance_in_cell = self.agent_instance_in_grid_location[target_type_nr].get((x_new, y_new))
            if target_instance_in_cell is not None:
                agent_instance.energy += target_instance_in_cell.energy
                self._deactivate_agent(target_instance_in_cell)
                self.n_active_agent_type[target_type_nr] -= 1
                self.agent_age_of_death_list_type[target_type_nr].append(target_instance_in_cell.age)
                if target_type_nr == self.prey_type_nr:
                    self.n_eaten_prey += 1
                elif target_type_nr == self.grass_type_nr:
                    self.n_eaten_grass += 1
                self.rewards[agent_instance.agent_name] += catch_reward
                self.rewards[target_instance_in_cell.agent_name] += death_reward
                if agent_instance.energy > energy_gain_threshold:
                    potential_new_agents = self._non_active_agent_instance_list(agent_instance)
                    if potential_new_agents:
                        self._reproduce_new_agent(agent_instance, potential_new_agents)
                        self.n_active_agent_type[agent_type_nr] += 1
                        if agent_type_nr == self.predator_type_nr:
                            self.n_born_predator += 1
                        elif agent_type_nr == self.prey_type_nr:
                            self.n_born_prey += 1
                    self.rewards[agent_instance.agent_name] += reproduction_reward
        else:
            self._deactivate_agent(agent_instance)
            self.n_active_agent_type[agent_type_nr] -= 1
            self.agent_age_of_death_list_type[agent_type_nr].append(agent_instance.age)
            if agent_type_nr == self.predator_type_nr:
                self.n_starved_predator += 1
            elif agent_type_nr == self.prey_type_nr:
                self.n_starved_prey += 1
            self.rewards[agent_instance.agent_name] += death_reward

    def process_grass(self):
        for grass_name in self.possible_agent_name_list_type[self.grass_type_nr]:
            grass_instance = self.agent_name_to_instance_dict[grass_name]
            grass_energy_gain = min(
                grass_instance.energy_gain_per_step,
                max(self.max_energy_level_grass - grass_instance.energy, 0),
            )
            grass_instance.energy += grass_energy_gain
            if grass_instance.energy >= self.initial_energy_type[self.grass_type_nr] and not grass_instance.is_active:
                self._activate_agent(grass_instance)
                self.n_active_agent_type[self.grass_type_nr] += 1
            elif grass_instance.energy < self.initial_energy_type[self.grass_type_nr] and grass_instance.is_active:
                self._deactivate_agent(grass_instance)
                self.n_active_agent_type[self.grass_type_nr] -= 1

    def record_metrics(self):
        self.n_cycles += 1
        self._record_population_metrics()

    def process_end_of_cycle(self):
        """
        Common logic for handling the end of a step cycle.
        """
        self._reset_rewards()
        for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr]:
            self.process_agent(
                predator_instance,
                self.predator_type_nr,
                self.prey_type_nr,
                self.predator_creation_energy_threshold,
                self.catch_reward_prey,
                self.reproduction_reward_predator,
                self.death_reward_predator
            )
        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
            self.process_agent(
                prey_instance,
                self.prey_type_nr,
                self.grass_type_nr,
                self.prey_creation_energy_threshold,
                self.catch_reward_grass,
                self.reproduction_reward_prey,
                self.death_reward_prey
            )
        self.process_grass()
        self.record_metrics()


class PredPreyGrassAECEnv(PredPreyGrassBaseEnv):
    def step(self, action, agent_instance, is_last_step_of_cycle):
        if agent_instance.is_active:
            self._apply_agent_action(agent_instance, action)
        if is_last_step_of_cycle:
            self.process_end_of_cycle()


class PredPreyGrassParallelEnv(PredPreyGrassBaseEnv):
    def step(self, actions: Dict[str, int]):
        for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr]:
            self._apply_agent_action(predator_instance, actions[predator_instance.agent_name])
        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
            self._apply_agent_action(prey_instance, actions[prey_instance.agent_name])
        self.process_end_of_cycle()
