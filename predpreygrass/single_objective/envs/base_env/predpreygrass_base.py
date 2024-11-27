# discretionary libraries
from predpreygrass.single_objective.utils.env import PredPreyGrassSuperBaseEnv

# external libraries
from typing import Dict


class PredPreyGrassAECEnv(PredPreyGrassSuperBaseEnv):
    """
    Pred/Prey/Grass PettingZoo multi-agent learning AEC environment. This environment 
    transfers the energy of eaten prey/grass to the predator/prey while the grass
    regrows over time. The environment is a 2D grid world where agents can move
    in four cardinal directions. 
    """
  
    def step(self, action, agent_instance, is_last_step_of_cycle):
        
        if agent_instance.is_active:
            self._apply_agent_action(agent_instance, action)

        # apply the engagement rules and reap rewards
        if is_last_step_of_cycle:
            #print(f"n_active_agent_type: {self.n_active_agent_type}")
            self._reset_rewards()
            # removes agents, reap rewards and eventually (re)create agents
            for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr].copy():
                if predator_instance.energy > 0:
                    # new is the position of the predator after the move
                    x_new, y_new = predator_instance.position
                    predator_instance.energy += predator_instance.energy_gain_per_step
                    predator_instance.age += 1
                    # predator_instance.energy += predator_action_energy
                    # engagement with environment: "other agents"
                    prey_instance_in_predator_cell = (
                        self.agent_instance_in_grid_location[self.prey_type_nr][
                            (x_new, y_new)
                        ]
                    )
                    if prey_instance_in_predator_cell is not None:
                        predator_instance.energy += prey_instance_in_predator_cell.energy
                        self._deactivate_agent(prey_instance_in_predator_cell)
                        self.n_active_agent_type[self.prey_type_nr] -= 1
                        self.agent_age_of_death_list_type[self.prey_type_nr].append(prey_instance_in_predator_cell.age)
                        self.n_eaten_prey += 1
                        self.rewards[
                            predator_instance.agent_name
                        ] += self.catch_reward_prey
                        self.rewards[
                            prey_instance_in_predator_cell.agent_name
                        ] += self.death_reward_prey
                        if (
                            predator_instance.energy
                            > self.predator_creation_energy_threshold
                        ):          
        
                            potential_new_predator_instance_list = (
                                self._non_active_agent_instance_list(predator_instance)
                            )

                            if potential_new_predator_instance_list:
                                self._reproduce_new_agent(
                                    predator_instance, potential_new_predator_instance_list
                                )
                                self.n_active_agent_type[self.predator_type_nr] += 1
                                self.n_born_predator += 1
                            self.rewards[
                                predator_instance.agent_name
                            ] += self.reproduction_reward_predator
                else:
                    self._deactivate_agent(predator_instance)
                    self.n_active_agent_type[self.predator_type_nr] -= 1
                    self.agent_age_of_death_list_type[self.predator_type_nr].append(predator_instance.age)
                    self.n_starved_predator += 1
                    self.rewards[
                        predator_instance.agent_name
                    ] += self.death_reward_predator

            for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
                if prey_instance.energy > 0:
                    # new is the position of the predator after the move
                    x_new, y_new = prey_instance.position
                    prey_instance.age += 1
                    prey_instance.energy += prey_instance.energy_gain_per_step
                    grass_instance_in_prey_cell = self.agent_instance_in_grid_location[
                        self.grass_type_nr][(x_new, y_new)]
                    if grass_instance_in_prey_cell is not None:
                        prey_instance.energy += grass_instance_in_prey_cell.energy
                        self._deactivate_agent(grass_instance_in_prey_cell) 
                        self.n_active_agent_type[self.grass_type_nr] -= 1
                        self.n_eaten_grass += 1
                        self.rewards[prey_instance.agent_name] += self.catch_reward_grass
                        if prey_instance.energy > self.prey_creation_energy_threshold:
                            potential_new_prey_instance_list = (
                                self._non_active_agent_instance_list(prey_instance)
                            )
                            if potential_new_prey_instance_list:
                                self._reproduce_new_agent( prey_instance, potential_new_prey_instance_list)
                                self.n_active_agent_type[self.prey_type_nr] += 1
                                self.n_born_prey += 1

                            # weather or not reproduction actually occurred, the prey gets rewarded
                            self.rewards[prey_instance.agent_name] += self.reproduction_reward_prey
                else:
                    self._deactivate_agent(prey_instance)
                    self.n_active_agent_type[self.prey_type_nr] -= 1
                    self.agent_age_of_death_list_type[self.prey_type_nr].append(prey_instance.age)
                    self.n_starved_prey += 1
                    self.rewards[prey_instance.agent_name] += self.death_reward_prey

            # process grass (re)growth
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

                elif (
                    grass_instance.energy < self.initial_energy_type[self.grass_type_nr]
                    and grass_instance.is_active
                ):
                    self._deactivate_agent(grass_instance)
                    self.n_active_agent_type[self.grass_type_nr] -= 1
     
            # 3] record step metrics
            self._record_population_metrics()
            self.n_cycles += 1

class PredPreyGrassParallelEnv(PredPreyGrassSuperBaseEnv):
    """
    Pred/Prey/Grass PettingZoo multi-agent learning Parallel environment. This environment 
    transfers the energy of eaten prey/grass to the predator/prey while the grass
    regrows over time. The environment is a 2D grid world where agents can move
    in four cardinal directions. 
    """
   
    def step(self, actions: Dict[str, int]) -> None:
        """
        Executes a step in the environment based on the provided actions.

        Parameters:
            actions (Dict[str, int]): Dictionary containing actions for each agent.
        """
        # 1] apply actions (i.e. movements) for all active agents in parallel
        for predator_instance in self.active_agent_instance_list_type[
            self.predator_type_nr
        ]:
            self._apply_agent_action(
                predator_instance, actions[predator_instance.agent_name]
            )
        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
            self._apply_agent_action(prey_instance, actions[prey_instance.agent_name])

        self._reset_rewards()

        # 2] apply rules of engagement for all agents
        for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr].copy():  
            # make a copy to make removal during iteration possible
            if predator_instance.energy > 0:
                # engagement with environment: "nature and time"
                predator_instance.age += 1
                predator_instance.energy += predator_instance.energy_gain_per_step
                # predator_instance.energy += predator_action_energy
                # engagement with environment: "other agents"
                prey_instance_in_predator_cell = self.agent_instance_in_grid_location[
                    self.prey_type_nr, *predator_instance.position
                ]
                if prey_instance_in_predator_cell:
                    predator_instance.energy += prey_instance_in_predator_cell.energy
                    self._deactivate_agent(prey_instance_in_predator_cell)
                    self.n_active_agent_type[self.prey_type_nr] -= 1
                    self.n_eaten_prey += 1
                    self.agent_age_of_death_list_type[self.prey_type_nr].append(
                        prey_instance_in_predator_cell.age
                    )
                    self.rewards[predator_instance.agent_name] += self.catch_reward_prey
                    self.rewards[
                        prey_instance_in_predator_cell.agent_name
                    ] += self.death_reward_prey
                    if (
                        predator_instance.energy
                        > self.predator_creation_energy_threshold
                    ):
                        potential_new_predator_instance_list = (
                            self._non_active_agent_instance_list(predator_instance)
                        )
                        if potential_new_predator_instance_list:
                            self._reproduce_new_agent(
                                predator_instance, potential_new_predator_instance_list
                            )
                            self.n_active_agent_type[self.predator_type_nr] += 1
                            self.n_born_predator += 1
                        # weather or not reproduction actually occurred, the predator gets rewarded
                        self.rewards[
                            predator_instance.agent_name
                        ] += self.reproduction_reward_predator
            else:  # predator_instance.energy <= 0
                self._deactivate_agent(predator_instance)
                self.n_active_agent_type[self.predator_type_nr] -= 1
                self.agent_age_of_death_list_type[self.predator_type_nr].append(predator_instance.age)
                self.n_starved_predator += 1
                self.rewards[predator_instance.agent_name] += self.death_reward_predator

        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr].copy():  
            if prey_instance.energy > 0:
                # engagement with environment: "nature and time"
                prey_instance.age += 1
                prey_instance.energy += prey_instance.energy_gain_per_step
                # engagement with environmeny: "other agents"
                grass_instance_in_prey_cell = self.agent_instance_in_grid_location[
                    self.grass_type_nr, *prey_instance.position
                ]
                if grass_instance_in_prey_cell:
                    prey_instance.energy += grass_instance_in_prey_cell.energy
                    self._deactivate_agent(grass_instance_in_prey_cell)
                    self.n_active_agent_type[self.grass_type_nr] -= 1
                    self.n_eaten_grass += 1
                    self.rewards[prey_instance.agent_name] += self.catch_reward_grass
                    if prey_instance.energy > self.prey_creation_energy_threshold:
                        potential_new_prey_instance_list = (
                            self._non_active_agent_instance_list(prey_instance)
                        )
                        if potential_new_prey_instance_list:
                            self._reproduce_new_agent(
                                prey_instance, potential_new_prey_instance_list
                            )
                            self.n_active_agent_type[self.prey_type_nr] += 1
                            self.n_born_prey += 1
                        # weather or not reproduction actually occurred, the prey gets rewarded
                        self.rewards[
                            prey_instance.agent_name
                        ] += self.reproduction_reward_prey
            else:  
                self._deactivate_agent(prey_instance)
                self.n_active_agent_type[self.prey_type_nr] -= 1
                self.n_starved_prey += 1
                self.agent_age_of_death_list_type[self.prey_type_nr].append(prey_instance.age)
                self.rewards[prey_instance.agent_name] += self.death_reward_prey

        # process grass (re)growth
        for grass_instance in self.possible_agent_instance_list_type[
            self.grass_type_nr
        ]:
            grass_energy_gain = min(
                grass_instance.energy_gain_per_step,
                max(self.max_energy_level_grass - grass_instance.energy, 0),
            )
            grass_instance.energy += grass_energy_gain
            if (
                grass_instance.energy >= self.initial_energy_grass
                and not grass_instance.is_active
            ):
                self._activate_agent(grass_instance)
                self.n_active_agent_type[self.grass_type_nr] += 1
            elif (
                grass_instance.energy < self.initial_energy_grass
                and grass_instance.is_active
            ):
                self._deactivate_agent(grass_instance)
                self.n_active_agent_type[self.grass_type_nr] -= 1
        # 3] record step metrics
        self._record_population_metrics()
        self.n_cycles += 1
