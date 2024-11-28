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
            # one iteration insteadd of three separate loops for al biotic agents (Predator, Prey, Grass) 
            for agent_instance in self.possible_agent_instance_list:
                agent_type_nr = agent_instance.agent_type_nr
                if agent_instance.is_active:
                    if agent_instance.energy > 0:
                        x_new, y_new = agent_instance.position
                        agent_instance.energy += agent_instance.energy_gain_per_step
                        if agent_type_nr == 1:
                            prey_instance_in_predator_cell = (
                                self.agent_instance_in_grid_location[self.prey_type_nr][
                                    (x_new, y_new)
                                ]
                            )

                            if prey_instance_in_predator_cell is not None:
                                agent_instance.energy += prey_instance_in_predator_cell.energy
                                self._deactivate_agent(prey_instance_in_predator_cell)
                                self.n_active_agent_type[self.prey_type_nr] -= 1
                                self.agent_age_of_death_list_type[self.prey_type_nr].append(prey_instance_in_predator_cell.age)
                                self.n_eaten_prey += 1
                                self.rewards[
                                    agent_instance.agent_name
                                ] += self.catch_reward_prey
                                self.rewards[
                                    prey_instance_in_predator_cell.agent_name
                                ] += self.death_reward_prey
                                if (
                                    agent_instance.energy
                                    > self.predator_creation_energy_threshold
                                ):          
                
                                    potential_new_predator_instance_list = (
                                        self._non_active_agent_instance_list(agent_instance)
                                    )

                                    if potential_new_predator_instance_list:
                                        self._reproduce_new_agent(
                                            agent_instance, potential_new_predator_instance_list
                                        )
                                        self.n_active_agent_type[agent_type_nr] += 1
                                        self.n_born_predator += 1
                                    self.rewards[
                                        agent_instance.agent_name
                                    ] += self.reproduction_reward_predator

                        elif agent_type_nr == 2:

                            grass_instance_in_prey_cell = self.agent_instance_in_grid_location[
                                self.grass_type_nr][(x_new, y_new)]
                            if grass_instance_in_prey_cell is not None:
                                agent_instance.energy += grass_instance_in_prey_cell.energy
                                self._deactivate_agent(grass_instance_in_prey_cell) 
                                self.n_active_agent_type[self.grass_type_nr] -= 1
                                self.n_eaten_grass += 1
                                self.rewards[agent_instance.agent_name] += self.catch_reward_grass
                                if agent_instance.energy > self.prey_creation_energy_threshold:
                                    potential_new_prey_instance_list = (
                                        self._non_active_agent_instance_list(agent_instance)
                                    )
                                    if potential_new_prey_instance_list:
                                        self._reproduce_new_agent( agent_instance, potential_new_prey_instance_list)
                                        self.n_active_agent_type[agent_type_nr] += 1
                                        self.n_born_prey += 1

                                    # weather or not reproduction actually occurred, the prey gets rewarded
                                    self.rewards[agent_instance.agent_name] += self.reproduction_reward_prey

                        elif agent_type_nr == 3:
                            grass_energy_gain = min(
                                agent_instance.energy_gain_per_step,
                                max(self.max_energy_level_grass - agent_instance.energy, 0),
                            )
                            agent_instance.energy += grass_energy_gain


                    else:
                        self._deactivate_agent(agent_instance)
                        self.n_active_agent_type[agent_type_nr] -= 1
                        self.agent_age_of_death_list_type[agent_type_nr].append(agent_instance.age)
                        if agent_type_nr == 1:
                            self.n_starved_predator += 1
                            self.rewards[agent_instance.agent_name] += self.death_reward_predator
                        elif agent_type_nr == 2:
                            self.n_starved_prey += 1
                            self.rewards[agent_instance.agent_name] += self.death_reward_prey

                elif agent_type_nr == 3:
                    grass_energy_gain = min(
                        agent_instance.energy_gain_per_step,
                        max(self.max_energy_level_grass - agent_instance.energy, 0),
                    )
                    agent_instance.energy += grass_energy_gain
                    if agent_instance.energy >= self.initial_energy_type[agent_type_nr]:
                        self._activate_agent(agent_instance)
                        self.n_active_agent_type[self.grass_type_nr] += 1
    
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
        raise NotImplementedError("Parallel environment not implemented for this experiment.")
 