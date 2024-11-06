"""
pred/prey/grass PettingZoo multi-agent learning environment
this environment transfers the energy of eaten prey/grass to the predator/prey
"""
# discretionary libraries
from predpreygrass.envs._so_predpreygrass_v0.agents.so_discrete_agent import (
    DiscreteAgent
)
from predpreygrass.utils.renderer import render
# external libraries
from gymnasium.utils import seeding
from gymnasium import spaces
import numpy as np
import pygame
import os
import random
from typing import List, Dict, Optional
import types


class PredPreyGrass:
    def __init__(
        self,
        x_grid_size: int = 25,
        y_grid_size: int = 25,
        max_cycles: int = 10000,
        n_possible_predator: int = 18,
        n_possible_prey: int = 24,
        n_possible_grass: int = 30,
        n_initial_active_predator: int = 6,
        n_initial_active_prey: int = 8,
        max_observation_range: int = 7,
        obs_range_predator: int = 5,
        obs_range_prey: int = 7,
        render_mode: Optional[str] = None,
        energy_gain_per_step_predator: float = -0.15,
        energy_gain_per_step_prey: float = -0.05,
        energy_gain_per_step_grass: float = 0.2,
        initial_energy_predator: float = 5.0,
        initial_energy_prey: float = 5.0,
        initial_energy_grass: float = 3.0,
        cell_scale: int = 40,
        x_pygame_window: int = 0,
        y_pygame_window: int = 0,
        regrow_grass: bool = True,
        prey_creation_energy_threshold: float = 10.0,
        predator_creation_energy_threshold: float = 10.0,
        create_prey: bool = True,
        create_predator: bool = True,
        step_reward_predator: float = 0.0,
        step_reward_prey: float = 0.0,
        step_reward_grass: float = 0.0,
        catch_reward_prey: float = 0.0,
        catch_reward_grass: float = 0.0,
        death_reward_prey: float = 0.0,
        death_reward_predator: float = 0.0,
        reproduction_reward_prey: float = 10.0,
        reproduction_reward_predator: float = 10.0,
        catch_prey_energy: float = 5.0,
        catch_grass_energy: float = 3.0,
        watch_grid_model: bool = False,
        show_energy_chart: bool = True,
        max_energy_level_grass: float = 4.0,
        spawning_area_predator: dict = dict(
            {
                "x_begin": 0,
                "y_begin": 0,
                "x_end": 24,
                "y_end": 24,
            }
        ),
        spawning_area_prey: dict = dict(
            {
                "x_begin": 0,
                "y_begin": 0,
                "x_end": 24,
                "y_end": 24,
            }
        ),
        spawning_area_grass: dict = dict(
            {
                "x_begin": 0,
                "y_begin": 0,
                "x_end": 24,
                "y_end": 24,
            }
        ),
        num_episodes: int = 100,
    ):
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.max_cycles = max_cycles
        self.n_possible_predator = n_possible_predator
        self.n_possible_prey = n_possible_prey
        self.n_possible_grass = n_possible_grass
        self.n_initial_active_predator = n_initial_active_predator
        self.n_initial_active_prey = n_initial_active_prey
        self.max_observation_range = max_observation_range
        self.obs_range_predator = obs_range_predator
        self.obs_range_prey = obs_range_prey
        self.render_mode = render_mode
        self.energy_gain_per_step_predator = energy_gain_per_step_predator
        self.energy_gain_per_step_prey = energy_gain_per_step_prey
        self.energy_gain_per_step_grass = energy_gain_per_step_grass
        self.cell_scale = cell_scale
        self.initial_energy_predator = initial_energy_predator
        self.initial_energy_prey = initial_energy_prey
        self.initial_energy_grass = initial_energy_grass
        self.x_pygame_window = x_pygame_window
        self.y_pygame_window = y_pygame_window
        self.catch_reward_grass = catch_reward_grass
        self.catch_reward_prey = catch_reward_prey
        self.regrow_grass = regrow_grass
        self.prey_creation_energy_threshold = prey_creation_energy_threshold
        self.predator_creation_energy_threshold = predator_creation_energy_threshold
        self.create_prey = create_prey
        self.create_predator = create_predator
        self.death_reward_prey = death_reward_prey
        self.death_reward_predator = death_reward_predator
        self.reproduction_reward_prey = reproduction_reward_prey
        self.reproduction_reward_predator = reproduction_reward_predator
        self.catch_prey_energy = catch_prey_energy
        self.catch_grass_energy = catch_grass_energy
        self.show_energy_chart = show_energy_chart
        self.step_reward_predator = step_reward_predator
        self.step_reward_prey = step_reward_prey
        self.step_reward_grass = step_reward_grass
        self.max_energy_level_grass = max_energy_level_grass
        self.spawning_area_predator = spawning_area_predator
        self.spawning_area_prey = spawning_area_prey
        self.spawning_area_grass = spawning_area_grass
        self.watch_grid_model = watch_grid_model
        self.num_episodes = num_episodes

        self._seed()

        self._initialize_variables()

        # observation spac
        obs_space = spaces.Box(
            low=0,
            high=max(self.max_energy_level_predator,self.max_energy_level_prey),  # maximum energy level of agents
            shape=(
                self.max_observation_range,
                self.max_observation_range,
                self.nr_observation_channels,
            ),
            dtype=np.float64,
        )
        self.observation_space = [obs_space for _ in range(self.n_possible_agents)]
        # end observations

        # actions
        self.motion_range: List[List[int]] = [
            [-1, 0],  # move left (in a pygame grid)
            [0, -1],  # move up
            [0, 0],  # stay
            [0, 1],  # move down
            [1, 0],  # move right
        ]
        self.n_actions_agent: int = len(self.motion_range)
        action_space_agent = spaces.Discrete(self.n_actions_agent)
        self.action_space = [action_space_agent for _ in range(self.n_possible_agents)]
        # end actions

        self.file_name: int = 0

        # visualization grid
        # pygame screen position window
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (
            self.x_pygame_window,
            self.y_pygame_window,
        )
        self.screen = None
        self.save_image_steps: bool = False  # save step images of the environment
        # width of energy chart
        self.width_energy_chart: int = 1800 if self.show_energy_chart else 0
        self.height_energy_chart: int = self.cell_scale * self.y_grid_size
        if self.n_possible_agent_type[self.predator_type_nr] > 18 or self.n_possible_agent_type[self.prey_type_nr] > 24:
            # too many agents to display on screen in energy chart
            self.show_energy_chart: bool = False
            self.width_energy_chart: int = 0
        # end visualization

        # Assign the imported render function to the instance using MethodType
        # to bind the function to the instance to be able to call it in 
        # another file (like so_predpreygrass.py); pretty ugly hack...
        # TODO import a class instead of a function
        self.render = types.MethodType(render, self)

    def reset(self):
        self._initialize_variables()
        # create agents of all types excluding "wall"-agents
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            agent_type_name = self.agent_type_name_list[agent_type_nr]
            # intialize all possible agents of a certain type (agent_type_nr)
            for _ in range(self.n_agent_type_list[agent_type_nr]):
                agent_id_nr = self.agent_id_counter
                agent_name = agent_type_name + "_" + str(agent_id_nr)
                self.agent_id_counter += 1
                agent_instance = DiscreteAgent(
                    agent_type_nr,
                    agent_id_nr,
                    agent_name,
                    # needed to detect if a cell is allready occupied by an agent of the same type
                    self.model_state[agent_type_nr], 
                    observation_range=self.obs_range_list[agent_type_nr],
                    motion_range=self.motion_range,
                    initial_energy=self.initial_energy_type[agent_type_nr],
                    energy_gain_per_step=self.energy_gain_per_step_list[agent_type_nr],
                )
                #  choose a cell for the agent which is not yet occupied by another agent of the same type
                #  and which is within the spawning area of the agent
                agent_instance.position = self._get_new_allowed_position(
                    agent_instance, self.spawning_area, self.model_state
                )
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                agent_instance.is_active = True
                agent_instance.energy = self.initial_energy_type[agent_type_nr]
                self._link_agent_to_grid(agent_instance)
                self.possible_agent_instance_list_type[agent_type_nr].append(agent_instance)

        agent_types = [self.predator_type_nr, self.prey_type_nr, self.grass_type_nr]

        for agent_type in agent_types:
            # Copy possible agent instances to active agent instances
            self.active_agent_instance_list_type[agent_type] = self.possible_agent_instance_list_type[agent_type].copy()

            # Create identical twins for the active agents for generic iterating purposes
            self.active_agent_instance_list_type[agent_type] = self.possible_agent_instance_list_type[agent_type].copy()

            # Create agent name lists from instance lists
            self.possible_agent_name_list_type[agent_type] = self._create_agent_name_list_from_instance_list(
                self.possible_agent_instance_list_type[agent_type]
            )

        self.start_index_type: list[int] = []
        self.start_index_type[self.predator_type_nr] = 0,
        self.start_index_type[self.prey_type_nr] = self.n_possible_agent_type[self.predator_type_nr],
        self.start_index_type[self.grass_type_nr] = self.n_possible_agent_type[self.predator_type_nr] + self.n_possible_agent_type[self.prey_type_nr] 
        

        # deactivate agents which can be created later at runtime
        for predator_instance in self.possible_agent_instance_list_type[self.predator_type_nr]:
            if predator_instance.agent_id_nr >= self.start_index_type[self.predator_type_nr] + self.n_initial_active_agent_type[self.predator_type_nr]:  
                self.active_agent_instance_list_type[self.predator_type_nr].remove(predator_instance)
                self.n_active_agent_type[self.predator_type_nr] -= 1
                self.total_energy_predator -= predator_instance.energy
                predator_instance.is_active = False
                predator_instance.energy = 0.0
                self._unlink_agent_from_grid(predator_instance)

        for prey_instance in self.possible_agent_instance_list_type[self.prey_type_nr]:
            if prey_instance.agent_id_nr >= self.start_index_type[self.prey_type_nr] + self.n_initial_active_agent_type[self.prey_type_nr]:
                self.active_agent_instance_list_type[self.prey_type_nr].remove(prey_instance)
                self.n_active_agent_type[self.prey_type_nr] -= 1
                self.total_energy_prey -= prey_instance.energy
                self._unlink_agent_from_grid(prey_instance)
                prey_instance.is_active = False
                prey_instance.energy = 0.0

        self.learning_agent_types = [self.predator_type_nr, self.prey_type_nr]
        for agent_type in self.learning_agent_types:
            for agent_instance in self.possible_agent_instance_list_type[agent_type]:
                if agent_instance.agent_id_nr >= self.start_index_type[self.prey_type_nr] + self.n_initial_active_agent_type[self.prey_type_nr]:
                    self.active_agent_instance_list_type[agent_type].remove(agent_instance)
                    self.n_active_agent_type[agent_type] -= 1
                    self.total_energy_agent_type[agent_type] -= agent_instance.energy
                    self._unlink_agent_from_grid(agent_instance)
                    agent_instance.is_active = False
                    agent_instance.energy = 0.0

        # define the learning agents
        self.active_agent_instance_list = (
            self.active_agent_instance_list_type[self.predator_type_nr] + self.active_agent_instance_list_type[self.prey_type_nr]
        )
        self.possible_agent_name_list = (
            self.possible_agent_name_list_type[self.predator_type_nr] + self.possible_agent_name_list_type[self.prey_type_nr]
        )        
        self.n_active_predator_list.insert(self.n_cycles, self.n_active_agent_type[self.predator_type_nr])
        self.n_active_prey_list.insert(self.n_cycles, self.n_active_agent_type[self.prey_type_nr])
        self.n_active_grass_list.insert(self.n_cycles, self.n_active_agent_type[self.grass_type_nr])

    def step(self, actions):
        """
        1. take actions of all active learning agents
        2. update energy changes of all agents as a concequence of actions:  # TODO
        3, update energy as a concequence of time: 
            - homeostasis: predator and prey lose energy over time
            - regrow and activate grass if energy is above threshold
        4. (de)activate agents if energy threshold is breached 
            - predator: if energy is below threshold, deactivate predator
            - prey: if energy is below threshold, deactivate prey
            - grass: if energy gets above threshold, activate grass
        3. apply rules of engagement
            - predator eats prey => predator energy += prey energy => prey dies => predator reproduces? if so: gets rewarded
            - prey eats grass => prey energy += grass energy => grass dies => prey reproduces? if so gets rewarded
        """
        # 1] apply actions (i.e. movements) for all active agents in parallel
        for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr]:
            self._apply_agent_action(predator_instance, actions[predator_instance.agent_name])
        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
            self._apply_agent_action(prey_instance, actions[prey_instance.agent_name])

        self._reset_rewards()

        # 2] apply rules of engagement for all agents
        # in that way active list does not get modified during iteration
        for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr][:]: # make a copy to make removal during iteration possible
            if predator_instance.energy > 0:
                # engagement with environment: "nature and time"
                predator_instance.age += 1
                predator_instance.energy += predator_instance.energy_gain_per_step
                #predator_instance.energy += predator_action_energy 
                # engagement with environment: "other agents"
                prey_instance_in_predator_cell = self.agent_instance_in_grid_location[self.prey_type_nr, *predator_instance.position]
                if prey_instance_in_predator_cell:
                    predator_instance.energy += prey_instance_in_predator_cell.energy
                    self._deactivate_agent(prey_instance_in_predator_cell)
                    self.n_active_agent_type[self.prey_type_nr] -= 1
                    self.n_eaten_prey += 1
                    self.prey_age_of_death_list.append(prey_instance_in_predator_cell.age)
                    self.rewards[predator_instance.agent_name] += self.catch_reward_prey
                    self.rewards[prey_instance_in_predator_cell.agent_name] += self.death_reward_prey
                    if predator_instance.energy > self.predator_creation_energy_threshold:
                        potential_new_predator_instance_list = self.non_active_agent_instance_list(predator_instance)
                        if potential_new_predator_instance_list:
                            self._reproduce_new_agent(predator_instance, potential_new_predator_instance_list)
                            self.n_active_agent_type[self.predator_type_nr] += 1
                            self.n_born_predator += 1
                        # weather or not reproduction actually occurred, the predator gets rewarded 
                        self.rewards[predator_instance.agent_name] += self.reproduction_reward_predator
            else: # predator_instance.energy <= 0
                self._deactivate_agent(predator_instance)
                self.n_active_agent_type[self.predator_type_nr] -= 1
                self.n_starved_predator += 1
                self.predator_age_of_death_list.append(predator_instance.age)
                self.rewards[predator_instance.agent_name] += self.death_reward_predator

        for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr][:]: # make a copy to make removal during iteration possible
            if prey_instance.energy > 0:
                # engagement with environment: "nature and time"
                prey_instance.age += 1
                prey_instance.energy += prey_instance.energy_gain_per_step
                # engagement with environmeny: "other agents"   
                grass_instance_in_prey_cell = self.agent_instance_in_grid_location[self.grass_type_nr, *prey_instance.position]
                if grass_instance_in_prey_cell:
                    prey_instance.energy += grass_instance_in_prey_cell.energy
                    self._deactivate_agent(grass_instance_in_prey_cell)
                    self.n_active_agent_type[self.grass_type_nr] -= 1
                    self.n_eaten_grass += 1
                    self.rewards[prey_instance.agent_name] += self.catch_reward_grass
                    if prey_instance.energy > self.prey_creation_energy_threshold:
                        potential_new_prey_instance_list = self.non_active_agent_instance_list(prey_instance)
                        if potential_new_prey_instance_list:
                            self._reproduce_new_agent(prey_instance, potential_new_prey_instance_list)
                            self.n_active_agent_type[self.prey_type_nr] += 1
                            self.n_born_prey += 1
                        # weather or not reproduction actually occurred, the prey gets rewarded 
                        self.rewards[prey_instance.agent_name] += self.reproduction_reward_prey
            # TODO replace by "else:" # i.e. prey_instance.energy <= 0
            else: # prey_instance.energy <= 0
                self._deactivate_agent(prey_instance)
                self.n_active_agent_type[self.prey_type_nr] -= 1
                self.n_starved_prey += 1
                self.prey_age_of_death_list.append(prey_instance.age)
                self.rewards[prey_instance.agent_name] += self.death_reward_prey

        # process grass (re)growth
        for grass_instance in self.possible_agent_instance_list_type[self.grass_type_nr]:
            grass_energy_gain = min(grass_instance.energy_gain_per_step, max(self.max_energy_level_grass - grass_instance.energy, 0))
            grass_instance.energy += grass_energy_gain
            if grass_instance.energy >= self.initial_energy_grass and not grass_instance.is_active:
                self._activate_agent(grass_instance)
                self.n_active_agent_type[self.grass_type_nr] += 1
            elif grass_instance.energy < self.initial_energy_grass and grass_instance.is_active:
                self._deactivate_agent(grass_instance)
                self.n_active_agent_type[self.grass_type_nr] -= 1
        # 3] record step metrics
        self._record_population_metrics()
        self.n_cycles += 1

    def observe(self, agent_name):
        max_obs_range = self.max_observation_range
        max_obs_offset = (max_obs_range - 1) // 2
        x_grid_size, y_grid_size = self.x_grid_size, self.y_grid_size
        nr_channels = self.nr_observation_channels

        # Retrieve agent position
        agent_instance = self.agent_name_to_instance_dict[agent_name]
        xp, yp = agent_instance.position
        observation_range_agent = agent_instance.observation_range

        # Observation clipping limits
        xlo = max(0, xp - max_obs_offset)
        xhi = min(x_grid_size, xp + max_obs_offset + 1)
        ylo = max(0, yp - max_obs_offset)
        yhi = min(y_grid_size, yp + max_obs_offset + 1)

        # Observation matrix initialization (filled with 1s in the wall channel only)
        observation = np.zeros((nr_channels, max_obs_range, max_obs_range), dtype=np.float64)
        observation[0].fill(1.0)

        # Calculate bounds for observation array assignment
        xolo = max(0, max_obs_offset - xp)
        xohi = xolo + (xhi - xlo)
        yolo = max(0, max_obs_offset - yp)
        yohi = yolo + (yhi - ylo)

        # Populate observation within visible area
        observation[:, xolo:xohi, yolo:yohi] = np.abs(self.model_state[:, xlo:xhi, ylo:yhi])

        # Apply mask to limit observation range based on agent's capabilities
        mask = (max_obs_range - observation_range_agent) // 2
        if mask > 0:
            observation[:, :mask, :] = 0
            observation[:, -mask:, :] = 0
            observation[:, :, :mask] = 0
            observation[:, :, -mask:] = 0
        elif mask < 0:
            raise ValueError("Error: observation_range_agent larger than max_observation_range")

        return observation

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    @property
    def is_no_grass(self):
        if self.n_active_agent_type[self.grass_type_nr] == 0:
            return True
        return False

    @property
    def is_no_prey(self):
        if self.n_active_agent_type[self.prey_type_nr] == 0:
            return True
        return False

    @property
    def is_no_predator(self):
        if self.n_active_agent_type[self.predator_type_nr] == 0:
            return True
        return False

    def _apply_agent_action(self, agent_instance, action):
        self._unlink_agent_from_grid(agent_instance)
        agent_instance.step(action)
        self._link_agent_to_grid(agent_instance)

    def non_active_agent_instance_list(self, agent_instance):
        return [
            agent_instance for agent_instance in self.possible_agent_instance_list_type[agent_instance.agent_type_nr]
            if not agent_instance.is_active
        ]

    def _reproduce_new_agent(self, parent_agent_instance, non_active_agent_instance_list ):
        agent_type_nr = parent_agent_instance.agent_type_nr
        parent_agent_instance.energy -= self.initial_energy_type[agent_type_nr]
        self.model_state[
            agent_type_nr, parent_agent_instance.position[0], parent_agent_instance.position[1]
        ] = parent_agent_instance.energy
        # activate a new agent
        new_agent_instance = non_active_agent_instance_list[-1]
        new_agent_instance.position = self._get_new_allowed_position(
            new_agent_instance, self.spawning_area, self.model_state
        )
        self._activate_agent(new_agent_instance)

    def _unlink_agent_from_grid(self, agent_instance):
        self.agent_instance_in_grid_location[agent_instance.agent_type_nr, *agent_instance.position] = None
        self.model_state[agent_instance.agent_type_nr, *agent_instance.position] = 0.0

    def _link_agent_to_grid(self, agent_instance):
        self.agent_instance_in_grid_location[agent_instance.agent_type_nr, *agent_instance.position] = agent_instance
        self.model_state[agent_instance.agent_type_nr, *agent_instance.position] = agent_instance.energy

    def _activate_agent(self, agent_instance):
        agent_instance.is_active = True
        agent_instance.energy = self.initial_energy_type[agent_instance.agent_type_nr]
        agent_instance.age = 0
        self.active_agent_instance_list_type[agent_instance.agent_type_nr].append(agent_instance)
        self._link_agent_to_grid(agent_instance)

    def _deactivate_agent(self, agent_instance):
        self._unlink_agent_from_grid(agent_instance)
        agent_instance.is_active = False
        agent_instance.energy = 0.0
        self.active_agent_instance_list_type[agent_instance.agent_type_nr].remove(agent_instance)

    def _get_new_allowed_position(
        self, new_agent_instance, spawning_area, model_state
    ):
        available_cell_list = [
            (x, y)
            for x in range(spawning_area[new_agent_instance.agent_type_nr]["x_begin"], spawning_area[new_agent_instance.agent_type_nr]["x_end"] + 1)
            for y in range(spawning_area[new_agent_instance.agent_type_nr]["y_begin"], spawning_area[new_agent_instance.agent_type_nr]["y_end"] + 1)
            if model_state[new_agent_instance.agent_type_nr, x, y] == 0.0
        ]
        return random.choice(available_cell_list)

    def _seed(self, seed=None, options=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _create_agent_name_list_from_instance_list(self, agent_instance_list):
        return [agent_instance.agent_name for agent_instance in agent_instance_list]
       
    def _reset_rewards(self):
        self.rewards = dict.fromkeys(self.possible_agent_name_list, 0.0)

    def _record_population_metrics(self):
        self.n_active_predator_list.append(self.n_active_agent_type[self.predator_type_nr])
        self.n_active_prey_list.append(self.n_active_agent_type[self.prey_type_nr])
        self.n_active_grass_list.append(self.n_active_agent_type[self.grass_type_nr])

    def _initialize_variables(self):
        # id
        self.agent_id_counter: int = 0

        self.n_cycles: int = 0

        # agent types definitions
        self.agent_type_name_list: List[str] = ["wall", "predator", "prey", "grass"]
        self.predator_type_nr = 1
        self.prey_type_nr = 2
        self.grass_type_nr = 3
        self.nr_observation_channels: int = len(self.agent_type_name_list)

        self.n_possible_agent_type: List[int] = [0, 0, 0, 0]
        self.n_possible_agent_type[self.predator_type_nr] = self.n_possible_predator
        self.n_possible_agent_type[self.prey_type_nr] = self.n_possible_prey
        self.n_possible_agent_type[self.grass_type_nr] = self.n_possible_grass

        self.n_initial_active_agent_type: List[int] = [0, 0, 0, 0]
        self.n_initial_active_agent_type[self.predator_type_nr] = self.n_initial_active_predator
        self.n_initial_active_agent_type[self.prey_type_nr] = self.n_initial_active_prey

        self.n_possible_agents: int = self.n_possible_agent_type[self.predator_type_nr] + self.n_possible_agent_type[self.prey_type_nr]


        self.possible_agent_instance_list_type: List[List[DiscreteAgent]] = [
            [] for _ in range(len(self.agent_type_name_list))
        ]
        self.active_agent_instance_list_type: List[List[DiscreteAgent]] = [
            [] for _ in range(len(self.agent_type_name_list))
        ]
        self.possible_agent_name_list_type: List[List[DiscreteAgent]] = [
            [] for _ in range(len(self.agent_type_name_list))
        ]
        self.active_agent_name_list_type: List[List[DiscreteAgent]] = [
            [] for _ in range(len(self.agent_type_name_list))
        ]

        self.n_agent_type_list: List[int] = [
            0,  # wall agents
            self.n_possible_agent_type[self.predator_type_nr],
            self.n_possible_agent_type[self.prey_type_nr],
            self.n_possible_agent_type[self.grass_type_nr],
        ]
        self.obs_range_list: List[int] = [
            0,  # wall has no observation range
            self.obs_range_predator,
            self.obs_range_prey,
            0,  # grass has no observation range
        ]
        self.initial_energy_type: List[int] = [
            0,
            self.initial_energy_predator,
            self.initial_energy_prey,
            self.initial_energy_grass,
        ]
        self.energy_gain_per_step_list: List[int] = [
            0,
            self.energy_gain_per_step_predator,
            self.energy_gain_per_step_prey,
            self.energy_gain_per_step_grass,
        ]
        # boundaries for the spawning of agents within the grid
        # Initialize a spawning area for the agents
        self.spawning_area = [{}, {}, {}, {}]
        self.spawning_area.insert(self.predator_type_nr, self.spawning_area_predator)
        self.spawning_area.insert(self.prey_type_nr, self.spawning_area_prey)
        self.spawning_area.insert(self.grass_type_nr, self.spawning_area_grass)


        self.possible_agent_name_list: List[str] = []
        self.active_agent_instance_list: List[DiscreteAgent] = []

        # episode population metrics
        self.n_active_agent_type: List[int] = [0, 0, 0, 0]
        self.n_active_agent_type[self.predator_type_nr] = self.n_possible_agent_type[self.predator_type_nr]
        self.n_active_agent_type[self.prey_type_nr] = self.n_possible_agent_type[self.prey_type_nr]
        self.n_active_agent_type[self.grass_type_nr] = self.n_possible_agent_type[self.grass_type_nr]

        self.n_active_predator_list: List[int] = []
        self.n_active_prey_list: List[int] = []
        self.n_active_grass_list: List[int] = []


        self.n_active_agent_type: List[int] = [
            0,
            self.n_active_agent_type[self.predator_type_nr],
            self.n_active_agent_type[self.prey_type_nr],
            self.n_active_agent_type[self.grass_type_nr],
        ]
        self.n_possible_agent_type: List[int] = [
            0,
            self.n_possible_agent_type[self.predator_type_nr],
            self.n_possible_agent_type[self.prey_type_nr],
            self.n_possible_agent_type[self.grass_type_nr],
        ]

        

        self.total_energy_predator: float = self.n_active_agent_type[self.predator_type_nr] * self.initial_energy_predator
        self.total_energy_prey: float = self.n_active_agent_type[self.prey_type_nr] * self.initial_energy_prey
        self.total_energy_grass: float = self.n_active_agent_type[self.grass_type_nr] * self.initial_energy_grass
        self.total_energy_learning_agents: float = self.total_energy_predator + self.total_energy_prey

        # note: prey can become inactive due to starvation or getting eaten by predators
        self.n_starved_predator: int = 0
        self.n_starved_prey: int = 0
        self.n_eaten_prey: int = 0
        self.n_eaten_grass: int = 0
        self.n_born_predator: int = 0
        self.n_born_prey: int = 0

        self.predator_age_of_death_list: List[int] = []
        self.prey_age_of_death_list: List[int] = []


        # initalize model state
        self.model_state: np.ndarray = np.zeros(
            (self.nr_observation_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )
        # lookup record for agent instances per grid location
        self.agent_instance_in_grid_location = np.empty(
            (len(self.agent_type_name_list), self.x_grid_size, self.y_grid_size), dtype=object
        )
        # intialization per agent type
        self.agent_types = [self.predator_type_nr, self.prey_type_nr, self.grass_type_nr]
        for agent_type_nr in self.agent_types:
            self.agent_instance_in_grid_location[agent_type_nr] = np.full(
                (self.x_grid_size, self.y_grid_size), None
            )
            self.possible_agent_name_list_type[agent_type_nr] = []
            self.active_agent_name_list_type[agent_type_nr] = []
            self.possible_agent_instance_list_type[agent_type_nr] = []
            self.active_agent_instance_list_type[agent_type_nr] = []            

        # lookup record for agent instances per agent name
        self.agent_name_to_instance_dict: Dict[str, DiscreteAgent] = {}

        # creation agent name lists
        predator_id_nr_range = range(0, self.n_possible_agent_type[self.predator_type_nr])
        prey_id_nr_range = range(self.n_possible_agent_type[self.predator_type_nr], self.n_possible_agent_type[self.prey_type_nr] + self.n_possible_agent_type[self.predator_type_nr])
        grass_id_nr_range = range(self.n_possible_agent_type[self.prey_type_nr] + self.n_possible_agent_type[self.predator_type_nr], self.n_possible_agent_type[self.prey_type_nr] + self.n_possible_agent_type[self.predator_type_nr] + self.n_possible_agent_type[self.grass_type_nr])
        self.possible_agent_name_list_type[self.predator_type_nr] = ["predator" + "_" + str(a) for a in predator_id_nr_range]
        self.possible_agent_name_list_type[self.prey_type_nr] = ["prey" + "_" + str(a) for a in prey_id_nr_range]
        self.possible_agent_name_list_type[self.grass_type_nr] = ["grass" + "_" + str(a) for a in grass_id_nr_range]
        self.possible_agent_name_list = (
            self.possible_agent_name_list_type[self.predator_type_nr] + self.possible_agent_name_list_type[self.prey_type_nr]
        )
        self._reset_rewards()

