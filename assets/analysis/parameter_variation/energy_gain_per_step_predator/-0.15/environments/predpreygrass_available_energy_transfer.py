"""
pred/prey/grass PettingZoo multi-agent learning environment

"""
import os
import numpy as np
import random
from typing import List, Dict, Optional, TypeVar
import pygame
from collections import defaultdict

import gymnasium
from gymnasium.utils import seeding, EzPickle
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from agents.discrete_agent import DiscreteAgent
from pettingzoo.utils.env import AgentID

# agent types
PREDATOR_TYPE_NR = 1
PREY_TYPE_NR = 2
GRASS_TYPE_NR = 3

class PredPreyGrass:
    def __init__(
        self,
        x_grid_size: int = 16,
        y_grid_size: int = 16,
        max_cycles: int = 10000,
        n_possible_predator: int = 6,
        n_possible_prey: int = 8,
        n_possible_grass: int = 30,
        n_initial_active_predator: int = 6,
        n_initial_active_prey: int = 8,
        max_observation_range: int = 7,
        obs_range_predator: int = 5,
        obs_range_prey: int = 7,
        render_mode: Optional[str] = None,
        energy_gain_per_step_predator: float = -0.3,
        energy_gain_per_step_prey: float = -0.05,
        energy_gain_per_step_grass: float = 0.2,
        initial_energy_predator: float = 5.0,
        initial_energy_prey: float = 5.0,
        initial_energy_grass: float = 3.0,
        cell_scale: int = 40,
        x_pygame_window: int = 0,
        y_pygame_window: int = 0,
        regrow_grass: bool = False,
        prey_creation_energy_threshold: float = 10.0,
        predator_creation_energy_threshold: float = 10.0,
        create_prey: bool = False,
        create_predator: bool = False,
        step_reward_predator: float = -0.3,
        step_reward_prey: float = -0.05,
        step_reward_grass: float = 0.2,
        catch_reward_prey: float = 5.0,
        catch_reward_grass: float = 3.0,
        death_reward_prey: float = -10.0,
        death_reward_predator: float = -10.0,
        reproduction_reward_prey: float = 10.0,
        reproduction_reward_predator: float = 10.0,
        catch_prey_energy: float = 5.0,
        catch_grass_energy: float = 3.0,
        show_energy_chart: bool = True,
        max_energy_level_grass: float = 4.0,
        spawning_area_predator: dict = dict({
            "x_begin": 0,
            "y_begin": 0,
            "x_end": 7,
            "y_end": 7,
        }),
        spawning_area_prey: dict = dict({
            "x_begin": 0,
            "y_begin": 0,
            "x_end": 7,
            "y_end": 7,
        }),
        spawning_area_grass: dict = dict({
            "x_begin": 0,
            "y_begin": 0,
            "x_end": 7,
            "y_end": 7,
        }),

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

        # agent types
        
        self.agent_type_name_list: List[str] = ["wall", "predator", "prey", "grass"]
 
        # boundaries for the spawning of agents within the grid
        # Initialize a spawning area for the agents
        self.spawning_area = [{},{},{},{}]
        self.spawning_area.insert(PREDATOR_TYPE_NR, self.spawning_area_predator)
        self.spawning_area.insert(PREY_TYPE_NR, self.spawning_area_prey)
        self.spawning_area.insert(GRASS_TYPE_NR, self.spawning_area_grass)

        # visualization
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
        if self.n_possible_predator > 18 or self.n_possible_prey > 24:
            # too many agents to display on screen in energy chart
            self.show_energy_chart: bool = False
            self.width_energy_chart: int = 0
        # end visualization

        self._seed()
        self.agent_id_counter: int = 0


        # episode population metrics
        self.n_possible_agents: int = self.n_possible_predator + self.n_possible_prey
        self.n_active_predator: int = self.n_possible_predator
        self.n_active_prey: int = self.n_possible_prey
        self.n_active_grass: int = self.n_possible_grass
        self.n_active_predator_list: List[int] = []
        self.n_active_prey_list: List[int] = []
        self.n_active_grass_list: List[int] = []
        self.total_energy_predator: float = 0.0
        self.total_energy_prey: float = 0.0
        self.total_energy_grass: float = 0.0
        self.total_energy_predator_list: List[float] = []
        self.total_energy_prey_list: List[float] = []
        self.total_energy_grass_list: List[float] = []
        self.n_starved_predator: int = 0
        self.n_starved_prey: int = 0  # note: prey can become inactive due to starvation or getting eaten by predators
        self.n_eaten_prey: int = 0
        self.n_born_predator: int = 0
        self.n_born_prey: int = 0
        self.predator_age_list: List[int] = []
        self.prey_age_list: List[int] = []

        self.active_predator_instance_list: List[
            DiscreteAgent
        ] = []  # list of all active ("living") predators
        self.active_prey_instance_list: List[DiscreteAgent] = []  # list of active prey
        self.active_grass_instance_list: List[DiscreteAgent] = []  # list of active grass
        self.active_agent_instance_list: List[DiscreteAgent] = []  # list of active predators and prey
        self.possible_predator_name_list: List[AgentID] = []
        self.possible_prey_name_list: List[AgentID] = []
        self.possible_grass_name_list: List[AgentID] = []
        self.possible_agent_name_list: List[AgentID] = []

        # lookup record for agent instances per grid location
        self.agent_instance_in_grid_location = np.empty(
            (len(self.agent_type_name_list), x_grid_size, y_grid_size), dtype=object
        )
        # intialization
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            self.agent_instance_in_grid_location[agent_type_nr] = np.full(
                (self.x_grid_size, self.y_grid_size), None
            )

        # lookup record for agent instances per agent name
        self.agent_name_to_instance_dict: Dict[AgentID, DiscreteAgent] = {}

        # creation agent name lists
        predator_id_nr_range = range(0, self.n_possible_predator)
        prey_id_nr_range = range(
            self.n_possible_predator, self.n_possible_prey + self.n_possible_predator
        )
        grass_id_nr_range = range(
            self.n_possible_prey + self.n_possible_predator,
            self.n_possible_prey + self.n_possible_predator + self.n_possible_grass,
        )
        self.possible_predator_name_list = [
            "predator" + "_" + str(a) for a in predator_id_nr_range
        ]
        self.possible_prey_name_list = ["prey" + "_" + str(a) for a in prey_id_nr_range]
        self.possible_grass_name_list = ["grass" + "_" + str(a) for a in grass_id_nr_range]
        self.possible_agent_name_list = self.possible_predator_name_list + self.possible_prey_name_list

        # observations
        self.nr_observation_channels: int = len(self.agent_type_name_list)
        obs_space = spaces.Box(
            low=0,
            high=100,  # maximum energy level of agents
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

        # records for removal of agents
        self.agent_energy_from_eating_dict = dict(
            zip(self.possible_agent_name_list, [0.0 for _ in self.possible_agent_name_list]))

        self.prey_who_remove_grass_dict: Dict[AgentID, bool] = dict(
            zip(self.possible_prey_name_list, [False for _ in self.possible_prey_name_list])
        )
        self.grass_to_be_removed_by_prey_dict: Dict[AgentID, bool] = dict(
            zip(self.possible_grass_name_list, [False for _ in self.possible_grass_name_list])
        )
        self.predator_who_remove_prey_dict: Dict[AgentID, bool] = dict(
            zip(self.possible_predator_name_list, [False for _ in self.possible_predator_name_list])
        )
        self.prey_to_be_removed_by_predator_dict: Dict[AgentID, bool] = dict(
            zip(self.possible_prey_name_list, [False for _ in self.possible_prey_name_list])
        )
        self.predator_to_be_removed_by_starvation_dict: Dict[AgentID, bool] = dict(
            zip(self.possible_predator_name_list, [False for _ in self.possible_predator_name_list])
        )
        self.prey_to_be_removed_by_starvation_dict: Dict[AgentID, bool] = dict(
            zip(self.possible_prey_name_list, [False for _ in self.possible_prey_name_list])
        )
        # end records for removal agents

        self.file_name: int = 0
        self.n_aec_cycles: int = 0

        # TODO: upperbound for observation space = max(energy levels of all agents)
        self.max_energy_level_prey = 25.0 # in kwargs later, init level = 5.0
        self.max_energy_level_predator = 25.0 # in kwargs later, init level = 5.0

    def position_new_agent_on_gridworld(self, agent_instance, spawning_area, model_state):
        """
        ouputs a random position for a new agent on the gridworld, within the spawning 
        area of the agent and not on a cell which is already occupied by an agent of 
        the same type
        """
        agent_type_nr = agent_instance.agent_type_nr
        # intialize an empty list to store the available cells for spawning a new agent
        available_cell_list = []

        # Iterate over the range of x and y coordinates within the spawning area of the agent
        for x in range(spawning_area[agent_type_nr]["x_begin"], spawning_area[agent_type_nr]["x_end"]+1):
            for y in range(spawning_area[agent_type_nr]["y_begin"], spawning_area[agent_type_nr]["y_end"]+1):
                # checks if the cell is not already occupied by an agent of the same type
                if model_state[agent_type_nr][x,y]==0.0:
                    # If the cell is empty, append the (x, y) coordinate to the list
                    available_cell_list.append((x, y))    
        # Return a random choice from the available cell list
        return random.choice(available_cell_list)

    def reset(self):
        # empty agent lists
        self.active_predator_instance_list = []
        self.active_prey_instance_list = []
        self.active_grass_instance_list = []
        self.active_agent_instance_list = []

        self.possible_predator_name_list = []
        self.possible_prey_name_list = []
        self.possible_grass_name_list = []
        self.possible_agent_name_list = []

        agent_type_instance_list: List[List[DiscreteAgent]] = [
            [] for _ in range(len(self.agent_type_name_list))
        ]
        # record of agent ages
        self.predator_age_list = []
        self.prey_age_list = []

        # initialization
        self.n_active_predator = self.n_possible_predator
        self.n_active_prey = self.n_possible_prey
        self.n_active_grass = self.n_possible_grass
        self.total_energy_predator = self.n_active_predator * self.initial_energy_predator
        self.total_energy_prey = self.n_active_prey * self.initial_energy_prey
        self.total_energy_grass = self.n_active_grass * self.initial_energy_grass

        self.n_agent_type_list: List[int] = [
            0,  # wall agents
            self.n_possible_predator,
            self.n_possible_prey,
            self.n_possible_grass,
        ]
        self.obs_range_list: List[int] = [
            0,  # wall has no observation range
            self.obs_range_predator,
            self.obs_range_prey,
            0,  # grass has no observation range
        ]
        self.initial_energy_list: List[int] = [
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

        self.agent_id_counter = 0
        self.agent_name_to_instance_dict = {}
        self.model_state: np.ndarray = np.zeros(
            (self.nr_observation_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )
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
                    self.model_state[
                        agent_type_nr
                    ],  # needed to detect if a cell is allready occupied by an agent of the same type
                    observation_range=self.obs_range_list[agent_type_nr],
                    motion_range=self.motion_range,
                    initial_energy=self.initial_energy_list[agent_type_nr],
                    energy_gain_per_step=self.energy_gain_per_step_list[agent_type_nr],
                )

                #  choose a cell for the agent which is not yet occupied by another agent of the same type
                #  and which is within the spawning area of the agent
                xinit, yinit = self.position_new_agent_on_gridworld(agent_instance, self.spawning_area, self.model_state)
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                agent_instance.position = (xinit, yinit)
                agent_instance.is_active = True
                agent_instance.energy = self.initial_energy_list[agent_type_nr]
                self.model_state[agent_type_nr, xinit, yinit] = agent_instance.energy
                agent_type_instance_list[agent_type_nr].append(agent_instance)
                self.agent_instance_in_grid_location[
                    agent_type_nr, xinit, yinit
                ] = agent_instance

        self.active_predator_instance_list = agent_type_instance_list[
            PREDATOR_TYPE_NR
        ]
        self.active_prey_instance_list = agent_type_instance_list[PREY_TYPE_NR]
        self.active_grass_instance_list = agent_type_instance_list[GRASS_TYPE_NR]

        self.possible_predator_name_list = self.create_possible_agent_name_list_from_instance_list(
            self.active_predator_instance_list
        )
        self.possible_prey_name_list = self.create_possible_agent_name_list_from_instance_list(
            self.active_prey_instance_list
        )
        self.possible_grass_name_list = self.create_possible_agent_name_list_from_instance_list(
            self.active_grass_instance_list
        )

        # deactivate agents which can be created later at runtime
        predator_name: AgentID
        for predator_name in self.possible_predator_name_list:
            predator_instance = self.agent_name_to_instance_dict[predator_name]
            if (
                predator_instance.agent_id_nr >= self.n_initial_active_predator
            ):  # number of initial active predators
                self.active_predator_instance_list.remove(predator_instance)
                self.n_active_predator -= 1
                self.total_energy_predator -= predator_instance.energy
                self.agent_instance_in_grid_location[
                    PREDATOR_TYPE_NR,
                    predator_instance.position[0],
                    predator_instance.position[1],
                ] = None
                self.model_state[
                    PREDATOR_TYPE_NR,
                    predator_instance.position[0],
                    predator_instance.position[1],
                ] = 0.0
                predator_instance.is_active = False
                predator_instance.energy = 0.0
        for prey_name in self.possible_prey_name_list:
            prey_instance = self.agent_name_to_instance_dict[prey_name]
            if (
                prey_instance.agent_id_nr
                >= self.n_possible_predator + self.n_initial_active_prey
            ):  # number of initial active prey
                self.active_prey_instance_list.remove(prey_instance)
                self.n_active_prey -= 1
                self.total_energy_prey -= prey_instance.energy
                self.agent_instance_in_grid_location[
                    PREY_TYPE_NR,
                    prey_instance.position[0],
                    prey_instance.position[1],
                ] = None
                self.model_state[
                    PREY_TYPE_NR,
                    prey_instance.position[0],
                    prey_instance.position[1],
                ] = 0.0
                prey_instance.is_active = False
                prey_instance.energy = 0.0

        # removal agents set to false
        self.prey_who_remove_grass_dict = dict(
            zip(self.possible_prey_name_list, [False for _ in self.possible_prey_name_list])
        )
        self.grass_to_be_removed_by_prey_dict = dict(
            zip(self.possible_grass_name_list, [False for _ in self.possible_grass_name_list])
        )
        self.predator_who_remove_prey_dict = dict(
            zip(self.possible_predator_name_list, [False for _ in self.possible_predator_name_list])
        )
        self.prey_to_be_removed_by_predator_dict = dict(
            zip(self.possible_prey_name_list, [False for _ in self.possible_prey_name_list])
        )
        self.prey_to_be_removed_by_starvation_dict = dict(
            zip(self.possible_prey_name_list, [False for _ in self.possible_prey_name_list])
        )
        self.predator_to_be_removed_by_starvation_dict = dict(
            zip(self.possible_predator_name_list, [False for _ in self.possible_predator_name_list])
        )


        # define the learning agents
        self.active_agent_instance_list = self.active_predator_instance_list + self.active_prey_instance_list
        self.possible_agent_name_list = self.possible_predator_name_list + self.possible_prey_name_list

        self.agent_reward_dict: Dict[str, float] = dict(
            zip(self.possible_agent_name_list, [0.0 for _ in self.possible_agent_name_list])
        )

        self.agent_energy_from_eating_dict = dict(
            zip(self.possible_agent_name_list, [0.0 for _ in self.possible_agent_name_list]))

        self.n_aec_cycles = 0

        # time series of active agents
        self.n_active_predator_list = []
        self.n_active_prey_list = []
        self.n_active_grass_list = []

        self.n_active_predator_list.insert(self.n_aec_cycles, self.n_active_predator)
        self.n_active_prey_list.insert(self.n_aec_cycles, self.n_active_prey)
        self.n_active_grass_list.insert(self.n_aec_cycles, self.n_active_grass)

        self.total_energy_predator_list.insert(self.n_aec_cycles, self.total_energy_predator)
        self.total_energy_prey_list.insert(self.n_aec_cycles, self.total_energy_prey)
        self.total_energy_grass_list.insert(self.n_aec_cycles, self.total_energy_grass)

        # episode population metrics
        self.n_starved_predator = 0
        self.n_starved_prey = 0
        self.n_eaten_prey = 0
        self.n_born_predator = 0
        self.n_born_prey = 0
    
    def step(self, action, agent_instance, is_last_step_of_cycle):
        if agent_instance.is_active:
            agent_type_nr = agent_instance.agent_type_nr

            if agent_type_nr == PREDATOR_TYPE_NR:
                if agent_instance.energy > 0:
                    agent_instance.age += 1
                    # move in Von Neumann neighborhood
                    self.move_agent(agent_instance, action)
                    x_new, y_new = agent_instance.position
                    is_prey_in_new_cell = self.agent_instance_in_grid_location[PREY_TYPE_NR][(x_new, y_new)] is not None
                    if is_prey_in_new_cell:
                        # found prey to eat and store records for last step of the cycle
                        # if energy of predator is above energy of all prey surrounding in neighborhood of attacked prey
                        self.earmarking_predator_catches_prey(agent_instance, x_new, y_new)
                    else:
                        # no prey instance in new cell
                        if self.model_state[PREY_TYPE_NR, x_new, y_new] > 0:
                            print("WARNING: Prey instance not found in in cel (", x_new,
                                   ",", y_new, 
                                   ") but model_state[PREY_TYPE_NR,", x_new,",", y_new,"] = "
                                  , self.model_state[PREY_TYPE_NR, x_new, y_new])
                            """
                            found_prey = False
                            for prey_instance in self.active_prey_instance_list:
                                if prey_instance.position[0] == x_new and prey_instance.position[1] == y_new:
                                    print("Prey instance found in active_prey_instance_list")
                                    print("Prey instance name: ", prey_instance.agent_name)
                                    print("Prey instance energy: ", prey_instance.energy)
                                    self.render_mode = "human"
                                    self.render()
                                    found_prey = True
                            if not found_prey:
                                print("Prey instance not found in active_prey_instance_list")
                            """

                else:
                    # store for inactivation at last step of the cycle
                    self.predator_to_be_removed_by_starvation_dict[agent_instance.agent_name] = True

            # If the agent is a prey and it's alive
            elif agent_type_nr == PREY_TYPE_NR:
                if agent_instance.energy > 0:
                    agent_instance.age += 1
                    # move in Von Neumann neighborhood
                    self.move_agent(agent_instance, action)
                    x_new, y_new = agent_instance.position
                    is_grass_in_new_cell = self.model_state[GRASS_TYPE_NR, x_new, y_new] > 0
                    if is_grass_in_new_cell:
                        # found grass to eat aand store records for last step of the cycle
                        self.earmarking_prey_eats_grass(agent_instance, x_new, y_new)
                else:
                    # mark for inactivation at last step of the cycle
                    self.prey_to_be_removed_by_starvation_dict[agent_instance.agent_name] = True


        if is_last_step_of_cycle:
            # reset rewards to zero
            self.reset_rewards()
            self.total_energy_predator = 0.0
            self.total_energy_prey = 0.0
            self.total_energy_grass = 0.0
            # removes agents, reap rewards and eventually (re)create agents 
            for predator_name in self.possible_predator_name_list:
                predator_instance = self.agent_name_to_instance_dict[predator_name]
                if predator_instance.is_active:
                    if self.predator_to_be_removed_by_starvation_dict[predator_name]:
                        self.remove_predator(predator_instance)
                    else:
                        # reap rewards and updates energy for predator which removes prey
                        self.reward_predator(predator_instance)
                        self.total_energy_predator += predator_instance.energy
                        if predator_instance.energy > self.predator_creation_energy_threshold:
                            # create new predator when energy level is above threshold
                            self.create_new_predator(predator_instance)

            for prey_name in self.possible_prey_name_list:
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                if prey_instance.is_active:
                    if (
                        self.prey_to_be_removed_by_predator_dict[prey_name]
                        or self.prey_to_be_removed_by_starvation_dict[prey_name]
                    ):
                        # remove prey which is selected to starve to death or eaten by predator, 
                        # from self.active_prey_instance_list
                        self.remove_prey(prey_instance)
                    else:
                        self.reward_prey(prey_instance)
                        self.total_energy_prey += prey_instance.energy
                        if prey_instance.energy > self.prey_creation_energy_threshold:
                            self.create_new_prey(prey_instance)

            for grass_name in self.possible_grass_name_list:
                grass_instance = self.agent_name_to_instance_dict[grass_name]
                if grass_instance.is_active:
                    # remove grass which gets eaten by a prey
                    if self.grass_to_be_removed_by_prey_dict[grass_name]:
                        self.remove_grass(grass_instance)
                    else:
                        # increase grass energy by self.energy_gain_per_step_grass, but not higher than self.max_energy_level_grass
                        grass_energy_gain = min(grass_instance.energy_gain_per_step, max(self.max_energy_level_grass - grass_instance.energy,0))
                        grass_instance.energy += grass_energy_gain
                        self.model_state[  
                            GRASS_TYPE_NR,
                            grass_instance.position[0],
                            grass_instance.position[1],
                        ] = grass_instance.energy
                        self.total_energy_grass += grass_instance.energy
                else:
                    # grass is inactive
                    grass_instance.energy += grass_instance.energy_gain_per_step
                    self.total_energy_grass += grass_instance.energy
                    # revive dead grass if energy regrows to at least self.initial_energy_grass
                    if grass_instance.energy >= self.initial_energy_grass:
                        self.n_active_grass += 1
                        self.active_grass_instance_list.append(grass_instance)
                        self.model_state[
                            GRASS_TYPE_NR,
                            grass_instance.position[0],
                            grass_instance.position[1],
                        ] = grass_instance.energy
                        grass_instance.is_active = True

            self.n_aec_cycles += 1

            # record number of active agents at the end of the cycle
            self.n_active_predator_list.insert(self.n_aec_cycles, self.n_active_predator)
            self.n_active_prey_list.insert(self.n_aec_cycles, self.n_active_prey)
            self.n_active_grass_list.insert(self.n_aec_cycles, self.n_active_grass)

            self.total_energy_predator_list.insert(self.n_aec_cycles, self.total_energy_predator)   
            self.total_energy_prey_list.insert(self.n_aec_cycles, self.total_energy_prey)
            self.total_energy_grass_list.insert(self.n_aec_cycles, self.total_energy_grass)

            self.reset_removal_records()


    def move_agent(self, agent_instance, action):
        self.agent_instance_in_grid_location[
            agent_instance.agent_type_nr,
            agent_instance.position[0],
            agent_instance.position[1],
        ] = None
        self.model_state[
            agent_instance.agent_type_nr,
            agent_instance.position[0],
            agent_instance.position[1],
        ] = 0.0
        agent_instance.step(action)
        self.model_state[
            agent_instance.agent_type_nr,
            agent_instance.position[0],
            agent_instance.position[1],
        ] = agent_instance.energy
        self.agent_instance_in_grid_location[
            agent_instance.agent_type_nr,
            agent_instance.position[0],
            agent_instance.position[1],
        ] = agent_instance

    def earmarking_predator_catches_prey(self, predator_instance, x_new, y_new):
        # set this option parameter to True to earmark a prey only if it is unaccompanied
        is_only_earmarked_if_prey_is_unaccompanied = False 
        if is_only_earmarked_if_prey_is_unaccompanied:
            # check if there is a prey in the Moore neighborhood of the attacked prey
            # (in order to ivestigate flocking behavior of prey, 
            # since accompanied prey cannot be earmarked for removal)
            is_accompanied_prey = False # initialization
            is_accompanied_prey = (
                (self.model_state[PREY_TYPE_NR, x_new, y_new-1] > 0  if y_new-1 >=0 else False) or
                (self.model_state[PREY_TYPE_NR, x_new, y_new+1] > 0 if y_new+1 < self.y_grid_size else False) +
                (self.model_state[PREY_TYPE_NR, x_new-1, y_new] > 0 if x_new-1 >=0 else False) + 
                (self.model_state[PREY_TYPE_NR, x_new-1, y_new-1] > 0 if x_new-1 >=0 and y_new-1 >=0 else False) + 
                (self.model_state[PREY_TYPE_NR, x_new-1, y_new+1] > 0 if x_new-1 >=0 and y_new+1 < self.y_grid_size else False) + 
                (self.model_state[PREY_TYPE_NR, x_new+1, y_new] > 0 if x_new+1 < self.x_grid_size else False)+
                (self.model_state[PREY_TYPE_NR, x_new+1, y_new+1] > 0 if x_new+1 < self.x_grid_size and y_new+1 < self.y_grid_size else False) +
                (self.model_state[PREY_TYPE_NR, x_new+1, y_new-1] > 0 if x_new+1 < self.x_grid_size and y_new-1 >=0 else False)
            )
            # if there is no other prey in the neighborhood of the attacked prey, the prey is earmarked for removal by the predator
            # otherwise it is not earmarked for removal 
            if not is_accompanied_prey:
                prey_instance_removed = self.agent_instance_in_grid_location[
                    PREY_TYPE_NR][
                    (x_new, y_new)
                ]
                self.predator_who_remove_prey_dict[predator_instance.agent_name] = True
                self.prey_to_be_removed_by_predator_dict[prey_instance_removed.agent_name] = True
                self.agent_energy_from_eating_dict[
                    predator_instance.agent_name
                ] = prey_instance_removed.energy
        else:
            # allways earmarking option
            prey_instance_removed = self.agent_instance_in_grid_location[
                PREY_TYPE_NR][
                (x_new, y_new)
            ]
            self.predator_who_remove_prey_dict[predator_instance.agent_name] = True
            self.prey_to_be_removed_by_predator_dict[prey_instance_removed.agent_name] = True
            self.agent_energy_from_eating_dict[
                predator_instance.agent_name
            ] = prey_instance_removed.energy

    def earmarking_prey_eats_grass(self, prey_instance, x_new, y_new):
        grass_instance_removed = self.agent_instance_in_grid_location[GRASS_TYPE_NR][
            (x_new, y_new)
        ]
        # book keeping for last step of the cycle actions
        # TODO: change to: agent_who_eats_dict and agent_who_gets_eaten_dict?
        self.prey_who_remove_grass_dict[prey_instance.agent_name] = True
        self.grass_to_be_removed_by_prey_dict[grass_instance_removed.agent_name] = True
        self.agent_energy_from_eating_dict[prey_instance.agent_name] = grass_instance_removed.energy

    def reset_rewards(self):
        self.agent_reward_dict = dict(
            zip(
                self.possible_agent_name_list,
                [0.0 for _ in self.possible_agent_name_list],
            )
        )

    def remove_predator(self, predator_instance):
        self.active_predator_instance_list.remove(predator_instance)
        self.n_active_predator -= 1
        self.n_starved_predator += 1
        self.agent_instance_in_grid_location[
            PREDATOR_TYPE_NR,
            predator_instance.position[0],
            predator_instance.position[1],
        ] = None
        self.model_state[
            PREDATOR_TYPE_NR,
            predator_instance.position[0],
            predator_instance.position[1],
        ] = 0.0
        predator_instance.is_active = False
        self.predator_age_list.append(predator_instance.age)
        predator_instance.energy = 0.0
        predator_instance.age = 0
        self.agent_reward_dict[
            predator_instance.agent_name
        ] += self.death_reward_predator

    def remove_prey(self, prey_instance):
        self.active_prey_instance_list.remove(prey_instance)
        self.n_active_prey -= 1
        if self.prey_to_be_removed_by_starvation_dict[prey_instance.agent_name]:
            self.n_starved_prey += 1
        else:
            self.n_eaten_prey += 1
        self.agent_instance_in_grid_location[
            PREY_TYPE_NR, prey_instance.position[0], prey_instance.position[1]
        ] = None
        self.model_state[
            PREY_TYPE_NR, prey_instance.position[0], prey_instance.position[1]
        ] = 0.0
        prey_instance.is_active = False
        self.prey_age_list.append(prey_instance.age)
        prey_instance.energy = 0.0
        prey_instance.age = 0
        self.agent_reward_dict[prey_instance.agent_name] += self.death_reward_prey

    def create_new_predator(self, parent_predator):
        non_active_predator_names = [
            name
            for name in self.possible_predator_name_list
            if not self.agent_name_to_instance_dict[name].is_active
        ]
        if non_active_predator_names:
            new_predator_name = non_active_predator_names[-1]
            new_predator_instance = self.agent_name_to_instance_dict[new_predator_name]
            new_predator_instance.is_active = True
            self.predator_to_be_removed_by_starvation_dict[new_predator_name] = False
            parent_predator.energy -= self.initial_energy_predator
            self.model_state[
                PREDATOR_TYPE_NR,
                parent_predator.position[0],
                parent_predator.position[1],
            ] = parent_predator.energy
            new_predator_instance.energy = self.initial_energy_predator
            new_predator_instance.age = 0
            self.active_predator_instance_list.append(new_predator_instance)
            self.n_active_predator += 1
            self.n_born_predator += 1
            x_new, y_new = self.position_new_agent_on_gridworld(new_predator_instance, self.spawning_area, self.model_state)

            new_predator_instance.position = (x_new, y_new)
            self.agent_instance_in_grid_location[
                PREDATOR_TYPE_NR, x_new, y_new
            ] = new_predator_instance
            self.model_state[PREDATOR_TYPE_NR, x_new, y_new] = (
                new_predator_instance.energy
            )
            self.agent_reward_dict[
                parent_predator.agent_name
            ] += self.reproduction_reward_predator

    def create_new_prey(self, parent_prey):
        non_active_prey_names = [
            name
            for name in self.possible_prey_name_list
            if not self.agent_name_to_instance_dict[name].is_active
        ]
        if non_active_prey_names:
            new_prey_name = non_active_prey_names[-1]
            new_prey_instance = self.agent_name_to_instance_dict[new_prey_name]
            new_prey_instance.is_active = True
            self.prey_to_be_removed_by_starvation_dict[new_prey_name] = False
            parent_prey.energy -= self.initial_energy_prey
            self.model_state[
                PREY_TYPE_NR, parent_prey.position[0], parent_prey.position[1]
            ] = parent_prey.energy
            new_prey_instance.energy = self.initial_energy_prey
            new_prey_instance.age = 0
            self.active_prey_instance_list.append(new_prey_instance)
            self.n_active_prey += 1
            self.n_born_prey += 1
            #x_new, y_new = self.find_new_position(PREY_TYPE_NR)
            x_new, y_new = self.position_new_agent_on_gridworld(new_prey_instance, self.spawning_area, self.model_state)

            new_prey_instance.position = (x_new, y_new)
            self.agent_instance_in_grid_location[PREY_TYPE_NR, x_new, y_new] = (
                new_prey_instance
            )
            self.model_state[PREY_TYPE_NR, x_new, y_new] = new_prey_instance.energy
            self.agent_reward_dict[
                parent_prey.agent_name
            ] += self.reproduction_reward_prey

    def reward_predator(self, predator_instance):
        predator_name = predator_instance.agent_name
        self.agent_reward_dict[predator_name] += self.step_reward_predator
        self.agent_reward_dict[predator_name] += (
            self.catch_reward_prey * self.predator_who_remove_prey_dict[predator_name]
        )
        predator_instance.energy += self.energy_gain_per_step_predator
        predator_instance.energy += (
            self.agent_energy_from_eating_dict[predator_name]
        )
        self.model_state[
            PREDATOR_TYPE_NR,
            predator_instance.position[0],
            predator_instance.position[1],
        ] = predator_instance.energy

    def reward_prey(self, prey_instance):
        prey_name = prey_instance.agent_name
        self.agent_reward_dict[prey_name] += self.step_reward_prey
        self.agent_reward_dict[prey_name] += (
            self.catch_reward_grass * self.prey_who_remove_grass_dict[prey_name]
        )
        prey_instance.energy += self.energy_gain_per_step_prey
        prey_instance.energy += (
            self.agent_energy_from_eating_dict[prey_name]
        )
        self.model_state[
            PREY_TYPE_NR,
            prey_instance.position[0],
            prey_instance.position[1],
        ] = prey_instance.energy

    def remove_grass(self, grass_instance):
        self.active_grass_instance_list.remove(grass_instance)
        self.n_active_grass -= 1
        self.model_state[
            GRASS_TYPE_NR, grass_instance.position[0], grass_instance.position[1]
        ] = 0.0
        grass_instance.energy = 0.0
        grass_instance.is_active = False

    def reset_removal_records(self):
        # reinit agents removal records to default at the end of the cycle
        self.agent_energy_from_eating_dict = dict(
            zip(self.possible_agent_name_list, [0.0 for _ in self.possible_agent_name_list])
        )
        self.prey_who_remove_grass_dict = dict(
            zip(
                self.possible_prey_name_list,
                [False for _ in self.possible_prey_name_list],
            )
        )
        self.grass_to_be_removed_by_prey_dict = dict(
            zip(
                self.possible_grass_name_list,
                [False for _ in self.possible_grass_name_list],
            )
        )
        self.predator_who_remove_prey_dict = dict(
            zip(
                self.possible_predator_name_list,
                [False for _ in self.possible_predator_name_list],
            )
        )
        self.prey_to_be_removed_by_predator_dict = dict(
            zip(
                self.possible_prey_name_list,
                [False for _ in self.possible_prey_name_list],
            )
        )
        self.prey_to_be_removed_by_starvation_dict = dict(
            zip(
                self.possible_prey_name_list,
                [False for _ in self.possible_prey_name_list],
            )
        )
        self.predator_to_be_removed_by_starvation_dict = dict(
            zip(
                self.possible_predator_name_list,
                [False for _ in self.possible_predator_name_list],
            )
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def create_possible_agent_name_list_from_instance_list(self, _active_agent_instance_list):
        _possible_agent_name_list = []
        for agent_instance in _active_agent_instance_list:
            _possible_agent_name_list.append(agent_instance.agent_name)
        return _possible_agent_name_list

    @property
    def is_no_grass(self):
        if self.n_active_grass == 0:
            return True
        return False

    @property
    def is_no_prey(self):
        if self.n_active_prey == 0:
            return True
        return False

    @property
    def is_no_predator(self):
        if self.n_active_predator == 0:
            return True
        return False

    def observe(self, agent_name):
        max_obs_range = self.max_observation_range
        max_obs_offset: int = int((max_obs_range - 1) / 2)
        x_grid_size: int = self.x_grid_size
        y_grid_size: int = self.y_grid_size

        def obs_clip(x, y, max_obs_offset, x_grid_size, y_grid_size):
            xld = x - max_obs_offset
            xhd = x + max_obs_offset
            yld = y - max_obs_offset
            yhd = y + max_obs_offset
            xlo, xhi, ylo, yhi = (
                np.clip(xld, 0, x_grid_size - 1),
                np.clip(xhd, 0, x_grid_size - 1),
                np.clip(yld, 0, y_grid_size - 1),
                np.clip(yhd, 0, y_grid_size - 1),
            )
            xolo, yolo = abs(np.clip(xld, -max_obs_offset, 0)), abs(
                np.clip(yld, -max_obs_offset, 0)
            )
            xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
            return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

        agent_instance = self.agent_name_to_instance_dict[agent_name]

        xp, yp = agent_instance.position[0], agent_instance.position[1]

        observation = np.zeros(
            (
                self.nr_observation_channels,
                self.max_observation_range,
                self.max_observation_range,
            ),
            dtype=np.float64,
        )
        # wall channel  filled with ones up front
        observation[0].fill(1.0)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = obs_clip(xp, yp, max_obs_offset, x_grid_size, y_grid_size)

        observation[0 : self.nr_observation_channels, xolo:xohi, yolo:yohi] = np.abs(
            self.model_state[0 : self.nr_observation_channels, xlo:xhi, ylo:yhi]
        )

        observation_range_agent = agent_instance.observation_range
        # mask is number of 'outer squares' of an observation surface set to zero
        mask = int((max_obs_range - observation_range_agent) / 2)
        if (
            mask > 0
        ):  # observation_range agent is smaller than default max_observation_range
            for j in range(mask):
                for i in range(self.nr_observation_channels):
                    observation[i][j, 0:max_obs_range] = 0
                    observation[i][max_obs_range - 1 - j, 0:max_obs_range] = 0
                    observation[i][0:max_obs_range, j] = 0
                    observation[i][0:max_obs_range, max_obs_range - 1 - j] = 0
            return observation
        elif mask == 0:
            return observation
        else:
            raise Exception(
                "Error: observation_range_agent larger than max_observation_range"
            )

    def render(self):
        def draw_grid_model():
            # Draw grid and borders
            for x in range(self.x_grid_size):
                for y in range(self.y_grid_size):
                    cell_pos = pygame.Rect(
                        self.cell_scale * x,
                        self.cell_scale * y,
                        self.cell_scale,
                        self.cell_scale,
                    )
                    cell_color = (255, 255, 255)
                    pygame.draw.rect(self.screen, cell_color, cell_pos)

                    border_pos = pygame.Rect(
                        self.cell_scale * x,
                        self.cell_scale * y,
                        self.cell_scale,
                        self.cell_scale,
                    )
                    border_color = (192, 192, 192)
                    pygame.draw.rect(self.screen, border_color, border_pos, 1)

            # Draw red border around total grid
            border_pos = pygame.Rect(
                0,
                0,
                self.cell_scale * self.x_grid_size,
                self.cell_scale * self.y_grid_size,
            )
            border_color = (255, 0, 0)
            pygame.draw.rect(self.screen, border_color, border_pos, 5)

        def draw_predator_observations():
            for predator_instance in self.active_predator_instance_list:
                position = predator_instance.position
                x = position[0]
                y = position[1]
                mask = int(
                    (self.max_observation_range - predator_instance.observation_range)
                    / 2
                )
                if mask == 0:
                    patch = pygame.Surface(
                        (
                            self.cell_scale * self.max_observation_range,
                            self.cell_scale * self.max_observation_range,
                        )
                    )
                    patch.set_alpha(128)
                    patch.fill((255, 152, 72))
                    ofst = self.max_observation_range / 2.0
                    self.screen.blit(
                        patch,
                        (
                            self.cell_scale * (x - ofst + 1 / 2),
                            self.cell_scale * (y - ofst + 1 / 2),
                        ),
                    )
                else:
                    patch = pygame.Surface(
                        (
                            self.cell_scale * predator_instance.observation_range,
                            self.cell_scale * predator_instance.observation_range,
                        )
                    )
                    patch.set_alpha(128)
                    patch.fill((255, 152, 72))
                    ofst = predator_instance.observation_range / 2.0
                    self.screen.blit(
                        patch,
                        (
                            self.cell_scale * (x - ofst + 1 / 2),
                            self.cell_scale * (y - ofst + 1 / 2),
                        ),
                    )

        def draw_prey_observations():
            for prey_instance in self.active_prey_instance_list:
                position = prey_instance.position
                x = position[0]
                y = position[1]
                # this hopefully can be improved with rllib..
                mask = int(
                    (self.max_observation_range - prey_instance.observation_range) / 2
                )
                if mask == 0:
                    patch = pygame.Surface(
                        (
                            self.cell_scale * self.max_observation_range,
                            self.cell_scale * self.max_observation_range,
                        )
                    )
                    patch.set_alpha(128)
                    patch.fill((72, 152, 255))
                    ofst = self.max_observation_range / 2.0
                    self.screen.blit(
                        patch,
                        (
                            self.cell_scale * (x - ofst + 1 / 2),
                            self.cell_scale * (y - ofst + 1 / 2),
                        ),
                    )
                else:
                    patch = pygame.Surface(
                        (
                            self.cell_scale * prey_instance.observation_range,
                            self.cell_scale * prey_instance.observation_range,
                        )
                    )
                    patch.set_alpha(128)
                    patch.fill((72, 152, 255))
                    ofst = prey_instance.observation_range / 2.0
                    self.screen.blit(
                        patch,
                        (
                            self.cell_scale * (x - ofst + 1 / 2),
                            self.cell_scale * (y - ofst + 1 / 2),
                        ),
                    )

        def draw_predator_instances():
            for predator_instance in self.active_predator_instance_list:
                position = predator_instance.position
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )

                col = (255, 0, 0)  # red

                pygame.draw.circle(self.screen, col, center, int(self.cell_scale / 2.3))  # type: ignore

        def draw_prey_instances():
            for prey_instance in self.active_prey_instance_list:
                position = prey_instance.position
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )

                col = (0, 0, 255)  # blue

                pygame.draw.circle(self.screen, col, center, int(self.cell_scale / 2.3))  # type: ignore

        def draw_grass_instances():
            for grass_instance in self.active_grass_instance_list:
                position = grass_instance.position
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )
                col = (0, 128, 0)  # green
                pygame.draw.circle(self.screen, col, center, int(self.cell_scale / 2.3))  # type: ignore

        def draw_agent_instance_id_nrs():
            font = pygame.font.SysFont("Comic Sans MS", self.cell_scale * 2 // 3)

            predator_positions = defaultdict(int)
            prey_positions = defaultdict(int)
            grass_positions = defaultdict(int)

            for predator_instance in self.active_predator_instance_list:
                prey_position = predator_instance.position
                x = prey_position[0]
                y = prey_position[1]
                predator_positions[(x, y)] = predator_instance.agent_id_nr

            for prey_instance in self.active_prey_instance_list:
                prey_position = prey_instance.position
                x = prey_position[0]
                y = prey_position[1]
                prey_positions[(x, y)] = prey_instance.agent_id_nr

            for grass_instance in self.active_grass_instance_list:
                grass_position = grass_instance.position
                x = grass_position[0]
                y = grass_position[1]
                grass_positions[(x, y)] = grass_instance.agent_id_nr

            for x, y in predator_positions:
                (pos_x, pos_y) = (
                    self.cell_scale * x + self.cell_scale // 6,
                    self.cell_scale * y + self.cell_scale // 1.2,
                )

                predator_id_nr__text = str(predator_positions[(x, y)])

                predator_text = font.render(predator_id_nr__text, False, (255, 255, 0))

                self.screen.blit(predator_text, (pos_x, pos_y - self.cell_scale // 2))

            for x, y in prey_positions:
                (pos_x, pos_y) = (
                    self.cell_scale * x + self.cell_scale // 6,
                    self.cell_scale * y + self.cell_scale // 1.2,
                )

                prey_id_nr__text = str(prey_positions[(x, y)])

                prey_text = font.render(prey_id_nr__text, False, (255, 255, 0))

                self.screen.blit(prey_text, (pos_x, pos_y - self.cell_scale // 2))

            for x, y in grass_positions:
                (pos_x, pos_y) = (
                    self.cell_scale * x + self.cell_scale // 6,
                    self.cell_scale * y + self.cell_scale // 1.2,
                )

                grass_id_nr__text = str(grass_positions[(x, y)])

                grass_text = font.render(grass_id_nr__text, False, (255, 255, 0))

                self.screen.blit(grass_text, (pos_x, pos_y - self.cell_scale // 2))

        def draw_white_canvas_energy_chart():
            # relative position of energy chart within pygame window
            x_position_energy_chart = self.cell_scale * self.x_grid_size
            y_position_energy_chart = 0  # self.y_pygame_window
            pos = pygame.Rect(
                x_position_energy_chart,
                y_position_energy_chart,
                self.width_energy_chart,
                self.height_energy_chart,
            )
            color = (255, 255, 255)  # white background
            pygame.draw.rect(self.screen, color, pos)  # type: ignore

        def draw_bar_chart_energy():
            # Constants
            BLACK = (0, 0, 0)
            RED = (255, 0, 0)
            BLUE = (0, 0, 255)

            # Create data array predators and prey
            data_predators = [
                self.agent_name_to_instance_dict[name].energy
                for name in self.possible_predator_name_list
            ]
            data_prey = [
                self.agent_name_to_instance_dict[name].energy
                for name in self.possible_prey_name_list
            ]

            # postion and size parameters energy chart
            width_energy_chart = self.width_energy_chart  # = 1800

            max_energy_value_chart = 30
            bar_width = 20
            offset_bars = 20
            x_screenposition = 400  # x_axis screen position?
            y_screenposition = 150
            y_axis_height = 500
            x_axis_width = width_energy_chart - 120  # = 1680
            x_screenposition_prey_bars = 1450
            title_x = 2000
            title_y = 120

            # Draw y-axis
            y_axis_x = (
                x_screenposition
                + (width_energy_chart - (bar_width * len(data_predators))) // 2
                - 10
            )
            y_axis_y = y_screenposition  # 50
            # x-axis
            x_axis_x = (
                x_screenposition
                + (width_energy_chart - (bar_width * len(data_predators))) // 2
            )
            x_axis_y = y_screenposition + y_axis_height  # 50 + 500 = 550
            x_start_prey_bars = x_screenposition_prey_bars + x_screenposition
            x_start_predator_bars = x_axis_x
            predator_legend_x = x_start_predator_bars
            predator_legend_y = y_screenposition + 550
            prey_legend_x = x_start_prey_bars
            prey_legend_y = y_screenposition + 550
            title_font_size = 30
            predator_legend_font_size = 30
            prey_legend_font_size = 30

            # Draw chart title
            chart_title = "Energy levels agents"
            title_color = BLACK  # black
            title_font = pygame.font.Font(None, title_font_size)
            title_text = title_font.render(chart_title, True, title_color)
            self.screen.blit(title_text, (title_x, title_y))
            # Draw legend title for predators
            predator_legend_title = "Predators"
            predator_legend_color = RED
            predator_legend_font = pygame.font.Font(None, predator_legend_font_size)
            predator_legend_text = predator_legend_font.render(
                predator_legend_title, True, predator_legend_color
            )
            self.screen.blit(
                predator_legend_text, (predator_legend_x, predator_legend_y)
            )
            # Draw legend title for prey
            prey_legend_title = "Prey"
            prey_legend_color = BLUE
            prey_legend_font = pygame.font.Font(None, prey_legend_font_size)
            prey_legend_text = prey_legend_font.render(
                prey_legend_title, True, prey_legend_color
            )
            self.screen.blit(prey_legend_text, (prey_legend_x, prey_legend_y))

            # Draw y-axis
            y_axis_color = BLACK
            pygame.draw.rect(
                self.screen, y_axis_color, (y_axis_x, y_axis_y, 5, y_axis_height)
            )
            # Draw x-axis
            x_axis_color = BLACK
            pygame.draw.rect(
                self.screen, x_axis_color, (x_axis_x, x_axis_y, x_axis_width, 5)
            )

            # Draw predator bars
            for i, value in enumerate(data_predators):
                bar_height = (value / max_energy_value_chart) * y_axis_height
                bar_x = x_start_predator_bars + i * (bar_width + offset_bars)
                bar_y = y_screenposition + y_axis_height - bar_height

                color = (255, 0, 0)  # red

                pygame.draw.rect(
                    self.screen, color, (bar_x, bar_y, bar_width, bar_height)
                )

            # Draw tick labels predators on x-axis
            for i, predator_name in enumerate(self.possible_predator_name_list):
                predator_instance = self.agent_name_to_instance_dict[predator_name]
                label = str(predator_instance.agent_id_nr)
                label_x = x_axis_x + i * (bar_width + offset_bars)
                label_y = x_axis_y + 10
                label_color = (255, 0, 0)  # red
                font = pygame.font.Font(None, 30)
                text = font.render(label, True, label_color)
                self.screen.blit(text, (label_x, label_y))

            # Draw prey bars
            for i, value in enumerate(data_prey):
                bar_height = (value / max_energy_value_chart) * y_axis_height
                bar_x = x_start_prey_bars + i * (bar_width + offset_bars)
                bar_y = y_screenposition + y_axis_height - bar_height

                color = (0, 0, 255)  # blue

                pygame.draw.rect(
                    self.screen, color, (bar_x, bar_y, bar_width, bar_height)
                )

            # Draw tick labels prey on x-axis
            for i, prey_name in enumerate(self.possible_prey_name_list):
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                label = str(prey_instance.agent_id_nr)
                label_x = x_start_prey_bars + i * (bar_width + offset_bars)
                label_y = x_axis_y + 10
                label_color = BLUE
                font = pygame.font.Font(None, 30)
                text = font.render(label, True, label_color)
                self.screen.blit(text, (label_x, label_y))

            # Draw tick points on y-axis
            num_ticks = max_energy_value_chart + 1
            tick_spacing = y_axis_height // (num_ticks - 1)
            for i in range(num_ticks):
                tick_x = y_axis_x - 5
                tick_y = y_screenposition + y_axis_height - i * tick_spacing
                tick_width = 10
                tick_height = 2
                tick_color = (0, 0, 0)  # black
                pygame.draw.rect(
                    self.screen, tick_color, (tick_x, tick_y, tick_width, tick_height)
                )

                # Draw tick labels every 5 ticks
                if i % 5 == 0:
                    label = str(i)
                    label_x = tick_x - 30
                    label_y = tick_y - 5
                    label_color = (0, 0, 0)  # black
                    font = pygame.font.Font(None, 30)
                    text = font.render(label, True, label_color)
                    self.screen.blit(text, (label_x, label_y))

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (
                        self.cell_scale * self.x_grid_size + self.width_energy_chart,
                        self.cell_scale * self.y_grid_size,
                    )
                )
                pygame.display.set_caption("PredPreyGrass - create agents")
            else:
                self.screen = pygame.Surface(
                    (
                        self.cell_scale * self.x_grid_size,
                        self.cell_scale * self.y_grid_size,
                    )
                )

        draw_grid_model()
        draw_prey_observations()
        draw_predator_observations()
        draw_grass_instances()
        draw_prey_instances()
        draw_predator_instances()
        draw_agent_instance_id_nrs()
        if self.show_energy_chart:
            draw_white_canvas_energy_chart()
            draw_bar_chart_energy()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            if self.save_image_steps:
                self.file_name += 1
                print(str(self.file_name) + ".png saved")
                directory = "./assets/images/"
                pygame.image.save(self.screen, directory + str(self.file_name) + ".png")

        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predpreygrass",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self.pred_prey_env = PredPreyGrass(
            *args, **kwargs
        )  #  this calls the code from PredPreyGrass

        self.agents = self.pred_prey_env.possible_agent_name_list

        self.possible_agents = self.agents[:]
        # added for optuna
        self.action_spaces = dict(zip(self.agents, self.pred_prey_env.action_space))  # type: ignore
        self.observation_spaces = dict(zip(self.agents, self.pred_prey_env.observation_space))  # type: ignore



    def reset(self, seed=None, options=None):
        if seed is not None:
            self.pred_prey_env._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents

        self.possible_agents = self.agents[:]
        self.agent_name_to_index_mapping = dict(
            zip(self.agents, list(range(self.num_agents)))
        )
        self._agent_selector = agent_selector(self.agents)

        # spaces
        self.action_spaces = dict(zip(self.agents, self.pred_prey_env.action_space))  # type: ignore
        self.observation_spaces = dict(zip(self.agents, self.pred_prey_env.observation_space))  # type: ignore
        self.steps = 0
        # this method "reset"
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.pred_prey_env.reset()  # this calls reset from PredPreyGrass

    def close(self):
        if not self.closed:
            self.closed = True
            self.pred_prey_env.close()

    def render(self):
        if not self.closed:
            return self.pred_prey_env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        agent_instance = self.pred_prey_env.agent_name_to_instance_dict[agent]
        self.pred_prey_env.step(action, agent_instance, self._agent_selector.is_last())

        for k in self.terminations:
            if self.pred_prey_env.n_aec_cycles >= self.pred_prey_env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = (
                    self.pred_prey_env.is_no_prey or self.pred_prey_env.is_no_predator
                )

        for agent_name in self.agents:
            self.rewards[agent_name] = self.pred_prey_env.agent_reward_dict[agent_name]
        self.steps += 1
        self._cumulative_rewards[
            self.agent_selection
        ] = 0  # cannot be left out for proper rewards
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()  # cannot be left out for proper rewards
        if self.render_mode == "human" and agent_instance.is_active:
            self.render()
            
    def observe(self, agent_name):
        agent_instance = self.pred_prey_env.agent_name_to_instance_dict[agent_name]
        obs = self.pred_prey_env.observe(agent_name)
        observation = np.swapaxes(obs, 2, 0)  # type: ignore
        # "black death": return observation of only zeros if agent is not alive
        if not agent_instance.is_active:
            observation = np.zeros(observation.shape)
        return observation

    def observation_space(self, agent: str):  # must remain
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]