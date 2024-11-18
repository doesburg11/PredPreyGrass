# discretionary libraries
from predpreygrass.aec_predpreygrass.agents.discrete_agent import DiscreteAgent

# external libraries
import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces

import os
import numpy as np
import random
from typing import Tuple, List, Dict, Optional, Set
import pygame
from collections import defaultdict


class PredPreyGrass:
    """
    pred/prey/grass PettingZoo multi-agent learning environment this environment 
    transfers the energy of eaten prey/grass to the predator/prey while the grass
    regrows over time. The environment is a 2D grid world where agents can move
    in four cardinal directions. 
    """
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

        # agent types definitions
        self.agent_type_name_list: List[str] = ["wall", "predator", "prey", "grass"]
        self.predator_type_nr = 1
        self.prey_type_nr = 2
        self.grass_type_nr = 3
        # end agent types definitions

        # boundaries for the spawning of agents within the grid
        # Initialize a spawning area for the agents
        self.spawning_area = [{}, {}, {}, {}]
        self.spawning_area.insert(self.predator_type_nr, self.spawning_area_predator)
        self.spawning_area.insert(self.prey_type_nr, self.spawning_area_prey)
        self.spawning_area.insert(self.grass_type_nr, self.spawning_area_grass)

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
        self.total_energy_learning_agents: float = 0.0
        self.total_energy_predator_list: List[float] = []
        self.total_energy_prey_list: List[float] = []
        self.total_energy_learning_agents_list: List[float] = []
        self.total_energy_grass_list: List[float] = []
        self.n_starved_predator: int = 0
        self.n_starved_prey: int = (
            0  # note: prey can become inactive due to starvation or getting eaten by predators
        )
        self.n_eaten_prey: int = 0
        self.n_eaten_grass: int = 0
        self.n_born_predator: int = 0
        self.n_born_prey: int = 0
        self.predator_age_list: List[int] = []
        self.prey_age_list: List[int] = []

        self.active_predator_instance_list: List[DiscreteAgent] = (
            []
        )  # list of all active ("living") predators
        self.active_prey_instance_list: List[DiscreteAgent] = []  # list of active prey
        self.active_grass_instance_list: List[DiscreteAgent] = (
            []
        )  # list of active grass
        self.active_agent_instance_list: List[DiscreteAgent] = (
            []
        )  # list of active predators and prey
        self.possible_predator_name_list: List[str] = []
        self.possible_prey_name_list: List[str] = []
        self.possible_grass_name_list: List[str] = []
        self.possible_agent_name_list: List[str] = []

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
        self.agent_name_to_instance_dict: Dict[str, DiscreteAgent] = {}

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
        self.possible_grass_name_list = [
            "grass" + "_" + str(a) for a in grass_id_nr_range
        ]
        self.possible_agent_name_list = (
            self.possible_predator_name_list + self.possible_prey_name_list
        )

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

        self.file_name: int = 0
        self.n_cycles: int = 0

        # TODO: upperbound for observation space = max(energy levels of all agents)
        self.max_energy_level_prey = 25.0  # in kwargs later, init level = 5.0
        self.max_energy_level_predator = 25.0  # in kwargs later, init level = 5.0

        self._initialize_variables()


    def reset(self):
        self._initialize_variables()

        self.total_energy_predator_list = []
        self.total_energy_prey_list = []
        self.total_energy_grass_list = []
        self.total_energy_learning_agents_list = []

        # record of agent ages
        self.predator_age_list = []
        self.prey_age_list = []

        # initialization
        self.n_active_predator = self.n_possible_predator
        self.n_active_prey = self.n_possible_prey
        self.n_active_grass = self.n_possible_grass
        self.total_energy_predator = (
            self.n_active_predator * self.initial_energy_predator
        )
        self.total_energy_prey = self.n_active_prey * self.initial_energy_prey
        self.total_energy_grass = self.n_active_grass * self.initial_energy_grass
        self.total_energy_learning_agents = (
            self.total_energy_predator + self.total_energy_prey
        )

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

        # reset model state
        self.model_state: np.ndarray = np.zeros(
            (self.nr_observation_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            self.agent_instance_in_grid_location[agent_type_nr] = np.full(
                (self.x_grid_size, self.y_grid_size), None
            )

        # create agents of all types excluding "wall"-agents

        for agent_type_nr in self.agent_types:
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
                xinit, yinit = self._get_new_allowed_position(
                    agent_instance, self.spawning_area, self.model_state
                )
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                agent_instance.is_active = True
                agent_instance.position = (xinit, yinit)
                agent_instance.energy = self.initial_energy_list[agent_type_nr]

                self._link_agent_to_grid(agent_instance)
                self.possible_agent_instance_list_type[agent_type_nr].append(
                    agent_instance
                )

             
        for agent_type in self.agent_types:
            # Copy possible agent instances to active agent instances
            self.active_agent_instance_list_type[agent_type] = (
                self.possible_agent_instance_list_type[agent_type].copy()
            )
            # Create agent name lists from instance lists
            self.possible_agent_name_list_type[agent_type] = (
                self._create_agent_name_list_from_instance_list(
                    self.possible_agent_instance_list_type[agent_type]
                )
            )


        self.active_predator_instance_list = self.possible_agent_instance_list_type[
            self.predator_type_nr
        ]
        self.active_prey_instance_list = self.possible_agent_instance_list_type[self.prey_type_nr]
        self.active_grass_instance_list = self.possible_agent_instance_list_type[self.grass_type_nr]

        self.possible_predator_name_list = (
            self._create_agent_name_list_from_instance_list(
                self.active_predator_instance_list
            )
        )
        self.possible_prey_name_list = (
            self._create_agent_name_list_from_instance_list(
                self.active_prey_instance_list
            )
        )
        self.possible_grass_name_list = (
            self._create_agent_name_list_from_instance_list(
                self.active_grass_instance_list
            )
        )


        # deactivate agents which can be created later at runtime
        for agent_type in self.learning_agent_types:
            for agent_instance in self.possible_agent_instance_list_type[agent_type]:
                if agent_instance.agent_id_nr >= (
                    self.start_index_type[agent_type]
                    + self.n_initial_active_agent_type[agent_type]
                ):
                    self.active_agent_instance_list_type[agent_type].remove(
                        agent_instance
                    )
                    self.n_active_agent_type[agent_type] -= 1
                    self.total_energy_agent_type[agent_type] -= agent_instance.energy
                    self._unlink_agent_from_grid(agent_instance)
                    agent_instance.is_active = False
                    agent_instance.energy = 0.0


        # deactivate agents which can be created later at runtime
        for predator_name in self.possible_predator_name_list:
            predator_instance = self.agent_name_to_instance_dict[predator_name]
            if (
                predator_instance.agent_id_nr >= self.n_initial_active_predator
            ):  # number of initial active predators
                self.active_predator_instance_list.remove(predator_instance)
                self.n_active_predator -= 1
                self.total_energy_predator -= predator_instance.energy
                self._unlink_agent_from_grid(predator_instance)
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
                self._unlink_agent_from_grid(prey_instance)
                prey_instance.is_active = False
                prey_instance.energy = 0.0

        # removal agents set to false
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

        # define the learning agents
        self.active_agent_instance_list = (
            self.active_predator_instance_list + self.active_prey_instance_list
        )
        self.possible_agent_name_list = (
            self.possible_predator_name_list + self.possible_prey_name_list
        )

        self.agent_energy_from_eating_dict = dict(
            zip(
                self.possible_agent_name_list,
                [0.0 for _ in self.possible_agent_name_list],
            )
        )

        self.n_cycles = 0

        # time series of active agents
        self.n_active_predator_list = []
        self.n_active_prey_list = []
        self.n_active_grass_list = []

        self.n_active_predator_list.insert(self.n_cycles, self.n_active_predator)
        self.n_active_prey_list.insert(self.n_cycles, self.n_active_prey)
        self.n_active_grass_list.insert(self.n_cycles, self.n_active_grass)

        self.total_energy_predator_list.insert(
            self.n_cycles, self.total_energy_predator
        )
        self.total_energy_prey_list.insert(self.n_cycles, self.total_energy_prey)
        self.total_energy_grass_list.insert(self.n_cycles, self.total_energy_grass)
        self.total_energy_learning_agents = (
            self.total_energy_predator + self.total_energy_prey
        )
        self.total_energy_learning_agents_list.insert(
            self.n_cycles, self.total_energy_learning_agents
        )

        # episode population metrics
        self.n_starved_predator = 0
        self.n_starved_prey = 0
        self.n_eaten_prey = 0
        self.n_eaten_grass: int = 0
        self.n_born_predator = 0
        self.n_born_prey = 0

    def step(self, action, agent_instance, is_last_step_of_cycle):
        
        if agent_instance.is_active:
            self._apply_agent_action(agent_instance, action)
            # TODO implement energy gain per step in the agent class step method
            agent_instance.energy += agent_instance.energy_gain_per_step

        # apply the engagement rules and reap rewards
        if is_last_step_of_cycle:
            self._reset_rewards()
            # removes agents, reap rewards and eventually (re)create agents
            for predator_instance in self.active_predator_instance_list.copy():
                if predator_instance.energy > 0:
                    # new is the position of the predator after the move
                    x_new, y_new = predator_instance.position
                    predator_instance.age += 1
                    prey_instance_in_predator_cell = (
                        self.agent_instance_in_grid_location[self.prey_type_nr][
                            (x_new, y_new)
                        ]
                    )
                    if prey_instance_in_predator_cell is not None:
                        #print(f"{predator_instance.agent_name} (energy {round(predator_instance.energy,1)})",end="")
                        predator_instance.energy += prey_instance_in_predator_cell.energy
                        #print(f" eats {prey_instance_in_predator_cell.agent_name} (energy {round(prey_instance_in_predator_cell.energy,1)})", end="")
                        #print(f" predator energy after eating: {round(predator_instance.energy,1)}")
                        self._unlink_agent_from_grid(prey_instance_in_predator_cell)
                        prey_instance_in_predator_cell.is_active = False
                        prey_instance_in_predator_cell.energy = 0.0
                        self.active_prey_instance_list.remove(
                            prey_instance_in_predator_cell
                        )
                        predator_instance.energy += prey_instance_in_predator_cell.energy

                        self.n_active_prey -= 1
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
                            #print(f"{predator_instance.agent_name} (energy {round(predator_instance.energy,1)}) creates a new predator",end="")
                            self.activate_new_predator(predator_instance)
                            #print(f" predator new energy: {round(predator_instance.energy,1)}")
                            self.position_agent_on_grid(predator_instance)
                            self.rewards[
                                predator_instance.agent_name
                            ] += self.reproduction_reward_predator
                else:

                    self._unlink_agent_from_grid(predator_instance)
                    predator_instance.is_active = False
                    predator_instance.energy = 0.0
                    #self.active_agent_instance_list_type[predator_instance.agent_type_nr].remove(
                    #    predator_instance
                    #)

                    self.active_predator_instance_list.remove(predator_instance)
                    self.n_active_predator -= 1
                    self.predator_age_of_death_list.append(predator_instance.age)
                    self.n_starved_predator += 1
                    self.rewards[
                        predator_instance.agent_name
                    ] += self.death_reward_predator

            for prey_instance in self.active_prey_instance_list:
                if prey_instance.energy > 0:
                    # new is the position of the predator after the move
                    x_new, y_new = prey_instance.position

                    prey_instance.age += 1
                    grass_instance_in_prey_cell = self.agent_instance_in_grid_location[
                        self.grass_type_nr
                    ][(x_new, y_new)]
                    if grass_instance_in_prey_cell is not None:
                        #print(f"{prey_instance.agent_name} (energy {round(prey_instance.energy,1)})",end="")
                        prey_instance.energy += grass_instance_in_prey_cell.energy
                        #print(f" eats {grass_instance_in_prey_cell.agent_name} (energy {round(grass_instance_in_prey_cell.energy,1)})", end="")
                        #print(f" prey energy after eating: {round(prey_instance.energy,1)}")
                        self._unlink_agent_from_grid(grass_instance_in_prey_cell)
                        self.active_grass_instance_list.remove(
                                grass_instance_in_prey_cell
                        )
                        grass_instance_in_prey_cell.energy = 0.0
                        grass_instance_in_prey_cell.is_active = False
                        self.n_active_grass -= 1
                        self.n_eaten_grass += 1
                        self.rewards[prey_instance.agent_name] += self.catch_reward_grass
                        if prey_instance.energy > self.prey_creation_energy_threshold:
                            #print(f"{prey_instance.agent_name} (energy {round(prey_instance.energy,1)}) creates a new prey",end="")
                            self.activate_new_prey(prey_instance)
                            #print(f" prey new energy: {round(prey_instance.energy,1)}")
                            self.position_agent_on_grid(prey_instance)
                            self.rewards[prey_instance.agent_name] += self.reproduction_reward_prey
                else:
                    self._unlink_agent_from_grid(prey_instance)
                    prey_instance.is_active = False
                    prey_instance.energy = 0.0
                    #self.active_agent_instance_list_type[prey_instance.agent_type_nr].remove(
                    #    prey_instance
                    #)


                    self.active_prey_instance_list.remove(prey_instance)
                    self.n_active_prey -= 1

                    self.prey_age_of_death_list.append(prey_instance.age)
                    self.n_starved_prey += 1
                    self.rewards[prey_instance.agent_name] += self.death_reward_prey

            for grass_name in self.possible_grass_name_list:
                grass_instance = self.agent_name_to_instance_dict[grass_name]
                grass_energy_gain = min(
                    grass_instance.energy_gain_per_step,
                    max(self.max_energy_level_grass - grass_instance.energy, 0),
                )
                grass_instance.energy += grass_energy_gain
                if grass_instance.energy >= self.initial_energy_grass:
                    if not grass_instance.is_active:
                        grass_instance.is_active = True
                        self.n_active_grass += 1
                        self.active_grass_instance_list.append(grass_instance)
                        self.position_agent_on_grid(grass_instance)
                else:
                    if grass_instance.is_active:
                        grass_instance.is_active = False
                        self.n_active_grass -= 1
                        self.active_grass_instance_list.appenremove(grass_instance)
                        self._unlink_agent_from_grid(grass_instance)


            self.n_cycles += 1

            # record number of active agents at the end of the cycle
            self.n_active_predator_list.insert(
                self.n_cycles, self.n_active_predator
            )
            self.n_active_prey_list.insert(self.n_cycles, self.n_active_prey)
            self.n_active_grass_list.insert(self.n_cycles, self.n_active_grass)

            self.total_energy_learning_agents = (
                self.total_energy_predator + self.total_energy_prey
            )
            self.total_energy_predator_list.insert(
                self.n_cycles, self.total_energy_predator
            )
            self.total_energy_prey_list.insert(
                self.n_cycles, self.total_energy_prey
            )
            self.total_energy_grass_list.insert(
                self.n_cycles, self.total_energy_grass
            )
            self.total_energy_learning_agents_list.insert(
                self.n_cycles, self.total_energy_learning_agents
            )

    def observe(self, agent_name: str) -> np.ndarray:
        """
        Returns the observation for the given agent.

        Parameters:
            agent_name (str): The name of the agent for which to return the observation.

        Returns:
            np.ndarray: The observation matrix for the agent.
        """
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
        observation = np.zeros(
            (nr_channels, max_obs_range, max_obs_range), dtype=np.float64
        )
        observation[0].fill(1.0)

        # Calculate bounds for observation array assignment
        xolo = max(0, max_obs_offset - xp)
        xohi = xolo + (xhi - xlo)
        yolo = max(0, max_obs_offset - yp)
        yohi = yolo + (yhi - ylo)

        # Populate observation within visible area
        observation[:, xolo:xohi, yolo:yohi] = np.abs(
            self.model_state[:, xlo:xhi, ylo:yhi]
        )

        # Apply mask to limit observation range based on agent's capabilities
        mask = (max_obs_range - observation_range_agent) // 2
        if mask > 0:
            observation[:, :mask, :] = 0
            observation[:, -mask:, :] = 0
            observation[:, :, :mask] = 0
            observation[:, :, -mask:] = 0
        elif mask < 0:
            raise ValueError(
                "Error: observation_range_agent larger than max_observation_range"
            )

        return observation

    def _apply_agent_action(self, agent_instance: 'DiscreteAgent', action: int) -> None:
        """
        Applies the given action to the agent instance.

        Parameters:
            agent_instance (DiscreteAgent): The agent instance to apply the action to.
            action (int): The action to apply.
        """
        self._unlink_agent_from_grid(agent_instance)
        agent_instance.step(action)
        self._link_agent_to_grid(agent_instance)

    def _unlink_agent_from_grid(self, agent_instance: 'DiscreteAgent') -> None:
        """
        Unlink the given agent from the grid (i.e., free its position).

        Args:
            agent_instance: The agent to unlink from the grid.
        """
        x, y = agent_instance.position
        self.agent_instance_in_grid_location[agent_instance.agent_type_nr, x, y] = None
        self.model_state[agent_instance.agent_type_nr, x, y] = 0.0

        # Add the cell back to available cells since it is now free
        self.available_cells_per_agent_type[agent_instance.agent_type_nr].add((x, y))
        
    def _link_agent_to_grid(self, agent_instance: 'DiscreteAgent') -> None:
        """
        Link the given agent to the grid (i.e., occupy its position).

        Args:
            agent_instance: The agent to link to the grid.
        """
        x, y = agent_instance.position
        self.agent_instance_in_grid_location[agent_instance.agent_type_nr, x, y] = agent_instance
        self.model_state[agent_instance.agent_type_nr, x, y] = agent_instance.energy

        # Remove the position from available cells for the specific agent type
        if (x, y) in self.available_cells_per_agent_type[agent_instance.agent_type_nr]:
            self.available_cells_per_agent_type[agent_instance.agent_type_nr].remove((x, y))

    def _get_new_allowed_position(self, agent_instance, spawning_area, model_state) -> Tuple[int, int]:
        """
        Get a new allowed position for the given agent within the spawning area.

        Args:
            agent_instance: The agent for which to find a new position.
            model_state: The current state of the model.

        Returns:
            Tuple[int, int]: A new allowed position (x, y) for the agent.
        """
        agent_type_nr = agent_instance.agent_type_nr

        # Fetch the available cells for the specific agent type within the spawning area
        possible_positions = [
            (x, y) for (x, y) in self.available_cells_per_agent_type[agent_type_nr]
            if spawning_area[agent_type_nr]["x_begin"] <= x <= spawning_area[agent_type_nr]["x_end"] and
               spawning_area[agent_type_nr]["y_begin"] <= y <= spawning_area[agent_type_nr]["y_end"] and
               model_state[agent_type_nr, x, y] == 0.0
        ]

        if not possible_positions:
            raise ValueError(f"No available positions left for spawning agent type {agent_type_nr}.")

        new_position = random.choice(possible_positions)

        # Remove the position from available cells since it will now be occupied
        self.available_cells_per_agent_type[agent_type_nr].remove(new_position)

        return new_position

    def _deactivate_agent(self, agent_instance: 'DiscreteAgent') -> None:
        """
        Deactivates an agent instance, setting its energy to 0 and unlinking it from the grid.

        Parameters:
            agent_instance (DiscreteAgent): The agent instance to deactivate.
        """
        self._unlink_agent_from_grid(agent_instance)
        agent_instance.is_active = False
        agent_instance.energy = 0.0
        self.active_agent_instance_list_type[agent_instance.agent_type_nr].remove(
            agent_instance
        )

    def _create_agent_name_list_from_instance_list(self, agent_instance_list: List['DiscreteAgent']) -> List[str]:
        """
        Creates a list of agent names from a list of agent instances.

        Parameters:
            agent_instance_list (List[DiscreteAgent]): List of agent instances.

        Returns:
            List[str]: List of agent names.
        """        
        return [agent_instance.agent_name for agent_instance in agent_instance_list]

    def activate_new_predator(self, parent_predator):
        non_active_predator_names = [
            name
            for name in self.possible_predator_name_list
            if not self.agent_name_to_instance_dict[name].is_active
        ]
        if non_active_predator_names:
            # reduce energy of parent predator by the energy needed for reproduction
            parent_predator.energy -= self.initial_energy_predator
            self.model_state[
                self.predator_type_nr,
                parent_predator.position[0],
                parent_predator.position[1],
            ] = parent_predator.energy
            # activate a new predator
            new_predator_name = non_active_predator_names[-1]
            new_predator_instance = self.agent_name_to_instance_dict[new_predator_name]
            new_predator_instance.is_active = True
            new_predator_instance.energy = self.initial_energy_predator
            new_predator_instance.age = 0
            self.active_predator_instance_list.append(new_predator_instance)
            self.n_active_predator += 1
            self.n_born_predator += 1
            x_new, y_new = self._get_new_allowed_position(
                new_predator_instance, self.spawning_area, self.model_state
            )
            new_predator_instance.position = (x_new, y_new)         
            self.position_agent_on_grid(new_predator_instance)

    def activate_new_prey(self, parent_prey):
        non_active_prey_names = [
            name
            for name in self.possible_prey_name_list
            if not self.agent_name_to_instance_dict[name].is_active
        ]
        if non_active_prey_names:
            parent_prey.energy -= self.initial_energy_prey
            self.model_state[
                self.prey_type_nr, parent_prey.position[0], parent_prey.position[1]
            ] = parent_prey.energy
            # activate a new prey
            new_prey_name = non_active_prey_names[-1]
            new_prey_instance = self.agent_name_to_instance_dict[new_prey_name]
            new_prey_instance.is_active = True
            new_prey_instance.energy = self.initial_energy_prey
            new_prey_instance.age = 0
            self.active_prey_instance_list.append(new_prey_instance)
            self.n_active_prey += 1
            self.n_born_prey += 1
            x_new, y_new = self._get_new_allowed_position(
                new_prey_instance, self.spawning_area, self.model_state
            )
            new_prey_instance.position = (x_new, y_new)
            self.position_agent_on_grid(new_prey_instance)

    def position_agent_on_grid(self, agent_instance):
        self.agent_instance_in_grid_location[
            agent_instance.agent_type_nr,
            agent_instance.position[0],
            agent_instance.position[1],
        ] = agent_instance
        self.model_state[
            agent_instance.agent_type_nr,
            agent_instance.position[0],
            agent_instance.position[1],
        ] = agent_instance.energy

    def _reset_rewards(self) -> None:
        """
        Resets the rewards for all agents.
        """
        self.rewards = dict.fromkeys(self.possible_agent_name_list, 0.0)

    def remove_predator(self, predator_instance):
        self.active_predator_instance_list.remove(predator_instance)
        self.n_active_predator -= 1
        self.n_starved_predator += 1
        self.agent_instance_in_grid_location[
            self.predator_type_nr,
            predator_instance.position[0],
            predator_instance.position[1],
        ] = None
        self.model_state[
            self.predator_type_nr,
            predator_instance.position[0],
            predator_instance.position[1],
        ] = 0.0
        predator_instance.is_active = False
        self.predator_age_list.append(predator_instance.age)
        predator_instance.energy = 0.0
        predator_instance.age = 0
        self.rewards[
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
            self.prey_type_nr, prey_instance.position[0], prey_instance.position[1]
        ] = None
        self.model_state[
            self.prey_type_nr, prey_instance.position[0], prey_instance.position[1]
        ] = 0.0
        prey_instance.is_active = False
        self.prey_age_list.append(prey_instance.age)
        prey_instance.energy = 0.0
        prey_instance.age = 0
        self.rewards[prey_instance.agent_name] += self.death_reward_prey

    def remove_grass(self, grass_instance):
        self.active_grass_instance_list.remove(grass_instance)
        self.n_active_grass -= 1
        self.n_eaten_grass += 1
        self.model_state[
            self.grass_type_nr, grass_instance.position[0], grass_instance.position[1]
        ] = 0.0
        grass_instance.energy = 0.0
        grass_instance.is_active = False

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
                self.predator_type_nr,
                parent_predator.position[0],
                parent_predator.position[1],
            ] = parent_predator.energy
            new_predator_instance.energy = self.initial_energy_predator
            new_predator_instance.age = 0
            self.active_predator_instance_list.append(new_predator_instance)
            self.n_active_predator += 1
            self.n_born_predator += 1
            x_new, y_new = self._get_new_allowed_position(
                new_predator_instance, self.spawning_area, self.model_state
            )

            new_predator_instance.position = (x_new, y_new)
            self.agent_instance_in_grid_location[self.predator_type_nr, x_new, y_new] = (
                new_predator_instance
            )
            self.model_state[self.predator_type_nr, x_new, y_new] = (
                new_predator_instance.energy
            )
            self.rewards[
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
                self.prey_type_nr, parent_prey.position[0], parent_prey.position[1]
            ] = parent_prey.energy
            new_prey_instance.energy = self.initial_energy_prey
            new_prey_instance.age = 0
            self.active_prey_instance_list.append(new_prey_instance)
            self.n_active_prey += 1
            self.n_born_prey += 1
            # x_new, y_new = self.find_new_position(self.prey_type_nr)
            x_new, y_new = self._get_new_allowed_position(
                new_prey_instance, self.spawning_area, self.model_state
            )

            new_prey_instance.position = (x_new, y_new)
            self.agent_instance_in_grid_location[self.prey_type_nr, x_new, y_new] = (
                new_prey_instance
            )
            self.model_state[self.prey_type_nr, x_new, y_new] = new_prey_instance.energy
            self.rewards[
                parent_prey.agent_name
            ] += self.reproduction_reward_prey

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

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
            x_screenposition = 380  # x_axis screen position?
            y_screenposition = 150
            y_axis_height = 500
            x_axis_width = width_energy_chart - 120  # = 1680
            x_screenposition_prey_bars = 1450
            title_x = x_screenposition + 1400
            title_y = 60

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
 
    def _initialize_variables(self) -> None:


        self.n_possible_agent_type: List[int] = [0, 0, 0, 0]
        self.n_possible_agent_type[self.predator_type_nr] = self.n_possible_predator
        self.n_possible_agent_type[self.prey_type_nr] = self.n_possible_prey
        self.n_possible_agent_type[self.grass_type_nr] = self.n_possible_grass

        self.n_initial_active_agent_type: List[int] = [0, 0, 0, 0]
        self.n_initial_active_agent_type[self.predator_type_nr] = (
            self.n_initial_active_predator
        )
        self.n_initial_active_agent_type[self.prey_type_nr] = self.n_initial_active_prey

        # episode population metrics
        self.n_active_agent_type: List[int] = [0, 0, 0, 0]
        self.n_active_agent_type[self.predator_type_nr] = self.n_possible_agent_type[
            self.predator_type_nr
        ]
        self.n_active_agent_type[self.prey_type_nr] = self.n_possible_agent_type[
            self.prey_type_nr
        ]
        self.n_active_agent_type[self.grass_type_nr] = self.n_possible_agent_type[
            self.grass_type_nr
        ]
        self.n_active_agent_type: List[int] = [
            0,
            self.n_active_agent_type[self.predator_type_nr],
            self.n_active_agent_type[self.prey_type_nr],
            self.n_active_agent_type[self.grass_type_nr],
        ]
        self.initial_energy_type: List[int] = [
            0,
            self.initial_energy_predator,
            self.initial_energy_prey,
            self.initial_energy_grass,
        ]


        self.total_energy_agent_type: List[int] = [0, 0, 0, 0]

        self.total_energy_agent_type[self.predator_type_nr] = (
            self.n_active_agent_type[self.predator_type_nr]
            * self.initial_energy_type[self.predator_type_nr]
        )
        self.total_energy_agent_type[self.prey_type_nr] = (
            self.n_active_agent_type[self.prey_type_nr]
            * self.initial_energy_type[self.prey_type_nr]
        )
        self.total_energy_agent_type[self.grass_type_nr] = (
            self.n_active_agent_type[self.grass_type_nr]
            * self.initial_energy_type[self.grass_type_nr]
        )

        # Track available cells in the grid per agent type using a dictionary of sets
        self.available_cells_per_agent_type: Dict[int, Set[Tuple[int, int]]] = {
            self.predator_type_nr: set(),
            self.prey_type_nr: set(),
            self.grass_type_nr: set()
        }

        # Initialize available cells for each agent type based on their spawning areas
        for agent_type_nr in [self.predator_type_nr, self.prey_type_nr, self.grass_type_nr]:
            area = self.spawning_area[agent_type_nr]
            self.available_cells_per_agent_type[agent_type_nr] = {
                (x, y)
                for x in range(area["x_begin"], area["x_end"] + 1)
                for y in range(area["y_begin"], area["y_end"] + 1)
            }



        self._reset_rewards()
        
        self.total_energy_predator = 0.0
        self.total_energy_prey = 0.0
        self.total_energy_grass = 0.0
        self.total_energy_learning_agents = (
            self.total_energy_predator + self.total_energy_prey
        )

        # intialization per agent type
        self.learning_agent_types = [self.predator_type_nr, self.prey_type_nr]
        self.agent_types = [
            self.predator_type_nr,
            self.prey_type_nr,
            self.grass_type_nr,
        ]
        self.n_possible_agents: int = (
            self.n_possible_agent_type[self.predator_type_nr]
            + self.n_possible_agent_type[self.prey_type_nr]
        )

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

        for agent_type_nr in self.agent_types:
            self.agent_instance_in_grid_location[agent_type_nr] = np.full(
                (self.x_grid_size, self.y_grid_size), None
            )
            self.possible_agent_name_list_type[agent_type_nr] = []
            self.active_agent_name_list_type[agent_type_nr] = []
            self.possible_agent_instance_list_type[agent_type_nr] = []
            self.active_agent_instance_list_type[agent_type_nr] = []


        self.start_index_type: list[int] = []
        # Initialize start_index_type with the correct size
        self.start_index_type = [0] * len(self.agent_type_name_list)

        self.start_index_type[self.predator_type_nr] = 0
        self.start_index_type[self.prey_type_nr] = self.n_possible_agent_type[
            self.predator_type_nr
        ]
        self.start_index_type[self.grass_type_nr] = (
            self.n_possible_agent_type[self.predator_type_nr]
            + self.n_possible_agent_type[self.prey_type_nr]
        )

        self.predator_age_of_death_list: List[int] = []
        self.prey_age_of_death_list: List[int] = []

