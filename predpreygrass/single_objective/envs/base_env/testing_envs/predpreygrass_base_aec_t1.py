# discretionary libraries
from predpreygrass.aec_predpreygrass.agents.discrete_agent import DiscreteAgent
from predpreygrass.utils.renderer import Renderer


# external libraries
from gymnasium.utils import seeding
from gymnasium import spaces
import os
import numpy as np
import random
from typing import Tuple, List, Optional
import pygame


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
        self.obs_range_predator = obs_range_predator
        self.obs_range_prey = obs_range_prey
        self.energy_gain_per_step_predator = energy_gain_per_step_predator
        self.energy_gain_per_step_prey = energy_gain_per_step_prey
        self.energy_gain_per_step_grass = energy_gain_per_step_grass
        self.initial_energy_predator = initial_energy_predator
        self.initial_energy_prey = initial_energy_prey
        self.initial_energy_grass = initial_energy_grass
        self.catch_reward_grass = catch_reward_grass
        self.catch_reward_prey = catch_reward_prey
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
        self.step_reward_predator = step_reward_predator
        self.step_reward_prey = step_reward_prey
        self.step_reward_grass = step_reward_grass
        self.spawning_area_predator = spawning_area_predator
        self.spawning_area_prey = spawning_area_prey
        self.spawning_area_grass = spawning_area_grass

        self.watch_grid_model = watch_grid_model
        self.num_episodes = num_episodes
        self.max_observation_range = max_observation_range
        self.render_mode = render_mode
        self.cell_scale = cell_scale
        self.x_pygame_window = x_pygame_window
        self.y_pygame_window = y_pygame_window
        self.show_energy_chart = show_energy_chart
        self.regrow_grass = regrow_grass
        self.max_energy_level_grass = max_energy_level_grass

        
        self._initialize_variables()
        # TODO move to config file
        self.torus = True

        # Create a Renderer instance if rendering is needed
        if self.render_mode is not None:
            self.renderer = Renderer(
                env=self,
                cell_scale=cell_scale,
                show_energy_chart=show_energy_chart,
                x_pygame_window=x_pygame_window,
                y_pygame_window=y_pygame_window,
            )
        else:
            self.renderer = None

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

        # observations
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
        
    def reset(self):
        self._initialize_variables()
        # Pre-compute common attributes for agents to avoid repeated lookups
        initial_energy_type = self.initial_energy_type
        obs_range_list_type = self.obs_range_list_type
        spawning_area_list_type = self.spawning_area_list_type
        energy_gain_per_step_type = self.energy_gain_per_step_type
        possible_agent_instance_list_type = self.possible_agent_instance_list_type
        active_agent_instance_list_type = self.active_agent_instance_list_type
        total_energy_agent_type = self.total_energy_agent_type
        agent_name_to_instance_dict = self.agent_name_to_instance_dict

        # Create agents of all types (excluding "wall" agents)
        for agent_type_nr in self.agent_types:
            agent_type_name = self.agent_type_name_list[agent_type_nr]
            possible_agents = possible_agent_instance_list_type[agent_type_nr]
            model_state = self.model_state[agent_type_nr]
            observation_range = obs_range_list_type[agent_type_nr]
            initial_energy = initial_energy_type[agent_type_nr]
            energy_gain_per_step = energy_gain_per_step_type[agent_type_nr]

            # Initialize all possible agents of a certain type (agent_type_nr)
            for _ in range(self.n_possible_agent_type[agent_type_nr]):
                agent_id_nr = self.agent_id_counter
                agent_name = f"{agent_type_name}_{agent_id_nr}"
                self.agent_id_counter += 1

                # Create agent instance with precomputed values to avoid repeated lookups
                agent_instance = DiscreteAgent(
                    agent_type_nr,
                    agent_id_nr,
                    agent_name,
                    model_state,
                    observation_range=observation_range,
                    motion_range=self.motion_range,
                    initial_energy=initial_energy,
                    energy_gain_per_step=energy_gain_per_step,
                    torus=self.torus,
                )
                # Choose a cell for the agent which is not yet occupied by another agent of the same type
                xinit, yinit = self._get_new_allowed_position(
                    agent_instance, spawning_area_list_type, self.model_state
                )
                
                # Update agent instance attributes and link it to grid
                agent_instance.is_active = True
                agent_instance.position = (xinit, yinit)
                agent_instance.energy = initial_energy
                agent_name_to_instance_dict[agent_name] = agent_instance
                self._link_agent_to_grid(agent_instance)
                possible_agents.append(agent_instance)

        # Copy possible agent instances to active agent instances and create agent name lists
        for agent_type_nr in self.agent_types:
            possible_agents = possible_agent_instance_list_type[agent_type_nr]
            active_agent_instance_list_type[agent_type_nr] = possible_agents[:]
            self.possible_agent_name_list_type[agent_type_nr] = [agent.agent_name for agent in possible_agents]

        # Deactivate agents which can be created later at runtime
        for agent_type in self.learning_agent_types:
            start_index = self.start_index_type[agent_type]
            n_initial_active = self.n_initial_active_agent_type[agent_type]
            active_agents = active_agent_instance_list_type[agent_type]

            agents_to_remove = possible_agent_instance_list_type[agent_type][start_index + n_initial_active:]

            for agent_instance in agents_to_remove:
                if agent_instance.is_active:
                    active_agents.remove(agent_instance)
                    self.n_active_agent_type[agent_type] -= 1
                    total_energy_agent_type[agent_type] -= agent_instance.energy
                    self._unlink_agent_from_grid(agent_instance)
                    agent_instance.is_active = False
                    agent_instance.energy = 0.0

        # Removal agents set to false using dictionary comprehensions
        prey_names = self.possible_agent_name_list_type[self.prey_type_nr]
        grass_names = self.possible_agent_name_list_type[self.grass_type_nr]
        predator_names = self.possible_agent_name_list_type[self.predator_type_nr]

        # Define the learning agents
        self.possible_learning_agent_name_list = predator_names + prey_names
        self.n_cycles = 0

        # Time series of active agents and total energy
        for agent_type_nr in (self.predator_type_nr, self.prey_type_nr, self.grass_type_nr):
            self.n_active_agent_list_type[agent_type_nr].insert(self.n_cycles, self.n_active_agent_type[agent_type_nr])
            self.total_energy_agent_list_type[agent_type_nr].insert(self.n_cycles, total_energy_agent_type[agent_type_nr])

        self.total_energy_learning_agents = total_energy_agent_type[self.predator_type_nr] + total_energy_agent_type[self.prey_type_nr]
        self.total_energy_learning_agents_list.insert(self.n_cycles, self.total_energy_learning_agents)

        # Episode population metrics
        self.n_starved_predator = 0
        self.n_starved_prey = 0
        self.n_eaten_prey = 0
        self.n_eaten_grass = 0
        self.n_born_predator = 0
        self.n_born_prey = 0

    def step(self, action, agent_instance, is_last_step_of_cycle):
        if agent_instance.is_active:
            self._apply_agent_action(agent_instance, action)

        # Apply the engagement rules and reap rewards
        if is_last_step_of_cycle:
            self._reset_rewards()

            # Filter out agents with positive energy in advance
            predator_instances = [agent for agent in self.active_agent_instance_list_type[self.predator_type_nr] if agent.energy > 0]
            prey_instances = [agent for agent in self.active_agent_instance_list_type[self.prey_type_nr] if agent.energy > 0]
            grass_instances = [grass for grass in self.possible_agent_instance_list_type[self.grass_type_nr] if grass.energy < self.max_energy_level_grass]

            # Process predators
            for predator_instance in predator_instances:
                x_new, y_new = predator_instance.position
                predator_instance.energy += predator_instance.energy_gain_per_step
                predator_instance.age += 1

                prey_instance = self.agent_instance_in_grid_location[self.prey_type_nr].get((x_new, y_new))
                if prey_instance:
                    predator_instance.energy += prey_instance.energy
                    self._deactivate_agent(prey_instance)
                    self.n_active_agent_type[self.prey_type_nr] -= 1
                    self.agent_age_of_death_list_type[self.prey_type_nr].append(prey_instance.age)
                    self.n_eaten_prey += 1
                    self.rewards[predator_instance.agent_name] += self.catch_reward_prey
                    self.rewards[prey_instance.agent_name] += self.death_reward_prey

                    if predator_instance.energy > self.predator_creation_energy_threshold:
                        potential_new_predators = self._non_active_agent_instance_list(predator_instance)
                        if potential_new_predators:
                            self._reproduce_new_agent(predator_instance, potential_new_predators)
                            self.n_active_agent_type[self.predator_type_nr] += 1
                            self.n_born_predator += 1
                        self.rewards[predator_instance.agent_name] += self.reproduction_reward_predator

            # Handle predators that have starved
            for predator_instance in self.active_agent_instance_list_type[self.predator_type_nr]:
                if predator_instance.energy <= 0:
                    self._deactivate_agent(predator_instance)
                    self.n_active_agent_type[self.predator_type_nr] -= 1
                    self.agent_age_of_death_list_type[self.predator_type_nr].append(predator_instance.age)
                    self.n_starved_predator += 1
                    self.rewards[predator_instance.agent_name] += self.death_reward_predator

            # Process prey
            for prey_instance in prey_instances:
                x_new, y_new = prey_instance.position
                prey_instance.age += 1
                prey_instance.energy += prey_instance.energy_gain_per_step

                grass_instance = self.agent_instance_in_grid_location[self.grass_type_nr].get((x_new, y_new))
                if grass_instance:
                    prey_instance.energy += grass_instance.energy
                    self._deactivate_agent(grass_instance)
                    self.n_active_agent_type[self.grass_type_nr] -= 1
                    self.n_eaten_grass += 1
                    self.rewards[prey_instance.agent_name] += self.catch_reward_grass

                    if prey_instance.energy > self.prey_creation_energy_threshold:
                        potential_new_preys = self._non_active_agent_instance_list(prey_instance)
                        if potential_new_preys:
                            self._reproduce_new_agent(prey_instance, potential_new_preys)
                            self.n_active_agent_type[self.prey_type_nr] += 1
                            self.n_born_prey += 1
                        self.rewards[prey_instance.agent_name] += self.reproduction_reward_prey

            # Handle prey that have starved
            for prey_instance in self.active_agent_instance_list_type[self.prey_type_nr]:
                if prey_instance.energy <= 0:
                    self._deactivate_agent(prey_instance)
                    self.n_active_agent_type[self.prey_type_nr] -= 1
                    self.agent_age_of_death_list_type[self.prey_type_nr].append(prey_instance.age)
                    self.n_starved_prey += 1
                    self.rewards[prey_instance.agent_name] += self.death_reward_prey

            # Process grass (re)growth more efficiently by iterating over only inactive grass agents
            for grass_instance in grass_instances:
                grass_instance.energy = min(
                    grass_instance.energy + grass_instance.energy_gain_per_step,
                    self.max_energy_level_grass,
                )
                if grass_instance.energy >= self.initial_energy_type[self.grass_type_nr] and not grass_instance.is_active:
                    self._activate_agent(grass_instance)
                    self.n_active_agent_type[self.grass_type_nr] += 1
                elif grass_instance.energy < self.initial_energy_type[self.grass_type_nr] and grass_instance.is_active:
                    self._deactivate_agent(grass_instance)
                    self.n_active_agent_type[self.grass_type_nr] -= 1

            # Record step metrics
            self._record_population_metrics()
            self.n_cycles += 1

    def observe(self, agent_name: str, torus: bool = True) -> np.ndarray:
        """
        Returns the observation for the given agent.

        Parameters:
            agent_name (str): The name of the agent for which to return the observation.
            torus (bool): Whether to treat the grid as a torus (wrap-around).

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

        # Initialize observation matrix (filled with 1s in the wall channel only)
        observation = np.ones((nr_channels, max_obs_range, max_obs_range), dtype=np.float64)
        observation[1:] = 0.0

        # Precompute torus wrap-around adjustments if applicable
        wrap = (lambda coord, size: coord % size) if torus else (lambda coord, size: coord if 0 <= coord < size else None)

        # Populate observation within visible area using vectorization where possible
        x_indices = [(wrap(xp + dx, x_grid_size), dx + max_obs_offset) for dx in range(-max_obs_offset, max_obs_offset + 1)]
        y_indices = [(wrap(yp + dy, y_grid_size), dy + max_obs_offset) for dy in range(-max_obs_offset, max_obs_offset + 1)]

        for x, obs_x in x_indices:
            if x is None:
                continue
            for y, obs_y in y_indices:
                if y is None:
                    continue
                # Populate the observation matrix with the model state
                observation[:, obs_x, obs_y] = self.model_state[:, x, y]

        # Apply mask to limit observation range based on agent's capabilities
        if observation_range_agent < max_obs_range:
            mask = (max_obs_range - observation_range_agent) // 2
            observation[:, :mask, :] = 0
            observation[:, -mask:, :] = 0
            observation[:, :, :mask] = 0
            observation[:, :, -mask:] = 0

        return observation


    def render(self) -> Optional[np.ndarray]:
        """
        Renders the environment if a renderer is available.

        Returns:
            Optional[np.ndarray]: Rendered frame or None if no renderer is set.
        """
        if self.renderer:
            return self.renderer.render()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

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

    def _get_new_allowed_position(self, agent_instance, spawning_area_list_type, model_state) -> Tuple[int, int]:
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
            if self.spawning_area_list_type[agent_type_nr]["x_begin"] <= x <= self.spawning_area_list_type[agent_type_nr]["x_end"] and
               self.spawning_area_list_type[agent_type_nr]["y_begin"] <= y <= self.spawning_area_list_type[agent_type_nr]["y_end"] and
               model_state[agent_type_nr, x, y] == 0.0
        ]

        if not possible_positions:
            raise ValueError(f"No available positions left for spawning agent type {agent_type_nr}.")

        new_position = random.choice(possible_positions)

        # Remove the position from available cells since it will now be occupied
        self.available_cells_per_agent_type[agent_type_nr].remove(new_position)

        return new_position

    def _activate_agent(self, agent_instance: 'DiscreteAgent') -> None:
        """
        Activates an agent instance, setting its energy and age, and linking it to the grid.

        Parameters:
            agent_instance (DiscreteAgent): The agent instance to activate.
        """
        agent_instance.is_active = True
        agent_instance.energy = self.initial_energy_type[agent_instance.agent_type_nr]
        agent_instance.age = 0
        self.active_agent_instance_list_type[agent_instance.agent_type_nr].append(
            agent_instance
        )
        self._link_agent_to_grid(agent_instance)

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

    def _reproduce_new_agent(
        self, parent_agent_instance: 'DiscreteAgent', non_active_agent_instance_list: List['DiscreteAgent']
    ) -> None:
        """
        Reproduces a new agent based on the parent agent instance.

        Parameters:
            parent_agent_instance (DiscreteAgent): The parent agent instance.
            non_active_agent_instance_list (List[DiscreteAgent]): List of non-active agent instances to activate.
        """

        agent_type_nr = parent_agent_instance.agent_type_nr
        parent_agent_instance.energy -= self.initial_energy_type[agent_type_nr]
        self.model_state[
            agent_type_nr,
            parent_agent_instance.position[0],
            parent_agent_instance.position[1],
        ] = parent_agent_instance.energy
        # activate a new agent
        new_agent_instance = non_active_agent_instance_list[-1]
        new_agent_instance.position = self._get_new_allowed_position(
            new_agent_instance, self.spawning_area_list_type[new_agent_instance.agent_type_nr], self.model_state
        )
        self._activate_agent(new_agent_instance)

    def _non_active_agent_instance_list(self, agent_instance: 'DiscreteAgent') -> List['DiscreteAgent']:
        """
        Returns a list of non-active agent instances of the same type as the given agent.

        Parameters:
            agent_instance (DiscreteAgent): The agent instance to check against.

        Returns:
            List[DiscreteAgent]: A list of non-active agent instances of the same type.
        """
        return [
            agent_instance
            for agent_instance in self.possible_agent_instance_list_type[
                agent_instance.agent_type_nr
            ]
            if not agent_instance.is_active
        ]

    def _create_agent_name_list_from_instance_list(self, agent_instance_list: List['DiscreteAgent']) -> List[str]:
        """
        Creates a list of agent names from a list of agent instances.

        Parameters:
            agent_instance_list (List[DiscreteAgent]): List of agent instances.

        Returns:
            List[str]: List of agent names.
        """        
        return [agent_instance.agent_name for agent_instance in agent_instance_list]

    def _record_population_metrics(self) -> None:
        """
        Records population metrics for predators, prey, and grass.
        """
        # record number of active agents at the end of the cycle
        self.n_active_agent_list_type[self.predator_type_nr].insert(
            self.n_cycles, self.n_active_agent_type[self.predator_type_nr]
        )
        self.n_active_agent_list_type[self.prey_type_nr].insert(self.n_cycles, self.n_active_agent_type[self.prey_type_nr])
        self.n_active_agent_list_type[self.grass_type_nr].insert(self.n_cycles, self.n_active_agent_type[self.grass_type_nr])

        self.total_energy_learning_agents = (
            self.total_energy_agent_type[self.predator_type_nr] + self.total_energy_agent_type[self.prey_type_nr]
        )
        self.total_energy_agent_list_type[self.predator_type_nr].insert(
            self.n_cycles, self.total_energy_agent_type[self.predator_type_nr]
        )
        self.total_energy_agent_list_type[self.prey_type_nr].insert(
            self.n_cycles, self.total_energy_agent_type[self.prey_type_nr]
        )
        self.total_energy_agent_list_type[self.grass_type_nr].insert(
            self.n_cycles, self.total_energy_agent_type[self.grass_type_nr]
        )
        self.total_energy_learning_agents_list.insert(
            self.n_cycles, self.total_energy_learning_agents
        )

    def _reset_rewards(self) -> None:
        """
        Resets the rewards for all agents.
        """
        self.rewards = dict.fromkeys(self.possible_learning_agent_name_list, 0.0)

    def _initialize_variables(self) -> None:
        # Agent types definitions
        self.agent_type_name_list = ["wall", "predator", "prey", "grass"]
        self.predator_type_nr, self.prey_type_nr, self.grass_type_nr = 1, 2, 3
        self.agent_id_counter = 0

        # Episode population metrics
        self.n_possible_agent_type = [0, self.n_possible_predator, self.n_possible_prey, self.n_possible_grass]
        self.n_initial_active_agent_type = [0, self.n_initial_active_predator, self.n_initial_active_prey, self.n_possible_grass]
        self.n_active_agent_type = [0, self.n_possible_predator, self.n_possible_prey, self.n_possible_grass]

        # Initialization per agent type
        self.learning_agent_types = [self.predator_type_nr, self.prey_type_nr]
        self.agent_types = [self.predator_type_nr, self.prey_type_nr, self.grass_type_nr]
        self.n_possible_agents = sum(self.n_possible_agent_type[1:3])

        # Translation of input parameters
        self.obs_range_list_type = [0, self.obs_range_predator, self.obs_range_prey, 0]
        self.energy_gain_per_step_type = [0, self.energy_gain_per_step_predator, self.energy_gain_per_step_prey, self.energy_gain_per_step_grass]
        self.initial_energy_type = [0, self.initial_energy_predator, self.initial_energy_prey, self.initial_energy_grass]
        self.catch_reward_list_type = [0, 0, self.catch_reward_prey, self.catch_reward_grass]
        self.step_reward_list_type = [0, self.step_reward_predator, self.step_reward_prey, self.step_reward_grass]
        self.spawning_area_list_type = [{}, self.spawning_area_predator, self.spawning_area_prey, self.spawning_area_grass]

        # Internal initializations
        list_length = len(self.agent_type_name_list)
        self.agent_instance_in_grid_location = np.full((list_length, self.x_grid_size, self.y_grid_size), None, dtype=object)
        self.nr_observation_channels: int = len(self.agent_type_name_list)
        # reset model state
        self.model_state: np.ndarray = np.zeros(
            (self.nr_observation_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )
        self.agent_name_to_instance_dict = {}
        self.total_energy_agent_type = [0.0] * list_length
        self.possible_agent_instance_list_type = [[] for _ in range(list_length)]
        self.active_agent_instance_list_type = [[] for _ in range(list_length)]
        self.agent_age_of_death_list_type = [[] for _ in range(list_length)]
        self.possible_agent_name_list_type = [[] for _ in range(list_length)]
        self.n_active_agent_list_type = [[] for _ in range(list_length)]
        self.total_energy_agent_list_type = [[] for _ in range(list_length)]
        self.total_energy_learning_agents_list = []

        # Initialize start_index_type and agent ID/name ranges
        self.start_index_type = [sum(self.n_possible_agent_type[:i]) for i in range(list_length)]
        self.agent_id_nr_range_type = [
            range(self.start_index_type[i], self.start_index_type[i + 1]) if i < list_length - 1 else range(self.start_index_type[i], self.start_index_type[i] + self.n_possible_agent_type[i])
            for i in range(list_length)
        ]
        for agent_type in self.agent_types:
            self.possible_agent_name_list_type[agent_type] = [
                f"{self.agent_type_name_list[agent_type]}_{a}" for a in self.agent_id_nr_range_type[agent_type]
            ]
        self.possible_learning_agent_name_list = (
            self.possible_agent_name_list_type[self.predator_type_nr] + self.possible_agent_name_list_type[self.prey_type_nr]
        )
        for agent_type in self.agent_types:
            self.total_energy_agent_type[agent_type] = (
                self.n_active_agent_type[agent_type] * self.initial_energy_type[agent_type]
            )
        self.total_energy_learning_agents = sum(
            self.total_energy_agent_type[agent_type] for agent_type in self.learning_agent_types
        )

        # Seed, file name, and cycle initialization
        self._seed()
        self.file_name = 0
        self.n_cycles = 0

        # Track available cells in the grid per agent type
        self.available_cells_per_agent_type = {
            agent_type: {
                (x, y)
                for x in range(self.spawning_area_list_type[agent_type]["x_begin"], self.spawning_area_list_type[agent_type]["x_end"] + 1)
                for y in range(self.spawning_area_list_type[agent_type]["y_begin"], self.spawning_area_list_type[agent_type]["y_end"] + 1)
            }
            for agent_type in self.agent_types
        }

        # Population metrics
        self.n_starved_predator = 0
        self.n_starved_prey = 0
        self.n_eaten_prey = 0
        self.n_eaten_grass = 0
        self.n_born_predator = 0
        self.n_born_prey = 0

        self._reset_rewards()
