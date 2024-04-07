"""
pred, prey, grass RlLib environment. Fixed rewards and parallel step function.
"""
import os
import numpy as np
import random
from typing import List
import pygame
from collections import defaultdict
from collections import OrderedDict # for ordered dictionaries, to be used as self.observations

from gymnasium.utils import seeding
from gymnasium.spaces import Discrete, Dict, Box

from ray.rllib import MultiAgentEnv

from agents.discrete_agent import DiscreteAgent

class PredPreyGrassEnv(MultiAgentEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predpreygrass_fixed_rewards",
        "is_parallelizable": True,
        "render_fps": 5,
    }    
    #the environment parameters are set in the env_config dictionary
    def __init__(self, configuration):
        super().__init__() 
        # if configuration is None, set it to an empty dictionary
        configuration = configuration or {} 
        # set defaults if no config["key"] is provided
        # environment configuration
        self.x_grid_size = configuration.get("x_grid_size",16)
        self.y_grid_size = configuration.get("y_grid_size",16)
        self.max_cycles = configuration.get("max_cycles",500)
        # agent configuration
        # predator configuration
        self.n_initial_predator = configuration.get("n_initial_predator",3)
        self.obs_range_predator = configuration.get("obs_range_predator",5)
        self.initial_energy_predator = configuration.get("initial_energy_predator",10.0)
        self.energy_loss_per_step_predator = configuration.get("energy_loss_per_step_predator",-0.1)
        self.catch_prey_reward = configuration.get("catch_reward_prey",5.0)
        # prey configuration
        self.n_initial_prey = configuration.get("n_initial_prey",3)
        self.obs_range_prey = configuration.get("obs_range_range_prey",7)
        self.initial_energy_prey = configuration.get("initial_energy_prey",10.0)
        self.energy_loss_per_step_prey = configuration.get("energy_loss_per_step_prey",-0.05)
        self.catch_grass_reward = configuration.get("catch_reward_grass",3.0)
        # grass configuration
        self.n_initial_grass = configuration.get("n_initial_grass",30)
        # visualization configuration
        self.render_mode = configuration.get("render_mode",None)
        self.cell_scale = configuration.get("cell_scale",40)
        self.x_pygame_window = configuration.get("x_pygame_window",0)
        self.y_pygame_window = configuration.get("y_pygame_window",0)


        self._skip_env_checking = False

        # pygame position window
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x_pygame_window, 
                                                        self.y_pygame_window)
        #print("configuration",configuration)

        self._seed()

        self.render_mode = "human"
        pygame.init()
        self.closed = False


        # agent types
        self.agent_id_counter = 0
        self.agent_type_name_list = ["wall", "predator", "prey", "grass"]  # different types of agents 
        self.predator_type_nr = self.agent_type_name_list.index("predator") #0
        self.prey_type_nr = self.agent_type_name_list.index("prey")  #1
        self.grass_type_nr = self.agent_type_name_list.index("grass")  #2
        
        # lists of agents
        # initialization
        self.n_active_predator = self.n_initial_predator
        self.n_active_prey = self.n_initial_prey  
        self.n_active_grass = self.n_initial_grass
        self.predator_instance_list = [] # list of all living predator
        self.prey_instance_list = [] # list of all living prey
        self.grass_instance_list = [] # list of all living grass
        self.agent_instance_list = [] # list of all living agents
        self.predator_name_list = []
        self.prey_name_list = []
        self.grass_name_list = []
 
        self.agent_instance_in_grid_location = np.empty((len(self.agent_type_name_list), self.x_grid_size, self.y_grid_size), dtype=object)
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            self.agent_instance_in_grid_location[agent_type_nr] = np.full((self.x_grid_size, self.y_grid_size), None)


        self.agent_name_to_instance_dict = dict()
        self.n_agents = self.n_active_predator + self.n_active_prey

        # creation agent type lists
        self.predator_name_list =  ["predator" + "_" + str(a) for a in range(self.n_active_predator)]
        self.prey_name_list =  ["prey" + "_" + str(a) for a in range(self.n_active_predator, self.n_active_prey+self.n_active_predator)]
        self.agent_name_list = self.predator_name_list + self.prey_name_list

        # this is to integrate to rllib structure
        self.agents = self.agent_name_list 
        self._agent_ids = set(self.agents)  # rllib integration
        self.possible_agents = self.agents[:]
        self.num_agents = len(self.agents)
        # end creation agent type lists



        # observations
        self.nr_observation_channels = len(self.agent_type_name_list)

        # observation space with channels first; same shape as observation shapes
        self.obs_shape_predator = (self.nr_observation_channels, self.obs_range_predator, self.obs_range_predator)
        self.obs_shape_prey = (self.nr_observation_channels, self.obs_range_prey, self.obs_range_prey)

        self.obs_space_predator = Box(
            low=0,
            high=1,
            shape=self.obs_shape_predator,
            dtype=np.int32,
        )

        self.obs_space_prey = Box(
            low=0,
            high=1,
            shape=self.obs_shape_prey,
            dtype=np.int32,
        )


        self._observation_space_predators = [self.obs_space_predator for _ in range(self.n_initial_predator)]
        self._observation_space_prey = [self.obs_space_prey for _ in range(self.n_initial_prey)]
        self._observation_space_agents = self._observation_space_predators + self._observation_space_prey
        self._agents = self.predator_name_list + self.prey_name_list # double to self.agents unnecessary?
 
        # this is to integrate to rllib structure
        self.observation_space_predators = Dict(dict(zip(self._agents, self._observation_space_predators)))
        self.observation_space_prey = Dict(dict(zip(self._agents, self._observation_space_prey)))
        self.observation_space = Dict(dict(zip(self._agents, self._observation_space_agents)))
        #print("self.observation_space",self.observation_space)
        #print("isinstance(self.observation_space, gym.spaces.Dict)",isinstance(self.observation_space, gym.spaces.Dict))

        # check observation space and observations sample
        self._spaces_in_preferred_format = True 
        
        observations_sample = self.observation_space.sample()
        if self._check_if_obs_space_maps_agent_id_to_sub_space(): 
            #print("Observation space maps correctly agent ids to spaces of individual agents")
            pass
        else:
            raise ValueError("Observation space NOT correctly maps agent ids to spaces of individual agents")            
        if self.observation_space_contains(observations_sample):
            #print("Observation sample is correctly an element of observation space")
            pass
        else:
            raise ValueError("Observation sample is NOT correctly an element of the observation space")
        

        obs_predator = np.zeros(self.obs_shape_predator, dtype=np.int32)
        obs_prey = np.zeros(self.obs_shape_prey, dtype=np.int32)

        self._observation = [obs_predator for _ in range(self.n_initial_predator)] +\
                            [obs_prey for _ in range(self.n_initial_prey)]

        self.observations = dict() # or ordered dictionary
         #self.observations = observations_sample

        # end observations


        # actions
        self.motion_range = [
            [-1, 0], # move left
            [0, -1], # move up
            [0, 0], # stay
            [0, 1], # move down
            [1, 0], # move right
        ]    
        self.n_actions_agent=len(self.motion_range)
        action_space_agent = Discrete(self.n_actions_agent)  
        self._action_space = [action_space_agent for _ in range(self.n_agents)] 
        self.action_space = Dict(dict(zip(self.agents, self._action_space)))
        self._action_space_in_preferred_format = self._check_if_action_space_maps_agent_id_to_sub_space() # True
        # end actions

        # removal agents
        self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.predator_who_remove_prey_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list])) 
        self.prey_to_be_removed_by_predator_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.prey_to_be_removed_by_starvation_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        # end removal agents

        # visualization
        self.screen = None
        self.save_image_steps = False
        self.width_energy_chart = 800
        self.height_energy_chart = self.cell_scale * self.y_grid_size
        # end visualization

        self.file_name = 0
        self.n_cycles = 0


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        # empty agent lists
        self._agent_ids = set()
        self.predator_instance_list: List[DiscreteAgent] = []
        self.prey_instance_list: List[DiscreteAgent] = []
        self.grass_instance_list: List[DiscreteAgent] =[]
        self.agent_instance_list: List[DiscreteAgent] = []

        self.predator_name_list: List[str] = []
        self.prey_name_list: List[str] = []
        self.grass_name_list: List[str] = []
        self.agent_name_list: List[str] = []

        self.agent_type_instance_list = [[] for _ in range(4)]


        # initialization
        self.n_active_predator: int = self.n_initial_predator
        self.n_active_prey: int = self.n_initial_prey  
        self.n_active_grass: int = self.n_initial_grass


        self.n_agent_instance_list = [0, self.n_active_predator, self.n_active_prey, self.n_active_grass]
        self.obs_range_list = [0, self.obs_range_predator, self.obs_range_prey, 0]
        self.initial_energy_list = [0, self.initial_energy_predator, self.initial_energy_prey, 2]
        self.energy_loss_per_step_list = [0, self.energy_loss_per_step_predator, self.energy_loss_per_step_prey, 0]

        self.agent_id_counter = 0
        self.agent_name_to_instance_dict = {}        
        self.model_state = np.zeros((self.nr_observation_channels, self.x_grid_size, self.y_grid_size), dtype=np.int32)
        

        # create agents of all types excluding "wall"
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            agent_type_name = self.agent_type_name_list[agent_type_nr]
            #empty cell list: an array of tuples with the coordinates of empty cells
            empty_cell_list = [(i, j) for i in range(self.x_grid_size) for j in range(self.y_grid_size)] # all cells are empty
            for _ in range(self.n_agent_instance_list[agent_type_nr]): 
                agent_id_nr = self.agent_id_counter           
                agent_name = agent_type_name + "_" + str(agent_id_nr)
                # suggested by copilot to integrate working with rllib
                if agent_type_name != "grass":
                    self._agent_ids.add(agent_name)
                self.agent_id_counter+=1
                agent_instance = DiscreteAgent(
                    agent_type_nr, 
                    agent_id_nr,
                    agent_name,
                    self.model_state[agent_type_nr], # to avoid occpuying the same cell by two agents
                    observation_range= self.obs_range_list[agent_type_nr],
                    motion_range=self.motion_range,
                    initial_energy=self.initial_energy_list[agent_type_nr],
                    catch_grass_reward=self.catch_grass_reward,
                    catch_prey_reward=self.catch_prey_reward,
                    energy_loss_per_step=self.energy_loss_per_step_list[agent_type_nr]
                )
                position = np.zeros(2, dtype=np.int32)  # x and y position
                
                #  updates lists en records
                xinit, yinit = random.choice(empty_cell_list)
                position[0], position[1] = xinit, yinit

                empty_cell_list.remove((xinit,yinit)) # occupied cell removed from empty_cell_list
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                #agent_instance.position = (xinit, yinit)  
                agent_instance.position = position  
                agent_instance.is_alive = True
                agent_instance.energy = self.initial_energy_list[agent_type_nr]
                self.model_state[agent_type_nr, xinit, yinit] = 1
                self.agent_type_instance_list[agent_type_nr].append(agent_instance) 
                self.agent_instance_in_grid_location[agent_type_nr, xinit, yinit]  = agent_instance


        self.predator_instance_list = self.agent_type_instance_list[self.predator_type_nr]
        self.prey_instance_list = self.agent_type_instance_list[self.prey_type_nr]
        self.grass_instance_list = self.agent_type_instance_list[self.grass_type_nr]

        self.predator_name_list =  self.create_agent_name_list_from_instance_list(
            self.predator_instance_list
        )
        self.prey_name_list =  self.create_agent_name_list_from_instance_list(
            self.prey_instance_list
        )
        self.grass_name_list =  self.create_agent_name_list_from_instance_list(
            self.grass_instance_list
        )

        # add grass and prey agents to position dict

        # removal agents set to false
        self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))

        # define the learning agents
        self.agent_instance_list = self.predator_instance_list + self.prey_instance_list        
        self.agent_name_list = self.predator_name_list + self.prey_name_list
        self.observations = self._get_obs()
        # print("self.observations",self.observations)

        self.n_cycles = 0
        
        self.steps = 0
        self.agents = self.possible_agents
        self._agent_ids = set(self.agents)  # rllib integration

             
        self.possible_agents = self.agents[:]
        self.agent_name_to_index_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.steps = 0
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminateds = dict(zip(self.agents, [False for _ in self.agents]))
        self.terminateds["__all__"] = False
        self.truncateds = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncateds["__all__"] = False
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))


        return self.observations, self.infos 
        
    def step(self, actions):
        # reset rewards to zero during every step
        self.rewards = dict(zip(self.agent_name_list, [0.0 for _ in self.agent_name_list]))
        """
        print("cycle",self.n_cycles)
        print(self._agent_ids)
        print()
        """

        for agent_instance in self.agent_instance_list:
            if self.terminateds[agent_instance.agent_name]:
                continue
            if agent_instance.is_alive:
                agent_type_nr = agent_instance.agent_type_nr
                agent_name = agent_instance.agent_name
                agent_energy = agent_instance.energy

                # If the agent is a predator and it's alive
                if agent_type_nr == self.predator_type_nr: 
                    if agent_energy > 0: # If predator has energy
                        # Move the predator and update the model state
                        self.agent_instance_in_grid_location[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] = None
                        self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] -= 1
                        agent_instance.move(actions[agent_name])    
                        self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] += 1
                        self.agent_instance_in_grid_location[agent_type_nr, agent_instance.position[0], agent_instance.position[1]] = agent_instance
                        # If there's prey at the new position, remove (eat) one at random
                        x_new_position_predator, y_new_position_predator = agent_instance.position
                        if self.model_state[self.prey_type_nr, x_new_position_predator, y_new_position_predator] > 0:
                            prey_instance_removed = self.agent_instance_in_grid_location[self.prey_type_nr][(x_new_position_predator, y_new_position_predator)]
                            self.predator_who_remove_prey_dict[agent_name] = True
                            self.prey_to_be_removed_by_predator_dict[prey_instance_removed.agent_name] = True
                
                            # remove prey which gets eaten by a predator 
                            self.prey_instance_list.remove(prey_instance_removed)
                            self.n_active_prey -= 1
                            self.agent_instance_in_grid_location[self.prey_type_nr,prey_instance_removed.position[0],prey_instance_removed.position[1]] = None
                            self.model_state[self.prey_type_nr,prey_instance_removed.position[0],prey_instance_removed.position[1]] -= 1
                            prey_instance_removed.is_alive = False
                            prey_instance_removed.energy = 0.0
                            self._agent_ids.remove(prey_instance_removed.agent_name)
                            self.terminateds[prey_instance_removed.agent_name] = True

                            # reap rewards for predator which removes prey
                            catch_reward_predator = agent_instance.catch_prey_reward * self.predator_who_remove_prey_dict[agent_name] 
                            self.rewards[agent_name] += catch_reward_predator
                            agent_instance.energy += catch_reward_predator
                            #print(agent_instance.name," has energy: ",round(agent_instance.energy,1))
                        
                        step_reward_predator = agent_instance.energy_loss_per_step
                        self.rewards[agent_name] += step_reward_predator
                        agent_instance.energy += step_reward_predator 


                    else:  # If predator has no energy, it starves to death
                        self.predator_instance_list.remove(agent_instance)
                        self.n_active_predator -= 1
                        self.agent_instance_in_grid_location[self.predator_type_nr,agent_instance.position[0],agent_instance.position[1]] = None
                        self.model_state[self.predator_type_nr,agent_instance.position[0],agent_instance.position[1]] -= 1
                        agent_instance.is_alive = False
                        agent_instance.energy = 0.0
                        self._agent_ids.remove(agent_name)
                        self.terminateds[agent_name] = True

                # If the agent is a prey and it's alive
                elif agent_type_nr == self.prey_type_nr:
                    if agent_energy > 0:  # If prey has energy
                        # Move the prey and update the model state
                        self.agent_instance_in_grid_location[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] = None
                        #self.remove_agent_instance_from_position_dict(agent_instance)
                        self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] -= 1
                        agent_instance.move(actions[agent_instance.agent_name])
                        self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] += 1
                        self.agent_instance_in_grid_location[agent_type_nr, agent_instance.position[0], agent_instance.position[1]] = agent_instance
                        #self.add_agent_instance_to_position_dict(agent_instance)
                        x_new_position_prey, y_new_position_prey = agent_instance.position

                        # If there's grass at the new position, remove (eat) one at random
                        if self.model_state[self.grass_type_nr, x_new_position_prey, y_new_position_prey] > 0: 
                            grass_instance_removed = self.agent_instance_in_grid_location[self.grass_type_nr][(x_new_position_prey, y_new_position_prey)]                        
                            self.prey_who_remove_grass_dict[agent_name] = True

                            # Immediately remove the grass instance from the position dict
                            self.agent_instance_in_grid_location[self.grass_type_nr,x_new_position_prey,y_new_position_prey] = None
                            # remove grass which gets eaten by a prey
                            #removes grass_name from 'grass_name_list'
                            self.grass_instance_list.remove(grass_instance_removed)
                            self.n_active_grass -= 1
                            self.agent_instance_in_grid_location[self.grass_type_nr,grass_instance_removed.position[0],grass_instance_removed.position[1]] = None
                            self.model_state[self.grass_type_nr,grass_instance_removed.position[0],grass_instance_removed.position[1]] -= 1


                            # reap rewards for prey which removes grass
                            catch_reward_prey = agent_instance.catch_grass_reward * self.prey_who_remove_grass_dict[agent_name] 
                            # this is not cumulative reward, rewards is set to zero before 'last' step
                            self.rewards[agent_name] += catch_reward_prey
                            agent_instance.energy += catch_reward_prey

                        step_reward_prey = agent_instance.energy_loss_per_step
                        self.rewards[agent_name] += step_reward_prey
                        agent_instance.energy += step_reward_prey

                    else: # prey starves to death
                        self.prey_to_be_removed_by_starvation_dict[agent_name] = True
                        # remove prey which starves to death
                        self.prey_instance_list.remove(agent_instance)
                        self.n_active_prey -= 1
                        self.agent_instance_in_grid_location[self.prey_type_nr,agent_instance.position[0],agent_instance.position[1]] = None
                        self.model_state[self.prey_type_nr,agent_instance.position[0],agent_instance.position[1]] -= 1
                        agent_instance.is_alive = False
                        agent_instance.energy = 0.0
                        self._agent_ids.remove(agent_name)
                        self.terminateds[agent_name] = True



        self.n_cycles = self.n_cycles + 1
        
        #reinit agents records to default at the end of the cycle
        self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.predator_who_remove_prey_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list])) 
        self.prey_to_be_removed_by_predator_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))

        self.observations = self._get_obs()
        if self.n_cycles > self.max_cycles:
            for agent_name in self.agents:
                self.truncateds[agent_name] = True
            self.truncateds["__all__"] = True
        self.terminateds["__all__"] = self.is_no_grass or self.is_no_prey or self.is_no_predator 


        self.steps += 1
        for agent_instance in self.agent_instance_list:
            agent_name = agent_instance.agent_name
            if self.terminateds[agent_name]:
                self.observations[agent_name] = None
                self.rewards[agent_name] = None
                self.infos[agent_name] = None
                

        return self.observations, self.rewards, self.terminateds, self.truncateds, self.infos

  
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def create_agent_name_list_from_instance_list(self, _agent_instance_list):
        _agent_name_list = []
        for agent_instance in _agent_instance_list:
            _agent_name_list.append(agent_instance.agent_name)
        return _agent_name_list


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

    def _get_obs(self):
        _observations = dict()

        for agent_name in self.agents:
            agent_instance = self.agent_name_to_instance_dict[agent_name]
            obs_range = agent_instance.observation_range

            xp, yp = agent_instance.position[0], agent_instance.position[1]

            observation = np.zeros((self.nr_observation_channels, obs_range, obs_range), dtype=np.int32)
            # wall channel  filled with ones up front
            observation[0].fill(1)  

            xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(obs_range, xp, yp)

            observation[0:self.nr_observation_channels, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:self.nr_observation_channels, xlo:xhi, ylo:yhi])
            _observations[agent_name] = observation
        return _observations


    def obs_clip(self, observation_range, x, y):
        obs_offset = int((observation_range - 1) / 2) 
        xld = x - obs_offset
        xhd = x + obs_offset
        yld = y - obs_offset
        yhd = y + obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_grid_size - 1),
            np.clip(xhd, 0, self.x_grid_size - 1),
            np.clip(yld, 0, self.y_grid_size - 1),
            np.clip(yhd, 0, self.y_grid_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -obs_offset, 0)), abs(
            np.clip(yld, -obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def render(self):

        def draw_grid_model(self):
            x_len, y_len = (self.x_grid_size, self.y_grid_size)
            for x in range(x_len):
                for y in range(y_len):
                    # Draw white cell
                    cell_pos = pygame.Rect(
                        self.cell_scale * x,
                        self.cell_scale * y,
                        self.cell_scale,
                        self.cell_scale,
                    )
                    cell_color = (255, 255, 255)  # white background
                    pygame.draw.rect(self.screen, cell_color, cell_pos)

                    # Draw black border around cells
                    border_pos = pygame.Rect(
                        self.cell_scale * x,
                        self.cell_scale * y,
                        self.cell_scale,
                        self.cell_scale,
                    )
                    border_color = (192, 192, 192)  # light grey border around cells
                    pygame.draw.rect(self.screen, border_color, border_pos, 1)

            # Draw red border around total grid
            border_pos = pygame.Rect(
                0,
                0,
                self.cell_scale * x_len,
                self.cell_scale * y_len,
            )
            border_color = (255, 0, 0) # red
            pygame.draw.rect(self.screen, border_color, border_pos, 5) # type: ignore

        def draw_predator_observations(self):
            for predator_instance in self.predator_instance_list:
                position =  predator_instance.position 
                x = position[0]
                y = position[1]
                patch = pygame.Surface(
                    (self.cell_scale * predator_instance.observation_range, self.cell_scale * predator_instance.observation_range)
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

        def draw_prey_observations(self):
            for prey_instance in self.prey_instance_list:
                position =  prey_instance.position 
                x = position[0]
                y = position[1]
                patch = pygame.Surface(
                    (self.cell_scale * prey_instance.observation_range, self.cell_scale * prey_instance.observation_range)
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

        def draw_predator_instances(self):
            for predator_instance in self.predator_instance_list:
                position =  predator_instance.position 
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )

                col = (255, 0, 0) # red

                pygame.draw.circle(self.screen, col, center, int(self.cell_scale / 3)) # type: ignore

        def draw_prey_instances(self):
            for prey_instance in self.prey_instance_list:
                position =  prey_instance.position 
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )

                col = (0, 0, 255) # blue

                pygame.draw.circle(self.screen, col, center, int(self.cell_scale / 3)) # type: ignore

        def draw_grass_instances(self):
            for grass_instance in self.grass_instance_list:

                position =  grass_instance.position 
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )

                col = (0, 128, 0) # green

                pygame.draw.circle(self.screen, col, center, int(self.cell_scale / 3)) # type: ignore

        def draw_agent_instance_id_nrs(self):
            font = pygame.font.SysFont("Comic Sans MS", self.cell_scale * 2 // 3)

            predator_positions = defaultdict(int)
            prey_positions = defaultdict(int)
            grass_positions = defaultdict(int)

            for predator_instance in self.predator_instance_list:
                prey_position =  predator_instance.position 
                x = prey_position[0]
                y = prey_position[1]
                predator_positions[(x, y)] = predator_instance.agent_id_nr

            for prey_instance in self.prey_instance_list:
                prey_position =  prey_instance.position 
                x = prey_position[0]
                y = prey_position[1]
                prey_positions[(x, y)] = prey_instance.agent_id_nr

            for grass_instance in self.grass_instance_list:
                grass_position =  grass_instance.position 
                x = grass_position[0]
                y = grass_position[1]
                grass_positions[(x, y)] = grass_instance.agent_id_nr

            for x, y in predator_positions:
                (pos_x, pos_y) = (
                    self.cell_scale * x + self.cell_scale // 3.4,
                    self.cell_scale * y + self.cell_scale // 1.2,
                )

                predator_id_nr__text =str(predator_positions[(x, y)])

                predator_text = font.render(predator_id_nr__text, False, (255, 255, 0))

                self.screen.blit(predator_text, (pos_x, pos_y - self.cell_scale // 2))

            for x, y in prey_positions:
                (pos_x, pos_y) = (
                    self.cell_scale * x + self.cell_scale // 3.4,
                    self.cell_scale * y + self.cell_scale // 1.2,
                )

                prey_id_nr__text =str(prey_positions[(x, y)])

                prey_text = font.render(prey_id_nr__text, False, (255, 255, 0))

                self.screen.blit(prey_text, (pos_x, pos_y - self.cell_scale // 2))

            for x, y in grass_positions:
                (pos_x, pos_y) = (
                    self.cell_scale * x + self.cell_scale // 3.4,
                    self.cell_scale * y + self.cell_scale // 1.2,
                )

                grass_id_nr__text =str(grass_positions[(x, y)])

                grass_text = font.render(grass_id_nr__text, False, (255, 255, 0))

                self.screen.blit(grass_text, (pos_x, pos_y - self.cell_scale // 2))

        def draw_white_canvas_energy_chart(self):
            # relative position of energy chart within pygame window
            x_position_energy_chart = self.cell_scale*self.x_grid_size
            y_position_energy_chart = 0 #self.y_pygame_window
            pos = pygame.Rect(
                x_position_energy_chart,
                y_position_energy_chart,
                self.width_energy_chart,
                self.height_energy_chart,
            )
            color = (255, 255, 255) # white background                
            pygame.draw.rect(self.screen, color, pos) # type: ignore

        def draw_bar_chart_energy(self):
            chart_title = "Energy levels agents"
            # Draw chart title
            title_x = 1000
            title_y = 30
            title_color = (0, 0, 0)  # black
            font = pygame.font.Font(None, 30)
            title_text = font.render(chart_title, True, title_color)
            self.screen.blit(title_text, (title_x, title_y))
            # Draw predator bars
            data_predators = []
            for predator_name in self.predator_name_list:
                predator_instance = self.agent_name_to_instance_dict[predator_name]
                predator_energy = predator_instance.energy
                data_predators.append(predator_energy)
                #print(predator_name," has energy", round(predator_energy,1))
            x_screenposition = 400   
            y_screenposition = 50
            bar_width = 20
            offset_bars = 20
            height = 500
            max_energy_value_chart = 30
            for i, value in enumerate(data_predators):
                bar_height = (value / max_energy_value_chart) * height
                bar_x = x_screenposition + (self.width_energy_chart - (bar_width * len(data_predators))) // 2 + i * (bar_width+offset_bars)
                bar_y = y_screenposition + height - bar_height

                color = (255, 0, 0)  # blue

                pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, bar_height))

            # Draw y-axis
            y_axis_x = x_screenposition + (self.width_energy_chart - (bar_width * len(data_predators))) // 2 - 10
            y_axis_y = y_screenposition
            y_axis_height = height + 10
            y_axis_color = (0, 0, 0)  # black
            pygame.draw.rect(self.screen, y_axis_color, (y_axis_x, y_axis_y, 5, y_axis_height))

            # Draw x-axis
            x_axis_x = x_screenposition + 15 + (self.width_energy_chart - (bar_width * len(data_predators))) // 2 - 10
            x_axis_y = y_screenposition + height
            x_axis_width = self.width_energy_chart - 120
            x_axis_color = (0, 0, 0)  # black
            pygame.draw.rect(self.screen, x_axis_color, (x_axis_x, x_axis_y, x_axis_width, 5))

            # Draw tick labels predators on x-axis
            for i, predator_name in enumerate(self.predator_name_list):
                predator_instance = self.agent_name_to_instance_dict[predator_name]
                label = str(predator_instance.agent_id_nr)
                label_x = x_axis_x + i * (bar_width + offset_bars)
                label_y = x_axis_y + 10
                label_color = (255, 0, 0)  # red
                font = pygame.font.Font(None, 30)
                text = font.render(label, True, label_color)
                self.screen.blit(text, (label_x, label_y))
            # Draw tick labels prey on x-axis
            for i, prey_name in enumerate(self.prey_name_list):
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                label = str(prey_instance.agent_id_nr)
                label_x = 330 + x_axis_x + i * (bar_width + offset_bars)
                label_y = x_axis_y + 10
                label_color = (0, 0, 255)  # blue
                font = pygame.font.Font(None, 30)
                text = font.render(label, True, label_color)
                self.screen.blit(text, (label_x, label_y))



            # Draw tick points on y-axis
            num_ticks = max_energy_value_chart + 1 
            tick_spacing = height // (num_ticks - 1)
            for i in range(num_ticks):
                tick_x = y_axis_x - 5
                tick_y = y_screenposition + height - i * tick_spacing
                tick_width = 10
                tick_height = 2
                tick_color = (0, 0, 0)  # black
                pygame.draw.rect(self.screen, tick_color, (tick_x, tick_y, tick_width, tick_height))

                # Draw tick labels every 5 ticks
                if i % 5 == 0:
                    label = str(i)
                    label_x = tick_x - 30
                    label_y = tick_y - 5
                    label_color = (0, 0, 0)  # black
                    font = pygame.font.Font(None, 30)
                    text = font.render(label, True, label_color)
                    self.screen.blit(text, (label_x, label_y))

            # Draw prey bars
            data_prey = []
            for prey_name in self.prey_name_list:
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                prey_energy = prey_instance.energy
                data_prey.append(prey_energy)
                #print(prey_name," has energy", round(prey_energy,1))
            x_screenposition = 750   
            y_screenposition = 50
            bar_width = 20
            offset_bars = 20
            height = 500
            for i, value in enumerate(data_prey):
                bar_height = (value / max_energy_value_chart) * height
                bar_x = x_screenposition + (self.width_energy_chart - (bar_width * len(data_prey))) // 2 + i * (bar_width+offset_bars)
                bar_y = y_screenposition + height - bar_height

                color = (0, 0, 255)  # blue

                pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, bar_height))


        
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.cell_scale * self.x_grid_size +self.width_energy_chart, 
                     self.cell_scale * self.y_grid_size)
                )
                pygame.display.set_caption("PredPreyGrass")
            else:
                self.screen = pygame.Surface(
                    (self.cell_scale * self.x_grid_size, self.cell_scale * self.y_grid_size)
                )

        draw_grid_model(self)

        draw_prey_observations(self)
        draw_predator_observations(self)

        draw_grass_instances(self)
        draw_prey_instances(self)
        draw_predator_instances(self)

        draw_agent_instance_id_nrs(self)

        draw_white_canvas_energy_chart(self)

        draw_bar_chart_energy(self)


        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            if self.save_image_steps:
                self.file_name+=1
                print(str(self.file_name)+".png saved")
                directory= "./assets/images/"
                pygame.image.save(self.screen, directory+str(self.file_name)+".png")
        
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )


