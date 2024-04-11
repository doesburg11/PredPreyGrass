"""
pred, prey, grass PettingZoo environment. Instead of gaining the rewards of the energy of
the prey or grass that is caught, this version equates fixed pre-defined rewards,
"""
import os
import numpy as np
import random
from typing import List
import pygame
from collections import defaultdict

import gymnasium
from gymnasium.utils import seeding, EzPickle
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from agents.discrete_agent import DiscreteAgent


class PredPrey:
    def __init__(
        self,
        x_grid_size: int = 16,
        y_grid_size: int = 16,
        max_cycles: int = 500,
        n_possible_predator: int = 6,
        n_possible_prey: int = 8,
        n_possible_grass: int = 30,
        n_initial_active_predator: int = 10,
        n_initial_active_prey: int = 10,
        max_observation_range: int = 7,
        obs_range_predator: int = 5,
        obs_range_prey: int = 7,
        render_mode = None,
        energy_gain_per_step_predator = -0.1,
        energy_gain_per_step_prey = -0.1,  
        energy_gain_per_step_grass = 1.0,   
        initial_energy_predator = 10.0,
        initial_energy_prey = 10.0,
        initial_energy_grass = 5.0,
        cell_scale: int = 40,
        x_pygame_window : int = 0,
        y_pygame_window : int = 0,
        catch_grass_reward=5.0,
        catch_prey_reward=5.0,
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
        self.catch_grass_reward = catch_grass_reward
        self.catch_prey_reward = catch_prey_reward

        #self.n_initial_active_predator = 6
        #self.n_initial_active_prey = 8

        # pygam position window
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x_pygame_window, 
                                                        self.y_pygame_window)

        self._seed()
        self.agent_id_counter = 0
        self.action_range = 3

        # agent types
        self.agent_type_name_list = ["wall", "predator", "prey", "grass"]  # different types of agents 
        self.predator_type_nr = self.agent_type_name_list.index("predator") #0
        self.prey_type_nr = self.agent_type_name_list.index("prey")  #1
        self.grass_type_nr = self.agent_type_name_list.index("grass")  #2
        
        # lists of agents
        # initialization
        self.n_active_predator = self.n_possible_predator
        self.n_active_prey = self.n_possible_prey  
        self.n_active_grass = self.n_possible_grass

        self.predator_instance_list = [] # list of all living predator
        self.prey_instance_list = [] # list of all living prey
        self.grass_instance_list = [] # list of all living grass
        self.agent_instance_list = [] # list of all living agents
        self.predator_name_list = []
        self.prey_name_list = []
        self.grass_name_list = []
 
        self.agent_instance_in_grid_location = np.empty((len(self.agent_type_name_list), x_grid_size, y_grid_size), dtype=object)
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            self.agent_instance_in_grid_location[agent_type_nr] = np.full((self.x_grid_size, self.y_grid_size), None)


        self.agent_name_to_instance_dict = dict()
        self.n_agents = self.n_possible_predator + self.n_possible_prey

        

        # creation agent type lists
        self.predator_name_list =  ["predator" + "_" + str(a) for a in range(self.n_possible_predator)]
        self.prey_name_list =  ["prey" + "_" + str(a) for a in range(self.n_possible_predator, self.n_possible_prey+self.n_possible_predator)]
        self.agent_name_list = self.predator_name_list + self.prey_name_list


        # observations
        max_agents_overlap = max(self.n_possible_predator, self.n_possible_prey, self.n_possible_grass)
        self.max_obs_offset = int((self.max_observation_range - 1) / 2) 
        self.nr_observation_channels = len(self.agent_type_name_list)
        obs_space = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.max_observation_range, self.max_observation_range, self.nr_observation_channels),
            dtype=np.float32,
        )
        self.observation_space = [obs_space for _ in range(self.n_agents)]  # type: ignore
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
        action_space_agent = spaces.Discrete(self.n_actions_agent)  
        self.action_space = [action_space_agent for _ in range(self.n_agents)] # type: ignore
        # end actions


        # removal agents
        self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.grass_to_be_removed_by_prey_dict = dict(zip(self.grass_name_list, [False for _ in self.grass_name_list]))
        self.predator_who_remove_prey_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list])) 
        self.prey_to_be_removed_by_predator_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.predator_to_be_removed_by_starvation_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list]))
        self.prey_to_be_removed_by_starvation_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        # end removal agents


        # visualization
        self.screen = None
        self.save_image_steps = False
        self.width_energy_chart = 1800
        self.height_energy_chart = self.cell_scale * self.y_grid_size
        # end visualization

        self.file_name = 0
        self.n_aec_cycles = 0


    def reset(self):
        # empty agent lists
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
        self.n_active_predator: int = self.n_possible_predator
        self.n_active_prey: int = self.n_possible_prey  
        self.n_active_grass: int = self.n_possible_grass


        self.n_agent_type_list = [0, self.n_possible_predator, self.n_possible_prey, self.n_possible_grass]
        self.obs_range_list = [0, self.obs_range_predator, self.obs_range_prey, 0]
        self.initial_energy_list = [0, self.initial_energy_predator, self.initial_energy_prey, self.initial_energy_grass]
        self.energy_gain_per_step_list = [0, self.energy_gain_per_step_predator, self.energy_gain_per_step_prey, self.energy_gain_per_step_grass]

        self.agent_id_counter = 0
        self.agent_name_to_instance_dict = {}        
        self.model_state = np.zeros((self.nr_observation_channels, self.x_grid_size, self.y_grid_size), dtype=np.float32)
        

        # create agents of all types excluding "wall"-agents
        for agent_type_nr in range(1, len(self.agent_type_name_list)):
            agent_type_name = self.agent_type_name_list[agent_type_nr]
            #empty cell list: an array of tuples with the coordinates of empty cells, at initialization all cells are empty
            empty_cell_list = [(i, j) for i in range(self.x_grid_size) for j in range(self.y_grid_size)] 
            # intialize all possible agents of a certain type
            for _ in range(self.n_agent_type_list[agent_type_nr]): 
                agent_id_nr = self.agent_id_counter           
                agent_name = agent_type_name + "_" + str(agent_id_nr)
                self.agent_id_counter+=1
                agent_instance = DiscreteAgent(
                    agent_type_nr, 
                    agent_id_nr,
                    agent_name,
                    self.model_state[agent_type_nr], # needed to detect if a cell is allready occupied by an agent of the same type 
                    observation_range= self.obs_range_list[agent_type_nr],
                    motion_range=self.motion_range,
                    initial_energy=self.initial_energy_list[agent_type_nr],
                    catch_grass_reward=self.catch_grass_reward,
                    catch_prey_reward=self.catch_prey_reward,
                    energy_gain_per_step=self.energy_gain_per_step_list[agent_type_nr]
                )
                
                #  updates lists en records
                xinit, yinit = random.choice(empty_cell_list)
                empty_cell_list.remove((xinit,yinit)) # occupied cell removed from empty_cell_list
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                agent_instance.position = (xinit, yinit)  
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

        # deactivate initial agents by setting energy to zero
        # the ultimate deactivation is handled in the step function
        for predator_instance in self.predator_instance_list:
            if predator_instance.agent_id_nr >= self.n_initial_active_predator: # number of initial active predators
               predator_instance.energy = 0.0
        for prey_instance in self.prey_instance_list:
            if prey_instance.agent_id_nr >= self.n_possible_predator+self.n_initial_active_prey: # number of initial active predators
                prey_instance.energy = 0.0

        # removal agents set to false
        self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.grass_to_be_removed_by_prey_dict = dict(zip(self.grass_name_list, [False for _ in self.grass_name_list]))

        # define the learning agents
        self.agent_instance_list = self.predator_instance_list + self.prey_instance_list        
        self.agent_name_list = self.predator_name_list + self.prey_name_list

        self.agent_reward_dict = dict(zip(self.agent_name_list, [0.0 for _ in self.agent_name_list]))
    
        self.n_aec_cycles = 0
        
    def step(self, action, agent_instance, is_last):
        # Extract agent details
        if agent_instance.is_alive:
            agent_type_nr = agent_instance.agent_type_nr
            agent_name = agent_instance.agent_name
            agent_energy = agent_instance.energy

            # If the agent is a predator and it's alive
            if agent_type_nr == self.predator_type_nr: 
                if agent_energy > 0: # If predator has energy
                    # Move the predator and update the model state
                    self.agent_instance_in_grid_location[self.predator_type_nr,agent_instance.position[0],agent_instance.position[1]] = None
                    self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] -= 1
                    agent_instance.step(action)
                    self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] += 1
                    self.agent_instance_in_grid_location[self.predator_type_nr, agent_instance.position[0], agent_instance.position[1]] = agent_instance
                    # If there's prey at the new position, remove (eat) one at random
                    x_new_position_predator, y_new_position_predator = agent_instance.position
                    if self.model_state[self.prey_type_nr, x_new_position_predator, y_new_position_predator] > 0:
                        prey_instance_removed = self.agent_instance_in_grid_location[self.prey_type_nr][(x_new_position_predator, y_new_position_predator)]
                        self.predator_who_remove_prey_dict[agent_name] = True
                        self.prey_to_be_removed_by_predator_dict[prey_instance_removed.agent_name] = True
                else:  # If predator has no energy, it starves to death
                    self.predator_to_be_removed_by_starvation_dict[agent_name] = True

            # If the agent is a prey and it's alive
            elif agent_type_nr == self.prey_type_nr:
                if agent_energy > 0:  # If prey has energy
                    # Move the prey and update the model state
                    self.agent_instance_in_grid_location[self.prey_type_nr,agent_instance.position[0],agent_instance.position[1]] = None
                    #self.remove_agent_instance_from_position_dict(agent_instance)
                    self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] -= 1
                    agent_instance.step(action)
                    self.model_state[agent_type_nr,agent_instance.position[0],agent_instance.position[1]] += 1
                    self.agent_instance_in_grid_location[self.prey_type_nr, agent_instance.position[0], agent_instance.position[1]] = agent_instance
                    #self.add_agent_instance_to_position_dict(agent_instance)
                    x_new_position_prey, y_new_position_prey = agent_instance.position

                    # If there's grass at the new position, remove (eat) one at random
                    # second conditons is to prevent searching an empty list when another prey in the same cycle
                    # already selected the only grass instance for removal earlier in the same cycle
                    if self.model_state[self.grass_type_nr, x_new_position_prey, y_new_position_prey] > 0: 
                        grass_instance_removed = self.agent_instance_in_grid_location[self.grass_type_nr][(x_new_position_prey, y_new_position_prey)]                        
                        grass_name_removed = grass_instance_removed.agent_name
                        self.prey_who_remove_grass_dict[agent_name] = True
                        self.grass_to_be_removed_by_prey_dict[grass_name_removed] = True
                        # Immediately remove the grass instance from the position dict
                        # to prevent eating the same grass instance twice by different prey in the same cycle
                        # This way, the same grass agent cannot be selected for removal by another prey agent in the same cycle.
                        #self.remove_agent_instance_from_position_dict(grass_instance_removed)
                        #self.agent_instance_in_grid_location[self.grass_type_nr,x_new_position_prey,y_new_position_prey] = None
                else: # prey starves to death
                    self.prey_to_be_removed_by_starvation_dict[agent_name] = True

        # reset rewards to zero in every step of an agent
        self.agent_reward_dict = dict(zip(self.agent_name_list, 
                                        [0.0 for _ in self.agent_name_list]))

        if is_last: # removes agents and reap rewards at the end of the cycle
            #print("last step of cycle: ", self.n_aec_cycles)

            for predator_name in self.predator_name_list:
                predator_instance = self.agent_name_to_instance_dict[predator_name]
                if predator_instance.is_alive:
                    # remove predator which starves to death
                    if self.predator_to_be_removed_by_starvation_dict[predator_name]:
                        self.predator_instance_list.remove(predator_instance)
                        self.n_active_predator -= 1
                        self.agent_instance_in_grid_location[self.predator_type_nr,predator_instance.position[0],predator_instance.position[1]] = None
                        self.model_state[self.predator_type_nr,predator_instance.position[0],predator_instance.position[1]] -= 1
                        predator_instance.is_alive = False
                        predator_instance.energy = 0.0
                        # not yet implemented
                        #self.remove_agent_instance_from_position_dict(predator_instance)
                    else: # reap rewards for predator which removes prey
                        catch_reward_predator = predator_instance.catch_prey_reward * self.predator_who_remove_prey_dict[predator_name] 
                        step_reward_predator = predator_instance.energy_gain_per_step
                        self.agent_reward_dict[predator_name] += step_reward_predator
                        self.agent_reward_dict[predator_name] += catch_reward_predator
                        # TODO: energy and rewards integrated in future?
                        predator_instance.energy += step_reward_predator 
                        predator_instance.energy += catch_reward_predator
                        #print(predator_name," has energy: ",round(predator_instance.energy,1))
            for prey_name in self.prey_name_list:
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                if prey_instance.is_alive:
                    # remove prey which gets eaten by a predator or starves to death
                    if self.prey_to_be_removed_by_predator_dict[prey_name] or self.prey_to_be_removed_by_starvation_dict[prey_name]:
                        # to be implemented: put in exit strategy function for all agents
                        self.prey_instance_list.remove(prey_instance)
                        self.n_active_prey -= 1
                        self.agent_instance_in_grid_location[self.prey_type_nr,prey_instance.position[0],prey_instance.position[1]] = None
                        self.model_state[self.prey_type_nr,prey_instance.position[0],prey_instance.position[1]] -= 1
                        prey_instance.is_alive = False
                        prey_instance.energy = 0.0
                        #self.remove_agent_instance_from_position_dict(prey_instance)
                    else: # reap rewards for prey which removes grass
                        #self.agent_reward_dict[prey_name] += -number_of_predators_in_observation
                        catch_reward_prey = prey_instance.catch_grass_reward * self.prey_who_remove_grass_dict[prey_name] 
                        step_reward_prey = prey_instance.energy_gain_per_step
                        # this is not cumulative reward, agent_reward_dict is set to zero before 'last' step
                        self.agent_reward_dict[prey_name] += step_reward_prey
                        self.agent_reward_dict[prey_name] += catch_reward_prey
                        prey_instance.energy += step_reward_prey
                        prey_instance.energy += catch_reward_prey
            for grass_name in self.grass_name_list:
                grass_instance = self.agent_name_to_instance_dict[grass_name]
                # remove grass which gets eaten by a prey
                grass_instance.energy += grass_instance.energy_gain_per_step
                if self.grass_to_be_removed_by_prey_dict[grass_name]:
                    #removes grass_name from 'grass_name_list'
                    self.grass_instance_list.remove(grass_instance)
                    self.n_active_grass -= 1
                    #self.agent_instance_in_grid_location[self.grass_type_nr,grass_instance.position[0],grass_instance.position[1]] = None
                    self.model_state[self.grass_type_nr,grass_instance.position[0],grass_instance.position[1]] -= 1
                    grass_instance.energy = 0.0
                    grass_instance.is_alive = False
                    #TODO next line crashes but is needed to remove grass from position dict
                    #self.remove_agent_instance_from_position_dict(grass_instance)
 
                # revive dead grass if energy regrows to self.initial_energy_grass, which effectively means that grass regrowths after 5 AEC cycles
                if not grass_instance.is_alive and grass_instance.energy > self.initial_energy_grass:
                    self.n_active_grass += 1
                    self.grass_instance_list.append(grass_instance)
                    self.model_state[self.grass_type_nr,grass_instance.position[0],grass_instance.position[1]] += 1
                    grass_instance.is_alive = True
                             

            self.n_aec_cycles = self.n_aec_cycles + 1
            
            
            #reinit agents records to default at the end of the cycle
            self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
            self.grass_to_be_removed_by_prey_dict = dict(zip(self.grass_name_list, [False for _ in self.grass_name_list]))
            self.predator_who_remove_prey_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list])) 
            self.prey_to_be_removed_by_predator_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
            self.predator_to_be_removed_by_starvation_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list]))
            # end reinit agents records to default at the end of the cycle

        if self.render_mode == "human" and agent_instance.is_alive:
            self.render()

    def observation_space(self, agent):
        return self.observation_spaces[agent] # type: ignore

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

    def observe(self, agent_name):

        agent_instance = self.agent_name_to_instance_dict[agent_name]
        
        xp, yp = agent_instance.position[0], agent_instance.position[1]

        observation = np.zeros((self.nr_observation_channels, self.max_observation_range, self.max_observation_range), dtype=np.float32)
        # wall channel  filled with ones up front
        observation[0].fill(1.0)  

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        observation[0:self.nr_observation_channels, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:self.nr_observation_channels, xlo:xhi, ylo:yhi])
        
        observation_range_agent = agent_instance.observation_range
        max = self.max_observation_range
        #mask is number of 'outer squares' of an observation surface set to zero
        mask = int((max - observation_range_agent)/2)
        if mask > 0: # observation_range agent is smaller than default max_observation_range
            for j in range(mask):
                for i in range(self.nr_observation_channels):
                    observation[i][j,0:max] = 0
                    observation[i][max-1-j,0:max] = 0
                    observation[i][0:max,j] = 0
                    observation[i][0:max,max-1-j] = 0
            return observation
        elif mask == 0:
            return observation
        else:
            raise Exception(
                "Error: observation_range_agent larger than max_observation_range"
                )

    def obs_clip(self, x, y):
        xld = x - self.max_obs_offset
        xhd = x + self.max_obs_offset
        yld = y - self.max_obs_offset
        yhd = y + self.max_obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_grid_size - 1),
            np.clip(xhd, 0, self.x_grid_size - 1),
            np.clip(yld, 0, self.y_grid_size - 1),
            np.clip(yhd, 0, self.y_grid_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.max_obs_offset, 0)), abs(
            np.clip(yld, -self.max_obs_offset, 0)
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
                mask = int((self.max_observation_range - predator_instance.observation_range)/2)
                if mask == 0:
                    patch = pygame.Surface(
                        (self.cell_scale * self.max_observation_range, self.cell_scale * self.max_observation_range)
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
                mask = int((self.max_observation_range - prey_instance.observation_range)/2)
                if mask == 0:
                    patch = pygame.Surface(
                        (self.cell_scale * self.max_observation_range, self.cell_scale * self.max_observation_range)
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
                #print(grass_instance.agent_name," at position ", position)
                x = position[0]
                y = position[1]

                center = (
                    int(self.cell_scale * x + self.cell_scale / 2),
                    int(self.cell_scale * y + self.cell_scale / 2),
                )

                col = (0, 128, 0) # green

                #col = (0, 0, 255) # blue

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
            # Constants
            BLACK = (0, 0, 0)
            RED = (255, 0, 0)
            BLUE = (0, 0, 255)

            # Create data array predators and prey
            data_predators = [self.agent_name_to_instance_dict[name].energy for name in self.predator_name_list]
            data_prey = [self.agent_name_to_instance_dict[name].energy for name in self.prey_name_list]

            #postion and size parameters energy chart
            width_energy_chart = self.width_energy_chart #= 1800

            max_energy_value_chart = 30
            bar_width = 20
            offset_bars = 20
            x_screenposition = 0   #x_axis screen position?
            y_screenposition = 50
            y_axis_height = 500
            x_axis_width = width_energy_chart - 120 # = 1680
            x_screenposition_prey_bars = 1450   
            title_x = 1400
            title_y = 30

            # Draw y-axis
            y_axis_x = x_screenposition + (width_energy_chart - (bar_width * len(data_predators))) // 2 - 10
            y_axis_y = y_screenposition # 50
            # x-axis
            x_axis_x = x_screenposition + (width_energy_chart - (bar_width * len(data_predators))) // 2
            x_axis_y = y_screenposition + y_axis_height # 50 + 500 = 550
            x_start_prey_bars = x_screenposition_prey_bars + x_screenposition 
            x_start_predator_bars = x_axis_x
            predator_legend_x = x_start_predator_bars
            predator_legend_y = 600
            prey_legend_x = x_screenposition_prey_bars
            prey_legend_y = 600
            title_font_size = 30
            predator_legend_font_size = 30
            prey_legend_font_size = 30





            # Draw chart title
            chart_title = "Energy levels agents"
            title_color = BLACK  # black
            title_font= pygame.font.Font(None, title_font_size)
            title_text = title_font.render(chart_title, True, title_color)
            self.screen.blit(title_text, (title_x, title_y))
            # Draw legend title for predators
            predator_legend_title= "Predators"
            predator_legend_color = RED  
            predator_legend_font = pygame.font.Font(None, predator_legend_font_size)
            predator_legend_text = predator_legend_font.render(predator_legend_title, True, predator_legend_color)
            self.screen.blit(predator_legend_text, (predator_legend_x, predator_legend_y))
            # Draw legend title for prey
            prey_legend_title= "Prey"
            prey_legend_color = BLUE
            prey_legend_font = pygame.font.Font(None, prey_legend_font_size)
            prey_legend_text = prey_legend_font.render(prey_legend_title, True, prey_legend_color)
            self.screen.blit(prey_legend_text, (prey_legend_x, prey_legend_y))

            # Draw y-axis
            y_axis_color = BLACK
            pygame.draw.rect(self.screen, y_axis_color, (y_axis_x, y_axis_y, 5, y_axis_height))
            # Draw x-axis
            x_axis_color = BLACK
            pygame.draw.rect(self.screen, x_axis_color, (x_axis_x, x_axis_y, x_axis_width, 5))

            # Draw predator bars
            for i, value in enumerate(data_predators):
                bar_height = (value / max_energy_value_chart) * y_axis_height
                bar_x = x_screenposition + (width_energy_chart - (bar_width * len(data_predators))) // 2 + i * (bar_width+offset_bars)
                bar_x = x_start_predator_bars + i * (bar_width+offset_bars)
                bar_y = y_screenposition + y_axis_height - bar_height

                color = (255, 0, 0)  # red

                pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, bar_height))

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

            # Draw prey bars
            for i, value in enumerate(data_prey):
                bar_height = (value / max_energy_value_chart) * y_axis_height
                bar_x = x_start_prey_bars + i * (bar_width+offset_bars)
                bar_y = y_screenposition + y_axis_height - bar_height

                color = (0, 0, 255)  # blue

                pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, bar_height))

            # Draw tick labels prey on x-axis
            for i, prey_name in enumerate(self.prey_name_list):
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                label = str(prey_instance.agent_id_nr)
                label_x = x_start_prey_bars+ i * (bar_width+offset_bars)
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
                pygame.display.set_caption("PredPreyGrass - create agents")
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


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predpreygrass_create_agents",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self.pred_prey_env = PredPrey(*args, **kwargs) #  this calls the code from PredPrey

        self.agents = self.pred_prey_env.agent_name_list 

        self.possible_agents = self.agents[:]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.pred_prey_env._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents
             
        self.possible_agents = self.agents[:]
        self.agent_name_to_index_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)

        # spaces
        # self = raw_env
        self.action_spaces = dict(zip(self.agents, self.pred_prey_env.action_space)) # type: ignore
        self.observation_spaces = dict(zip(self.agents, self.pred_prey_env.observation_space)) # type: ignore
        self.steps = 0
        # this method "reset"
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.pred_prey_env.reset()  # this calls reset from PredPrey

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
        self.pred_prey_env.step(
            action, agent_instance, self._agent_selector.is_last()
        )

        for k in self.terminations:
            if self.pred_prey_env.n_aec_cycles >= self.pred_prey_env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = \
                    self.pred_prey_env.is_no_grass or \
                    self.pred_prey_env.is_no_prey or \
                    self.pred_prey_env.is_no_predator
                
        for agent_name in self.agents:
        #for agent_name in self.pred_prey_env.agent_name_list:
        #for agent_name in self.possible_agents:
            self.rewards[agent_name] = self.pred_prey_env.agent_reward_dict[agent_name]
        self.steps += 1
        self._cumulative_rewards[self.agent_selection] = 0  # cannot be left out for proper rewards
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()  # cannot be left out for proper rewards
        if self.render_mode == "human":
            self.render()

    def observe(self, agent_name):
        agent_instance = self.pred_prey_env.agent_name_to_instance_dict[agent_name]
        obs = self.pred_prey_env.observe(agent_name)
        observation = np.swapaxes(obs, 2, 0) # type: ignore
        # return observation of only zeros if agent is not alive
        if not agent_instance.is_alive:
            observation = np.zeros(observation.shape)
        return observation
            
    def observation_space(self, agent: str):  # must remain
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
    
