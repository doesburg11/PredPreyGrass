"""
[v38]
This is a significant modified version of the original predprey.py file from branch v37.
It attempts to implement the following changes:
- abstraction in the name giving of the agents 
- a divide of possible_agents into active and inactive agents
- at reset also inactive agents are created, to make it in the future possible to create agents
at runtime
- at capturing grass or prey, the energy of the captured agent is added to the capturing agent
- the capturing agent is rewarded for capturing grass or prey with the amount of energy of the 
captured agent
- the model state is changed from the amount of agents to the amount of energy of the agents
- the observation space is changed from the amount of agents to the amount of energy of the agents
- observations are changed from the amount of agents to the amount of energy of the agents

Conclusions: 
-this envrionment works and terminates properly with a random policy
-this envrionment however does NOT work with a PPO policy (yet). 
-it raises with PPO: "ValueError: when an agent is dead, the only valid action is None"
"""

# noqa: D212, D415

from collections import defaultdict
import numpy as np
import pygame
import random
import time

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import os
# position of the pygame window on the screen
x_pygame_window = 0
y_pygame_window = 0


class DiscreteAgent():
    def __init__(
        self,
        x_grid_size,
        y_grid_size,
        agent_type_nr, # 0: wall, 1: prey, 2: grass, 3: predator
        agent_id_nr,
        observation_range=7,
        n_channels=4, # n channels is the number of observation channels
        flatten=False,  
        motion_range = [],
        initial_energy=10,
        energy_loss_per_step=-0.1

    ):
        #identification agent
        self.agent_type_nr = agent_type_nr   # also channel number of agent 
        self.agent_id_nr = agent_id_nr       # unique integer per agent
        self.alive = False

        #(fysical) boundaries/limitations agent in observing (take in) and acting (take out)
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.observation_range = observation_range
        self.observation_shape = (n_channels * observation_range**2 + 1,) if flatten else \
            (observation_range, observation_range, n_channels)
        self.motion_range = motion_range
        self.n_actions_agent=len(self.motion_range)   
        self.action_space_agent = spaces.Discrete(self.n_actions_agent) 
        self.position = np.zeros(2, dtype=np.int32)  # x and y position
        self.energy = initial_energy  # still to implement
        self.energy_loss_per_step = energy_loss_per_step

    def step(self, action):
        # returns new position of agent "self" given action "action"
        
        next_position = np.zeros(2, dtype=np.int32) 
        next_position[0], next_position[1] = self.position[0], self.position[1]

        next_position += self.motion_range[action]
        # masking
        if not (0 <= next_position[0] < self.x_grid_size and 
                            0 <= next_position[1] < self.y_grid_size):
            return self.position   # if moved out of borders: dont move
        else:
            self.position = next_position
            return self.position

class PredPrey:
    def __init__(
        self,
        x_grid_size: int = 10,
        y_grid_size: int = 10,
        max_cycles: int = 500,
        agent_type_name_list =              ["wall", "predator", "prey", "grass"],  # different types of agents
        n_possible_agent_list =             [0, 10, 20, 40],  # 0: wall, 1: predator, 2: prey, 3: grass
        n_initial_agent_list =              [0, 6, 16, 40],  
        max_observation_range: int = 7,
        obs_range_agent_list =              [0, 5, 7, 0],
        energy_loss_per_step_agent_list =   [0, -0.1, -0.1, 0],
        initial_energy_list =               [0, 1.0, 1.0, 1.0], 
        catch_reward = [[0, 0, 0, 0], # 0: wall
                        [0, 0, 5, 0], # 1: predator
                        [0, 0, 0, 2], # 2: prey
                        [0, 0, 0, 0]], # 3: grass
        render_mode = None,
        cell_scale: int = 40,
        x_pygame_window : int = 0,
        y_pygame_window : int = 0,
        ):
        #parameter init
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.max_cycles = max_cycles
        self.n_possible_agent_list = n_possible_agent_list
        self.n_initial_agent_list = n_initial_agent_list
        self.initial_energy_list = initial_energy_list
        self.max_observation_range = max_observation_range
        self.obs_range_agent_list = obs_range_agent_list
        self.energy_loss_per_step_agent_list = energy_loss_per_step_agent_list

        # running numbers
        self.n_active_agent_list = self.n_initial_agent_list

        # agent types
        self.agent_type_name_list = agent_type_name_list
        self.predator_type_nr = self.agent_type_name_list.index("predator") #1
        self.prey_type_nr = self.agent_type_name_list.index("prey")  #2
        self.grass_type_nr = self.agent_type_name_list.index("grass")  #3


        # visualization parameters
        self.render_mode = render_mode
        self.cell_scale = cell_scale
        self.x_pygame_window = x_pygame_window
        self.y_pygame_window = y_pygame_window
        # end vizualization parameters

        self.nr_observation_channels = len(self.agent_type_name_list)

        # creation agent type lists
        self.predator_name_list =  ["predator" + "_" + str(a) for a in range(self.n_possible_agent_list[1] )]
        self.prey_name_list =  ["prey" + "_" + str(a) for a in range(self.n_possible_agent_list[1] , self.n_possible_agent_list[1] +self.n_possible_agent_list[2] )]
        self.agents = self.predator_name_list + self.prey_name_list

        self.n_agents = self.n_possible_agent_list[1] + self.n_possible_agent_list[2]  # predators and prey
        self.Predator_0_instance = None


        # actions
        self.action_range = 3
        action_offset = int((self.action_range - 1) / 2) 
        action_range_iterator = list(range(-action_offset, action_offset+1))
        self.motion_range = []
        action_nr = 0
        for d_x in action_range_iterator:
            for d_y in action_range_iterator:
                if abs(d_x) + abs(d_y) <= action_offset:
                    self.motion_range.append([d_x,d_y])        
                    action_nr += 1
     
        self.n_actions_agent=len(self.motion_range)
        action_space_agent = spaces.Discrete(self.n_actions_agent)  
        self.action_space = [action_space_agent for _ in range(self.n_agents)] # type: ignore
                  
        # end actions

        self.n_actions_agent=len(self.motion_range)
        action_space_agent = spaces.Discrete(self.n_actions_agent)  
        self.action_space = [action_space_agent for _ in range(self.n_agents)] # type: ignore




        # observations
        max_energy_in_channel_cell = 50
        self.max_obs_offset = int((self.max_observation_range - 1) / 2) 
        self.nr_observation_channels = len(self.agent_type_name_list)
        obs_space = spaces.Box(
            low=0,
            high=max_energy_in_channel_cell,
            shape=(self.max_observation_range, self.max_observation_range, self.nr_observation_channels),
            dtype=np.float32,
        )
        self.observation_space = [obs_space for _ in range(self.n_agents)]  # type: ignore
        # end observations

 

        # visualization
        self.screen = None
        self.save_image_steps = False
        self.energy_chart = False
        if self.energy_chart:
             self.width_energy_chart = 1000
        else:
            self.width_energy_chart = 0
        self.height_energy_chart = self.cell_scale * self.y_grid_size
        # end visualization
        self.file_name = 0
        self.n_aec_cycles = 0



    def reset(self):
        # initialization
        self.n_active_agent_list[1] = self.n_initial_agent_list[1]  # predator
        self.n_active_agent_list[2] = self.n_initial_agent_list[2]  # prey
        self.n_active_agent_list[3] = self.n_initial_agent_list[3]  # grass

        self.agent_id_counter = 0
        self.agent_name_to_instance_dict = {}        
        self.model_state = np.zeros((self.nr_observation_channels, self.x_grid_size, self.y_grid_size), dtype=np.float32)

        # Initialize as a list of empty lists
        self.active_agent_instance_list = [[] for _ in range(len(self.agent_type_name_list))]
        self.inactive_agent_instance_list = [[] for _ in range(len(self.agent_type_name_list))]
        self.active_agent_name_list = [[] for _ in range(len(self.agent_type_name_list))]
        self.inactive_agent_name_list = [[] for _ in range(len(self.agent_type_name_list))]

        self.grass_instance_in_grid_location_dict = {}
        self.prey_instance_in_grid_location_dict = {}
        self.agent_instance_in_grid_location_dict = {
            'grass': self.grass_instance_in_grid_location_dict,
            'prey': self.prey_instance_in_grid_location_dict
        }
        for agent_type_name in self.agent_instance_in_grid_location_dict: 
            for i in range(self.x_grid_size):
                for j in range(self.y_grid_size):
                    self.agent_instance_in_grid_location_dict[agent_type_name][(i,j)] = []
        # end of initialization empty lists



        #creation of agents
        for agent_type_nr in range(1, len(self.agent_type_name_list)):

            for _ in range(self.n_active_agent_list[agent_type_nr]):
                agent_instance = DiscreteAgent(
                    self.x_grid_size, 
                    self.y_grid_size, 
                    agent_type_nr,  # predator type number =1
                    self.agent_id_counter,
                    observation_range=self.obs_range_agent_list[agent_type_nr],
                    motion_range=self.motion_range,
                    initial_energy=self.initial_energy_list[agent_type_nr],
                    energy_loss_per_step=self.energy_loss_per_step_agent_list[agent_type_nr]
                )
                agent_name = self.agent_type_name_list[agent_type_nr] + "_" + str(self.agent_id_counter)
                # add attributes to agent
                agent_instance.alive = True
                agent_instance.position = np.array([np.random.randint(0, self.x_grid_size),np.random.randint(0, self.y_grid_size)])    # random position
                agent_instance.agent_name = agent_name
                # update model state
                self.model_state[agent_type_nr, agent_instance.position[0],agent_instance.position[1]] += agent_instance.energy
                # add agent to lists
                self.active_agent_instance_list[agent_type_nr].append(agent_instance)
                self.active_agent_name_list[agent_type_nr].append(agent_name)
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                self.alive=True
                self.agent_id_counter+=1
            for _ in range(self.n_active_agent_list[agent_type_nr],self.n_possible_agent_list[agent_type_nr]):
                agent_instance = DiscreteAgent(
                    self.x_grid_size, 
                    self.y_grid_size, 
                    self.predator_type_nr,  # predator type number =1
                    self.agent_id_counter,
                    observation_range=self.obs_range_agent_list[agent_type_nr],
                    motion_range=self.motion_range,
                    initial_energy=self.initial_energy_list[agent_type_nr],
                    energy_loss_per_step=self.energy_loss_per_step_agent_list[agent_type_nr]
                )
                agent_name = self.agent_type_name_list[agent_type_nr] + "_" + str(self.agent_id_counter)
                # add attributes to agent
                agent_instance.alive = False
                agent_instance.agent_name = agent_name
                agent_instance.energy = 0
                # add agent to lists
                self.inactive_agent_instance_list[agent_type_nr].append(agent_instance)
                self.inactive_agent_name_list[agent_type_nr].append(agent_name)
                self.agent_name_to_instance_dict[agent_name] = agent_instance
                self.agent_id_counter+=1
        # end of creation of agents
        #aec_agents = self.active_agent_name_list[self.predator_type_nr] + self.inactive_agent_name_list[self.predator_type_nr] + self.active_agent_name_list[self.prey_type_nr] + self.inactive_agent_name_list[self.prey_type_nr]
        #print("aec_agents = ", aec_agents)

        self.predator_instance_list = self.active_agent_instance_list[self.predator_type_nr] + self.inactive_agent_instance_list[self.predator_type_nr] 
        self.prey_instance_list = self.active_agent_instance_list[self.prey_type_nr] + self.inactive_agent_instance_list[self.prey_type_nr] 
        self.grass_instance_list = self.active_agent_instance_list[self.grass_type_nr] + self.inactive_agent_instance_list[self.grass_type_nr] 
        self.predator_name_list = self.active_agent_name_list[self.predator_type_nr] + self.inactive_agent_name_list[self.predator_type_nr] 
        self.prey_name_list = self.active_agent_name_list[self.prey_type_nr] + self.inactive_agent_name_list[self.prey_type_nr] 
        self.grass_name_list = self.active_agent_name_list[self.grass_type_nr] + self.inactive_agent_name_list[self.grass_type_nr] 
        self.agents = self.predator_name_list + self.prey_name_list
        self.n_aec_cycles = 0

        self.active_grass_instance_list = self.active_agent_instance_list[self.grass_type_nr] 
        self.active_prey_instance_list = self.active_agent_instance_list[self.prey_type_nr]

        for grass_instance in self.active_grass_instance_list:
            self.add_agent_instance_to_position_dict(grass_instance)

        for prey_instance in self.active_prey_instance_list:
            self.add_agent_instance_to_position_dict(prey_instance)

        # removal agents during one cycle
        self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.grass_to_be_removed_by_prey_dict = dict(zip(self.grass_name_list, [False for _ in self.grass_name_list]))
        self.prey_to_be_removed_by_predator_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        self.predator_to_be_removed_by_starvation_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list]))
        self.prey_to_be_removed_by_starvation_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
        # end removal agents
        

    def step(self, action, agent_instance, is_last):
        # reset rewards to zero during every step
        self.agent_reward_dict = dict(zip(self.agents,[0.0 for _ in self.agents]))
        # Extract agent details
        agent_type_nr = agent_instance.agent_type_nr
        agent_name = agent_instance.agent_name
        # If the agent is a predator and it's alive
        if agent_type_nr == self.predator_type_nr and agent_instance.alive: 
            if agent_instance.energy > 0: # and if predator has energy
                # Update the agent's energy due to the energy loss from stepping
                # Move the predator and update the model state
                old_position = agent_instance.position
                # loss energy update in agent and model state
                agent_instance.energy += agent_instance.energy_loss_per_step
                self.model_state[agent_type_nr, agent_instance.position[0], agent_instance.position[1]] += agent_instance.energy_loss_per_step
                # agent leaves old position
                self.model_state[agent_type_nr, old_position[0], old_position[1]] -= agent_instance.energy
                agent_instance.step(action)
                self.model_state[agent_type_nr, agent_instance.position[0], agent_instance.position[1]] += agent_instance.energy
                x_new_position_predator, y_new_position_predator = agent_instance.position
                # check if there is prey energy at the new position and eat it all
                prey_energy_at_new_spot = self.model_state[self.prey_type_nr, x_new_position_predator, y_new_position_predator] 
                if prey_energy_at_new_spot > 0:
                    # list of prey instances in the same cell as the predator, gets converted to list of prey names; indepnedent copy
                    prey_name_list_in_cell_predator = self.create_agent_name_list_from_instance_list(
                        self.agent_instance_in_grid_location_dict[self.agent_type_name_list[self.prey_type_nr]][
                            (x_new_position_predator, y_new_position_predator)])                   
                    # empties list of prey instances in the same cell as the predator, but leaves iterator untouched              
                    cumulative_prey_energy_at_new_spot = 0
                    for prey_name in prey_name_list_in_cell_predator:
                        prey_instance = self.agent_name_to_instance_dict[prey_name]
                        cumulative_prey_energy_at_new_spot += prey_instance.energy
                        self.prey_to_be_removed_by_predator_dict[prey_instance.agent_name] = True
                        #print(prey_instance.agent_name," with energy ", round(prey_instance.energy,2), 
                        #      " is removed from position ",prey_instance.position, " by ", agent_name)
                        agent_instance.energy += prey_instance.energy
                        self.model_state[self.prey_type_nr, x_new_position_predator, y_new_position_predator] -= prey_instance.energy
                        self.model_state[self.predator_type_nr, x_new_position_predator, y_new_position_predator] += prey_instance.energy
                        prey_instance.energy = 0.0
                        self.remove_agent_from_position_dict(prey_instance)
            else:  # If predator has no energy, it starves to death
                self.predator_to_be_removed_by_starvation_dict[agent_name] = True

        # If the agent is a prey and it's alive
        elif agent_type_nr == self.prey_type_nr and agent_instance.alive:
            if agent_instance.energy > 0:  # If prey has energy
                old_position = agent_instance.position
                # 0 is left, 2 is up, 3 is stay, 4 is down, 5 is right
                # loss energy update in agent and model state
                agent_instance.energy += agent_instance.energy_loss_per_step
                self.remove_agent_from_position_dict(agent_instance)
                self.model_state[agent_type_nr, agent_instance.position[0], agent_instance.position[1]] += agent_instance.energy_loss_per_step
                # agent leaves old position
                self.model_state[agent_type_nr, old_position[0], old_position[1]] -= agent_instance.energy
                agent_instance.step(action)
                self.model_state[agent_type_nr, agent_instance.position[0], agent_instance.position[1]] += agent_instance.energy
                self.add_agent_instance_to_position_dict(agent_instance)
                x_new_position_prey, y_new_position_prey = agent_instance.position
                # check if there is grass energy at the new position and eat it all
                grass_energy_at_new_spot = self.model_state[self.grass_type_nr, x_new_position_prey, y_new_position_prey]
                if grass_energy_at_new_spot > 0 and self.agent_instance_in_grid_location_dict[self.agent_type_name_list[self.grass_type_nr]][(x_new_position_prey, y_new_position_prey)]:
                    self.prey_who_remove_grass_dict[agent_name] = True
                    grass_name_list_in_cell_prey = self.create_agent_name_list_from_instance_list(
                        self.agent_instance_in_grid_location_dict[self.agent_type_name_list[self.grass_type_nr]][
                            (x_new_position_prey, y_new_position_prey)])
                    # empties list of prey instances in the same cell as the predator, but leaves iterator untouched     
                    cumulative_grass_energy_at_new_spot = 0         
                    for grass_name in grass_name_list_in_cell_prey:
                        grass_instance = self.agent_name_to_instance_dict[grass_name]
                        cumulative_grass_energy_at_new_spot += grass_instance.energy
                        self.grass_to_be_removed_by_prey_dict[grass_instance.agent_name] = True
                        #print(grass_instance.agent_name," with energy ", round(grass_instance.energy,2), 
                        #      " is removed from position ",grass_instance.position, " by ", agent_name)
                        agent_instance.energy += grass_instance.energy
                        self.model_state[self.grass_type_nr, x_new_position_prey, y_new_position_prey] -= grass_instance.energy
                        self.model_state[self.prey_type_nr, x_new_position_prey, y_new_position_prey] += grass_instance.energy
                        grass_instance.energy = 0.0 
                        self.remove_agent_from_position_dict(grass_instance)
            else: # prey starves to death
                self.prey_to_be_removed_by_starvation_dict[agent_name] = True


        if is_last: # removes agents from list and reap rewards at the end of the cycle
            for predator_name in self.predator_name_list:
                predator_instance = self.agent_name_to_instance_dict[predator_name]
                if predator_instance.alive:
                    # remove predator which starves to death
                    if self.predator_to_be_removed_by_starvation_dict[predator_name]:
                        self.n_active_agent_list[self.predator_type_nr] -= 1
                        predator_instance.energy = 0.0
                        predator_instance.alive = False
                        self.active_agent_instance_list[self.predator_type_nr].remove(predator_instance)
                    else: # step rewards for predator an update
                        self.agent_reward_dict[predator_name] = predator_instance.energy

            for prey_name in self.prey_name_list:
                prey_instance = self.agent_name_to_instance_dict[prey_name]
                if prey_instance.alive:
                    # remove prey which gets eaten by a predator or starves to death
                    if self.prey_to_be_removed_by_predator_dict[prey_name] or self.prey_to_be_removed_by_starvation_dict[prey_name]:
                        self.n_active_agent_list[self.prey_type_nr] -= 1
                        prey_instance.alive = False
                        prey_instance.energy = 0.0
                        self.active_agent_instance_list[self.prey_type_nr].remove(prey_instance)
                    else: # reap rewards for prey which removes grass
                        self.agent_reward_dict[prey_name] = prey_instance.energy

            for grass_name in self.grass_name_list:
                grass_instance = self.agent_name_to_instance_dict[grass_name]
                if self.grass_to_be_removed_by_prey_dict[grass_name]:
                    self.n_active_agent_list[self.grass_type_nr] -= 1
                    self.active_agent_instance_list[self.grass_type_nr].remove(grass_instance)

            self.n_aec_cycles = self.n_aec_cycles + 1
            
            
            #reinit agents records to default at the end of the cycle
            self.prey_who_remove_grass_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
            self.grass_to_be_removed_by_prey_dict = dict(zip(self.grass_name_list, [False for _ in self.grass_name_list]))
            self.prey_to_be_removed_by_predator_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
            self.predator_to_be_removed_by_starvation_dict = dict(zip(self.predator_name_list, [False for _ in self.predator_name_list]))
            self.prey_to_be_removed_by_starvation_dict = dict(zip(self.prey_name_list, [False for _ in self.prey_name_list]))
            # end reinit agents

        if self.render_mode == "human":
            self.render()
            """
            print("self.model_state[self.predator_type_nr] = ")
            print(np.transpose(self.model_state[self.predator_type_nr]))
            print("self.model_state[self.prey_type_nr] = ")
            print(np.transpose(self.model_state[self.prey_type_nr]))
            print("self.model_state[self.grass_type_nr] = ")    
            print(np.transpose(self.model_state[self.grass_type_nr]))
            print()
            """
        
               
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


    def add_agent_instance_to_position_dict(self, agent_instance):
        # Determine the correct dictionary based on agent type
        agent_type = self.agent_type_name_list[agent_instance.agent_type_nr]
        if agent_type == 'grass':
            position_dict = self.agent_instance_in_grid_location_dict['grass']
        elif agent_type == 'prey':
            position_dict = self.agent_instance_in_grid_location_dict['prey']
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Add the agent instance to the dictionary
        position = tuple(agent_instance.position)
        if position not in position_dict:
            position_dict[position] = []
        position_dict[position].append(agent_instance)

    def remove_agent_from_position_dict(self, agent_instance):
        # Determine the correct dictionary based on agent type
        agent_type = self.agent_type_name_list[agent_instance.agent_type_nr]
        if agent_type == 'grass':
            position_dict = self.agent_instance_in_grid_location_dict['grass']
        elif agent_type == 'prey':
            position_dict = self.agent_instance_in_grid_location_dict['prey']
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Remove the agent instance from the dictionary
        position = tuple(agent_instance.position)
        position_dict[position].remove(agent_instance)



    @property
    def is_no_grass(self):
        if self.n_active_agent_list[self.grass_type_nr] == 0:
            return True
        return False

    @property
    def is_no_prey(self):
        if self.n_active_agent_list[self.prey_type_nr] == 0:
            return True
        return False

    @property
    def is_no_predator(self):
        if self.n_active_agent_list[self.predator_type_nr] == 0:
            return True
        return False

    def observe(self, agent_name):

        agent_instance = self.agent_name_to_instance_dict[agent_name]
        
        xp, yp = agent_instance.position[0], agent_instance.position[1]

        # returns a flattened array of all the observations
        observation = np.zeros((self.nr_observation_channels, self.max_observation_range, self.max_observation_range), dtype=np.float32)
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
            for predator_instance in self.active_agent_instance_list[self.predator_type_nr]:
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
            for prey_instance in self.active_agent_instance_list[self.prey_type_nr]:
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
            for predator_instance in self.active_agent_instance_list[self.predator_type_nr]:
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
            for prey_instance in self.active_agent_instance_list[self.prey_type_nr]:
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
            for grass_instance in self.active_agent_instance_list[self.grass_type_nr]:
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

            for predator_instance in self.active_agent_instance_list[self.predator_type_nr]:
                prey_position =  predator_instance.position 
                x = prey_position[0]
                y = prey_position[1]
                predator_positions[(x, y)] = predator_instance.agent_id_nr

            for prey_instance in self.active_agent_instance_list[self.prey_type_nr]:
                prey_position =  prey_instance.position 
                x = prey_position[0]
                y = prey_position[1]
                prey_positions[(x, y)] = prey_instance.agent_id_nr

            for grass_instance in self.active_agent_instance_list[self.grass_type_nr]:
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
            y_position_energy_chart = 0 # self.y_pygame_window
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
            x_screenposition = 350   
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
                label_x = 400 + x_axis_x + i * (bar_width + offset_bars)
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
        if self.energy_chart:
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
        "name": "predprey_37",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self.pred_prey_env = PredPrey(*args, **kwargs) #  this calls the code from PredPrey

        self.agents = self.pred_prey_env.agents

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

    def observe(self, _agent_name):
        _agent_instance = self.pred_prey_env.agent_name_to_instance_dict[_agent_name]
        obs = self.pred_prey_env.observe(_agent_name)
        observation = np.swapaxes(obs, 2, 0) # type: ignore
        # return observation of only zeros if agent is not alive
        #WHY IS THIS ONLY FOR PREY??
        if _agent_instance.agent_type_nr==self.pred_prey_env.prey_type_nr and \
              not _agent_instance.alive:
            shape=observation.shape
            observation = np.zeros(shape)
        return observation
            
    def observation_space(self, agent: str):  # must remain
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    