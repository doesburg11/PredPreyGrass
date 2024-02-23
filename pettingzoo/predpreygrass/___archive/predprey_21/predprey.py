"""
PD: implements PredatorPreyGrass, compared to pursuit_v4: 
Installation (besides requirements.txt):
-"libGL error: failed to load driver: swrast" requires 'conda install -c conda-forge gcc=12.1.0'
(https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin)

[v0] a Moore neighborhood with additional parameter moore_neighborhood = False
as default
[v1] parameres xb, yb for changing the size of the center non inhabitable white square
[v2] simplify the code
-removed manual policy
-removed optional rectangular obstacle in center and all possible obstacle variations in two_d_maps.py
-removed two_d_maps.py
-removed map_matrix
-model_state remained untouched (base) because it is input for observation; 
 model_state[0] has become obsolete, model_state[3] was already obsolete?
-removed agent_utils.py and integrated create_agents() into predprey_base.py)
-integrated Agent into discrete_agent.py and removed _utils_2.py
-integrated AgentLayer into discrete_agent.py and removed agent_layer.py
-integrated controllers into discrete_agent.py and removed controllers.py
-moved discrete_agent.py one directory level lower and removed directory utils
[v4] integrate files
-integrate discrete_agent.py into predprey_base.py
-integrate predprey_base.py into predprey.py
-more specific les abstract coding
-remove Agent from DiscreteAgent inheritance
-create n_action_pursuers from the moore or von neumann choice, so that reset does not have to be called 
beforehand
-Both evaders and pursuers can move in Moore or Von Neumann directions
[v6]
-surround option removed
-n_catch removed
[v7]
-created new agent predators and renamed evaders to grass and pursuers to prey
-added predators to grid but not yet activated
[v8]
-remove predprey_v8.py
-remove wrappers
-add parallel_env to executables (random and training)
-remove local_ratio(=1)
-remove unused methods in class PredPrey and DiscreteAgent
-simplify safely_observe
9) [v9]
-renaming, simplify
-convert-ready to bi-agent (prey/grass or predator/prey)
-number the grid cell with (x,y) coordinates 
            GRID COORDINATES MOORE: x_size=y_size=16
            ------------------------------------------
            |(0,0)...........(12,0)...........(15,0) |
            |  .                .                .   |
            |  .                .                .   |
            |  .      (11,9) (12,9)  (13,9)      .   |
            |(0,10).. (11,10 (12,10) (13,10)..(15,10)|
            |  .      (11,11)(12,11) (13,11)     .   |
            |  .                .                .   |
            |  .                .                .   |
            |(0,15)..........(12,15)..........(15,15)|
            ------------------------------------------

                       action = [     0,       1,       2,       3,      4,      5,      6,     7,     8]
            self.motion_range = [[-1,-1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0],[-1,1],[0,1],[-1,-1]]
            ------------------------------------------
            |                 .                      |
            |                  .                     |
            |         (-1,-1) (0,-1) (1,-1)          |
            |         (-1,0   (0,0)  (1,0)           |
            |         (-1,1)  (0,1)  (1,1)           |
            |                  .                     |
            |                  .                     |
            |                                        |
            ------------------------------------------

            GRID COORDINATES VON NEUMANN: x_size=y_size=16
            ------------------------------------------
            |(0,0)...........(12,0)...........(15,0) |
            |  .                .                .   |
            |  .                .                .   |
            |  .             (12,9)              .   |
            |(0,10).. (11,10 (12,10) (13,10)..(15,10)|
            |  .             (12,11)             .   |
            |  .                .                .   |
            |  .                .                .   |
            |(0,15)..........(12,15)..........(15,15)|
            ------------------------------------------
                       action = [    0,       1,      2,     3,      4]
            self.motion_range = [[0,-1], [-1, 0], [1, 0], [0,1], [0, 0]]
            ------------------------------------------
            |                 .                      |
            |                  .                     |
            |                 (0,-1)                 |
            |         (-1,0   (0,0)  (1,0)           |
            |                 (0,1)                  |
            |                  .                     |
            |                  .                     |
            |                                        |
            ------------------------------------------

-change predators into prey_9 (pre with moore movements)

different actions range and observation range per agent?
    -at creation of a single agent: give observation as input to create attribute (self.observation_range)
    -varying action spaces Moore/not Moore does not work. Only option that does work is
    Moore=False for Both agents
[v10] 
-remove prey_9 again (former predators)
-draw_prey_instances (based on instance list rather than layer)
-draw_gras_instances (based on instance list rather than layer)
-draw agent_instance_id_nr (instead of using the array index nr)
[v11]
-store diverse observation spaces per agent into array
-this is accomplished by setting an overall max_observation_range(=7 for example)
 and setting an observation range per agent which is smaller or equal to this range.
 If range is maller (=5 for example) then the outer ring(s) of the max_observation_range
 are all set to zero. The observations all need identical shapes, so this shortcut is 
 used in this manner. 
 -removed shared_reward (had no function)
 -obtains reward from last() per agent
 -add to cumulative reward
 -use agent_selector.next() and agent_selector.is_last() to count-up n_aec_cycles+=1
[v12]
 -average n_cycle added
 -create_agents makes a standalone list. does not append to existing list anymore
 -self.prey1_instance list and self.prey2_instance_list added (prey=prey1+prey2). To be able to 
 create stats on both groups.
[v13]
 -implement average reward stats  voor both types of prey
 -observation range made fully flexible below the max_observation_range
 -remove hard coded groups prey1_name_list and prey2_name_list and flexibilize
 -rename env into raw_env and pred_prey_env in sb3_predprey_vector.py
[v14]
-pixel-scale into kwargs
remove array-index related arrays and change them to dicts
 -self.rewards in PredPrey
 -self.grasses_gone
[v15]
-remove agent_idx number and replace by agent_id_name in predprey.step()
and agent_layer. Change move_agent (line 125) from agent_idx to agent_instance.
-cleanup
[v16]
-urgency_reward and catch_reward  'personalized' and moved to DisrceteAgent and renamed to homeostatic_energy_per_aec_cycle
and catch_grass_reward respectively
-removed agent_name_to_id_nr_dict (can be fetched by agent_name_to_instance_dict.agent_id_nr)
-removed urgency_reward and catch_reward from pred_prey args
[v17]
-let pred_prey handle the agent_intialization of self.agents (=pred_prey_env.agent_names_list)
-rename self.frames to self.n_aec_cycles and use in main program (remove agent_selector.last())
-surpress warnings by quick fixes vscode when possible
[v18]
-remove RandomPolicy and SingleActionPolicy controller classes
[v19]
-clean up DiscreteAgent
"""
"""
[v20]
-return agent_selector to show end of cycle section in main program, can help to show a rewards after cycle, which
went wrong earlier
-rewards as args in DiscreteAgent
-rename Prey1 to Predator
-rename Prey2 to Prey
-cleanup,make better distinction between predprey and raw_env corresponding variables
-hard coded number of channels to 3 in order to add "predator" in 'agent_type_names'
-removed self.agents in PredPrey (apparently no use)
[v21]
-changed revisions text to 'predprey.py'
-Color change in graphs

TODO
-still to implement: prey_layer and predator_layer
-ultimately make new observation channel for predators
-Multiple type of agents
-implement minimum energy levels
-Death and birth of agents
-pixel-scale dynamic on x_size/y?size?
-personalize reward per agent
-personalize (not)Moore per agent

"""

# noqa: D212, D415

from collections import defaultdict
import numpy as np
import pygame

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
    

class DiscreteAgent():
    def __init__(
        self,
        x_size,
        y_size,
        agent_type_nr, # 0: wall, 1: prey, 2: grass, 3: predator
        agent_id_nr,
        agent_id_name,
        observation_range=7,
        n_channels=3, # n channels is the number of observation channels
        flatten=False,
        moore_neighborhood=False,
        initial_energy=100,
        catch_grass_reward=5.0,
        homeostatic_energy_per_aec_cycle=-0.1

    ):
        #identification agent
        self.agent_type_nr = agent_type_nr   # also channel number of agent 
        self.agent_id_name = agent_id_name   # string like "prey_1"
        self.agent_id_nr = agent_id_nr       # unique integer per agent

        #(fysical) boundaries/limitations agent in observing (take in) and acting (take out)
        self.x_size = x_size
        self.y_size = y_size
        self.observation_range = observation_range
        self.observation_shape = (n_channels * observation_range**2 + 1,) if flatten else \
            (observation_range, observation_range, n_channels)
        self.moore_neighborhood = moore_neighborhood
        if moore_neighborhood:
            self.possible_actions = [
                0,  # move left/up
                1,  # move up
                2,  # move right/up
                3,  # move left
                4,  # stay
                5,  # move right
                6,  # move left/down 
                7,  # move down
                8,  # move right/down
            ]  
            self.motion_range = [[-1,-1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0],[-1,1],[0,1],[-1,-1]]
        else:  #Von Neumann neighborhood
            self.possible_actions = [
                0,  # move up
                1,  # move left
                2,  # move right
                3,  # move down
                4,  # stay
            ]  
            self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]] 
        self.n_actions_agent=len(self.possible_actions)   
        self.action_space_agent = spaces.Discrete(self.n_actions_agent) 
        self.position = np.zeros(2, dtype=np.int32)  # x and y position
        self.terminal = False # alive at creation
        self.energy = initial_energy  # still to implement
        self.homeostatic_energy_per_aec_cycle = homeostatic_energy_per_aec_cycle
        self.catch_grass_reward = catch_grass_reward

    def step(self, action):
        # returns new position of agent "self" given action "action"
        next_position = np.zeros(2, dtype=np.int32) 
        next_position[0], next_position[1] = self.position[0], self.position[1]
        next_position += self.motion_range[action]
        if self.terminal or not (0 <= next_position[0] < self.x_size and 
                                 0 <= next_position[1] < self.y_size):
            return self.position   # if dead or reached goal or if moved out of borders: dont move
        else:
            self.position = next_position
            return self.position

class AgentLayer:
    def __init__(self, xs, ys, ally_agents_instance_list):

        self.ally_agents_instance_list = ally_agents_instance_list
        self.n_ally_agents = len(ally_agents_instance_list)
        self.global_state_ally_agents = np.zeros((xs, ys), dtype=np.int32)

    def n_ally_layer_agents(self):
        return self.n_ally_agents

    def move_agent_instance(self, agent_instance, action):
        return agent_instance.step(action)

    def get_position_agent_instance(self, agent_instance):
        return agent_instance.position
  
    def remove_agent_instance(self, agent_instance):
        self.ally_agents_instance_list.remove(agent_instance)
        self.n_ally_agents -= 1

    def get_global_state_ally_agents(self):
        global_state_ally_agents = self.global_state_ally_agents
        global_state_ally_agents.fill(0)
        for ally_agent_instance in self.ally_agents_instance_list:
            x, y = ally_agent_instance.position
            global_state_ally_agents[x, y] += 1
        return global_state_ally_agents

class PredPrey:
    def __init__(
        self,
        x_size: int = 16,
        y_size: int = 16,
        max_cycles: int = 500,
        n_predator: int = 4,
        n_prey: int = 4,
        n_grasses: int = 10,
        max_observation_range: int =7,
        obs_range_predator: int =7,
        obs_range_prey: int =7,
        freeze_grasses: bool = False,
        render_mode=None,
        moore_neighborhood_prey: bool = False,        
        moore_neighborhood_grasses: bool = False,
        pixel_scale: int = 40

    ):
        #parameter init
        self.x_size = x_size
        self.y_size = y_size
        self.max_cycles = max_cycles
        self.n_predator = n_predator
        self.n_prey = n_prey
        self.n_grasses = n_grasses
        self.max_observation_range = max_observation_range
        self.obs_range_predator = obs_range_predator        
        self.obs_range_prey = obs_range_prey
        self.freeze_grasses = freeze_grasses
        self.render_mode = render_mode
        self.moore_neighborhood_prey = moore_neighborhood_prey
        self.moore_neighborhood_grasses = moore_neighborhood_grasses
        self.pixel_scale = pixel_scale

        self._seed()
        self.agent_id_counter = 0

        # agent types
        self.agent_type_names = ["wall", "prey", "grass", "predator"]  # different types of agents 
        self.prey_type_nr = self.agent_type_names.index("prey")
        self.grass_type_nr = self.agent_type_names.index("grass")
        self.predator_type_nr = self.agent_type_names.index("predator")
        
        # lists of agents
        # initialization
        self.predator_instance_list = [] # list of all living predator
        self.prey_instance_list = [] # list of all living prey
        self.grass_instance_list = [] # list of all living grasses
        self.agent_instance_list = []
 
        self.agent_name_to_instance_dict = dict()
        # 'n_aec_agents' is the PredPrey equivalent of PettingZoo 'num_agents' in raw_env
        self.n_aec_agents = self.n_predator + self.n_prey

        # creation
        # 'agent_name_aec_list' is the PredPrey equivalent of PettingZoo 'agents' in raw_env
        self.agent_name_aec_list =  list(["predator" + "_" + str(a) for a in range(self.n_predator)]) +\
                                    list(["prey" + "_" + str(a) for a in range(self.n_predator, self.n_prey+self.n_predator)])
                                    
        # observations
        max_agents_overlap = max(self.n_aec_agents, self.n_grasses)
        self.max_obs_offset = int((self.max_observation_range - 1) / 2) 
        self.nr_observation_channels = 3 #len(self.agent_type_names) # HARD CODED
        obs_space = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.max_observation_range, self.max_observation_range, self.nr_observation_channels),
            dtype=np.float32,
        )
        self.observation_space = [obs_space for _ in range(self.n_aec_agents)]  # type: ignore
        self.obs_spaces_test = []
        # end observations

        # actions
        self.n_actions_prey = 9 if self.moore_neighborhood_prey else 5
        self.n_actions_grasses = 9 if self.moore_neighborhood_grasses else 5 
        act_space = spaces.Discrete(self.n_actions_prey)  #### MUST BE FLEXIBLE PER AGENT????
        self.action_space = [act_space for _ in range(self.n_aec_agents)] # type: ignore
        # end actions

        self.screen = None
        self.n_aec_cycles = 0

    def create_agent_instance_list(
            self, 
            n_agents,
            agent_type_nr,
            observation_range, 
            randomizer, 
            flatten=False, 
            moore_neighborhood=True
            ):

        _agent_instance_list = []

        agent_type_name = self.agent_type_names[agent_type_nr]
        for _ in range(n_agents): 
            agent_id_nr = self.agent_id_counter           
            agent_id_name = agent_type_name + "_" + str(agent_id_nr)
            self.agent_id_counter+=1
            xinit, yinit = (randomizer.integers(0, self.x_size), randomizer.integers(0, self.y_size))      

            agent_instance = DiscreteAgent(
                self.x_size, 
                self.y_size, 
                agent_type_nr, 
                agent_id_nr,
                agent_id_name,
                observation_range=observation_range, 
                flatten=flatten, 
                moore_neighborhood=moore_neighborhood
            )
            #  updates lists en records
            self.agent_name_to_instance_dict[agent_id_name] = agent_instance
            agent_instance.position = (xinit, yinit)
            _agent_instance_list.append(agent_instance)
        return _agent_instance_list

    def create_agent_name_list_from_instance_list(self, _agent_instance_list):
        _agent_name_list = []
        for agent_instance in _agent_instance_list:
            _agent_name_list.append(agent_instance.agent_id_name)
        return _agent_name_list

    def reset(self):
        # empty agent lists
        self.predator_instance_list =[]
        self.prey_instance_list =[]
        self.grass_instance_list = []
        self.agent_instance_list = []

        self.predator_name_list =[]
        self.prey_name_list =[]
        self.grass_name_list = []
        self.agent_name_list = []


        self.agent_id_counter = 0
        self.agent_name_to_instance_dict = {}        
        self.model_state = np.zeros((self.nr_observation_channels, self.x_size, self.y_size), dtype=np.float32)
  
        #list agents consisting of predator agents
        self.predator_instance_list = self.create_agent_instance_list(
            self.n_predator, self.predator_type_nr, self.obs_range_predator, self.np_random, moore_neighborhood=False
        )
        self.predator_name_list =  self.create_agent_name_list_from_instance_list(
            self.predator_instance_list
        )
        #list agents consisting of prey agents
        self.prey_instance_list = self.create_agent_instance_list(
            self.n_prey, self.prey_type_nr, self.obs_range_prey, self.np_random, moore_neighborhood=False
        )
        self.prey_name_list =  self.create_agent_name_list_from_instance_list(
            self.prey_instance_list
        )
        self.grass_instance_list = self.create_agent_instance_list(
            # grass observation range is zero
            self.n_grasses, self.grass_type_nr, 0, self.np_random, moore_neighborhood=self.moore_neighborhood_grasses
        ) 
        self.grass_name_list =  self.create_agent_name_list_from_instance_list(
            self.grass_instance_list
        )
        self.grasses_gone_dict = dict(zip(self.grass_name_list, [False for _ in self.grass_name_list]))
        self.agent_instance_list = self.predator_instance_list + self.prey_instance_list
        
        self.agent_name_list = self.predator_name_list + self.prey_name_list


        #still to implement prey_layer and predator_layer
        self.agent_layer = AgentLayer(self.x_size, self.y_size, self.agent_instance_list)
        self.grass_layer = AgentLayer(self.x_size, self.y_size, self.grass_instance_list)

        self.pred_prey_rewards_dict = dict(zip(self.agent_name_list, [0.0 for _ in self.agent_name_list]))

    
        self.model_state[self.prey_type_nr] = self.agent_layer.get_global_state_ally_agents()
        self.model_state[self.grass_type_nr] = self.grass_layer.get_global_state_ally_agents()

        self.n_aec_cycles = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent] # type: ignore

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def step(self, action, agent_instance, is_last):
        grass_layer = self.grass_layer  # updates only at end of cycle
        self.agent_layer.move_agent_instance(agent_instance, action)
        self.model_state[self.prey_type_nr] = self.agent_layer.get_global_state_ally_agents()
        self.pred_prey_rewards_dict = dict(zip(self.agent_name_list, [0.0 for _ in self.agent_name_list]))

        if is_last:
            agents_who_remove_grass_dict = self.remove_grass()
            for grass_instance in self.grass_instance_list:
                action = 4 if self.freeze_grasses else self.np_random.integers(self.n_actions_grasses)
                grass_layer.move_agent_instance(grass_instance, action)
            for agent_name in self.agent_name_list:
                agent_instance = self.agent_name_to_instance_dict[agent_name]
                self.pred_prey_rewards_dict[agent_name] += agent_instance.catch_grass_reward * agents_who_remove_grass_dict[agent_name] 
                self.pred_prey_rewards_dict[agent_name] += agent_instance.homeostatic_energy_per_aec_cycle
            self.n_aec_cycles = self.n_aec_cycles + 1

        # Update the remaining layers
        self.model_state[self.grass_type_nr] = self.grass_layer.get_global_state_ally_agents()

        if self.render_mode == "human":
            self.render()

    def draw_model_state(self):
        # -1 is building pixel flag
        x_len, y_len = (self.x_size, self.y_size)
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                color = (255, 255, 255) # white background                

                pygame.draw.rect(self.screen, color, pos) # type: ignore

    def draw_predator_observations(self):
        for predator_instance in self.predator_instance_list:
            position =  predator_instance.position 
            x = position[0]
            y = position[1]
            mask = int((self.max_observation_range - predator_instance.observation_range)/2)
            if mask == 0:
                patch = pygame.Surface(
                    (self.pixel_scale * self.max_observation_range, self.pixel_scale * self.max_observation_range)
                )
                patch.set_alpha(128)
                patch.fill((72, 152, 255))
                ofst = self.max_observation_range / 2.0
                self.screen.blit(
                    patch,
                    (
                        self.pixel_scale * (x - ofst + 1 / 2),
                        self.pixel_scale * (y - ofst + 1 / 2),
                    ),
                )
            else:
                patch = pygame.Surface(
                    (self.pixel_scale * predator_instance.observation_range, self.pixel_scale * predator_instance.observation_range)
                )
                patch.set_alpha(128)
                patch.fill((72, 152, 255))
                ofst = predator_instance.observation_range / 2.0
                self.screen.blit(
                    patch,
                    (
                        self.pixel_scale * (x - ofst + 1 / 2),
                        self.pixel_scale * (y - ofst + 1 / 2),
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
                    (self.pixel_scale * self.max_observation_range, self.pixel_scale * self.max_observation_range)
                )
                patch.set_alpha(128)
                patch.fill((255, 152, 72))
                ofst = self.max_observation_range / 2.0
                self.screen.blit(
                    patch,
                    (
                        self.pixel_scale * (x - ofst + 1 / 2),
                        self.pixel_scale * (y - ofst + 1 / 2),
                    ),
                )
            else:
                patch = pygame.Surface(
                    (self.pixel_scale * prey_instance.observation_range, self.pixel_scale * prey_instance.observation_range)
                )
                patch.set_alpha(128)
                patch.fill((255, 152, 72))
                ofst = prey_instance.observation_range / 2.0
                self.screen.blit(
                    patch,
                    (
                        self.pixel_scale * (x - ofst + 1 / 2),
                        self.pixel_scale * (y - ofst + 1 / 2),
                    ),
                )

    def draw_predator_instances(self):
        for predator_instance in self.predator_instance_list:
            position =  predator_instance.position 
            x = position[0]
            y = position[1]

            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )

            col = (255, 0, 0) # red

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3)) # type: ignore

    def draw_prey_instances(self):
        for prey_instance in self.prey_instance_list:
            position =  prey_instance.position 
            x = position[0]
            y = position[1]

            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )

            col = (0, 0, 255) # blue

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3)) # type: ignore

    def draw_grass_instances(self):
        for grass_instance in self.grass_instance_list:
            position =  grass_instance.position 
            x = position[0]
            y = position[1]

            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )

            col = (0, 128, 0) # green

            #col = (0, 0, 255) # blue

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3)) # type: ignore

    def draw_agent_instance_id_nrs(self):
        font = pygame.font.SysFont("Comic Sans MS", self.pixel_scale * 2 // 3)

        predator_positions = defaultdict(int)
        prey_positions = defaultdict(int)
        grass_positions = defaultdict(int)

        for predator_instance in self.predator_instance_list:
            prey_position =  predator_instance.position 
            x = prey_position[0]
            y = prey_position[1]
            prey_positions[(x, y)] = predator_instance.agent_id_nr

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
                self.pixel_scale * x + self.pixel_scale // 2.4,
                self.pixel_scale * y + self.pixel_scale // 1.2,
            )

            predator_id_nr__text =str(predator_positions[(x, y)])

            predator_text = font.render(predator_id_nr__text, False, (255, 255, 0))

            self.screen.blit(predator_text, (pos_x, pos_y - self.pixel_scale // 2))

        for x, y in prey_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2.4,
                self.pixel_scale * y + self.pixel_scale // 1.2,
            )

            prey_id_nr__text =str(prey_positions[(x, y)])

            prey_text = font.render(prey_id_nr__text, False, (255, 255, 0))

            self.screen.blit(prey_text, (pos_x, pos_y - self.pixel_scale // 2))

        for x, y in grass_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 3.2,
                self.pixel_scale * y + self.pixel_scale // 1.2,
            )

            grass_id_nr__text =str(grass_positions[(x, y)])

            grass_text = font.render(grass_id_nr__text, False, (255, 255, 0))

            self.screen.blit(grass_text, (pos_x, pos_y - self.pixel_scale // 2))

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )
                pygame.display.set_caption("PreyGrass")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )

        self.draw_model_state()

        self.draw_prey_observations()
        self.draw_predator_observations()

        self.draw_grass_instances()
        self.draw_prey_instances()
        self.draw_predator_instances()

        self.draw_agent_instance_id_nrs()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def save_image(self, file_name):
        self.render()
        capture = pygame.surfarray.array3d(self.screen) # type: ignore

        xl, xh = -self.max_obs_offset - 1, self.x_size + self.max_obs_offset + 1
        yl, yh = -self.max_obs_offset - 1, self.y_size + self.max_obs_offset + 1

        window = pygame.Rect(xl, yl, xh, yh)
        subcapture = capture.subsurface(window) # type: ignore

        pygame.image.save(subcapture, file_name)

    @property
    def is_terminal(self):
        # ev = self.grass_layer.get_global_state_ally_agents()  # grass positions
        # if np.sum(ev) == 0.0:
        if self.grass_layer.n_ally_layer_agents() == 0:
            return True
        return False

    def safely_observe_agent_name(self, agent_id_name):

        agent_instance = self.agent_name_to_instance_dict[agent_id_name]
        
        xp, yp = agent_instance.position[0], agent_instance.position[1]

        # returns a flattened array of all the observations
        observation = np.zeros((self.nr_observation_channels, self.max_observation_range, self.max_observation_range), dtype=np.float32)
        observation[0].fill(1.0)  

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        observation[0:3, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:3, xlo:xhi, ylo:yhi])
        
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
        else:
            return observation

    def obs_clip(self, x, y):
        xld = x - self.max_obs_offset
        xhd = x + self.max_obs_offset
        yld = y - self.max_obs_offset
        yhd = y + self.max_obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.max_obs_offset, 0)), abs(
            np.clip(yld, -self.max_obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def remove_grass(self):
        removed_grass_names_list = []
        agents_who_remove_grass_dict = dict(zip(self.agent_name_list, [False for _ in self.agent_name_list]))
        for grass_name in self.grass_name_list:
            if self.grasses_gone_dict[grass_name]:
                continue
            grass_instance = self.agent_name_to_instance_dict[grass_name]
            x, y = self.grass_layer.get_position_agent_instance(
                grass_instance
                )
            if self.model_state[self.prey_type_nr, x, y] > 0:   # at least one prey on spot of grass
                # remove grass
                removed_grass_names_list.append(grass_name)
                self.grasses_gone_dict[grass_name] = True
                for agent_instance in self.agent_instance_list:  # identify all prey on the spot of the grass
                    xpp, ypp = self.agent_layer.get_position_agent_instance(agent_instance)
                    if xpp == x and ypp == y:
                        agents_who_remove_grass_dict[agent_instance.agent_id_name] = True
                    
        for grass_name in removed_grass_names_list:
            self.grass_layer.remove_agent_instance(
                self.agent_name_to_instance_dict[grass_name]
                )
        return agents_who_remove_grass_dict 

class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predprey_21",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self.pred_prey_env = PredPrey(*args, **kwargs) #  this calls the code from PredPrey

        self.agents = self.pred_prey_env.agent_name_aec_list 

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
        #is_last_agent = self._agent_selector.is_last()
        agent_instance = self.pred_prey_env.agent_name_to_instance_dict[agent]
        self.pred_prey_env.step(
            action, agent_instance, self._agent_selector.is_last()
        )

        for k in self.terminations:
            if self.pred_prey_env.n_aec_cycles >= self.pred_prey_env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.pred_prey_env.is_terminal
        for agent_name in self.agents:
            self.rewards[agent_name] = self.pred_prey_env.pred_prey_rewards_dict[agent_name]
        self.steps += 1
        self._cumulative_rewards[self.agent_selection] = 0  # cannot be left out for proper rewards
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()  # cannot be left out for proper rewards
        if self.render_mode == "human":
            self.render()

    def observe(self, agent_id_name):
        obs = self.pred_prey_env.safely_observe_agent_name(agent_id_name)
        observation = np.swapaxes(obs, 2, 0) # type: ignore
        return observation
    
    def observation_space(self, agent: str):  # must remain
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    
