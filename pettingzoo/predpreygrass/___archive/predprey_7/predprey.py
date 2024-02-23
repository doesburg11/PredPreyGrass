# noqa: D212, D415

import abc
from collections import defaultdict
from typing import Optional
import numpy as np
import pygame

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

    
# Implements multi-agent controllers
class PredPreyPolicy(abc.ABC):
    @abc.abstractmethod
    def act(self, state: np.ndarray) -> int:
        raise NotImplementedError


class DiscreteAgent():
    # constructor
    def __init__(
        self,
        xs,
        ys,
        agent_type_nr, # 0: wall, 1: prey, 2: grass
        agent_id_name,
        obs_range=3,
        n_channels=3, # n channels is the number of observation channels
        flatten=False,
        moore_neighborhood=False
    ):
        
        self.xs = xs
        self.ys = ys
        self.agent_type_nr = agent_type_nr   # also channel number of agent 
        self.agent_id_name = agent_id_name   # string like "prey_1"


        if moore_neighborhood:
            #print("moore_neighborhood")
            self.eactions = [
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
            self.motion_range = [[-1,1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0],[-1,-1],[0,-1],[1,-1]]           
        else:  #Von Neumann neighborhood
            self.eactions = [
                0,  # move left
                1,  # move right
                2,  # move up
                3,  # move down
                4,  # stay
            ]  
            self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]

        self.n_actions=len(self.eactions)

        self.current_pos = np.zeros(2, dtype=np.int32)  # x and y position
        self.last_pos = np.zeros(2, dtype=np.int32) # what is that? en why it is used?
        self.temp_pos = np.zeros(2, dtype=np.int32) # what is that? en why it is used?

        self.terminal = False

        self._obs_range = obs_range

        if flatten:
            self._obs_shape = (n_channels * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, n_channels)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape)

    @property
    def action_space(self):
        return spaces.Discrete(len(self.eactions))

    # Dynamics Functions
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # if dead or reached goal dont move
        if self.terminal:
            return cpos
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]

        # transition is deterministic
        #print(self.agent_id_name)
        #print("a ",a)
        #print(self.motion_range)
        tpos += self.motion_range[a]
        x = tpos[0]
        y = tpos[1]

        # check bounds
        if not self.inbounds(x, y):
            return cpos
        # if bumped into building, then stay
        else:
            lpos[0] = cpos[0]
            lpos[1] = cpos[1]
            cpos[0] = x
            cpos[1] = y
            return cpos

    def get_state(self):
        return self.current_pos

    # Helper Functions
    def inbounds(self, x, y):
        if 0 <= x < self.xs and 0 <= y < self.ys:
            return True
        return False

    def nactions(self):
        return len(self.eactions)

    def set_position(self, xs, ys):
        self.current_pos[0] = xs
        self.current_pos[1] = ys

    def current_position(self):
        return self.current_pos

    def last_position(self):
        return self.last_pos


class AgentLayer:
    def __init__(self, xs, ys, allies):
        """
        Stores charteristics of an agent type (Predators, prey and grasses) 
        of agents in a grid (global_state): 

        inputs: 
        ============================================
        xs, ys = size of grid
        self.allies = list of agents (instances)
        nagents = implicit number of agents
        =============================================
        xs: x size of map
        ys: y size of map
        allies: list of ally agents, list of DiscreteAgents
        seed: seed

        methods:



        Each ally agent must support:
        - move(action)
        - current_position()
        - nactions()
        - set_position(x, y)
        """
        self.allies = allies
        self.nagents = len(allies)
        # global state of the ally agents only, not other agents
        self.global_state = np.zeros((xs, ys), dtype=np.int32)

    def n_agents(self):
        return self.nagents

    def move_agent(self, agent_idx, action):
        return self.allies[agent_idx].step(action)

    def set_position(self, agent_idx, x, y):
        self.allies[agent_idx].set_position(x, y)

    def get_position(self, agent_idx):
        """Returns the position of the given agent."""
        return self.allies[agent_idx].current_position()

    def get_nactions(self, agent_idx):
        return self.allies[agent_idx].nactions()

    def remove_agent(self, agent_idx):
        # idx is between zero and nagents
        self.allies.pop(agent_idx)
        self.nagents -= 1

    def get_state_matrix(self):
        """Returns a matrix representing the positions of all allies.

        Example: matrix contains the number of allies at give (x,y) position
        0 0 0 1 0 0 0
        0 2 0 2 0 0 0
        0 0 0 0 0 0 1
        1 0 0 0 0 0 5
        """
        gs = self.global_state
        gs.fill(0)
        for ally in self.allies:
            x, y = ally.current_position()
            gs[x, y] += 1
        return gs

    def get_state(self):
        pos = np.zeros(2 * len(self.allies))
        idx = 0
        for ally in self.allies:
            pos[idx : (idx + 2)] = ally.get_state()
            idx += 2
        return pos


class RandomPolicy(PredPreyPolicy):
    # constructor
    def __init__(self, n_actions, rng):
        self.rng = rng
        self.n_actions = n_actions

    def set_rng(self, rng):
        self.rng = rng

    def act(self, state):
        return self.rng.integers(self.n_actions)


class PredPrey:
    def __init__(
        self,
        x_size: int = 16,
        y_size: int = 16,
        max_cycles: int = 500,
        shared_reward: bool = False,
        n_prey: int = 8,
        n_predators: int = 5,
        n_grasses: int = 30,
        obs_range: int = 7,
        freeze_grasses: bool = False,
        catch_reward: float = 5.0,
        urgency_reward: float = -0.1,
        render_mode=None,
        moore_neighborhood_prey: bool = False,        
        moore_neighborhood_grasses: bool = False,
        predator_controller: Optional[PredPreyPolicy] = None,  
        grass_controller: Optional[PredPreyPolicy] = None,  #??? find out what it is doing
        prey_controller: Optional[PredPreyPolicy] = None,

    ):
        
        self.x_size = x_size
        self.y_size = y_size
        self.max_cycles = max_cycles
        self.shared_reward = shared_reward
        self.n_grasses = n_grasses
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.obs_range = obs_range
        self.freeze_grasses = freeze_grasses
        self.catch_reward = catch_reward
        self.urgency_reward = urgency_reward
        self.render_mode = render_mode
        self.moore_neighborhood_prey = moore_neighborhood_prey
        self.moore_neighborhood_grasses = moore_neighborhood_grasses

        self._seed()
        self.num_agents = self.n_prey
        self.local_ratio = 1.0 - float(self.shared_reward) #TODO make continious?
        self.latest_reward_state = np.array([0.0 for _ in range(self.num_agents)])
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.obs_offset = int((self.obs_range - 1) / 2)

        # different types of agents 
        # TODO: "agent" "wall" can be terminated if no obstacles in grid exist. 
        # Observation can be reduced with one chanel in that case
        self.agent_type_names = ["predator", "prey", "grass"]
        self.predator_type_nr = self.agent_type_names.index("predator")
        self.prey_type_nr = self.agent_type_names.index("prey")
        self.grass_type_nr = self.agent_type_names.index("grass")
        #print("self.prey_type_nr ",self.prey_type_nr)
        #print("self.grass_type_nr ",self.grass_type_nr)
        self.nr_observation_channels = len(self.agent_type_names)
        
        # lists of agents
        self.predators = [] # list of all predators
        self.prey = [] # list of all prey
        self.grasses = [] # list of all grasses
        self.list_of_agent_type_lists = [self.predators, self.prey, self.grasses]

        self.n_actions_predators = 0
        self.n_actions_walls = 0
        self.n_actions_prey = 9 if self.moore_neighborhood_prey else 5
        self.n_actions_grasses = 9 if self.moore_neighborhood_grasses else 5 

        #controllers: still to find out how this is working
        if self.freeze_grasses:
            self.grass_controller = (
                # SingleActionPolicy(4) is NO move in both Von Nemann as well as Moore neighborhoods
                SingleActionPolicy(4)
                if grass_controller is None
                else grass_controller
            )
            self.prey_controller = (
                SingleActionPolicy(4)
                if prey_controller is None
                else prey_controller
            )
            self.predator_controller = (
                SingleActionPolicy(4)
                if predator_controller is None
                else predator_controller
            )
        else:
            
            self.predator_controller = (
                
                RandomPolicy(self.n_actions_predators, self.np_random)
                if predator_controller is None
                else predator_controller
            )
            self.grass_controller = (
                
                RandomPolicy(self.n_actions_grasses, self.np_random)
                if grass_controller is None
                else grass_controller
            )
            self.prey_controller = (
                RandomPolicy(self.n_actions_prey, self.np_random)
                if prey_controller is None
                else prey_controller
            )
 
        max_agents_overlap = max(self.n_prey, self.n_grasses, self.n_predators)
        obs_space = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.obs_range, self.obs_range, self.nr_observation_channels),
            dtype=np.float32,
        )
        act_space = spaces.Discrete(self.n_actions_prey)
        self.action_space = [act_space for _ in range(self.n_prey)]

        self.observation_space = [obs_space for _ in range(self.n_prey)]

        self.screen = None

        self.surround_mask = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

        self.pixel_scale = 30

        self.frames = 0
        #self.reset()

   
    def create_agents(
            self, 
            n_agents,
            agent_type_nr,
            obs_range, 
            randomizer, 
            flatten=False, 
            moore_neighborhood=True
            ):
        """
        Creates "n_agents" agent instances of type "agent_type_nr" in a "state_grid" with channel 
        "agent_type_nr" and adds those agents to the existing list of agentts. List can also be empty.
        """
        #print("CREATE AGENTS STARTS")
        agents = []
        # TODO change if's into CASE
        if agent_type_nr==1: # 1=prey
            agents = self.prey
            #print("prey CREATED")
            """
            if not self.prey:
                print("New preyes added to existing list of prey")
            """
        elif agent_type_nr==2: # 2=grasses
            agents = self.grasses 
            #print("grasses CREATED")
            """
            if not self.grasses:
                print("New grasses added to existing list of grasses")
            """
        elif agent_type_nr==0: # 0=predators
            agents = self.predators 
            #print("grasses CREATED")
            """
            if not self.grasses:
                print("New grasses added to existing list of grasses")
            """
        agent_type_name = self.agent_type_names[agent_type_nr]
        #print(agent_type_name)

        for i in range(n_agents):            
            agent_id_name = agent_type_name + "_" + str(i)
            # create the random position on the grid of the agent
            xinit, yinit = (randomizer.integers(0, self.x_size), randomizer.integers(0, self.y_size))      

            agent = DiscreteAgent(
                self.x_size, 
                self.y_size, 
                agent_type_nr, 
                agent_id_name,
                obs_range=obs_range, 
                flatten=flatten, 
                moore_neighborhood=moore_neighborhood
            )
            agent.set_position(xinit, yinit)
            agent_type_nr = 1
            agents.append(agent)
        return agents


    def reset(self):
        #print("RESET STARTS")
        # empty agent lists
        self.prey =[]
        self.grasses = []
        self.predators = []
        # set all "dones" False; TODO change tot "done"? and additionally "self.prey_gone"=False?
        self.grasses_gone = np.array([False for i in range(self.n_grasses)])
        self.predators_gone = np.array([False for i in range(self.n_grasses)])


        self.model_state = np.zeros((self.nr_observation_channels,) + (self.x_size, self.y_size), dtype=np.float32)

        self.grasses_gone.fill(False)
        self.predators_gone.fill(False)
     
        self.prey = self.create_agents(
            self.n_prey, self.prey_type_nr, self.obs_range, self.np_random, moore_neighborhood=self.moore_neighborhood_prey

        )

        self.prey_layer = AgentLayer(self.x_size, self.y_size, self.prey)

        self.grasses = self.create_agents(
            self.n_grasses, self.grass_type_nr, self.obs_range, self.np_random, moore_neighborhood=self.moore_neighborhood_grasses
        )
        #print(self.state_grid[self.grass_type_nr])
        self.grass_layer = AgentLayer(self.x_size, self.y_size, self.grasses)

        self.predators = self.create_agents(
            self.n_predators, self.predator_type_nr, self.obs_range, self.np_random, moore_neighborhood=self.moore_neighborhood_grasses
        )
        self.predators_layer = AgentLayer(self.x_size, self.y_size, self.predators)


        self.latest_reward_state = np.array([0.0 for _ in range(self.num_agents)])        
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.model_state[1] = self.prey_layer.get_state_matrix()
        self.model_state[2] = self.grass_layer.get_state_matrix()

        self.frames = 0

        return self.safely_observe(0)


    def observation_space(self, agent):
        return self.observation_spaces[agent]


    def action_space(self, agent):
        return self.action_spaces[agent]


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    @property
    def agents(self):
        return self.prey

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        try:
            policies = [self.grass_controller, self.prey_controller, self.predator_controller]
            for policy in policies:
                try:
                    policy.set_rng(self.np_random)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return [seed_]

    def get_param_values(self):
        return self.__dict__

    def step_agent(self, agent, is_last):
        """
        -try to incorporate the grasses as agents as well
        """
        pass


    def step(self, action, agent_id, is_last):
        """
        what happens here? 
        -the action nr gets executed on the prey agent with agent_id
        -1) the prey agent with agent_id moves
        -2) removes grasses which are at new spot in last step of AEC
        """
        agent_layer = self.prey_layer
        opponent_layer = self.grass_layer
        opponent_controller = self.grass_controller

        # actual action application, change the prey layer
        agent_layer.move_agent(agent_id, action)

        # Update only the prey layer
        self.model_state[1] = self.prey_layer.get_state_matrix()

        self.latest_reward_state = np.array([0.0 for _ in range(self.num_agents)])

        if is_last:
            # Possibly change the grass layer
            prey_who_remove = self.remove_agents()

            for i in range(opponent_layer.n_agents()):
                # controller input should be an observation, but doesn't matter right now
                a = opponent_controller.act(self.model_state)
                opponent_layer.move_agent(i, a)

            self.latest_reward_state += self.catch_reward * prey_who_remove
            self.latest_reward_state += self.urgency_reward
            self.frames = self.frames + 1

        # Update the remaining layers
        self.model_state[2] = self.grass_layer.get_state_matrix()

        global_val = self.latest_reward_state.mean()
        local_val = self.latest_reward_state
        self.latest_reward_state = (
            self.local_ratio * local_val + (1 - self.local_ratio) * global_val
        )

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
                col = (0, 0, 0)
                pygame.draw.rect(self.screen, col, pos)

    def draw_prey_observations(self):
        for i in range(self.prey_layer.n_agents()):
            x, y = self.prey_layer.get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(128)
            patch.fill((255, 152, 72))
            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )

    def draw_prey(self):
        for i in range(self.prey_layer.n_agents()):
            x, y = self.prey_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )

            col = (0, 0, 255) # blue

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_grasses(self):
        for i in range(self.grass_layer.n_agents()):
            x, y = self.grass_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (0, 128, 0) # green
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_predators(self):
        for i in range(self.predators_layer.n_agents()):
            x, y = self.predators_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0) # red

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_agent_counts(self):
        font = pygame.font.SysFont("Comic Sans MS", self.pixel_scale * 2 // 3)

        agent_positions = defaultdict(int)
        grass_positions = defaultdict(int)

        for i in range(self.grass_layer.n_agents()):
            x, y = self.grass_layer.get_position(i)
            grass_positions[(x, y)] += 1

        for i in range(self.prey_layer.n_agents()):
            x, y = self.prey_layer.get_position(i)
            agent_positions[(x, y)] += 1

        for x, y in grass_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = grass_positions[(x, y)]
            count_text: str
            if agent_count < 1:
                count_text = ""
            elif agent_count < 10:
                count_text = str(agent_count)
            else:
                count_text = "+"

            text = font.render(count_text, False, (0, 255, 255))

            self.screen.blit(text, (pos_x, pos_y))

        for x, y in agent_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = agent_positions[(x, y)]
            count_text: str
            if agent_count < 1:
                count_text = ""
            elif agent_count < 10:
                count_text = str(agent_count)
            else:
                count_text = "+"

            text = font.render(count_text, False, (255, 255, 0))

            self.screen.blit(text, (pos_x, pos_y - self.pixel_scale // 2))

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
                pygame.display.set_caption("PredatorPreyGrass")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )

        self.draw_model_state()

        self.draw_prey_observations()

        self.draw_grasses()
        self.draw_prey()
        self.draw_predators()
        self.draw_agent_counts()

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
        capture = pygame.surfarray.array3d(self.screen)

        xl, xh = -self.obs_offset - 1, self.x_size + self.obs_offset + 1
        yl, yh = -self.obs_offset - 1, self.y_size + self.obs_offset + 1

        window = pygame.Rect(xl, yl, xh, yh)
        subcapture = capture.subsurface(window)

        pygame.image.save(subcapture, file_name)


    @property
    def is_terminal(self):
        # ev = self.grass_layer.get_state_matrix()  # grass positions
        # if np.sum(ev) == 0.0:
        if self.grass_layer.n_agents() == 0:
            return True
        return False

    def update_ally_controller(self, controller):
        self.ally_controller = controller

    def update_opponent_controller(self, controller):
        self.opponent_controller = controller

    def n_agents(self):
        return self.prey_layer.n_agents()

    def safely_observe(self, i):
        agent_layer = self.prey_layer
        obs = self.collect_obs(agent_layer, i)
        return obs

    def collect_obs(self, agent_layer, i):
        for j in range(self.n_agents()):
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # returns a flattened array of all the observations
        obs = np.zeros((3, self.obs_range, self.obs_range), dtype=np.float32)
        obs[0].fill(1.0)  # border walls set to -0.1?
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        obs[0:3, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:3, xlo:xhi, ylo:yhi])
        """
        print()
        print("observation agent nr ", agent_idx)
        print(obs[3])
        """

        return obs

    def obs_clip(self, x, y):
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(
            np.clip(yld, -self.obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def remove_agents(self):
        removed_evade = []

        ai = 0
        rems = 0
        prey_who_remove = np.zeros(self.n_prey, dtype=bool)
        for i in range(self.n_grasses):
            if self.grasses_gone[i]:
                continue
            x, y = self.grass_layer.get_position(ai)
            if self.model_state[1, x, y] > 0:   # prey on spot of grass
                # remove grass
                removed_evade.append(ai - rems)
                self.grasses_gone[i] = True
                rems += 1
                for j in range(self.n_prey):  # identify all prey on the spot of the grass
                    xpp, ypp = self.prey_layer.get_position(j)
                    if xpp == x and ypp == y:
                        prey_who_remove[j] = True
            ai += 1

        for ridx in removed_evade:
            self.grass_layer.remove_agent(ridx)
        return prey_who_remove


_env = PredPrey

__all__ = ["env", "parallel_env", "raw_env"]


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predprey_v7",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        self.env = _env(*args, **kwargs) #  this calls the code from PredPrey
        #rendering
        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        #agents
        self.agents = ["prey_" + str(a) for a in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        #print("self.env.action_space ", self.env.action_space)
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.steps = 0
        self.closed = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.env.reset()  # this calls reset from PredPrey

    def close(self):
        if not self.closed:
            self.closed = True
            self.env.close()

    def render(self):
        if not self.closed:
            return self.env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_last()
        )
        for k in self.terminations:
            if self.env.frames >= self.env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.env.is_terminal
        for k in self.agents:
            self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]]
        self.steps += 1

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        o = self.env.safely_observe(self.agent_name_mapping[agent])
        return np.swapaxes(o, 2, 0)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]


class SingleActionPolicy(PredPreyPolicy):
    def __init__(self, a):
        self.action = a

    def act(self, state):
        return self.action

