import abc
import numpy as np
from gymnasium import spaces

class Agent:
    def __new__(cls, *args, **kwargs):
        agent = super().__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return f"<{type(self).__name__} instance>"

class DiscreteAgent(Agent):
    # constructor
    def __init__(
        self,
        xs,
        ys,
        obs_range=3,
        n_channels=3,
        flatten=False,
        moore_neighborhood=False
    ):
        # n channels is the number of observation channels


        self.xs = xs
        self.ys = ys

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
        else:  #Newton neighborhood
            #print("Newton neighborhood")
            self.eactions = [
                0,  # move left
                1,  # move right
                2,  # move up
                3,  # move down
                4,  # stay
            ]  
            self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]

        self.current_pos = np.zeros(2, dtype=np.int32)  # x and y position
        self.last_pos = np.zeros(2, dtype=np.int32)
        self.temp_pos = np.zeros(2, dtype=np.int32)

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
        #print(a)
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
        """Initializes the AgentLayer class. 
        AgentLayer stores charteristics of an agent type (Predators, Pursuers and Evaders) 
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
        #print("agent ", agent_idx)
        #print("self.allies ", self.allies)
        return self.allies[agent_idx].step(action)

    def set_position(self, agent_idx, x, y):
        self.allies[agent_idx].set_position(x, y)

    def get_position(self, agent_idx):
        """Returns the position of the given agent."""
        """
        print()
        print("agent_idx ", agent_idx)
        print()
        """
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
    
# Implements multi-agent controllers
class PredPreyPolicy(abc.ABC):
    @abc.abstractmethod
    def act(self, state: np.ndarray) -> int:
        raise NotImplementedError


class RandomPolicy(PredPreyPolicy):
    # constructor
    def __init__(self, n_actions, rng):
        self.rng = rng
        self.n_actions = n_actions

    def set_rng(self, rng):
        self.rng = rng

    def act(self, state):
        return self.rng.integers(self.n_actions)


class SingleActionPolicy(PredPreyPolicy):
    def __init__(self, a):
        self.action = a

    def act(self, state):
        return self.action

