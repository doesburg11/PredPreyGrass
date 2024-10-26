

# discretionary libraries   
from predpreygrass.envs._so_predpreygrass_v0.so_predpreygrass_base_par import PredPreyGrass as predpreygrass

# external libraries
from pettingzoo import ParallelEnv
from pettingzoo.utils import (
    parallel_to_aec, 
    wrappers,
)
import numpy as np
import pygame

# wrapped AEC env class
def env(**kwargs):
    env = raw_env(render_mode="human")
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env
# non-wrapped AEC env class
def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


# paralel env class
class parallel_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "so_predpreygrass_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(self, *args, **kwargs):

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self.predpreygrass = predpreygrass(
            *args, **kwargs
        )  #  this calls the code from PredPreyGrass

        self.agents = self.predpreygrass.possible_agent_name_list
        self.possible_agents = self.agents[:]
        self.action_spaces = dict(zip(self.agents, self.predpreygrass.action_space))  # type: ignore
        self.observation_spaces = dict(zip(self.agents, self.predpreygrass.observation_space))  # type: ignore


    def reset(self, seed=None, options=None):
        if seed is not None:
            self.predpreygrass._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]

        self.possible_agents = self.agents[:]
        self.agent_name_to_index_mapping = dict(
            zip(self.agents, list(range(self.num_agents)))
        )

        # spaces
        self.action_spaces = dict(zip(self.agents, self.predpreygrass.action_space))  # type: ignore
        self.observation_spaces = dict(zip(self.agents, self.predpreygrass.observation_space))  # type: ignore
        self.steps = 0
        # this method "reset"
        # initialise rewards and observations
        self.rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self._cumulative_rewards = dict(
            zip(self.agents, 
                [0 for _ in self.agents])
        )
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.predpreygrass.reset()  # this calls reset from PredPreyGrass
        self.num_moves = 0

        observations = {agent: self.observe(agent) for agent in self.agents}

        return observations, self.infos

    def close(self):
        if not self.closed:
            self.closed = True
            self.predpreygrass.close()

    def render(self):
        if not self.closed:
            return self.predpreygrass.render()

    def step(self, actions):
        self.predpreygrass.step(actions)

        for agent_name in self.agents:
            self.rewards[agent_name] = self.predpreygrass.rewards[agent_name]
        if self.render_mode == "human":
            self.render()
            #self.predpreygrass.check(2) # TODO implement render method in ansi

        observations = {agent: self.observe(agent) for agent in self.agents}       

        return observations, self.rewards, self.terminations, self.truncations, self.infos

    def observe(self, agent_name):
        agent_instance = self.predpreygrass.agent_name_to_instance_dict[agent_name]
        obs = self.predpreygrass.observe(agent_name)
        observation = np.swapaxes(obs, 2, 0)  # type: ignore
        # "black death": return observation of only zeros if agent is not alive
        if not agent_instance.is_active:
            observation = np.zeros(observation.shape)
        return observation

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
