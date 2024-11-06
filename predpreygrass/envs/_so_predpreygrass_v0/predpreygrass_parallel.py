# discretionary libraries   
from predpreygrass.envs._so_predpreygrass_v0.predpreygrass_base_parallel import PredPreyGrass as predpreygrass

# external libraries
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import numpy as np
import pygame


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(**kwargs):
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env

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
        self.predpreygrass = predpreygrass(*args, **kwargs)  
        self.agents = self.predpreygrass.possible_agent_name_list
        self.possible_agents = self.agents[:]
        self.action_spaces = dict(zip(self.agents, self.predpreygrass.action_space))  
        self.observation_spaces = dict(zip(self.agents, self.predpreygrass.observation_space)) 

    # "options" is required  by conversions.py
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.predpreygrass._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents[:]  # use slice to create a copy
        self.possible_agents = self.agents[:]
        self.agent_name_to_index_mapping = dict(
            zip(self.agents, list(range(self.num_agents)))
        )

        # spaces
        self.action_spaces = dict(zip(self.agents, self.predpreygrass.action_space))  
        self.observation_spaces = dict(zip(self.agents, self.predpreygrass.observation_space))  
        self.rewards = dict(zip(self.agents, [0.0 for _ in self.agents]))
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}        
        self.predpreygrass.reset()  
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        return self.observations, self.infos

    def step(self, actions):
        self.predpreygrass.step(actions)
        self.observations = {agent: self.observe(agent) for agent in self.agents}       
        self.rewards = {agent_name: self.predpreygrass.rewards[agent_name] for agent_name in self.agents}
        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def observe(self, agent_name):
        agent_instance = self.predpreygrass.agent_name_to_instance_dict[agent_name]
        obs = self.predpreygrass.observe(agent_name)
        observation = np.swapaxes(obs, 2, 0)  
        # return observation of only zeros if agent is not active ("black death")
        if not agent_instance.is_active:
            observation = np.zeros(observation.shape)
        # TODO 
        return observation
    """
    # TODO more efficent implementation according to chatGPT: avoid unnecessary swaps if inactive 
    def observe(self, agent_name):
        agent_instance = self.predpreygrass.agent_name_to_instance_dict[agent_name]
        obs = self.predpreygrass.observe(agent_name)

        # If agent is not active ("black death"), return zeros
        if not agent_instance.is_active:
            return np.zeros((obs.shape[2], obs.shape[1], obs.shape[0]))  # Shape after swapaxes

        # Otherwise, return the swapped observation
        return np.swapaxes(obs, 2, 0)
    """



    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
    
    def close(self):
        if not self.closed:
            self.closed = True
            self.predpreygrass.close()

    def render(self):
        if not self.closed:
            return self.predpreygrass.render()

    
