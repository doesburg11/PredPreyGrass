# discretionary libraries   
from predpreygrass.single_objective.envs.base_env.predpreygrass_base import PredPreyGrassAECEnv as predpreygrass

# external libraries
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import numpy as np
import pygame


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predpreygrass_aec_v0",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self.predpreygrass = predpreygrass(*args, **kwargs)  
        self.agents = self.predpreygrass.possible_learning_agent_name_list
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: space for agent, space in zip(self.agents, self.predpreygrass.action_space)}
        self.observation_spaces = {agent: space for agent, space in zip(self.agents, self.predpreygrass.observation_space)}

    # "options" is required  by conversions.py
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.predpreygrass._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.action_spaces = {agent: space for agent, space in zip(self.agents, self.predpreygrass.action_space)}
        self.observation_spaces = {agent: space for agent, space in zip(self.agents, self.predpreygrass.observation_space)}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.predpreygrass.reset()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        agent_instance = self.predpreygrass.agent_name_to_instance_dict[agent]
        for agent_name in self.agents:
            self.rewards[agent_name] = 0
        self.predpreygrass.step(action, agent_instance, self._agent_selector.is_last())
        for k in self.terminations:
            if self.predpreygrass.n_cycles >= self.predpreygrass.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = (
                    self.predpreygrass.is_no_prey or self.predpreygrass.is_no_predator
                )
        self.rewards = {agent_name: self.predpreygrass.rewards[agent_name] for agent_name in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards() 
        if self.render_mode == "human" and agent_instance.is_active:
            self.render()

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

    def close(self):
        if not self.closed:
            self.closed = True
            self.predpreygrass.close()

    def render(self):
        if not self.closed:
            return self.predpreygrass.render()

