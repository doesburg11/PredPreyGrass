from predpreygrass.envs._mo_predpreygrass_v0.mo_predpreygrass_base import PredPreyGrass as _env

import numpy as np

from gymnasium.utils import EzPickle
from pettingzoo.utils import agent_selector, wrappers
from momaland.utils.env import MOAECEnv

from momaland.utils.conversions import mo_aec_to_parallel
    
import pygame


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


#parallel_env = mo_aec_to_parallel(env)

def parallel_env(**kwargs):
    """Returns the wrapped env in `parallel` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A fully wrapped parallel env.
    """
    env = raw_env(**kwargs)
    env = mo_aec_to_parallel(env)
    return env

class raw_env(MOAECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predpreygrass_v0",
        "is_parallelizable": True,
        "render_fps": 5,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        self.render_mode = kwargs.get("render_mode")
        pygame.init()
        self.closed = False

        self._env = _env(
            *args, **kwargs
        )  #  this calls the code from mo_predpreygrass_base.py

        self.agents = self._env.possible_agent_name_list
        self.possible_agents = self.agents[:]
        self.action_spaces = dict(zip(self.agents, self._env.action_space))  # type: ignore
        self.observation_spaces = dict(zip(self.agents, self._env.observation_space))  # type: ignore
        self.reward_spaces = dict(zip(self.agents, self._env.reward_space))  # type: ignore


    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env._seed(seed=seed)
        self.steps = 0
        self.agents = self.possible_agents

        self.possible_agents = self.agents[:]
        self.agent_name_to_index_mapping = dict(
            zip(self.agents, list(range(self.num_agents)))
        )
        self._agent_selector = agent_selector(self.agents)

        # spaces
        self.action_spaces = dict(zip(self.agents, self._env.action_space))  # type: ignore
        self.observation_spaces = dict(zip(self.agents, self._env.observation_space))  # type: ignore
        self.steps = 0
        # this method "reset"
        # initialise rewards and observations
        self.rewards = self._env.agent_reward_dict
        self._cumulative_rewards = dict(
            zip(self.agents, 
            [ [0.0, 0.0] for _ in self.agents])
        )
        
 
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._env.reset()  # this calls reset from PredPreyGrass

    def close(self):
        if not self.closed:
            self.closed = True
            self._env.close()

    def render(self):
        if not self.closed:
            return self._env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        agent_instance = self._env.agent_name_to_instance_dict[agent]

        #self.rewards = dict(zip(self.agents, [ [0,0] for _ in self.agents]))

        self._env.step(action, agent_instance, self._agent_selector.is_last())

        for k in self.terminations:
            if self._env.n_aec_cycles >= self._env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = (
                    self._env.is_no_prey or self._env.is_no_predator
                )

        # clear rewards
        for agent_name in self.agents:
            self.rewards[agent_name] = self._env.agent_reward_dict[agent_name]
            if self.rewards[agent_name][0] > 0 or self.rewards[agent_name][1] > 0:
                #print(f"agent: {agent_name}, reward: {self.rewards[agent_name]}")
                pass

        self.steps += 1
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [ [] for _ in self.agents]))
        self._accumulate_rewards() 
        if self.render_mode == "human" and agent_instance.is_active:
            self.render()

    def observe(self, agent_name):
        agent_instance = self._env.agent_name_to_instance_dict[agent_name]
        obs = self._env.observe(agent_name)
        observation = np.swapaxes(obs, 2, 0)  # type: ignore
        # "black death": return observation of only zeros if agent is not alive
        if not agent_instance.is_active:
            observation = np.zeros(observation.shape)
        return observation

    def observation_space(self, agent: str):  
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
    
    def reward_space(self, agent: str):
        return self.reward_spaces[agent]
