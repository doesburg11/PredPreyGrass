from collections import defaultdict
from typing import Optional

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo.predprey.predprey_3.discrete_agent import (
    DiscreteAgent, 
    AgentLayer,
    PredPreyPolicy,
    RandomPolicy,
    SingleActionPolicy,
)


class PredPrey:
    def __init__(
        self,
        x_size: int = 16,
        y_size: int = 16,
        max_cycles: int = 500,
        shared_reward: bool = False,
        n_evaders: int = 30,
        n_pursuers: int = 8,
        n_predators: int = 4,
        obs_range_pursuers: int = 7,
        obs_range_predators: int = 5,
        obs_range_evaders: int = 0,     
        n_catch: int = 2,
        freeze_evaders: bool = False,
        evader_controller: Optional[PredPreyPolicy] = None,
        pursuer_controller: Optional[PredPreyPolicy] = None,
        predator_controller: Optional[PredPreyPolicy] = None,
        tag_reward: float = 0.01,
        catch_reward: float = 5.0,
        urgency_reward: float = -0.1,
        surround: bool = True,
        render_mode=None,
        moore_neighborhood_evaders: bool = False,
        moore_neighborhood_pursuers: bool = True,
        moore_neighborhood_predators: bool = True,

    ):
        """In evade pursuit a set of pursuers must 'tag' a set of evaders.

        Required arguments:
            x_size, y_size: World size
            shared_reward: whether the rewards should be shared between all agents
            n_evaders
            n_pursuers
            obs_range_pursuers: how far each agent can see
        Optional arguments:
        pursuer controller: stationary policy of ally pursuers
        evader controller: stationary policy of opponent evaders

        tag_reward: reward for 'tagging' a single evader

        max_cycles: after how many frames should the game end
        n_catch: how surrounded evader needs to be, before removal
        freeze_evaders: toggle evaders move or not
        catch_reward: reward for pursuer who catches an evader
        urgency_reward: reward added in each step
        surround: toggles surround condition for evader removal
        The 2D 'map' has a rectangle building centered in the middle.
        The size of the retangle is defined by xb and yb: if xb=yb=1.0 then the 
        rectangle is 0. If xb=yb=0.0 then the rectangle covers the whole env.
        So xb and xy define essentially the free moveable space (from 0 to 1)


        """
        self.x_size = x_size
        self.y_size = y_size
        self.max_cycles = max_cycles
        self._seed()

        self.shared_reward = shared_reward
        self.local_ratio = 1.0 - float(self.shared_reward)
        self.moore_neighborhood_evaders = moore_neighborhood_evaders,
        self.moore_neighborhood_pursuers = moore_neighborhood_pursuers
        self.moore_neighborhood_predators = moore_neighborhood_predators

        self.n_evaders = n_evaders
        self.n_pursuers = n_pursuers
        self.n_predators = n_predators
        self.num_agents = self.n_pursuers + self.n_predators

        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        """
        print()
        print("initialization self.latest_reward_state ", self.latest_reward_state)
        print()
        """
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        # can see self.obs_range grids around them by default
        self.obs_range_pursuers = obs_range_pursuers
        self.obs_range_predators = obs_range_predators
        self.obs_range_evaders = obs_range_evaders
        # assert self.obs_range_pursuers % 2 != 0, "obs_range_pursuers should be odd"
        self.obs_offset_pursuers = int((self.obs_range_pursuers - 1) / 2)
        self.obs_offset_predators = int((self.obs_range_predators - 1) / 2)

        self.pursuers = self.create_agents(
            self.n_pursuers, self.obs_range_pursuers, self.np_random,
            moore_neighborhood=self.moore_neighborhood_pursuers
        )

        self.predators = self.create_agents(
            self.n_predators, self.obs_range_predators, self.np_random,
            moore_neighborhood=self.moore_neighborhood_predators
        )

        self.evaders = self.create_agents(
            self.n_evaders, self.obs_range_evaders, self.np_random, 
            moore_neighborhood=self.moore_neighborhood_evaders
        )

        self.agent_list = self.pursuers

        for a in self.predators:
            self.agent_list.append(a)        

        #print("len(self.agent_list) ",len(self.agent_list))

        self.predator_layer = AgentLayer(x_size, y_size, self.predators)

        self.pursuer_layer = AgentLayer(x_size, y_size, self.pursuers)

        self.agent_list_layer = AgentLayer(x_size, y_size, self.agent_list)

        #print(self.pursuer_layer)

        self.evader_layer = AgentLayer(x_size, y_size, self.evaders)

        print(self.evader_layer.get_state_matrix)

        self.n_catch = n_catch

        self.n_act_purs = self.pursuer_layer.get_nactions(0)
        self.n_act_pred = self.predator_layer.get_nactions(0)
        self.n_act_ev = self.evader_layer.get_nactions(0)

        #print("self.n_act_purs ",self.n_act_purs)
        #print("self.n_act_pred ",self.n_act_pred)

        self.freeze_evaders = freeze_evaders

        if self.freeze_evaders:
            self.evader_controller = (
                # SingleActionPolicy(4) is NO move in both Von Nemann as well as Moore neighborhoods
                SingleActionPolicy(4)
                if evader_controller is None
                else evader_controller
            )
            self.pursuer_controller = (
                SingleActionPolicy(4)
                if pursuer_controller is None
                else pursuer_controller
            )
            self.predator_controller = (
                SingleActionPolicy(4)
                if predator_controller is None
                else predator_controller
            )
        else:
            self.evader_controller = (
                RandomPolicy(self.n_act_purs, self.np_random)
                if evader_controller is None
                else evader_controller
            )
            self.pursuer_controller = (
                RandomPolicy(self.n_act_ev, self.np_random)
                if pursuer_controller is None
                else pursuer_controller
            )
            self.predator_controller = (
                RandomPolicy(self.n_act_purs, self.np_random)
                if predator_controller is None
                else predator_controller
            )

        self.current_agent_layer = np.zeros((x_size, y_size), dtype=np.int32)

        self.tag_reward = tag_reward

        self.catch_reward = catch_reward

        self.urgency_reward = urgency_reward

        self.nr_observation_channels = 4

        #self.ally_actions = np.zeros(self.n_act_purs, dtype=np.int32)
        #self.evader_actions = np.zeros(self.n_act_ev, dtype=np.int32)

        max_agents_overlap = max(self.n_pursuers, self.n_evaders, self.n_predators)

        obs_space_purs = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.obs_range_pursuers, self.obs_range_pursuers, self.nr_observation_channels),
            dtype=np.float32,
        )

        obs_space_pred = spaces.Box(
            low=0,
            high=max_agents_overlap,
            shape=(self.obs_range_predators, self.obs_range_predators, self.nr_observation_channels),
            dtype=np.float32,
        )

        act_space_purs = spaces.Discrete(self.n_act_purs)
        act_space_pred = spaces.Discrete(self.n_act_pred)

        # build multiagent action space
        action_space = []
        for _ in range(self.n_pursuers):
            action_space.append(act_space_purs)
        for _ in range(self.n_predators):
            action_space.append(act_space_pred)

        self.action_space = action_space
                       
        #print(self.action_space)


        #self.observation_space = [obs_space for _ in range(self.n_pursuers)]
        # build multiagent observation space
        observation_space = []
        for _ in range(self.n_pursuers):
            observation_space.append(obs_space_purs)
        for _ in range(self.n_predators):
            observation_space.append(obs_space_pred)
        
        self.observation_space = observation_space

        #print(self.observation_space)
        #self.act_dims = [self.n_act_purs for i in range(self.n_pursuers)]

        #print("self.act_dims ",self.act_dims)

        self.evaders_gone = np.array([False for i in range(self.n_evaders)])

        self.surround = surround

        self.render_mode = render_mode
        self.screen = None

        self.surround_mask = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])


        self.model_state = np.zeros((self.nr_observation_channels,) + (self.x_size, self.y_size), dtype=np.float32)
        """
        print()
        print("model state:")
        print(self.model_state)
        """


        self.pixel_scale = 30

        self.frames = 0
        self.reset()
    
    def create_agents(self, n_agents, obs_range, randomizer, flatten=False, moore_neighborhood=False):
        """
        Initializes the agents on a map (map_matrix).
        -n_agents: the number of agents to put on the map
        """
        agents = []
        for _ in range(n_agents):
            xinit, yinit = (randomizer.integers(0, self.x_size), randomizer.integers(0, self.y_size))      

            agent = DiscreteAgent(
                self.x_size, self.y_size, obs_range=obs_range, flatten=flatten, moore_neighborhood=moore_neighborhood
            )
            agent.set_position(xinit, yinit)
            agents.append(agent)
        return agents


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    ##################################################################
    # The functions below are the interface with MultiAgentSimulator #
    ##################################################################

    @property
    def agents(self):
        return self.pursuers

    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        try:
            policies = [self.evader_controller, self.pursuer_controller, self.predator_controller]
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

    def reset(self):
        self.evaders_gone.fill(False)
     
        self.pursuers = self.create_agents(
            self.n_pursuers, self.obs_range_pursuers, self.np_random, moore_neighborhood=self.moore_neighborhood_pursuers
        )

        self.predators = self.create_agents(
            self.n_predators, self.obs_range_predators, self.np_random, moore_neighborhood=self.moore_neighborhood_predators
        )

        self.agent_list = self.pursuers

        for a in self.predators:
            self.agent_list.append(a)        

        self.evaders = self.create_agents(
            self.n_evaders, self.obs_range_evaders, self.np_random, moore_neighborhood=self.moore_neighborhood_evaders
        )

        self.pursuer_layer = AgentLayer(self.x_size, self.y_size, self.pursuers)

        self.predator_layer = AgentLayer(self.x_size, self.y_size, self.predators)

        self.agent_list_layer = AgentLayer(self.x_size, self.y_size, self.agent_list)

        self.evader_layer = AgentLayer(self.x_size, self.y_size, self.evaders)
        """
        self.latest_reward_state = [0 for _ in range(self.num_agents)]
        print()
        print("second initialization self.latest_reward_state ", self.latest_reward_state)
        print()
        """
        self.latest_done_state = [False for _ in range(self.num_agents)]
        self.latest_obs = [None for _ in range(self.num_agents)]

        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[2] = self.evader_layer.get_state_matrix()
        self.model_state[3] = self.predator_layer.get_state_matrix()

        self.frames = 0

        return self.safely_observe(0)

    def step(self, action, agent_id, is_last):
        #print("agent_id ",agent_id)
        #agent_layer = self.pursuer_layer
        agent_layer = self.agent_list_layer

        #agent = self.agent_list[self.agent_name_mapping[self.agent_selection]]
        evader_layer = self.evader_layer
        evader_controller = self.evader_controller

        # actual action application, change the pursuer pr predator layer
        #print("agent_layer.n_agents() ",agent_layer.n_agents())
        agent_layer.move_agent(agent_id, action)

        # Update only the pursuer and the predator layer
        self.model_state[1] = self.pursuer_layer.get_state_matrix()
        self.model_state[3] = self.predator_layer.get_state_matrix()

        self.latest_reward_state = self.reward() / self.num_agents
        """
        print()
        print("self.latest_reward_state ", self.latest_reward_state)
        print()
        print("self.reward() ",self.reward())
        print()
        """


        if is_last:
            # Possibly change the evader layer
            ev_remove, pr_remove, pursuers_who_remove = self.remove_agents()

            for i in range(evader_layer.n_agents()):
                # controller input should be an observation, but doesn't matter right now
                a = evader_controller.act(self.model_state)
                evader_layer.move_agent(i, a)

            self.latest_reward_state += self.catch_reward * pursuers_who_remove
            self.latest_reward_state += self.urgency_reward
            self.frames = self.frames + 1

        # Update the remaining layers
        self.model_state[2] = self.evader_layer.get_state_matrix()

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

    def draw_pursuers_observations(self):
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range_pursuers, self.pixel_scale * self.obs_range_pursuers)
            )
            patch.set_alpha(128)
            patch.fill((255, 152, 72))
            ofst = self.obs_range_pursuers / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )

    def draw_pursuers(self):
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_evaders(self):
        for i in range(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (0, 0, 255)

            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def draw_agent_counts(self):
        font = pygame.font.SysFont("Comic Sans MS", self.pixel_scale * 2 // 3)

        agent_positions = defaultdict(int)
        evader_positions = defaultdict(int)

        for i in range(self.evader_layer.n_agents()):
            x, y = self.evader_layer.get_position(i)
            evader_positions[(x, y)] += 1

        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            agent_positions[(x, y)] += 1

        for x, y in evader_positions:
            (pos_x, pos_y) = (
                self.pixel_scale * x + self.pixel_scale // 2,
                self.pixel_scale * y + self.pixel_scale // 2,
            )

            agent_count = evader_positions[(x, y)]
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
                pygame.display.set_caption("Pursuit")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )

        self.draw_model_state()

        self.draw_pursuers_observations()

        self.draw_evaders()
        self.draw_pursuers()
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

        xl, xh = -self.obs_offset_pursuers - 1, self.x_size + self.obs_offset_pursuers + 1
        yl, yh = -self.obs_offset_pursuers - 1, self.y_size + self.obs_offset_pursuers + 1

        window = pygame.Rect(xl, yl, xh, yh)
        subcapture = capture.subsurface(window)

        pygame.image.save(subcapture, file_name)

    def reward(self):

        es = self.evader_layer.get_state_matrix()  # evader positions
        print(es)
        # rewards pursuers
        rewards = [
            self.tag_reward
            * np.sum(
                es[
                    np.clip(
                        self.pursuer_layer.get_position(i)[0]
                        + self.surround_mask[:, 0],
                        0,
                        self.x_size - 1,
                    ),
                    np.clip(
                        self.pursuer_layer.get_position(i)[1]
                        + self.surround_mask[:, 1],
                        0,
                        self.y_size - 1,
                    ),
                ]
            )
            for i in range(self.n_pursuers)
        ]
        """
        print()
        print("np.array(rewards) ",np.array(rewards))
        print()
        """
        
        return np.array(rewards)

    @property
    def is_terminal(self):
        # ev = self.evader_layer.get_state_matrix()  # evader positions
        # if np.sum(ev) == 0.0:
        if self.evader_layer.n_agents() == 0:
            return True
        return False

    def update_ally_controller(self, controller):
        self.ally_controller = controller

    def update_evader_controller(self, controller):
        self.evader_controller = controller

    def n_agents(self):
        return self.pursuer_layer.n_agents() + self.predator_layer.n_agents() #pd: augmented

    def safely_observe(self, i):
        agent_layer = self.pursuer_layer
        obs = self.collect_obs(agent_layer, i)
        return obs

    def collect_obs(self, agent_layer, i):
        """
        print()
        print("self.n_agents() ",self.n_agents())
        print("agent_layer.n_agents() ", agent_layer.n_agents())
        print()
        """
        for j in range(self.n_agents()):
            #print("i ", i, " j ",j)
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # returns a flattened array of all the observations
        obs = np.zeros((self.nr_observation_channels, self.obs_range_pursuers, self.obs_range_pursuers), dtype=np.float32)
        obs[0].fill(1.0)  # border walls set to -0.1?
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)
        obs[0:self.nr_observation_channels, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:self.nr_observation_channels, xlo:xhi, ylo:yhi])
        """
        print()
        print("observation agent nr ", agent_idx)
        print(obs)
        """        

        return obs

    def obs_clip(self, x, y):
        xld = x - self.obs_offset_pursuers
        xhd = x + self.obs_offset_pursuers
        yld = y - self.obs_offset_pursuers
        yhd = y + self.obs_offset_pursuers
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_offset_pursuers, 0)), abs(
            np.clip(yld, -self.obs_offset_pursuers, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def remove_agents(self):
        """Remove agents that are caught.

        Return tuple (n_evader_removed, n_pursuer_removed, purs_sur)
        purs_sur: bool array, which pursuers surrounded an evader
        """
        n_pursuer_removed = 0
        n_evader_removed = 0
        removed_evade = []
        removed_pursuit = []

        ai = 0
        rems = 0
        xpur, ypur = np.nonzero(self.model_state[1])
        purs_sur = np.zeros(self.n_pursuers, dtype=bool)
        for i in range(self.n_evaders):
            if self.evaders_gone[i]:
                continue
            x, y = self.evader_layer.get_position(ai)
            if self.surround:
                pos_that_catch = self.surround_mask + self.evader_layer.get_position(ai)
                truths = np.array(
                    [
                        np.equal([xi, yi], pos_that_catch).all(axis=1)
                        for xi, yi in zip(xpur, ypur)
                    ]
                )
                if np.sum(truths.any(axis=0)) == self.need_to_surround(x, y):
                    removed_evade.append(ai - rems)
                    self.evaders_gone[i] = True
                    rems += 1
                    tt = truths.any(axis=1)
                    for j in range(self.n_pursuers):
                        xpp, ypp = self.pursuer_layer.get_position(j)
                        tes = np.concatenate((xpur[tt], ypur[tt])).reshape(
                            2, len(xpur[tt])
                        )
                        tem = tes.T == np.array([xpp, ypp])
                        if np.any(np.all(tem, axis=1)):
                            purs_sur[j] = True
                ai += 1
            else:
                if self.model_state[1, x, y] >= self.n_catch:
                    # add prob remove?
                    removed_evade.append(ai - rems)
                    self.evaders_gone[i] = True
                    rems += 1
                    for j in range(self.n_pursuers):
                        xpp, ypp = self.pursuer_layer.get_position(j)
                        if xpp == x and ypp == y:
                            purs_sur[j] = True
                ai += 1

        ai = 0
        for i in range(self.pursuer_layer.n_agents()):
            x, y = self.pursuer_layer.get_position(i)
            # can remove pursuers probabilitcally here?
        for ridx in removed_evade:
            self.evader_layer.remove_agent(ridx)
            n_evader_removed += 1
        for ridx in removed_pursuit:
            self.pursuer_layer.remove_agent(ridx)
            n_pursuer_removed += 1
        return n_evader_removed, n_pursuer_removed, purs_sur

    def need_to_surround(self, x, y):
        """Compute the number of surrounding grid cells.

        Compute the number of surrounding grid cells in x,y position that are open
        (no wall or obstacle)
        """
        tosur = 4
        if x == 0 or x == (self.x_size - 1):
            tosur -= 1
        if y == 0 or y == (self.y_size - 1):
            tosur -= 1
        neighbors = self.surround_mask + np.array([x, y])
        for n in neighbors:
            xn, yn = n
            if not 0 < xn < self.x_size or not 0 < yn < self.y_size:
                continue
        return tosur
