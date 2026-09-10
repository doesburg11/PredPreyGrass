"""Pack Hunt Opponent Shaping -- RLlib environment.

Fixed-population, scripted-prey environment for the intra-predator
producer-scrounger dilemma described in this module's README. Deliberately
does not model reproduction, death, or energy decay -- see README Section 3
for why. This is a first, minimal implementation to make the mechanic
watchable and testable; it has not yet been used for any training run.

Round mechanics (README Sections 5-6): a round ends either in a *catch*
(reward split among predators within `sharing_radius`) or, after
`round_timeout_steps` without one, in an *escape* (no reward). Either way the
prey respawns and the next round starts immediately -- the episode boundary
is a fixed step budget, not a fixed round count.

Engagement (README Section 5) is evaluated every step, not once per round:
a predator within `engagement_radius` of the prey pays `engagement_cost`
*that step* and counts toward that step's catch-probability roll. This means
sustained pursuit across a long round costs more than a late dash-in, which
is the intended effort-cost asymmetry, not an approximation of it.
"""

from typing import Optional

import gymnasium
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PackHuntEnv(MultiAgentEnv):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}
        self.config = config

        self.grid_size = config.get("grid_size", 10)
        self.n_predators = config.get("n_predators", 3)
        self.max_episode_steps = config.get("max_episode_steps", 500)
        self.round_timeout_steps = config.get("round_timeout_steps", 15)
        self.capture_radius = config.get("capture_radius", 1)
        self.engagement_radius = config.get("engagement_radius", 2)
        self.sharing_radius = config.get("sharing_radius", 3)
        self.engagement_cost = config.get("engagement_cost", 0.05)
        self.catch_prob_per_engaged = config.get("catch_prob_per_engaged", 0.25)
        self.capture_reward = config.get("capture_reward", 5.0)
        self.prey_respawn_min_distance = config.get("prey_respawn_min_distance", 3)
        self.gamma = config.get("gamma", 0.95)

        self.agents = self.possible_agents = [f"predator_{i}" for i in range(self.n_predators)]

        self._move_deltas = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

        obs_dim = 4 + 3 * (self.n_predators - 1) + 3
        self.observation_spaces = {
            a: gymnasium.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        self.action_spaces = {a: gymnasium.spaces.Discrete(5) for a in self.agents}

        self._rng = np.random.default_rng(config.get("seed"))

        # Populated by reset()/step(); read directly by the renderer.
        self.predator_positions = {}
        self.prey_position = (0, 0)
        self.current_step = 0
        self.round_step = 0
        self.round_index = 0
        self.last_round_engaged = {a: False for a in self.agents}
        self.last_round_outcome = 0.5  # 0=escape, 1=catch, 0.5=no prior round yet
        self.engaged_this_step = {a: False for a in self.agents}
        self.last_event = None  # "catch" | "escape" | None, for the renderer's flash
        self.last_event_step = -1000
        self.last_event_position = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        occupied = set()
        self.predator_positions = {}
        for a in self.agents:
            pos = self._random_free_cell(occupied)
            self.predator_positions[a] = pos
            occupied.add(pos)
        self.prey_position = self._respawn_prey_position()

        self.current_step = 0
        self.round_step = 0
        self.round_index = 0
        self.last_round_engaged = {a: False for a in self.agents}
        self.last_round_outcome = 0.5
        self.engaged_this_step = {a: False for a in self.agents}
        self.last_event = None
        self.last_event_step = -1000
        self.last_event_position = None

        observations = {a: self._build_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, action_dict):
        # 1. Move predators.
        for a, action in action_dict.items():
            dr, dc = self._move_deltas[int(action)]
            r, c = self.predator_positions[a]
            self.predator_positions[a] = (
                int(np.clip(r + dr, 0, self.grid_size - 1)),
                int(np.clip(c + dc, 0, self.grid_size - 1)),
            )

        # 2. Scripted prey: move to the neighboring cell (or stay) that
        #    maximizes distance to the nearest predator, ties broken randomly.
        self.prey_position = self._prey_evasion_move()

        # 3. Engagement + catch roll for this step.
        distances = {a: _manhattan(pos, self.prey_position) for a, pos in self.predator_positions.items()}
        engaged = {a: distances[a] <= self.engagement_radius for a in self.agents}
        self.engaged_this_step = engaged

        rewards = {a: 0.0 for a in self.agents}
        for a in self.agents:
            if engaged[a]:
                rewards[a] -= self.engagement_cost

        anyone_in_capture_range = any(distances[a] <= self.capture_radius for a in self.agents)
        round_ended = False
        if anyone_in_capture_range:
            k = sum(engaged.values())
            p_catch = 1.0 - (1.0 - self.catch_prob_per_engaged) ** k
            if self._rng.random() < p_catch:
                sharers = [a for a in self.agents if distances[a] <= self.sharing_radius]
                share = self.capture_reward / len(sharers)
                for a in sharers:
                    rewards[a] += share
                self.last_round_engaged = dict(engaged)
                self.last_round_outcome = 1.0
                self.last_event = "catch"
                round_ended = True

        self.round_step += 1
        if not round_ended and self.round_step >= self.round_timeout_steps:
            self.last_round_engaged = dict(engaged)
            self.last_round_outcome = 0.0
            self.last_event = "escape"
            round_ended = True

        if round_ended:
            # Recorded before respawning, purely so the renderer can freeze
            # on the true capture/escape location for a few frames instead
            # of jumping straight to the next round's fresh prey position --
            # observations always reflect the immediate, correct respawn.
            self.last_event_position = self.prey_position
            self.prey_position = self._respawn_prey_position()
            self.round_step = 0
            self.round_index += 1
            # +1: current_step is incremented below, *after* this block, but
            # the renderer always reads current_step post-increment -- this
            # keeps last_event_step in the same post-increment frame of
            # reference so "steps_since_event == 0" means what it says.
            self.last_event_step = self.current_step + 1

        self.current_step += 1
        truncated_all = self.current_step >= self.max_episode_steps
        terminations = {a: False for a in self.agents}
        terminations["__all__"] = False
        truncations = {a: truncated_all for a in self.agents}
        truncations["__all__"] = truncated_all

        observations = {a: self._build_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    def _build_observation(self, agent_id):
        g = float(self.grid_size)
        self_pos = self.predator_positions[agent_id]
        prey_pos = self.prey_position
        vec = [self_pos[0] / g, self_pos[1] / g, prey_pos[0] / g, prey_pos[1] / g]
        for other in self.agents:
            if other == agent_id:
                continue
            other_pos = self.predator_positions[other]
            vec += [
                other_pos[0] / g,
                other_pos[1] / g,
                1.0 if self.last_round_engaged.get(other, False) else 0.0,
            ]
        vec += [
            1.0 if self.last_round_engaged.get(agent_id, False) else 0.0,
            float(self.last_round_outcome),
            min(self.round_step / self.round_timeout_steps, 1.0),
        ]
        return np.asarray(vec, dtype=np.float32)

    def _random_free_cell(self, occupied):
        while True:
            pos = (int(self._rng.integers(0, self.grid_size)), int(self._rng.integers(0, self.grid_size)))
            if pos not in occupied:
                return pos

    def _respawn_prey_position(self):
        while True:
            pos = (int(self._rng.integers(0, self.grid_size)), int(self._rng.integers(0, self.grid_size)))
            if all(_manhattan(pos, p) >= self.prey_respawn_min_distance for p in self.predator_positions.values()):
                return pos

    def _prey_evasion_move(self):
        best_moves = []
        best_dist = -1
        for dr, dc in self._move_deltas.values():
            r = int(np.clip(self.prey_position[0] + dr, 0, self.grid_size - 1))
            c = int(np.clip(self.prey_position[1] + dc, 0, self.grid_size - 1))
            candidate = (r, c)
            nearest = min(_manhattan(candidate, p) for p in self.predator_positions.values())
            if nearest > best_dist:
                best_dist = nearest
                best_moves = [candidate]
            elif nearest == best_dist:
                best_moves.append(candidate)
        idx = int(self._rng.integers(0, len(best_moves)))
        return best_moves[idx]

    def close(self):
        pass
