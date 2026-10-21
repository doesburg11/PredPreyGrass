"""SELFish escape-based flocking environment.

This is a separate, compact reproduction of the 2019 paper's prey/predator
task. It is intentionally not based on the later Predator-Prey-Grass world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


DISCRETE_TURNS = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])


@dataclass(frozen=True)
class Settings:
    width: float = 40.0
    height: float = 40.0
    prey_count: int = 10
    neighbors: int = 5
    max_steps: int = 10_000
    prey_speed: float = 1.0
    predator_speed: float = 1.0
    predator_leap_speed: float = 1.5
    predator_turn_limit: float = 45.0
    predator_target_radius: float = 10.0
    predator_target_hold: int = 10
    predator_leap_interval: int = 50
    predator_leap_duration: int = 5
    agent_radius: float = 1.0
    catch_distance: float = 2.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("arena dimensions must be positive")
        if self.prey_count < 2:
            raise ValueError("prey_count must be at least 2")
        if self.neighbors < 1 or self.neighbors >= self.prey_count:
            raise ValueError("neighbors must be in [1, prey_count)")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.predator_target_hold < 1 or self.predator_leap_interval < 1:
            raise ValueError("predator timing settings must be positive")
        if self.predator_leap_duration < 0:
            raise ValueError("predator_leap_duration must be nonnegative")


def wrap_delta(delta: np.ndarray, width: float, height: float) -> np.ndarray:
    """Shortest signed displacement on the rectangular torus."""
    out = np.asarray(delta, dtype=float).copy()
    out[..., 0] -= width * np.round(out[..., 0] / width)
    out[..., 1] -= height * np.round(out[..., 1] / height)
    return out


def signed_degrees(angle: float) -> float:
    """Map an angle in degrees to (-180, 180]."""
    result = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if result == -180.0 else result


PeerPolicy = Callable[[np.ndarray], np.ndarray]


class SelfishFlockingEnv:
    """One focal learning prey and a homogeneous cohort on a torus.

    A caller supplies a frozen peer policy for the other prey at each step.
    This matches the paper's episode-wise policy sharing during training.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.rng = np.random.default_rng()
        self.steps = 0
        self.positions = np.empty((0, 2), dtype=float)
        self.headings = np.empty(0, dtype=float)
        self.predator_position = np.zeros(2, dtype=float)
        self.predator_heading = 0.0
        self.predator_target = 0
        self.target_age = 0
        self.leap_remaining = 0
        self.caught = 0
        self.focal = 0

    def reset(self, seed: int | None = None, focal: int = 0) -> np.ndarray:
        if not 0 <= focal < self.settings.prey_count:
            raise ValueError("focal must identify a prey in the population")
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.caught = 0
        self.focal = focal
        self.positions = self.rng.uniform(
            (0.0, 0.0),
            (self.settings.width, self.settings.height),
            size=(self.settings.prey_count, 2),
        )
        self.headings = self.rng.uniform(0.0, 360.0, size=self.settings.prey_count)
        self.predator_position = self.rng.uniform(
            (0.0, 0.0), (self.settings.width, self.settings.height)
        )
        self.predator_heading = float(self.rng.uniform(0.0, 360.0))
        self.predator_target = int(self.rng.integers(self.settings.prey_count))
        self.target_age = 0
        self.leap_remaining = 0
        return self.observation(focal)

    def _distance_and_bearing(self, origin: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        delta = wrap_delta(target - origin, self.settings.width, self.settings.height)
        distance = float(np.hypot(delta[0], delta[1]))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0]))) % 360.0
        return distance, bearing

    def observation(self, agent: int) -> np.ndarray:
        """Rows are predator, self, and nearest conspecifics; each row is
        normalized (distance, relative bearing, absolute orientation)."""
        s = self.settings
        rows = [self._entity_row(agent, self.predator_position, self.predator_heading)]
        rows.append(np.array([0.0, 0.0, self.headings[agent] / 360.0], dtype=np.float32))
        delta = wrap_delta(self.positions - self.positions[agent], s.width, s.height)
        distances = np.hypot(delta[:, 0], delta[:, 1])
        distances[agent] = np.inf
        for other in np.argsort(distances)[: s.neighbors]:
            rows.append(self._entity_row(agent, self.positions[other], self.headings[other]))
        return np.concatenate(rows).astype(np.float32)

    def _entity_row(self, agent: int, position: np.ndarray, heading: float) -> np.ndarray:
        s = self.settings
        distance, bearing = self._distance_and_bearing(self.positions[agent], position)
        relative = signed_degrees(bearing - self.headings[agent])
        return np.array(
            [
                min(distance / max(s.width, s.height), 1.0),
                (relative + 180.0) / 360.0,
                (heading % 360.0) / 360.0,
            ],
            dtype=np.float32,
        )

    def _choose_predator_target(self) -> None:
        s = self.settings
        if self.target_age < s.predator_target_hold:
            self.target_age += 1
            return
        distances = np.hypot(
            *np.moveaxis(
                wrap_delta(self.positions - self.predator_position, s.width, s.height),
                -1,
                0,
            )
        )
        nearby = np.flatnonzero(distances <= s.predator_target_radius)
        if len(nearby) > 1:
            self.predator_target = int(self.rng.choice(nearby))
        else:
            self.predator_target = int(np.argmin(distances))
        self.target_age = 0

    def _advance_predator(self) -> None:
        s = self.settings
        self._choose_predator_target()
        _, bearing = self._distance_and_bearing(
            self.predator_position, self.positions[self.predator_target]
        )
        turn = np.clip(
            signed_degrees(bearing - self.predator_heading),
            -s.predator_turn_limit,
            s.predator_turn_limit,
        )
        self.predator_heading = (self.predator_heading + turn) % 360.0
        if self.steps % s.predator_leap_interval == 0:
            self.leap_remaining = s.predator_leap_duration
        speed = s.predator_leap_speed if self.leap_remaining > 0 else s.predator_speed
        radians = np.radians(self.predator_heading)
        self.predator_position = (
            self.predator_position + speed * np.array([np.cos(radians), np.sin(radians)])
        ) % np.array([s.width, s.height])
        self.leap_remaining = max(0, self.leap_remaining - 1)

    def step(
        self,
        action: int | float,
        peer_policy: PeerPolicy | None = None,
        *,
        discrete: bool = True,
        terminate_on_focal_catch: bool = True,
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Advance the episode by one frame.

        If ``discrete`` is false, ``action`` is a continuous turn in [-1, 1],
        scaled to the predator's ±45-degree turn range used by DDPG.
        """
        s = self.settings
        if self.steps >= s.max_steps:
            raise RuntimeError("step called after the episode has finished; call reset")
        if discrete:
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
                raise ValueError("discrete action must be an integer")
            if not 0 <= int(action) < len(DISCRETE_TURNS):
                raise ValueError(f"discrete action must be in [0, {len(DISCRETE_TURNS)})")
            focal_turn = float(DISCRETE_TURNS[int(action)])
        else:
            focal_turn = float(np.clip(action, -1.0, 1.0) * s.predator_turn_limit)

        turns = np.zeros(s.prey_count, dtype=float)
        turns[self.focal] = focal_turn
        peers = [agent for agent in range(s.prey_count) if agent != self.focal]
        if peer_policy is not None:
            peer_observations = np.stack([self.observation(agent) for agent in peers])
            peer_actions = np.asarray(peer_policy(peer_observations), dtype=float).reshape(-1)
            if len(peer_actions) != len(peers):
                raise ValueError("peer_policy must return one action per peer observation")
            if discrete:
                if (
                    not np.all(np.isfinite(peer_actions))
                    or not np.all(peer_actions == np.floor(peer_actions))
                    or np.any((peer_actions < 0) | (peer_actions >= len(DISCRETE_TURNS)))
                ):
                    raise ValueError("discrete peer actions must be integers in [0, 5)")
                turns[peers] = DISCRETE_TURNS[peer_actions.astype(int)]
            else:
                if not np.all(np.isfinite(peer_actions)):
                    raise ValueError("continuous peer actions must be finite")
                turns[peers] = np.clip(peer_actions, -1.0, 1.0) * s.predator_turn_limit
        else:
            turns[peers] = (
                self.rng.choice(DISCRETE_TURNS, size=len(peers))
                if discrete
                else self.rng.uniform(-1.0, 1.0, size=len(peers)) * s.predator_turn_limit
            )

        # Actions are simultaneous: every prey moves from the same pre-step state.
        radians = np.radians(self.headings + turns)
        self.headings = (self.headings + turns) % 360.0
        self.positions = (
            self.positions
            + s.prey_speed * np.stack((np.cos(radians), np.sin(radians)), axis=1)
        ) % np.array([s.width, s.height])
        self.steps += 1
        self._advance_predator()

        distances = np.hypot(
            *np.moveaxis(
                wrap_delta(self.positions - self.predator_position, s.width, s.height),
                -1,
                0,
            )
        )
        newly_caught = np.flatnonzero(distances < s.catch_distance)
        self.caught += len(newly_caught)
        focal_caught = bool(np.any(newly_caught == self.focal))
        for agent in newly_caught:
            self.positions[agent] = self.rng.uniform(
                (0.0, 0.0), (s.width, s.height)
            )
            self.headings[agent] = float(self.rng.uniform(0.0, 360.0))

        done = (terminate_on_focal_catch and focal_caught) or self.steps >= s.max_steps
        reward = -1000.0 if focal_caught else 1.0
        info = {
            "caught": int(len(newly_caught)),
            "focal_caught": focal_caught,
            "steps": self.steps,
        }
        return self.observation(self.focal), reward, done, info
