"""Random-policy viewer for pack_hunt_opponent_shaping.

Runs the environment with uniformly random predator actions and renders it
with pygame, so the engagement/effort-cost/catch mechanic can be watched and
sanity-checked before any training code exists. No learning happens here.
"""

import random

import numpy as np
import pygame

from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.config.config_env_pack_hunt_opponent_shaping import (
    config_env,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.predpreygrass_rllib_env import (
    PackHuntEnv,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.utils.pygame_renderer import (
    PackHuntRenderer,
)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    cfg = dict(config_env)
    seed = cfg.get("seed")
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        cfg["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)

    env = PackHuntEnv(cfg)
    observations, _ = env.reset(seed=seed)
    for i, (agent_id, space) in enumerate(env.action_spaces.items()):
        space.seed(seed + i * 9973)

    renderer = PackHuntRenderer(env.grid_size)
    clock = pygame.time.Clock()

    n_catches = 0
    n_escapes = 0
    running = True
    truncated_all = False
    while running and not truncated_all:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(action_dict)

        if env.last_event_step == env.current_step:
            if env.last_event == "catch":
                n_catches += 1
            elif env.last_event == "escape":
                n_escapes += 1

        renderer.update(env)
        truncated_all = truncations["__all__"]
        clock.tick(renderer.target_fps)

    print(f"Episode ended at step {env.current_step}: {n_catches} catches, {n_escapes} escapes.")
    renderer.close()
    env.close()
