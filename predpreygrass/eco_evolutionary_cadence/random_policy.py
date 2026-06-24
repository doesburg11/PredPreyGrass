"""
Random policy for the PredPreyGrass environment.
No backward stepping is implemented in this version,
because that is pointless for debugging and testing
with a random policy.
"""

from predpreygrass.eco_evolutionary.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.eco_evolutionary.config.config_env_eco_evolutionary import config_env
from predpreygrass.eco_evolutionary.utils.pygame_grid_renderer_rllib import PyGameRenderer

# external libraries
import pygame
import random
import numpy as np



def env_creator(config):
    return PredPreyGrass(config)


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
    env = env_creator(cfg)
    observations, _ = env.reset(seed=seed)

    # Seed each action space with a distinct but reproducible seed to avoid
    # agents sampling identical action sequences in lockstep.
    action_spaces = env.action_spaces
    assert action_spaces is not None
    for i, (agent_id, space) in enumerate(action_spaces.items()):
        space.seed(seed + i * 9973)

    # Debug: print one observation shape to confirm the configured channel count.
    if observations:
        first_agent = next(iter(observations))
        print(f"Sample observation shape for {first_agent}: {observations[first_agent].shape}")

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(
        grid_size,
        enable_speed_slider=False,
        enable_tooltips=False,
        predator_obs_range=cfg.get("predator_obs_range"),
        prey_obs_range=cfg.get("prey_obs_range"),
        show_fov=True,
        fov_alpha=40,
        fov_agents=["predator_0", "prey_0"],
    )
    clock = pygame.time.Clock()

    # Run loop until termination
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # --- Step forward using random actions ---
        action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
        env.step(action_dict)
        # print(f"Step {env.current_step}")
        # print(f"{terminations}")

        # --- Update visualizer ---
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
        )

        terminated =  env.terminations["__all__"]
        truncated = env.truncations["__all__"]

        # Frame rate control
        clock.tick(visualizer.target_fps)

    print(f"Episode ended at step: {env.current_step}")
    visualizer.close()
    env.close()
