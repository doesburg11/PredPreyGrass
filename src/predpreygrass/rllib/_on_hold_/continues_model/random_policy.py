"""
Random policy for the PredPreyGrass environment.
No backward stepping is implemented in this version,
because that is pointless for debugging and testing
with a random policy.
"""
from predpreygrass.rllib._on_hold_continues_model.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib._on_hold_continues_model.config.config_env_random import config_env
from predpreygrass.rllib._on_hold_continues_model.utils.pygame_grid_renderer_rllib import PyGameRenderer

# external libraries
import pygame


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    env = env_creator(config_env)
    observations, _ = env.reset(seed=config_env.get("seed", 42))

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, enable_speed_slider=False, enable_tooltips=False)
    clock = pygame.time.Clock()

    # Run loop until termination
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # --- Step forward using random actions ---
        action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(action_dict)

        # --- Update visualizer ---
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
        )

        terminated = any(terminations.values())
        truncated = any(truncations.values())

        # Frame rate control
        clock.tick(visualizer.target_fps)

    visualizer.close()
    env.close()
