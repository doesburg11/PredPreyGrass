from predpreygrass.non_evolutionary.base_environment_seasonal.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.base_environment_seasonal.utils.pygame_grid_renderer_rllib import PyGameRenderer
import pygame


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    seed = 3
    env = env_creator({})
    observations, _ = env.reset(seed=seed)
    active_agents = list(observations.keys())

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, ennable_speed_slider=False)

    clock = pygame.time.Clock()
    target_fps = 10  # Adjust frame rate

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in active_agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        active_agents = [
            a for a in observations
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]

        visualizer.update(
            agent_positions=env.agent_positions,
            grass_positions=env.grass_positions,
            agent_energies=env.agent_energies,
            grass_energies=env.grass_energies,
            agents_just_ate=env.agents_just_ate,
            step=env.current_step,
        )

        terminated = terminations.get("__all__", False)
        truncated = truncations.get("__all__", False)

        clock.tick(target_fps)

    visualizer.close()
    env.close()
