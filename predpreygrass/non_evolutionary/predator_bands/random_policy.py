from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.predator_bands.utils.pygame_grid_renderer_rllib import PyGameRenderer
import pygame


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random-policy viewer for predator_bands.")
    parser.add_argument("--threats", type=int, default=0, help="number of roaming threats (default 0 = none)")
    parser.add_argument("--bands", type=int, default=None, help="number of bands (default: config, 3)")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    seed = args.seed
    env_config = {"num_threats": args.threats}
    if args.bands is not None:
        env_config["num_bands"] = args.bands
    env = env_creator(env_config)
    observations, _ = env.reset(seed=seed)
    active_agents = list(observations.keys())

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, ennable_speed_slider=False)

    clock = pygame.time.Clock()
    target_fps = 10

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
            fruit_positions=env.fruit_positions,
            fruit_energies=env.fruit_energies,
            agents_just_ate=env.agents_just_ate,
            step=env.current_step,
            agent_bands=env.agent_band,
            agent_fruit_stores=env.agent_fruit_store,
            threat_positions=env.threat_positions,
        )

        terminated = terminations.get("__all__", False)
        truncated = truncations.get("__all__", False)

        clock.tick(target_fps)

    visualizer.close()
    env.close()
