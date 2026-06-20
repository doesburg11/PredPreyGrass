"""
Random-policy test runner for the network_reciprocity environment.

All agents act randomly (movement only). Cooperation is automatic — cooperator_prey
donate energy to adjacent prey each step; defector_prey never do.

Run this script to verify:
  1. The environment steps without errors.
  2. Cooperation stats are tracked correctly.
  3. Cooperator and defector populations evolve differently over time.

The PyGame window colour-codes prey:
  green  = cooperator_prey
  orange = defector_prey
  red    = predator
  yellow = grass
"""
from predpreygrass.network_reciprocity.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.network_reciprocity.config.config_env import config_env
from predpreygrass.base_environment.utils.pygame_grid_renderer_rllib import PyGameRenderer

import pygame


STATS_INTERVAL = 50   # print cooperation stats every N steps
N_EPISODES = 3        # number of episodes to run


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


def print_stats(stats: dict):
    print(
        f"Step {stats['step']:4d} | "
        f"cooperators={stats['cooperators']:3d}  defectors={stats['defectors']:3d}  "
        f"predators={stats['predators']:3d} | "
        f"coop_fraction={stats['cooperator_fraction']:.2f}  "
        f"clustering={stats['cooperator_clustering']:.2f}"
    )


if __name__ == "__main__":
    for episode in range(N_EPISODES):
        env = env_creator(config_env)
        observations, _ = env.reset(seed=episode)

        grid_size = (env.grid_size, env.grid_size)
        visualizer = PyGameRenderer(grid_size, ennable_speed_slider=False)

        clock = pygame.time.Clock()
        target_fps = 10

        print(f"\n=== Episode {episode + 1} ===")
        print_stats(env.get_cooperation_stats())

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(action_dict)

            visualizer.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                agents_just_ate=env.agents_just_ate,
                step=env.current_step,
            )

            if env.current_step % STATS_INTERVAL == 0:
                print_stats(env.get_cooperation_stats())

            terminated = terminations.get("__all__", False)
            truncated = truncations.get("__all__", False)

            clock.tick(target_fps)

        print(f"--- Episode {episode + 1} ended at step {env.current_step} ---")
        print_stats(env.get_cooperation_stats())

        visualizer.close()
        env.close()
