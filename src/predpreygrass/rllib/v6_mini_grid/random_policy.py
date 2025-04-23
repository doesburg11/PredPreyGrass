from predpreygrass.utils.renderer import MatPlotLibRenderer, CombinedEvolutionVisualizer
from predpreygrass.rllib.v6_mini_grid.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v6_mini_grid.config.config_env_random import config_env

from time import sleep
import numpy as np

verbose_grid_state = False
verbose_observation = False

seed_value = None # 42  # Set seed for reproducibility

if __name__ == "__main__":
    env = PredPreyGrass(config=config_env)

    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)

    observations, _ = env.reset(seed=seed_value)

    if verbose_grid_state:
        print("\nRESET:")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.possible_agents + env.grass_agents

    grid_visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5, show_gridlines=False, scale=2)
    combined_evolution_visualizer = CombinedEvolutionVisualizer()


    for step in range(env.max_steps):
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        combined_evolution_visualizer.record(
            agent_ids=env.agents,
            internal_ids=env.agent_internal_ids,
            agent_ages=env.agent_ages
        )

        if verbose_observation:
            for agent in env.agents:
                print(f"\nAgent: {agent} position: {env.agent_positions[agent]}")
                print("Walls")
                print(observations[agent][0])
                print("Predators")
                print(observations[agent][1])
                print("Prey")
                print(observations[agent][2])
                print("Grass")
                print(observations[agent][3])
                print()

        if verbose_grid_state:
            print(f"Step {step}:")
            print("-----------------------------------------")
            print("Actions : ", action_dict)
            env._print_grid_from_positions()
            env._print_grid_from_state()
            print("-----------------------------------------")

        merged_positions = {**env.agent_positions, **env.grass_positions}
        grid_visualizer.update(merged_positions, step)

        if terminations["__all__"]:
            print("Environment terminated by termination.")
            break
        if truncations["__all__"]:
            print("Environment terminated by truncation.")
            break

        #sleep(0.1)

    combined_evolution_visualizer.plot()
    grid_visualizer.close()


