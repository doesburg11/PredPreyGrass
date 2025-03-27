from predpreygrass.utils.renderer import MatPlotLibRenderer, CombinedEvolutionVisualizer
from predpreygrass.rllib.predpreygrass_rllib_env_moore_speed_age import PredPreyGrass

from time import sleep
import numpy as np

verbose_grid_state = False
verbose_observation = False

seed_value = 42  # Set seed for reproducibility

if __name__ == "__main__":
    env = PredPreyGrass()

    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)

    observations, _ = env.reset(seed=seed_value)

    if verbose_grid_state:
        print("\nRESET:")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.possible_agents + env.grass_agents

    grid_visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)
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
            print("Environment terminated properly by termination.")
            print("Episode rewards: ", {k: round(v, 1) for k, v in env.cumulative_rewards.items()})
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

        sleep(0.1)

    grid_visualizer.close()
    combined_evolution_visualizer.plot()
    #average_age_visualizer.plot()
    # Plot the evolution of agent types
    #evolution_visualizer.plot()
