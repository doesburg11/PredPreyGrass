#from works_renderer import GridVisualizer
from predpreygrass.utils.renderer import MatPlotLibRenderer

from predpreygrass.rllib.predpreygrass_16 import PredPreyGrass  # Import your custom environment

from time import sleep
import numpy as np

verbose_grid_state = False
verbose_observation = False

seed_value = 42  # Set seed for reproducibility


if __name__ == "__main__":
    # Grid size
    env = PredPreyGrass()

    # Seed the action spaces
    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)

    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=seed_value)
    #print("Observation reset:")
    #print(observations)   
    if verbose_grid_state:
        print("\nRESET:")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    grid_size = (env.grid_size, env.grid_size)

    # Combine predator, prey, and grass agents
    all_agents = env.possible_agents + env.grass_agents

    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)

    for step in range(env.max_steps):  # Arbitrary large number to test termination
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        if verbose_observation:
            for agent in env.agents:
                print(f"\nAgent: {agent} position: {env.agent_positions[agent]}")
                print("Wallls")
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

        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}

        # Update visualization
        visualizer.update(merged_positions,step)

        # Check if the environment is properly terminating
        if terminations["__all__"]:
            print("Environment terminated properly by termination.")
            print("Episode rewards: ", {k: round(v, 1) for k, v in env.cumulative_rewards.items()})
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

        sleep(0.1)  # Slow down visualization
        pass

    visualizer.close()
