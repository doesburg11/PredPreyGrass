from predpreygrass.single_objective.envs.rllib.predpreygrass_16 import PredPreyGrass  # Import the optimized environment
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer

import numpy as np
from time import sleep

# Toggle verbose settings
verbose_grid_state = False
verbose_observation = False

if __name__ == "__main__":
    # Initialize environment
    env = PredPreyGrass()

    # Reset and get initial observations
    observations, _ = env.reset(seed=42)

    if verbose_grid_state:
        print("\nRESET:")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    # Grid size for visualization
    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.possible_agents + [f"grass_{i}" for i in range(env.initial_num_grass)]

    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)

    for step in range(env.max_steps):
        # Vectorized action selection (random sampling)
        action_array = np.random.randint(0, 5, size=len(env.agents))
        action_dict = {agent: action_array[i] for i, agent in enumerate(env.agents)}

        # Step the environment
        observations, rewards, terminations, truncations, info = env.step(action_dict)

        # Ensure agent positions are a NumPy array
        if isinstance(env.agent_positions, dict):  # Convert dictionary to NumPy array if needed
            agent_positions = np.vstack([env.agent_positions[agent] for agent in env.agents if agent in env.agent_positions])
        else:
            agent_positions = np.array(env.agent_positions)

        # Debugging agent observations
        if verbose_observation:
            for agent, pos in zip(env.agents, agent_positions):
                print(f"\nAgent: {agent} position: {tuple(pos)}")
                print("Walls")
                print(observations[agent][0])
                print("Predators")
                print(observations[agent][1])
                print("Prey")
                print(observations[agent][2])
                print("Grass")
                print(observations[agent][3])
                print()

        # Print grid state if verbose
        if verbose_grid_state:
            print(f"Step {step}:")
            print("-----------------------------------------")
            print("Actions : ", action_dict)
            env._print_grid_from_positions()
            env._print_grid_from_state()
            print("-----------------------------------------")

        # Fix: Only include active agents with valid positions
        merged_positions = {
            agent: tuple(pos)
            for agent, pos in zip(env.agents, agent_positions)
            if np.all(pos >= 0)  # Ensure valid position (skip inactive agents)
        }

        visualizer.update(merged_positions, step)

        # Handle environment termination
        if terminations["__all__"]:
            print("Environment terminated properly by termination.")
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

        sleep(0.1)  # Slow down visualization for clarity

    visualizer.close()
