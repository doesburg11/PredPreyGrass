#from works_renderer import GridVisualizer
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer

from predpreygrass.single_objective.envs.rllib.works_predpreygrass_1 import PredPreyGrass  # Import your custom environment

from time import sleep

if __name__ == "__main__":
    # Grid size
    env = PredPreyGrass()

    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=42)
    print("Initial Observations:")
    grid_size = (env.x_grid_size, env.y_grid_size)

    # Combine predator, prey, and grass agents
    all_agents = env.agents + env.grass_agents

    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=1)

    for step in range(1000):  # Arbitrary large number to test termination
        print(f"Step {step + 1}")
        # Generate random actions for all agents
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}

        # Perform a step
        observations, rewards, terminations, truncations, info = env.step(action_dict)

        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}

        # Update visualization
        visualizer.update(merged_positions)

        # Debug information
        print(f"Number of prey left: {env.num_prey}")
        print(f"Terminations: {terminations['__all__']}, Truncations: {truncations['__all__']}")

        # Check if the environment is properly terminating
        if terminations["__all__"]:
            print("Environment terminated properly.")
            break

        sleep(0.1)  # Slow down visualization

    visualizer.close()
