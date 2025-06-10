from predpreygrass.rllib.v2_speed.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.utils.renderer import MatPlotLibRenderer, EvolutionVisualizer

# External libraries

verbose_grid_state = False
verbose_observation = False
seed_value = 7  # Set seed for reproducibility

# Initialize tracking dictionary
speed_counts = {"speed_1_predator": [], "speed_2_predator": [], "speed_1_prey": [], "speed_2_prey": []}

if __name__ == "__main__":
    env = PredPreyGrass()

    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)

    observations, _ = env.reset(seed=seed_value)
    # for debugging puropses
    if verbose_grid_state:
        print("\nRESET:")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    # initialize the grid_visualizer
    grid_size = (env.grid_size, env.grid_size)
    all_entities = env.possible_agents + env.grass_agents
    grid_visualizer = MatPlotLibRenderer(grid_size, all_entities, trace_length=5)
    evolution_visualizer = EvolutionVisualizer()

    for step in range(env.max_steps):
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        evolution_visualizer.record_counts(env.agents)
        # for debugging puropses
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
        # for debugging puropses
        if verbose_grid_state:
            print(f"Step {step}:")
            print("-----------------------------------------")
            print("Actions : ", action_dict)
            env._print_grid_from_positions()
            env._print_grid_from_state()
            print("-----------------------------------------")

        # Update grid visualization
        merged_positions = {**env.agent_positions, **env.grass_positions}
        grid_visualizer.update(merged_positions, step)

        if terminations["__all__"]:
            print("Environment terminated by termination.")
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

        # sleep(0.1)

    grid_visualizer.close()
    # Plot the evolution of agent types
    evolution_visualizer.plot()
    env.close()
