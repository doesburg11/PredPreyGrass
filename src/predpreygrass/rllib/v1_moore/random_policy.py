from predpreygrass.utils.renderer import MatPlotLibRenderer, PopulationChart
from predpreygrass.rllib.v1_moore.predpreygrass_rllib_env import PredPreyGrass

# external libraries

verbose_grid_state = False
verbose_observation = False
seed_value = 7  # set seed for reproducibility


if __name__ == "__main__":
    env = PredPreyGrass()
    # seed the action spaces
    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)
    # reset the environment and get initial observations
    observations, _ = env.reset(seed=seed_value)
    # for debugging puropses
    if verbose_grid_state:
        print("\nRESET:")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    # initialize the grid_visualizer
    grid_size = (env.grid_size, env.grid_size)
    all_entities = env.possible_agents + env.grass_agents
    grid_visualizer = MatPlotLibRenderer(grid_size, all_entities, trace_length=5, show_gridlines=False)

    population_chart = PopulationChart()

    for step in range(env.max_steps):  # Arbitrary large number to test termination
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)
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

        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}

        # Update grid visualization
        grid_visualizer.update(merged_positions, step)
        # Update population chart data
        population_chart.record(step, env.agents)

        if terminations["__all__"]:
            print("Environment terminated by termination.")
            print("Episode rewards: ", {k: round(v, 1) for k, v in env.cumulative_rewards.items()})
            break
        if truncations["__all__"]:
            print("Environment terminated by truncation.")
            break

        # sleep(0.1)  # Slow down visualization
        pass

    population_chart.plot()
    grid_visualizer.close()
    env.close()
