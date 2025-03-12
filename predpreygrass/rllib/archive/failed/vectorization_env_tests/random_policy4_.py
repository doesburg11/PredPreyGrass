#from works_renderer import GridVisualizer
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer

from predpreygrass4_ import PredPreyGrass  # Import your custom environment

from time import sleep

verbose = False

if __name__ == "__main__":
    # Grid size
    env = PredPreyGrass()
    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=41)

    print("\nRESET:")
    env._print_grid_state()
    env._print_grid_from_state()

    #print(f"Initial obsrvations: {observations}")
    
    # Get the grid size
    grid_size = (env.grid_size, env.grid_size)
    
    # Combine predator, prey, and grass agents
    all_agents = env.agents + env.grass_agents
    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)
    for step in range(50):  # Arbitrary large number to test termination
        # Generate random actions for all agents
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        #print(f"Step {step}: {action_dict}")
        print(f"STEP {step}:")
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        env._print_grid_state()
        env._print_grid_from_state()
        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}
        # Update visualization
        visualizer.update(merged_positions,step)
        sleep(0.1)  # Slow down visualization
        pass
    visualizer.close()
    
    
