#from works_renderer import GridVisualizer
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer

from predpreygrass5_ import PredPreyGrass  # Import your custom environment

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
    step = 0
    while True:  # Run until termination condition is met
        print(f"STEP {step}:")
        
        # Generate random actions for all agents
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        print(f"terminations: {terminations}")
        
        env._print_grid_state()
        env._print_grid_from_state()
        
        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}
        # Update visualization
        visualizer.update(merged_positions, step)
        
        sleep(0.1)  # Slow down visualization
        
        # Check termination condition: Stop when all agents are terminated
        if all(terminations.values()):
            print(f"All agents terminated at step {step}. Stopping simulation.")
            break
        
        step += 1  # Increment step counter

    visualizer.close()