
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer

from predpreygrass_11 import PredPreyGrass  # Import your custom environmfrom predpreygrass_11 import PredPreyGrass  # Import your custom environment

from time import sleep
import numpy as np


config = {
    "max_steps": 200,
    "reward_predator_catch": 15.0,
    "reward_prey_survive": 0.5,
    "penalty_predator_miss": -0.2,
    "penalty_prey_caught": -20.0,
}
env = PredPreyGrass(config)
grid_size = (env.grid_size, env.grid_size) 




if __name__ == "__main__":
    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=42)
    #print(f"Initial observation for predator_0: {observations['predator_0']}")


    # Combine predator, prey, and grass agents
    all_agents = env.agents + env.grass_agents

    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=1)

    for step in range(1000):  # Arbitrary large number to test termination
        #print(f"Step {step + 1}")
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        #print(f"Action dict: {action_dict}")
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        #print(f"Observations: {observations}")
        #print(f"Rewards: {rewards}")
        #print(f"Terminations: {terminations}")

        # Merge (learning) agent and (non-learning) grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}
        # Update visualization
        visualizer.update(merged_positions)

        # Debug information
        #print(f"Number of prey left: {env.num_prey}")
        #print(f"Terminations: {terminations['__all__']}, Truncations: {truncations['__all__']}")

        # Check if the environment is properly terminating
        if terminations["__all__"]:
            print("Environment terminated properly.")
            break

        sleep(0.1)  # Slow down visualization

    visualizer.close()
