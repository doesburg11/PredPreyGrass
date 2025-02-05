#from works_renderer import GridVisualizer
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer

from predpreygrass_14 import PredPreyGrass  # Import your custom environment

from time import sleep

if __name__ == "__main__":
    # Grid size
    env = PredPreyGrass()

    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=42)
    #print("Observation reset:")
    #print(observations)   
    grid_size = (env.grid_size, env.grid_size)

    # Combine predator, prey, and grass agents
    all_agents = env.agents + env.grass_agents

    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)

    for step in range(env.max_steps):  # Arbitrary large number to test termination
        print(f"Step {step}:")
        print("-----------------------------------------")
        print("Agents:")
        print(env.agents)
        print()
        # Generate random actions for all agents
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        print("Actions:")
        print(action_dict)
        print()
        # Perform a step
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        print("Agent energies after full step:")
        print(env.agent_energies)
        print()
        print(f"Rewards step {step+1}: \n{rewards}")
        print("Accumulated rew:",{k: round(v, 1) for k, v in env.cumulative_rewards.items()})
        #print("Energies:",{k: round(v, 1) for k, v in env.agent_energies.items()})
        print()
        print(f"Positions agents step {step+1}:")
        print(env.agent_positions)
        #print("Observations step:")
        #print(observations)
        #print("Rewards:")
        #print(rewards)
        print()

        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}

        # Update visualization
        visualizer.update(merged_positions,step)

        # Debug information
        print(f"Number of predators left: {env.num_predators}")
        print(f"Number of prey left: {env.num_prey}")
        print(f"Number of agents left: {env.num_agents}")
        print(f"Number of grass left: {env.num_grass}")
        print("-----------------------------------------")
        #print(f"Terminations: {terminations['__all__']}, Truncations: {truncations['__all__']}")
        """
        if terminations["__all__"] or truncations["__all__"]:
            #print(f"Episode: Total:{env.episode_rewards}")
            print("Total episode: ", {k: round(v, 1) for k, v in env.episode_rewards.items()})
        """

        # Check if the environment is properly terminating
        if terminations["__all__"]:
            print("Environment terminated properly by termination.")
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

        sleep(0.1)  # Slow down visualization
        pass

    visualizer.close()
