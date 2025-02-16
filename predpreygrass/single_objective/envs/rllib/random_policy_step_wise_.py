import pygame
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer3
from predpreygrass_ import PredPreyGrass  # Import your custom environment
import numpy as np

# Ensure all elements are displayed
np.set_printoptions(threshold=np.inf)


verbose = False

def print_pretty_observations(observations):
    for agent, data in observations.items():
        print(f"\n{'='*20} {agent} {'='*20}\n")
        for i, channel in enumerate(data):
            rounded_channel = np.round(channel, 2)  # Ensure rounding to 2 decimal places
            print(f"Observation Channel {i }:\n")
            print(rounded_channel)
            print("\n" + "-"*50)  # Separator for better readability


if __name__ == "__main__":
    env = PredPreyGrass()
    observations, _ = env.reset(seed=42)
    #print("Observations reset:")
    #print_pretty_observations(observations)  
    #print("Grid World reset:")
    # Print each row on one line
    #print("Grid World reset:")
    #print(np.array2string(env.grid_world_state, separator=' ', threshold=np.inf, max_line_width=np.inf))

    #print(env.grid_world_state)
    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.agents + env.grass_agents
    visualizer = MatPlotLibRenderer3(grid_size, all_agents, trace_length=5)

    pygame.init()
    screen = pygame.display.set_mode((200, 200))  # Small window for event capturing
    pygame.display.set_caption("Click to proceed")

    observations, _ = env.reset(seed=42)
    #print("Observation reset:")
    #print_pretty_observations(observations)
    #print(observations)

    for step in range(env.max_steps):
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)
        #print(f"Agent energies after step: {step}")
        #print(env.agent_energies)
        #print()
        #print(f"Rewards {step}: \n{rewards}")
        #print("Accumulated rew:",{k: round(v, 1) for k, v in env.cumulative_rewards.items()})
        #print("Energies:",{k: round(v, 1) for k, v in env.agent_energies.items()})
        #print()
        #print(f"Positions agents step {step}:")
        #print(env.agent_positions)
        rounded_observations = {k: np.round(obs, 2).tolist() for k, obs in observations.items()}
        #print("Observations step:")
        #print_pretty_observations(observations)  
        #print("Grid World State:")
        #print(env.grid_world_state)
        #print("Rewards:")
        #print(rewards)
        #print()
        merged_positions = {**env.agent_positions, **env.grass_positions}
        visualizer.update(merged_positions, step)

        # Wait for a mouse click to proceed
        waiting_for_click = True
        while waiting_for_click:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    waiting_for_click = False  # Proceed to the next step
        print("-----------------------------------------------------------------------------")
        print(f"Predators step: {step}")
        print(np.array2string(np.round(env.grid_world_state[1], 2), separator='    ', threshold=np.inf, max_line_width=np.inf))
        print(f"Prey step: {step}")
        print(np.array2string(np.round(env.grid_world_state[2], 2), separator='    ', threshold=np.inf, max_line_width=np.inf))
        print(f"Grass step: {step}")
        print(np.array2string(np.round(env.grid_world_state[3], 2), separator='    ', threshold=np.inf, max_line_width=np.inf))
        print("-----------------------------------------------------------------------------")
        if terminations["__all__"]:
            print("Environment terminated properly by termination. Steps:", step)
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

    visualizer.close()
    pygame.quit()
