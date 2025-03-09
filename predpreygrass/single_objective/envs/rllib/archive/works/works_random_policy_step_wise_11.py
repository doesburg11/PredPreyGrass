import pygame
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer
from works_predpreygrass_11 import PredPreyGrass  # Import your custom environment
import numpy as np

# Ensure all elements are displayed
np.set_printoptions(threshold=np.inf)


verbose = True


if __name__ == "__main__":
    env = PredPreyGrass()
    observations, _ = env.reset(seed=42)

    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.agents + env.grass_agents
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)

    pygame.init()
    screen = pygame.display.set_mode((200, 200))  # Small window for event capturing
    pygame.display.set_caption("Click to proceed")

    observations, _ = env.reset(seed=42)

    for step in range(env.max_steps):
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        #print(f"Positions agents step {step}:")
        #print(env.agent_positions)
        observations, rewards, terminations, truncations, info = env.step(action_dict)

        rounded_observations = {k: np.round(obs, 2).tolist() for k, obs in observations.items()}

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
        if verbose:
            print(f"Step {step}:")
            print("-----------------------------------------")
            print("Actions :",list(action_dict.keys()))
            print('Agents  :',env.agents)
            print("Obs     :", list(observations.keys()))
            print("Reward  :", rewards)
            print("Terminations:", terminations)
            print("-----------------------------------------")


            env._print_grid_from_positions()
            env._print_grid_from_state()
            print("-----------------------------------------")

        if terminations["__all__"]:
            print("Environment terminated properly by termination. Steps:", step)
            break
        if truncations["__all__"]:
            print("Environment terminated properly by truncation.")
            break

    visualizer.close()
    pygame.quit()
