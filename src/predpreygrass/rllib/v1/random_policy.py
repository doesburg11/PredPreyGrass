from predpreygrass.rllib.v1.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.utils.pygame_renderer import PyGameRenderer, ViewerControlHelper, LoopControlHelper

# external libraries
import pygame


if __name__ == "__main__":
    verbose_grid_state = False
    verbose_observation = False
    seed_value = None  # 7  # set seed for reproducibility
    env = PredPreyGrass()
    # seed the action spaces
    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)
    # reset the environment and get initial observations
    observations, _ = env.reset(seed=seed_value)

    # initialize the grid_visualizer
    grid_size = (env.grid_size, env.grid_size)
    live_viewer = PyGameRenderer(grid_size)

    # Init pause state
    paused = False
    step_once = False
    clock = pygame.time.Clock()
    target_fps = 10  # 10 steps per second (adjust as desired)
    # Initialize control helper
    control = ViewerControlHelper()
    loop_helper = LoopControlHelper()
    # Initialize loop control
    simulation_terminated = False
    step = 0

    # for step in range(env.max_steps):
    while not loop_helper.simulation_terminated:
        control.handle_events()
        if loop_helper.should_step(control):
            action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
            # Run one env step
            observations, rewards, terminations, truncations, info = env.step(action_dict)
            # Update viewer after env step
            live_viewer.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                step=step
            )
            # Update termination flag AFTER env.step
            loop_helper.update_simulation_terminated(terminations, truncations)
            # Reset step_once
            control.step_once = False
            clock.tick(target_fps)
            step += 1
        else:
            # If paused → update viewer so tooltips work
            live_viewer.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                step=step
            )
            # Small sleep to avoid CPU busy loop
            pygame.time.wait(50)

    live_viewer.close()
    env.close()
