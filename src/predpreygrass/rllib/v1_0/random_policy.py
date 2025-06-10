from predpreygrass.rllib.v1_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.utils.pygame_renderer import PyGameRenderer, ViewerControlHelper, LoopControlHelper

# external libraries
import pygame


# Define environment registration
def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    """
    Sample a random action for the given agent using the environment's action space.

    Args:
        agent_id (str): The agent's ID.
        env: The environment instance (must provide env.action_spaces[agent_id]).

    Returns:
        int: A randomly sampled action.
    """
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    seed = 3  # set seed for reproducibility
    env = env_creator({})
    # reset the environment and get initial observations
    observations, _ = env.reset(seed=seed)

    # initialize the grid_visualizer
    grid_size = (env.grid_size, env.grid_size)
    visualizar = PyGameRenderer(grid_size)
    # Initialize viewer control + loop helper
    control = ViewerControlHelper()
    loop_helper = LoopControlHelper()

    # Optional: frame rate control
    clock = pygame.time.Clock()
    target_fps = 10  # Adjust as desired

    total_reward = 0
    predator_counts = []
    prey_counts = []
    time_steps = []

    # --- Setup snapshots for stepping backwards ---
    snapshots = []
    max_snapshots = 100  # Keep last 100 steps
    # Save initial snapshot
    snapshots.append(env.get_state_snapshot())

    # Run one evaluation episode
    while not loop_helper.simulation_terminated:
        control.handle_events()
        # Backward step handling
        if control.step_backward:
            if len(snapshots) > 1:
                snapshots.pop()  # Discard current step
                env.restore_state_snapshot(snapshots[-1])
                print(f"[ViewerControl] Step Backward → Step {env.current_step}")

                # --- REGENERATE obs to match restored state ---
                obs = {agent: env._get_observation(agent) for agent in env.agents}

                # --- Also rewind history lists ---
                if len(time_steps) > 0:
                    time_steps.pop()
                    predator_counts.pop()
                    prey_counts.pop()

                visualizar.update(
                    agent_positions=env.agent_positions,
                    grass_positions=env.grass_positions,
                    agent_energies=env.agent_energies,
                    grass_energies=env.grass_energies,
                    agents_just_ate=env.agents_just_ate,
                    step=env.current_step,
                )
                pygame.time.wait(100)
            control.step_backward = False
        # Normal step forward
        if loop_helper.should_step(control):
            action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}

            # Run one env step
            observations, rewards, terminations, truncations, info = env.step(action_dict)
            # Save snapshot AFTER step
            snapshots.append(env.get_state_snapshot())
            if len(snapshots) > max_snapshots:
                snapshots.pop(0)
            # Update viewer after env step
            visualizar.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                agents_just_ate=env.agents_just_ate,
                step=env.current_step,
            )
            # Update termination flag AFTER env.step
            loop_helper.update_simulation_terminated(terminations, truncations)
            # Reset step_once
            control.step_once = False
            clock.tick(target_fps)
        else:
            # If paused → update viewer so tooltips work
            visualizar.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                agents_just_ate=env.agents_just_ate,
                step=env.current_step,
            )
            # Small sleep to avoid CPU busy loop
            pygame.time.wait(50)

    # --- End of main loop ---
    visualizar.close()
    env.close()
