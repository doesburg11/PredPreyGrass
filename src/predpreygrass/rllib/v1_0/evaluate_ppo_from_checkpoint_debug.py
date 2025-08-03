"""
This script loads a trained PPO policy from a checkpoint and runs it in the PredPreyGrass environment
to visualize the agent behavior and collect statistics. The environment is rendered using PyGame, and
the simulation can be recorded as a video. The graphical interface allows for real-time interaction
with the simulation for debugging purposes, including a speed slider, pausing [BACKSPACE], stepping
forward [->], and stepping backward [<-] through the simulation steps. During pause, tooltips are
available to inspect agent IDs, positions, energies.
"""
# discretionary libraries
from predpreygrass.rllib.v1_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v1_0.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper, LoopControlHelper

# from predpreygrass.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper, LoopControlHelper

# external libraries
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import torch
import os
import matplotlib.pyplot as plt
import pygame
import cv2
import numpy as np

SAVE_MOVIE = False
MOVIE_FILENAME = "simulation.mp4"
MOVIE_FPS = 10


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None


def policy_pi(observation, policy_module, deterministic=True):
    """
    Compute the action for a single observation using the given policy module.

    Args:
        observation (np.array or torch.Tensor): The observation of the agent.
        policy_module: RLlib policy module (DefaultPPOTorchRLModule or similar).
        deterministic (bool): If True, take argmax (greedy). If False, sample from distribution.

    Returns:
        int: The selected action (Discrete).
    """
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)

    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})

    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("policy_pi: action_dist_inputs not found in action_output.")

    if deterministic:
        action = torch.argmax(logits, dim=-1).item()  # greedy action selection
    else:
        dist = torch.distributions.Categorical(logits=logits)  # sample from the categorical distribution
        action = dist.sample().item()

    return action


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrass", lambda config: env_creator(config))

    # Load trained model from checkpoint
    script_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(
        script_dir,
        "trained_policy",
        "PPO_2025-06-06_21-34-53",
        "PPO_PredPreyGrass_52139_00000_0_2025-06-06_21-34-53",
        "checkpoint_000030",
    )

    trained_algo = Algorithm.from_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully!")
    # Access RLModules from learner_group
    rl_modules = trained_algo.learner_group._learner.module  # Retrieves policy modules

    seed = 42  # set seed for reproducibility
    env = env_creator({})
    # Reset environment and get initial observations
    observations, _ = env.reset(seed=seed)

    # Initialize PyGameRenderer
    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, ennable_speed_slider=False)

    # Create movie
    if SAVE_MOVIE:
        screen_width = visualizer.screen.get_width()
        screen_height = visualizer.screen.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(MOVIE_FILENAME, fourcc, MOVIE_FPS, (screen_width, screen_height))

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
    # Avanced loop control added for step-wise back-and-forward evaluation debugging
    while not loop_helper.simulation_terminated:
        control.handle_events()
        # Backward step handling
        if control.step_backward:
            if len(snapshots) > 1:
                snapshots.pop()  # Discard current step
                env.restore_state_snapshot(snapshots[-1])
                print(f"[ViewerControl] Step Backward → Step {env.current_step}")

                # --- REGENERATE observations to match restored state ---
                observations = {agent: env._get_observation(agent) for agent in env.agents}

                # --- Also rewind history lists ---
                if len(time_steps) > 0:
                    time_steps.pop()
                    predator_counts.pop()
                    prey_counts.pop()

                visualizer.update(
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
            # Build action dict from PPO policy
            action_dict = {
                agent_id: policy_pi(
                    observations[agent_id],
                    rl_modules[policy_mapping_fn(agent_id)],
                    deterministic=True,  # or False if you want stochastic behavior
                )
                for agent_id in env.agents
            }
            # Step env
            observations, rewards, terminations, truncations, _ = env.step(action_dict)

            # Save snapshot AFTER step
            snapshots.append(env.get_state_snapshot())
            if len(snapshots) > max_snapshots:
                snapshots.pop(0)

            # Update viewer
            visualizer.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                agents_just_ate=env.agents_just_ate,
                step=env.current_step,
            )
            if SAVE_MOVIE:
                frame = pygame.surfarray.array3d(visualizer.screen)
                frame = np.transpose(frame, (1, 0, 2))  # Convert (width, height, channels) → (height, width, channels)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Pygame uses RGB, OpenCV uses BGR
                video_writer.write(frame)

            # Update loop control termination flag
            loop_helper.update_simulation_terminated(terminations, truncations)

            # Reset step_once
            control.step_once = False

            # Frame rate control
            clock.tick(target_fps)

            # Track stats
            num_predators = sum(1 for agent in env.agents if "predator" in agent)
            num_prey = sum(1 for agent in env.agents if "prey" in agent)

            time_steps.append(env.current_step)
            predator_counts.append(num_predators)
            prey_counts.append(num_prey)
            total_reward += sum(rewards.values())
        else:
            # If paused → update viewer so tooltips work
            visualizer.update(
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

    print(f"Evaluation complete! Total Reward: {total_reward}")

    # --- REWARD SUMMARY ---
    predator_rewards = []
    prey_rewards = []

    print("\n--- Reward Breakdown per Agent ---")
    for agent_id, reward in env.cumulative_rewards.items():
        print(f"{agent_id:15}: {reward:.2f}")
        if "predator" in agent_id:
            predator_rewards.append(reward)
        elif "prey" in agent_id:
            prey_rewards.append(reward)

    total_predator_reward = sum(predator_rewards)
    total_prey_reward = sum(prey_rewards)
    total_reward_all = total_predator_reward + total_prey_reward

    print("\n--- Aggregated Rewards ---")
    print(f"Total number of steps: {env.current_step-1}")
    print(f"Total Predator Reward: {total_predator_reward:.2f}")
    print(f"Total Prey Reward:     {total_prey_reward:.2f}")
    print(f"Total All-Agent Reward:{total_reward_all:.2f}")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, predator_counts, label="Predators", color="red")
    plt.plot(time_steps, prey_counts, label="Prey", color="blue")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Agents")
    plt.title("Agent Population Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if SAVE_MOVIE:
        video_writer.release()
        print(f"[VideoWriter] Saved movie to {MOVIE_FILENAME}")

    # Shutdown
    ray.shutdown()
    visualizer.close()
