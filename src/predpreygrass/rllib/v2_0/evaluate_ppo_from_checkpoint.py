"""
Evaluation code for evaluating a trained PPO agent from a checkpoint
"""
from predpreygrass.rllib.v2_0.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.rllib.v2_0.config.config_env_eval import config_env
from predpreygrass.utils.renderer import CombinedEvolutionVisualizer, PreyDeathCauseVisualizer
from predpreygrass.utils.pygame_renderer import PyGameRenderer, ViewerControlHelper, LoopControlHelper

# external libraries
import ray
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env
import torch
from datetime import datetime
import os
import json
import pygame
import cv2
import numpy as np

SAVE_MOVIE = False
MOVIE_FILENAME = "simulation.mp4"
MOVIE_FPS = 10

verbose_grid = False
verbose_actions = False
seed = 1  # 42 # Optional: set to integer for reproducibility

# Initialize Ray
ray.init(ignore_reinit_error=True)


def env_creator(config):
    return PredPreyGrass(config)


# Policy mapping function
def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    speed = parts[1]
    role = parts[2]
    return f"speed_{speed}_{role}"


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
    # === Set checkpoint paths ===
    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/v2_0/trained_policy"
    register_env("PredPreyGrass", lambda config: env_creator(config))
    # checkpoint_root = '/v5_move_energy/pred_obs_range/Pred_11_Prey_9/PPO_PredPreyGrass_109fe_00000_0_2025-04-19_10-41-19/'
    # checkpoint_root = '/v5_move_energy/reward_1.0/obs_range_Pred_11_Prey_9/PPO_PredPreyGrass_109fe_00000_0_2025-04-19_10-41-19/'
    checkpoint_root = "/PPO_2025-06-12_23-54-40/"
    checkpoint_dir = "checkpoint_iter_1000"
    checkpoint_path = os.path.abspath(ray_results_dir + checkpoint_root + checkpoint_dir)
    print(f"Checkpoint path: {checkpoint_path}")
    # === Get training directory and prepare eval output dir ===
    training_dir = os.path.dirname(checkpoint_path)
    print(f"Training directory: {training_dir}")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_output_dir = os.path.join(training_dir, f"eval_{checkpoint_dir}_{now}")
    print(f"Evaluation output directory: {eval_output_dir}")
    os.makedirs(eval_output_dir, exist_ok=True)

    # === Save config_env.json ===
    with open(os.path.join(eval_output_dir, "config_env.json"), "w") as f:
        json.dump(config_env, f, indent=4)

    # Load RLModules directly from subfolders
    module_paths = {
        pid: os.path.join(checkpoint_path, "learner_group", "learner", "rl_module", pid)
        for pid in ["speed_1_predator", "speed_2_predator", "speed_1_prey", "speed_2_prey"]
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}

    # Initialize the environment
    env = env_creator(config=config_env)  # PredPreyGrass()

    # Reset environment and get initial observations
    observations, _ = env.reset(seed=seed)

    # intitialize matplot lib renderer
    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.possible_agents + env.grass_agents
    visualizer = PyGameRenderer(grid_size)
    combined_evolution_visualizer = CombinedEvolutionVisualizer(
        destination_path=eval_output_dir,
        timestamp=now,
    )
    prey_death_cause_visualizer = PreyDeathCauseVisualizer(
        destination_path=eval_output_dir,
        timestamp=now,
    )

    # Create movie
    if SAVE_MOVIE:
        screen_width = visualizer.screen.get_width()
        screen_height = visualizer.screen.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(MOVIE_FILENAME, fourcc, MOVIE_FPS, (screen_width, screen_height))

    # Initialize viewer control + loop helper
    control = ViewerControlHelper()
    loop_helper = LoopControlHelper()
    # Add slider
    control.fps_slider_rect = visualizer.slider_rect
    control.fps_slider_update_fn = lambda new_fps: setattr(visualizer, "target_fps", new_fps)

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
                combined_evolution_visualizer.record(
                    agent_ids=env.agents, internal_ids=env.agent_internal_ids, agent_ages=env.agent_ages
                )
                prey_death_cause_visualizer.record(env.death_cause_prey)

                visualizer.update(
                    agent_positions=env.agent_positions,
                    grass_positions=env.grass_positions,
                    agent_energies=env.agent_energies,
                    grass_energies=env.grass_energies,
                    agents_just_ate=env.agents_just_ate,
                    step=env.current_step,
                )
                control.fps_slider_rect = visualizer.slider_rect

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
            combined_evolution_visualizer.record(
                agent_ids=env.agents, internal_ids=env.agent_internal_ids, agent_ages=env.agent_ages
            )
            prey_death_cause_visualizer.record(env.death_cause_prey)

            # Update viewer
            visualizer.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                agents_just_ate=env.agents_just_ate,
                step=env.current_step,
            )
            control.fps_slider_rect = visualizer.slider_rect

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
            # clock.tick(target_fps)
            clock.tick(visualizer.target_fps)

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

    # --- PREY DEATH CAUSE SUMMARY ---
    death_log_path = os.path.join(eval_output_dir, "prey_death_causes.txt")
    death_stats = {"eaten": 0, "starved": 0}

    with open(death_log_path, "w") as f:
        f.write("--- Prey Death Causes ---\n")
        for internal_id, cause in env.death_cause_prey.items():
            f.write(f"Prey internal_id {internal_id:4d}: {cause}\n")
            if cause in death_stats:
                death_stats[cause] += 1
        f.write("\n--- Summary ---\n")
        f.write(f"Total prey eaten   : {death_stats['eaten']}\n")
        f.write(f"Total prey starved : {death_stats['starved']}\n")

    print(f"Prey death summary written to: {death_log_path}")

    # --- REWARD SUMMARY ---
    reward_log_path = os.path.join(eval_output_dir, "reward_summary.txt")
    with open(reward_log_path, "w") as f:
        f.write(f"Total Reward: {total_reward}\n")
        f.write("\n--- Reward Breakdown per Agent ---\n")
        speed_1_predator_rewards = []
        speed_2_predator_rewards = []
        speed_1_prey_rewards = []
        speed_2_prey_rewards = []
        for agent_id, reward in env.cumulative_rewards.items():
            f.write(f"{agent_id:15}: {reward:.2f}\n")
            if "speed_1_predator" in agent_id:
                speed_1_predator_rewards.append(reward)
            elif "speed_2_predator" in agent_id:
                speed_2_predator_rewards.append(reward)
            elif "speed_1_prey" in agent_id:
                speed_1_prey_rewards.append(reward)
            elif "speed_2_prey" in agent_id:
                speed_2_prey_rewards.append(reward)

        total_speed_1_predator_reward = sum(speed_1_predator_rewards)
        total_speed_1_prey_reward = sum(speed_1_prey_rewards)
        total_reward_all_speed_1 = total_speed_1_predator_reward + total_speed_1_prey_reward
        total_speed_2_predator_reward = sum(speed_2_predator_rewards)
        total_speed_2_prey_reward = sum(speed_2_prey_rewards)
        total_reward_all_speed_2 = total_speed_2_predator_reward + total_speed_2_prey_reward

        f.write("\n--- Aggregated Rewards ---\n")
        f.write(f"Total number of steps            : {env.current_step-1}\n")
        f.write(f"Total Low-Speed Predator Reward  : {total_speed_1_predator_reward:.2f}\n")
        f.write(f"Total Low-Speed Prey Reward      : {total_speed_1_prey_reward:.2f}\n")
        f.write(f"Total Low-Speed Agent Reward     : {total_speed_1_predator_reward + total_speed_1_prey_reward:.2f}\n")
        f.write(f"Total High-Speed Predator Reward : {total_speed_2_predator_reward:.2f}\n")
        f.write(f"Total High-Speed Prey Reward     : {total_speed_2_prey_reward:.2f}\n")
        f.write(f"Total High-Speed Agent Reward    : {total_speed_2_predator_reward + total_speed_2_prey_reward:.2f}\n")

        print("\n--- Aggregated Rewards ---")
        print(f"Total number of steps            : {env.current_step-1}")
        print(f"Total Low-Speed Predator Reward  : {total_speed_1_predator_reward:.2f}")
        print(f"Total Low-Speed Prey Reward      : {total_speed_1_prey_reward:.2f}")
        print(f"Total Low-Speed Agent Reward     : {total_reward_all_speed_1:.2f}")
        print(f"Total High-Speed Predator Reward : {total_speed_2_predator_reward:.2f}")
        print(f"Total High-Speed Prey Reward     : {total_speed_2_prey_reward:.2f}")
        print(f"Total High-Speed Agent Reward    : {total_reward_all_speed_2:.2f}")

    combined_evolution_visualizer.plot()
    prey_death_cause_visualizer.plot()

    pygame.event.pump()  # Flush event queue
    pygame.quit()  # Ensure Pygame releases mouse cleanly

    # Shutdown Ray after evaluation
    ray.shutdown()
