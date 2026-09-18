"""
This script loads (pre) trained PPO policy modules (RLModules) directly from a
checkpoint and runs them in the PredPreyGrass environment
(predator_sexual_reproduction) for interactive debugging.

The simulation can be controlled in real-time using a graphical interface.
- [Space] Pause/Unpause
- [->] Step Forward
- [<-] Step Backward
- Tooltips are available to inspect agent IDs, positions, energies.

The environment is rendered using PyGame, and the simulation can be recorded as a video.
"""
# --- Project imports (predator_sexual_reproduction env + PyGame renderer) ---
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.predator_sexual_reproduction.utils.pygame_grid_renderer_rllib import (
    PyGameRenderer,
    ViewerControlHelper,
    LoopControlHelper,
)

# --- External libs ---
import os
import sys
import types

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env

SAVE_MOVIE = False
MOVIE_FILENAME = "simulation.mp4"
MOVIE_FPS = 10

# --- NumPy checkpoint compatibility shim ------------------------------------
try:
    import importlib
    if 'numpy._core.numeric' not in sys.modules:
        core_numeric = importlib.import_module('numpy.core.numeric')
        shim_pkg = types.ModuleType('numpy._core')
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = shim_pkg
        sys.modules['numpy._core.numeric'] = core_numeric
except Exception:
    pass


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator_male" in agent_id:
        return "predator_male_policy"
    if "predator_female" in agent_id:
        return "predator_female_policy"
    if "prey" in agent_id:
        return "prey_policy"
    raise KeyError(f"Unknown agent id '{agent_id}'; expected a predator_male, predator_female, or prey id.")


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)

    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})

    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("policy_pi: action_dist_inputs not found in action_output.")

    if deterministic:
        action = torch.argmax(logits, dim=-1).item()
    else:
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()

    return action


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrass", lambda config: env_creator(config))

    # --- Set your checkpoint path (directory that contains 'learner_group/learner/rl_module/...' ) ---
    checkpoint_path = os.path.join(
        os.path.expanduser("~"),
        "simulation_results",
        "ray_results",
        "PPO_PREDATOR_SEXUAL_REPRODUCTION_SEED42",  # placeholder -- update after training
        "checkpoint_000049",
    )

    expected_rlmodule_root = os.path.join(checkpoint_path, "learner_group", "learner", "rl_module")
    predator_male_path = os.path.join(expected_rlmodule_root, "predator_male_policy")
    predator_female_path = os.path.join(expected_rlmodule_root, "predator_female_policy")
    prey_path = os.path.join(expected_rlmodule_root, "prey_policy")
    if not (os.path.isdir(predator_male_path) and os.path.isdir(predator_female_path) and os.path.isdir(prey_path)):
        raise FileNotFoundError(
            "Could not find per-policy RLModule folders.\n"
            f"Looked for:\n  {predator_male_path}\n  {predator_female_path}\n  {prey_path}\n"
            "Make sure 'checkpoint_XXXX/learner_group/learner/rl_module/<policy_id>' exists.\n"
            "If your policy IDs differ, update policy_mapping_fn and module_paths below."
        )

    module_paths = {
        "predator_male_policy": predator_male_path,
        "predator_female_policy": predator_female_path,
        "prey_policy": prey_path,
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}
    print("Loaded RLModules:", list(rl_modules.keys()))

    seed = 42
    env = env_creator({})
    observations, _ = env.reset(seed=seed)
    active_agents = list(observations.keys())

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, ennable_speed_slider=False)

    video_writer = None
    if SAVE_MOVIE:
        screen_width = visualizer.screen.get_width()
        screen_height = visualizer.screen.get_height()
        video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")
        fourcc = video_writer_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(MOVIE_FILENAME, fourcc, MOVIE_FPS, (screen_width, screen_height))

    control = ViewerControlHelper()
    loop_helper = LoopControlHelper()

    clock = pygame.time.Clock()
    target_fps = 10

    total_reward = 0
    predator_male_counts = []
    predator_female_counts = []
    prey_counts = []
    time_steps = []

    snapshots = []
    max_snapshots = 100
    snapshots.append(env.get_state_snapshot())

    def render_current():
        visualizer.update(
            agent_positions=env.agent_positions,
            grass_positions=env.grass_positions,
            agent_energies=env.agent_energies,
            grass_energies=env.grass_energies,
            fruit_positions=env.fruit_positions,
            fruit_energies=env.fruit_energies,
            agents_just_ate=env.agents_just_ate,
            step=env.current_step,
        )

    while not loop_helper.simulation_terminated:
        control.handle_events()
        if control.step_backward:
            if len(snapshots) > 1:
                snapshots.pop()
                env.restore_state_snapshot(snapshots[-1])
                print(f"[ViewerControl] Step Backward → Step {env.current_step}")

                observations = {
                    agent: env._get_observation(agent)
                    for agent in env.agents
                    if agent in env.agent_positions
                }
                active_agents = list(observations.keys())

                if len(time_steps) > 0:
                    time_steps.pop()
                    predator_male_counts.pop()
                    predator_female_counts.pop()
                    prey_counts.pop()

                render_current()
                pygame.time.wait(100)
            control.step_backward = False
        if loop_helper.should_step(control):
            action_dict = {
                agent_id: policy_pi(
                    observations[agent_id],
                    rl_modules[policy_mapping_fn(agent_id)],
                    deterministic=True,
                )
                for agent_id in active_agents
            }
            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            active_agents = [
                a for a in observations
                if not terminations.get(a, False) and not truncations.get(a, False)
            ]

            snapshots.append(env.get_state_snapshot())
            if len(snapshots) > max_snapshots:
                snapshots.pop(0)

            render_current()
            if video_writer is not None:
                frame = pygame.surfarray.array3d(visualizer.screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            loop_helper.update_simulation_terminated(terminations, truncations)
            control.step_once = False
            clock.tick(target_fps)

            num_predator_male = sum(1 for agent in env.agents if "predator_male" in str(agent))
            num_predator_female = sum(1 for agent in env.agents if "predator_female" in str(agent))
            num_prey = sum(1 for agent in env.agents if "prey" in str(agent))
            time_steps.append(env.current_step)
            predator_male_counts.append(num_predator_male)
            predator_female_counts.append(num_predator_female)
            prey_counts.append(num_prey)
            total_reward += sum(rewards.values())
        else:
            render_current()
            pygame.time.wait(50)

    print(f"Evaluation complete! Total Reward: {total_reward}")

    predator_male_rewards, predator_female_rewards, prey_rewards = [], [], []
    print("\n--- Reward Breakdown per Agent ---")
    for agent_id, reward in env.cumulative_rewards.items():
        agent_name = str(agent_id)
        print(f"{agent_name:20}: {reward:.2f}")
        if "predator_male" in agent_name:
            predator_male_rewards.append(reward)
        elif "predator_female" in agent_name:
            predator_female_rewards.append(reward)
        elif "prey" in agent_name:
            prey_rewards.append(reward)

    total_predator_male_reward = sum(predator_male_rewards)
    total_predator_female_reward = sum(predator_female_rewards)
    total_prey_reward = sum(prey_rewards)
    total_reward_all = total_predator_male_reward + total_predator_female_reward + total_prey_reward

    print("\n--- Aggregated Rewards ---")
    print(f"Total number of steps: {env.current_step-1}")
    print(f"Total Predator Male Reward:   {total_predator_male_reward:.2f}")
    print(f"Total Predator Female Reward: {total_predator_female_reward:.2f}")
    print(f"Total Prey Reward:            {total_prey_reward:.2f}")
    print(f"Total All-Agent Reward:       {total_reward_all:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, predator_male_counts, label="Predator Males", color="red")
    plt.plot(time_steps, predator_female_counts, label="Predator Females", color="magenta")
    plt.plot(time_steps, prey_counts, label="Prey", color="blue")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Agents")
    plt.title("Agent Population Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if video_writer is not None:
        video_writer.release()
        print(f"[VideoWriter] Saved movie to {MOVIE_FILENAME}")

    ray.shutdown()
    visualizer.close()
