"""
Evaluation code for evaluating a trained PPO agent from a checkpoint.
This scripts ha an advanced viewer control system that allows stepping
back-and-forward through the simulation.
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
SAVE_EVAL_RESULTS = False  # Save plots of evolution and prey death causes
MOVIE_FILENAME = "simulation.mp4"
MOVIE_FPS = 10

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


def setup_environment_and_visualizer(now):
    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/v2_0/trained_policy"
    checkpoint_root = "/PPO_2025-06-12_23-54-40/"
    checkpoint_dir = "checkpoint_iter_1000"
    checkpoint_path = os.path.abspath(ray_results_dir + checkpoint_root + checkpoint_dir)

    training_dir = os.path.dirname(checkpoint_path)
    eval_output_dir = os.path.join(training_dir, f"eval_{checkpoint_dir}_{now}")

    module_paths = {
        pid: os.path.join(checkpoint_path, "learner_group", "learner", "rl_module", pid)
        for pid in ["speed_1_predator", "speed_2_predator", "speed_1_prey", "speed_2_prey"]
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}

    env = env_creator(config=config_env)
    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size)

    if SAVE_EVAL_RESULTS:
        os.makedirs(eval_output_dir, exist_ok=True)
        with open(os.path.join(eval_output_dir, "config_env.json"), "w") as f:
            json.dump(config_env, f, indent=4)
        ceviz = CombinedEvolutionVisualizer(destination_path=eval_output_dir, timestamp=now)
        pdviz = PreyDeathCauseVisualizer(destination_path=eval_output_dir, timestamp=now)
    else:
        ceviz = CombinedEvolutionVisualizer(destination_path=None, timestamp=now)
        pdviz = PreyDeathCauseVisualizer(destination_path=None, timestamp=now)

    return env, visualizer, rl_modules, ceviz, pdviz, eval_output_dir


def step_backwards_if_requested(control, env, snapshots, time_steps, predator_counts, prey_counts, visualizer, ceviz, pdviz):
    if control.step_backward:
        if len(snapshots) > 1:
            snapshots.pop()
            env.restore_state_snapshot(snapshots[-1])
            print(f"[ViewerControl] Step Backward → Step {env.current_step}")

            # REGENERATE observations
            observations = {agent: env._get_observation(agent) for agent in env.agents}

            # Rewind tracking
            if time_steps:
                time_steps.pop()
                predator_counts.pop()
                prey_counts.pop()

            ceviz.record(agent_ids=env.agents, internal_ids=env.agent_internal_ids, agent_ages=env.agent_ages)
            pdviz.record(env.death_cause_prey)

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
            return observations  # ✅ return updated obs

        control.step_backward = False
    return None


def step_forward(
    env,
    observations,
    rl_modules,
    control,
    visualizer,
    ceviz,
    pdviz,
    snapshots,
    predator_counts,
    prey_counts,
    time_steps,
    total_reward,
    clock,
    SAVE_MOVIE,
    video_writer,
):
    action_dict = {
        agent_id: policy_pi(
            observations[agent_id],
            rl_modules[policy_mapping_fn(agent_id)],
            deterministic=True,
        )
        for agent_id in env.agents
    }

    observations, rewards, terminations, truncations, _ = env.step(action_dict)

    snapshots.append(env.get_state_snapshot())
    if len(snapshots) > 100:
        snapshots.pop(0)

    ceviz.record(agent_ids=env.agents, internal_ids=env.agent_internal_ids, agent_ages=env.agent_ages)
    pdviz.record(env.death_cause_prey)

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
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    control.step_once = False
    clock.tick(visualizer.target_fps)

    predator_counts.append(sum(1 for a in env.agents if "predator" in a))
    prey_counts.append(sum(1 for a in env.agents if "prey" in a))
    time_steps.append(env.current_step)
    total_reward += sum(rewards.values())

    return observations, total_reward, terminations, truncations


def render_static_if_paused(env, visualizer):
    visualizer.update(
        agent_positions=env.agent_positions,
        grass_positions=env.grass_positions,
        agent_energies=env.agent_energies,
        grass_energies=env.grass_energies,
        agents_just_ate=env.agents_just_ate,
        step=env.current_step,
    )


def print_reward_summary(env, total_reward):
    print(f"\nEvaluation complete! Total Reward: {total_reward}")

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


def print_prey_death_summary(env):
    print("\n--- Prey Death Causes ---")
    death_stats = {"eaten": 0, "starved": 0}

    for internal_id, cause in env.death_cause_prey.items():
        print(f"Prey internal_id {internal_id:4d}: {cause}")
        if cause in death_stats:
            death_stats[cause] += 1

    print("\n--- Summary ---")
    print(f"Total prey eaten   : {death_stats['eaten']}")
    print(f"Total prey starved : {death_stats['starved']}")


def save_reward_summary_to_file(env, total_reward, eval_output_dir):
    reward_log_path = os.path.join(eval_output_dir, "reward_summary.txt")
    with open(reward_log_path, "w") as f:
        f.write(f"Total Reward: {total_reward}\n")
        f.write("\n--- Reward Breakdown per Agent ---\n")
        for agent_id, reward in env.cumulative_rewards.items():
            f.write(f"{agent_id:15}: {reward:.2f}\n")

        # Group rewards by type
        grouped_rewards = {"speed_1_predator": [], "speed_2_predator": [], "speed_1_prey": [], "speed_2_prey": []}
        for agent_id, reward in env.cumulative_rewards.items():
            for group in grouped_rewards:
                if group in agent_id:
                    grouped_rewards[group].append(reward)

        f.write("\n--- Aggregated Rewards ---\n")
        f.write(f"Total number of steps            : {env.current_step-1}\n")
        for group, rewards in grouped_rewards.items():
            total_group_reward = sum(rewards)
            f.write(f"Total {group.replace('_', ' ').title()} Reward: {total_group_reward:.2f}\n")


def save_prey_death_summary_to_file(env, eval_output_dir):
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


def run_post_evaluation_plots(combined_evolution_visualizer, prey_death_cause_visualizer):
    combined_evolution_visualizer.plot()
    prey_death_cause_visualizer.plot()


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    register_env("PredPreyGrass", lambda config: env_creator(config))

    env, visualizer, rl_modules, ceviz, pdviz, eval_output_dir = setup_environment_and_visualizer(now)
    observations, _ = env.reset(seed=seed)

    if SAVE_MOVIE:
        screen_width = visualizer.screen.get_width()
        screen_height = visualizer.screen.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(MOVIE_FILENAME, fourcc, MOVIE_FPS, (screen_width, screen_height))
    else:
        video_writer = None

    control = ViewerControlHelper()
    loop_helper = LoopControlHelper()
    control.fps_slider_rect = visualizer.slider_rect
    control.fps_slider_update_fn = lambda new_fps: setattr(visualizer, "target_fps", new_fps)
    control.visualizer = visualizer

    clock = pygame.time.Clock()
    total_reward = 0
    predator_counts, prey_counts, time_steps = [], [], []
    snapshots = [env.get_state_snapshot()]

    while not loop_helper.simulation_terminated:
        control.handle_events()

        new_obs = step_backwards_if_requested(
            control, env, snapshots, time_steps, predator_counts, prey_counts, visualizer, ceviz, pdviz
        )
        if new_obs is not None:
            observations = new_obs
            continue

        if loop_helper.should_step(control):
            observations, total_reward, terminations, truncations = step_forward(
                env,
                observations,
                rl_modules,
                control,
                visualizer,
                ceviz,
                pdviz,
                snapshots,
                predator_counts,
                prey_counts,
                time_steps,
                total_reward,
                clock,
                SAVE_MOVIE,
                video_writer,
            )
            loop_helper.update_simulation_terminated(terminations, truncations)
        else:
            render_static_if_paused(env, visualizer)
            pygame.time.wait(50)

    print_reward_summary(env, total_reward)
    print_prey_death_summary(env)

    if SAVE_EVAL_RESULTS:
        save_reward_summary_to_file(env, total_reward, eval_output_dir)
        save_prey_death_summary_to_file(env, eval_output_dir)

    run_post_evaluation_plots(ceviz, pdviz)

    pygame.event.pump()
    pygame.quit()
    ray.shutdown()
