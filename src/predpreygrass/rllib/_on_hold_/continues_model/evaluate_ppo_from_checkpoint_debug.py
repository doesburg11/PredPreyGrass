"""
Evaluation code for evaluating a trained PPO agent from a checkpoint.
This script has an advanced viewer control system that allows stepping
back-and-forward through the simulation. For a simplear evaluation script,
see `evaluate_ppo_from_checkpoint_headless.py`.
"""
from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.rllib.v3_1.config.config_env_eval import config_env
from predpreygrass.rllib.v3_1.utils.matplot_renderer import CombinedEvolutionVisualizer, PreyDeathCauseVisualizer
from predpreygrass.rllib.v3_1.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper, LoopControlHelper

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
import re
from collections import defaultdict


SAVE_EVAL_RESULTS = True
SAVE_MOVIE = False
MOVIE_FILENAME = "simulation.mp4"
MOVIE_FPS = 10


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    else:
        raise ValueError(f"Unrecognized agent_id format: {agent_id}")


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("policy_pi: action_dist_inputs not found in action_output.")
    if deterministic:
        return torch.argmax(logits, dim=-1).item()
    else:
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()


def setup_environment_and_visualizer(now):
    ray_results_dir = "/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    checkpoint_root = "v2_7_tune_default_benchmark/PPO_PredPreyGrass_86337_00000_0_2025-08-04_23-53-58/"
    checkpoint_dir = "checkpoint_000099"
    checkpoint_path = os.path.abspath(ray_results_dir + checkpoint_root + checkpoint_dir)

    # training_dir = os.path.dirname(checkpoint_path)
    eval_output_dir = os.path.join(checkpoint_path, f"eval_{checkpoint_dir}_{now}")

    rl_module_dir = os.path.join(checkpoint_path, "learner_group", "learner", "rl_module")
    module_paths = {}

    if os.path.isdir(rl_module_dir):
        for pid in os.listdir(rl_module_dir):
            path = os.path.join(rl_module_dir, pid)
            if os.path.isdir(path):
                module_paths[pid] = path
    else:
        raise FileNotFoundError(f"RLModule directory not found: {rl_module_dir}")

    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}

    env = env_creator(config=config_env)
    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(
        grid_size, 
        cell_size=32, 
        enable_speed_slider=True, 
        enable_tooltips=True,
        max_steps=config_env.get("max_steps", 1000)
    )

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


def step_backwards_if_requested(
    control, env, snapshots, time_steps, predator_counts, prey_counts, energy_by_type_series, visualizer, ceviz, pdviz
):
    if control.step_backward:
        if len(snapshots) > 1:
            snapshots.pop()
            env.restore_state_snapshot(snapshots[-1])
            print(f"[ViewerControl] Step Backward → Step {env.current_step}")
            observations = {agent: env._get_observation(agent) for agent in env.agents}
            if time_steps:
                time_steps.pop()
                predator_counts.pop()
                prey_counts.pop()
                energy_by_type_series.pop()

            ceviz.record(agent_ids=env.agents)
            pdviz.record(env.death_cause_prey)
            visualizer.update(
                grass_positions=env.grass_positions,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate,
                per_step_agent_data=env.per_step_agent_data,
            )
            control.fps_slider_rect = visualizer.slider_rect
            pygame.time.wait(100)
            control.step_backward = False
            return observations
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
    action_dict = {}
    for agent_id in env.agents:
        group = policy_mapping_fn(agent_id)
        if group in rl_modules:
            action_dict[agent_id] = policy_pi(observations[agent_id], rl_modules[group], deterministic=True)

    observations, rewards, terminations, truncations, _ = env.step(action_dict)
    # Inject unique ID per agent into step data
    for agent_id, agent_data in env.per_step_agent_data[-1].items():
        agent_data["unique_id"] = env.unique_agents[agent_id]

    snapshots.append(env.get_state_snapshot())
    if len(snapshots) > 100:
        snapshots.pop(0)

    ceviz.record(agent_ids=env.agents)
    if hasattr(ceviz, "record_energy"):
        ceviz.record_energy(env.get_total_energy_by_type())

    visualizer.update(
        grass_positions=env.grass_positions,
        grass_energies=env.grass_energies,
        step=env.current_step,
        agents_just_ate=env.agents_just_ate,
        per_step_agent_data=env.per_step_agent_data,
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
    energy_by_type_series.append(env.get_total_energy_by_type())
    total_reward += sum(rewards.values())

    return observations, total_reward, terminations, truncations


def render_static_if_paused(env, visualizer):
    visualizer.update(
        grass_positions=env.grass_positions,
        grass_energies=env.grass_energies,
        step=env.current_step,
        agents_just_ate=env.agents_just_ate,
        per_step_agent_data=env.per_step_agent_data,
    )


def parse_uid(uid):
    """
    Parse UID like 'type_1_predator_2_10' into sortable components:
    → ('type_1_predator', 2, 10)
    """
    match = re.match(r"(type_\d+_(?:predator|prey))_(\d+)_(\d+)", uid)
    if match:
        group, idx, reuse = match.groups()
        return group, int(idx), int(reuse)
    else:
        return uid, float("inf"), float("inf")  # fallback for malformed uids


def print_ranked_reward_summary(env, total_reward):
    group_rewards = defaultdict(list)

    for uid, stats in env.unique_agent_stats.items():
        reward = stats.get("cumulative_reward", 0.0)
        group, index, reuse = parse_uid(uid)
        group_rewards[group].append((uid, reward, index, reuse))

    print(f"\nEvaluation complete! Total Reward: {total_reward:.2f}")
    print("\n--- Ranked Reward Breakdown per Unique Agent ---")

    for group in sorted(group_rewards.keys()):
        print(f"\n## {group.replace('_', ' ').title()} ##")
        sorted_group = sorted(
            group_rewards[group], key=lambda x: (-x[1], x[2], x[3])  # sort by reward desc, then id asc, reuse asc
        )
        for uid, reward, _, _ in sorted_group:
            print(f"{uid:25}: {reward:.2f}")

    print("\n--- Aggregated Totals ---")
    print(f"Total number of steps: {env.current_step - 1}")
    for group in sorted(group_rewards.keys()):
        total = sum(r for _, r, _, _ in group_rewards[group])
        print(f"Total {group.replace('_', ' ').title():25}: {total:.2f}")
    print(f"Total All-Agent Reward:           {total_reward:.2f}")


def save_reward_summary_to_file(env, total_reward, output_dir):
    reward_log_path = os.path.join(output_dir, "reward_summary.txt")
    with open(reward_log_path, "w") as f:
        f.write(f"Total Reward: {total_reward}\n\n")
        f.write("--- Reward Breakdown per Agent ---\n")
        for agent_id, reward in env.cumulative_rewards.items():
            f.write(f"{agent_id:15}: {reward:.2f}\n")
        f.write("\n--- Aggregated Rewards ---\n")
        f.write(f"Total number of steps: {env.current_step-1}\n")
        grouped_rewards = {}
        for agent_id, reward in env.cumulative_rewards.items():
            group = policy_mapping_fn(agent_id)
            grouped_rewards.setdefault(group, []).append(reward)
        for group, rewards in grouped_rewards.items():
            f.write(f"Total {group.replace('_', ' ').title()} Reward: {sum(rewards):.2f}\n")


def run_post_evaluation_plots(ceviz, pdviz):
    if SAVE_EVAL_RESULTS:
        ceviz.plot()
        pdviz.plot()


def print_ranked_fitness_summary(env):
    print("\n--- Ranked Fitness Summary by Group ---")
    group_stats = defaultdict(list)
    for uid, stats in env.unique_agent_stats.items():
        group, index, reuse = parse_uid(uid)
        lifetime = (stats["death_step"] or env.current_step) - stats["birth_step"]
        group_stats[group].append(
            {
                "uid": uid,
                "reward": stats.get("cumulative_reward", 0.0),
                "lifetime": lifetime,
                "offspring": stats.get("offspring_count", 0),
                "efficiency": stats.get("offspring_count", 0) / max(stats.get("energy_spent", 1e-6), 1e-6),
            }
        )

    for group in sorted(group_stats.keys()):
        print(f"\n## {group.replace('_', ' ').title()} ##")
        sorted_group = sorted(group_stats[group], key=lambda x: (-x["offspring"], -x["reward"], -x["lifetime"]))
        for entry in sorted_group[:10]:  # top 10
            print(
                f"{entry['uid']:25} | "
                f"R={entry['reward']:.2f} | "
                f"Life={entry['lifetime']} | "
                f"Off={entry['offspring']} | "
                f"Eff={entry['efficiency']:.2f}"
            )


if __name__ == "__main__":
    seed = 4
    ray.init(ignore_reinit_error=True)
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
    energy_by_type_series = [env.get_total_energy_by_type()]

    while not loop_helper.simulation_terminated:
        control.handle_events()

        new_obs = step_backwards_if_requested(
            control, env, snapshots, time_steps, predator_counts, prey_counts, energy_by_type_series, visualizer, ceviz, pdviz
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

    # === Print total offspring by type ===
    offspring_counts = env.get_total_offspring_by_type()
    print("\n--- Offspring Counts by Type ---")
    for agent_type, count in offspring_counts.items():
        print(f"{agent_type:20}: {count}")

    # print_ranked_reward_summary(env, total_reward)

    # print("Death statistics:", env.death_agents_stats)

    if SAVE_EVAL_RESULTS:
        save_reward_summary_to_file(env, total_reward, eval_output_dir)
        energy_log_path = os.path.join(eval_output_dir, "energy_by_type.json")
        with open(energy_log_path, "w") as f:
            json.dump(energy_by_type_series, f, indent=2)
    # Always show plots on screen
    ceviz.plot()
    if SAVE_EVAL_RESULTS:
        # Export all unique agent fitness stats
        agent_fitness_path = os.path.join(eval_output_dir, "agent_fitness_stats.json")
        with open(agent_fitness_path, "w") as f:
            json.dump(env.unique_agent_stats, f, indent=2)

    print_ranked_fitness_summary(env)

    pygame.quit()
    ray.shutdown()
