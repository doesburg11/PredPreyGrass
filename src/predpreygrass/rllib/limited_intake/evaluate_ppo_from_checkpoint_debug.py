"""
This script loads (pre) trained PPO policy modules (RLModules) directly from a checkpoint
and runs them in the PredPreyGrass environment (limited_intake) for interactive debugging.

This version differs from ppg_2_policies in that it includes two types of predators and two types of prey, 
making distinct behaviors and characteristics possible per species. In this version, the "speed 2"
version of predator and prey are are faster and can cover more ground in one movement step.
Both speed 1 and speed 2 predators and prey are mutually trained. Evaluation of only speed 1 
predators and prey with only small change of mutation to speed 2 predators and prey generally 
leads to dominance of speed 2 agents and extinction of speed 1 agents as the trained model shows 
in the simulation.

The simulation can be controlled in real-time using a graphical interface.
- [Space] Pause/Unpause
- [->] Step Forward
- [<-] Step Backward
- Tooltips are available to inspect agent IDs, positions, energies.

The environment is rendered using PyGame, and the simulation can be recorded as a video. 
"""
from predpreygrass.rllib.limited_intake.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.rllib.limited_intake.config.config_env_limited_intake import config_env
from predpreygrass.rllib.limited_intake.utils.matplot_renderer import CombinedEvolutionVisualizer, PreyDeathCauseVisualizer
from predpreygrass.rllib.limited_intake.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper, LoopControlHelper

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


def _load_training_env_config_from_run(checkpoint_path, base_cfg):
    """
    Try to locate a run_config.json near the checkpoint and merge its env settings
    into the provided base_cfg. This aligns evaluation observations with the
    shapes the policy network was trained on (avoids matmul shape errors).
    """
    candidates = [
        os.path.join(os.path.dirname(checkpoint_path), "run_config.json"),
        os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "run_config.json"),
    ]
    training_env_cfg = None
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r") as f:
                    rc = json.load(f)
                # Prefer explicit env_config if present; else, fall back to top-level keys
                if isinstance(rc, dict):
                    if isinstance(rc.get("env_config"), dict):
                        training_env_cfg = rc["env_config"]
                    else:
                        # Heuristic: intersect with known keys in base_cfg
                        training_env_cfg = {k: rc[k] for k in base_cfg.keys() if k in rc}
                break
            except Exception:
                pass

    if not isinstance(training_env_cfg, dict):
        return base_cfg  # nothing to merge

    # Only override observation-critical keys to avoid changing wall layout requests
    obs_keys = {
        "grid_size",
        "num_obs_channels",
        "predator_obs_range",
        "prey_obs_range",
        "include_visibility_channel",
        "mask_observation_with_visibility",
        "respect_los_for_movement",
        # Action ranges can affect obs encoding in some setups; include for safety
        "type_1_action_range",
        "type_2_action_range",
    }
    merged = dict(base_cfg)
    for k in obs_keys:
        if k in training_env_cfg:
            merged[k] = training_env_cfg[k]
    return merged


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
    checkpoint_root = "PPO_limited_intake_TYPE_2_EATING_REWARDED_2025-10-08_09-51-32/PPO_PredPreyGrass_9b6a9_00000_0_2025-10-08_09-51-32"
    checkpoint_dir = "checkpoint_000099"
    checkpoint_path = os.path.join(ray_results_dir, checkpoint_root, checkpoint_dir)

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

    # Build config based on config_env and align observation-related keys with training run config
    cfg = dict(config_env)
    cfg = _load_training_env_config_from_run(checkpoint_path, cfg)

    env = env_creator(config=cfg)
    grid_size = (env.grid_size, env.grid_size)
    # Try to configure FOV overlay similar to random policy, but tolerate legacy renderer
    try:
        visualizer = PyGameRenderer(
            grid_size,
            cell_size=32,
            enable_speed_slider=True,
            enable_tooltips=True,
            max_steps=cfg.get("max_steps", 1000),
            predator_obs_range=cfg.get("predator_obs_range"),
            prey_obs_range=cfg.get("prey_obs_range"),
            show_fov=True,
            fov_alpha=40,
            fov_agents=["type_1_predator_0", "type_1_prey_0"],
            fov_respect_walls=True,
        )
    except TypeError:
        visualizer = PyGameRenderer(
            grid_size,
            cell_size=32,
            enable_speed_slider=True,
            enable_tooltips=True,
            max_steps=cfg.get("max_steps", 1000),
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
            try:
                visualizer.update(
                    grass_positions=env.grass_positions,
                    grass_energies=env.grass_energies,
                    step=env.current_step,
                    agents_just_ate=env.agents_just_ate,
                    per_step_agent_data=env.per_step_agent_data,
                    walls=getattr(env, "wall_positions", None),
                )
            except TypeError:
                # Fallback for legacy renderer without `walls` kwarg
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

    try:
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
            walls=getattr(env, "wall_positions", None),
        )
    except TypeError:
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
    try:
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
            walls=getattr(env, "wall_positions", None),
        )
    except TypeError:
        # Fallback for legacy renderer without `walls` kwarg
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
    def _get_group_rewards(env):
        group_rewards = defaultdict(list)
        for uid, stats in env.unique_agent_stats.items():
            reward = stats.get("cumulative_reward", 0.0)
            group, index, reuse = parse_uid(uid)
            group_rewards[group].append((uid, reward, index, reuse))
        return group_rewards

    def _format_ranked_reward_summary(env, total_reward, group_rewards):
        lines = []
        lines.append(f"Total Reward: {total_reward:.2f}\n")
        lines.append("--- Ranked Reward Breakdown per Unique Agent ---\n")
        for group in sorted(group_rewards.keys()):
            lines.append(f"\n## {group.replace('_', ' ').title()} ##\n")
            sorted_group = sorted(
                group_rewards[group], key=lambda x: (-x[1], x[2], x[3])
            )
            for uid, reward, _, _ in sorted_group:
                lines.append(f"{uid:25}: {reward:.2f}\n")
        lines.append("\n--- Aggregated Totals ---\n")
        lines.append(f"Total number of steps: {env.current_step - 1}\n")
        for group in sorted(group_rewards.keys()):
            total = sum(r for _, r, _, _ in group_rewards[group])
            lines.append(f"Total {group.replace('_', ' ').title():25}: {total:.2f}\n")
        lines.append(f"Total All-Agent Reward:           {total_reward:.2f}\n")
        return lines

    group_rewards = _get_group_rewards(env)
    lines = _format_ranked_reward_summary(env, total_reward, group_rewards)
    print("\nEvaluation complete! Total Reward: {:.2f}".format(total_reward))
    for line in lines[1:]:  # skip duplicate first line
        print(line.rstrip())


def save_reward_summary_to_file(env, total_reward, output_dir):
    reward_log_path = os.path.join(output_dir, "reward_summary.txt")
    def _get_group_stats(env):
        group_stats = defaultdict(list)
        for uid, stats in env.unique_agent_stats.items():
            group, index, reuse = parse_uid(uid)
            reward = stats.get("cumulative_reward", 0.0)
            lifetime = (stats.get("death_step") or env.current_step) - stats.get("birth_step", 0)
            offspring = stats.get("offspring_count", 0)
            off_per_step = offspring / lifetime if lifetime > 0 else 0.0
            group_stats[group].append({
                "uid": uid,
                "reward": reward,
                "lifetime": lifetime,
                "offspring": offspring,
                "off_per_step": off_per_step,
                "index": index,
                "reuse": reuse,
            })
        return group_stats

    def _format_ranked_fitness_summary(env, total_reward, group_stats):
        lines = []
        lines.append(f"Total Reward: {total_reward:.2f}\n")
        lines.append("--- Ranked Reward Breakdown per Unique Agent ---\n")
        for group in sorted(group_stats.keys()):
            lines.append(f"\n## {group.replace('_', ' ').title()} ##\n")
            sorted_group = sorted(
                group_stats[group], key=lambda x: (-x["reward"], x["index"], x["reuse"])
            )
            lines.append(f"{'Agent':25} | {'R':>8} | {'Life':>6} | {'Off':>4} | {'Off/100 Steps':>13}\n")
            for entry in sorted_group:
                lines.append(
                    f"{entry['uid']:25} | "
                    f"{entry['reward']:8.2f} | "
                    f"{entry['lifetime']:6} | "
                    f"{entry['offspring']:4} | "
                    f"{100*entry['off_per_step']:13.2f}\n"
                )
            # Print averages for this group
            n = len(sorted_group)
            if n > 0:
                avg_reward = sum(e['reward'] for e in sorted_group) / n
                avg_life = sum(e['lifetime'] for e in sorted_group) / n
                avg_offspring = sum(e['offspring'] for e in sorted_group) / n
                avg_off_per_step = sum(e['off_per_step'] for e in sorted_group) / n
                lines.append(f"{'(Averages)':25} | {avg_reward:8.2f} | {avg_life:6.1f} | {avg_offspring:4.2f} | {100*avg_off_per_step:13.2f}\n")
        lines.append("\n--- Aggregated Totals ---\n")
        lines.append(f"Total number of steps: {env.current_step - 1}\n")
        for group in sorted(group_stats.keys()):
            total = sum(e['reward'] for e in group_stats[group])
            lines.append(f"Total {group.replace('_', ' ').title():25}: {total:.2f}\n")
        lines.append(f"Total All-Agent Reward:           {total_reward:.2f}\n")
        return lines

    group_stats = _get_group_stats(env)
    lines = _format_ranked_fitness_summary(env, total_reward, group_stats)
    with open(reward_log_path, "w") as f:
        for line in lines:
            f.write(line)


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
                "off_per_step": stats.get("offspring_count", 0) / lifetime if lifetime > 0 else 0.0,
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
                f"Off/100 Steps={100*entry['off_per_step']:.2f}"
            )

        # Print averages for this group
        n = len(group_stats[group])
        if n > 0:
            avg_reward = sum(e['reward'] for e in group_stats[group]) / n
            avg_life = sum(e['lifetime'] for e in group_stats[group]) / n
            avg_offspring = sum(e['offspring'] for e in group_stats[group]) / n
            avg_off_per_step = sum(e['off_per_step'] for e in group_stats[group]) / n
            print(f"{'(Averages)':25} | "
                  f"R={avg_reward:.2f} | "
                  f"Life={avg_life:.1f} | "
                  f"Off={avg_offspring:.2f} | "
                  f"Off/100 Steps={100*avg_off_per_step:.2f}")


if __name__ == "__main__":
    seed = 5
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
