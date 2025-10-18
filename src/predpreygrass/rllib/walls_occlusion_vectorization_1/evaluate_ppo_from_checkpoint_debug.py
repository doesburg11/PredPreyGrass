"""
This script loads (pre) trained PPO policy modules (RLModules) directly from a checkpoint
and runs them in the PredPreyGrass environment (walls_occlusion_vectorization_1) for interactive debugging.

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
from predpreygrass.rllib.walls_occlusion_vectorization_1.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.rllib.walls_occlusion_vectorization_1.config.config_env_walls_occlusion_vectorization import config_env
from predpreygrass.rllib.walls_occlusion_vectorization_1.utils.matplot_renderer import CombinedEvolutionVisualizer, PreyDeathCauseVisualizer
from predpreygrass.rllib.walls_occlusion_vectorization_1.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper, LoopControlHelper

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


def detect_and_log_encounters(env, terminations, output_dir, save=True):
    """
    Inspect the current grid occupancy and record interesting encounter cases per cell:
    - predators encounter multiple prey
    - multiple predators encounter one prey
    - multiple predators encounter multiple prey
    - multiple prey encounter grass

    Records what happened using the environment's own outcomes:
    - which prey terminated in this step
    - which predators/prey are marked as having eaten (agents_just_ate)
    - current grass energy at that cell
    """
    # Build occupancy map: pos -> {predators: [...], prey: [...]} using current step data
    occupancy = defaultdict(lambda: {"predators": [], "prey": []})
    for agent_id, pos in env.agent_positions.items():
        key = tuple(pos)
        if "predator" in agent_id:
            occupancy[key]["predators"].append(agent_id)
        elif "prey" in agent_id:
            occupancy[key]["prey"].append(agent_id)

    # Also include prey that terminated this step at their final positions, so we can attribute prey deaths per cell
    terminated_positions = getattr(env, "_terminated_positions_this_step", {}) or {}
    for agent_id, pos in terminated_positions.items():
        if "prey" in agent_id:
            key = tuple(pos)
            if agent_id not in occupancy[key]["prey"]:
                occupancy[key]["prey"].append(agent_id)

    # Build quick grass lookup by position (there should be at most one grass per cell)
    grass_at = {}
    for gid, gpos in env.grass_positions.items():
        grass_at[tuple(gpos)] = gid

    # Prepare output path
    events_path = os.path.join(output_dir, "encounter_events.jsonl") if save else None

    def _json_default(o):
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None:
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()
        if isinstance(o, set):
            return list(o)
        return str(o)

    # Collect events
    events = []
    for pos, groups in occupancy.items():
        preds = groups["predators"]
        preys = groups["prey"]
        has_grass = pos in grass_at
        grass_id = grass_at.get(pos)
        grass_energy = float(env.grass_energies.get(grass_id, 0.0)) if grass_id is not None else 0.0


        # Determine event type(s), including pure/mixed multiple_prey_grass
        event_types = []
        if len(preds) == 1 and len(preys) > 1:
            event_types.append("one_predator_multiple_prey")
        if len(preds) > 1 and len(preys) == 1:
            event_types.append("multiple_predators_one_prey")
        if len(preds) > 1 and len(preys) > 1:
            event_types.append("multiple_predators_multiple_prey")
        # New: explicitly capture simple 1 vs 1 encounters
        if len(preds) == 1 and len(preys) == 1:
            event_types.append("one_predator_one_prey")
        # Split multiple_prey_grass into pure vs mixed
        if len(preys) > 1 and has_grass:
            if len(preds) == 0:
                event_types.append("multiple_prey_grass_pure")
            else:
                event_types.append("multiple_prey_grass_mixed")

        if not event_types:
            continue

        # Outcomes from this step
        prey_terminated = [a for a in preys if terminations.get(a, False)]
        predators_ate = [a for a in preds if a in env.agents_just_ate]
        prey_ate = [a for a in preys if a in env.agents_just_ate]

        pos_int = (int(pos[0]), int(pos[1]))
        ev = {
            "step": int(env.current_step),
            "position": pos_int,
            "event_types": event_types,
            "predators": preds,
            "prey": preys,
            "prey_terminated": prey_terminated,
            "predators_ate": predators_ate,
            "prey_ate": prey_ate,
            "grass_id": grass_id,
            "grass_energy": grass_energy,
        }
        events.append(ev)

    # Persist and print summaries
    if events:
        if save and events_path:
            os.makedirs(output_dir, exist_ok=True)
            with open(events_path, "a") as f:
                for ev in events:
                    f.write(json.dumps(ev, default=_json_default) + "\n")
        for ev in events:
            types_str = ", ".join(ev["event_types"])
            print(f"[ENCOUNTER] step={ev['step']} pos={ev['position']} types=[{types_str}] "
                  f"pred={len(ev['predators'])} prey={len(ev['prey'])} "
                  f"prey_term={ev['prey_terminated']} pred_ate={ev['predators_ate']} prey_ate={ev['prey_ate']} "
                  f"grass={ev['grass_id']} energy={ev['grass_energy']:.1f}")

    return events


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
    checkpoint_root = "PPO_WALLS_OCCLUSION_PROPER_TERMINATION_2025-10-16_23-33-32/PPO_PredPreyGrass_c3a90_00000_0_2025-10-16_23-33-32"
    checkpoint_dir = "checkpoint_000009"
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

    # Detect and log encounters for this step
    try:
        detect_and_log_encounters(env, terminations, output_dir=eval_output_dir, save=SAVE_EVAL_RESULTS)
    except Exception as e:
        print(f"[WARN] Encounter logging failed: {e}")

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


def summarize_encounter_events(output_dir):
    """
    Read encounter_events.jsonl and aggregate counts by event_types.
    Prints a compact table with per-type totals and simple outcome sums.
    """
    events_path = os.path.join(output_dir, "encounter_events.jsonl")
    if not os.path.isfile(events_path):
        print("[ENCOUNTER-SUMMARY] No encounter_events.jsonl found; skipping summary.")
        return

    from collections import defaultdict
    type_counts = defaultdict(int)
    type_cells = defaultdict(set)
    type_prey_term = defaultdict(int)
    type_pred_ate = defaultdict(int)
    type_prey_ate = defaultdict(int)
    type_pred_present = defaultdict(int)
    type_prey_present = defaultdict(int)

    total_lines = 0
    # Dedup sets: unique counts across all event types
    unique_prey_term_pairs = set()  # (step, prey_id)
    unique_pred_ate_pairs = set()   # (step, predator_id)
    unique_prey_ate_pairs = set()   # (step, prey_id) for prey that ate grass
    with open(events_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            total_lines += 1
            pos = tuple(ev.get("position", ()))
            preds = ev.get("predators", []) or []
            preys = ev.get("prey", []) or []
            prey_term = ev.get("prey_terminated", []) or []
            pred_ate = ev.get("predators_ate", []) or []
            prey_ate = ev.get("prey_ate", []) or []
            step = ev.get("step")
            for et in ev.get("event_types", []) or []:
                type_counts[et] += 1
                if pos:
                    type_cells[et].add(pos)
                type_prey_term[et] += len(prey_term)
                type_pred_ate[et] += len(pred_ate)
                type_prey_ate[et] += len(prey_ate)
                type_pred_present[et] += len(preds)
                type_prey_present[et] += len(preys)
            # Update unique, deduplicated sets by step+agent
            if step is not None:
                for a in prey_term:
                    unique_prey_term_pairs.add((step, a))
                for a in pred_ate:
                    unique_pred_ate_pairs.add((step, a))
                for a in prey_ate:
                    unique_prey_ate_pairs.add((step, a))

    if not type_counts:
        print("[ENCOUNTER-SUMMARY] No events found in log.")
        return

    # Print table header
    headers = [
        ("Event Type", 32),
        ("Events", 8),
        ("UniqueCells", 12),
        ("SumPreyTerm", 12),
        ("SumPredAte", 11),
        ("SumPreyAte", 11),
    ]
    title = "Encounter Summary (from encounter_events.jsonl)"
    print("\n" + title)
    print("-" * len(title))
    head_line = " | ".join(h[0].ljust(h[1]) for h in headers)
    print(head_line)
    print("-" * len(head_line))

    # Order by descending event count
    for et, cnt in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        row = [
            et.ljust(32),
            str(cnt).rjust(8),
            str(len(type_cells[et])).rjust(12),
            str(type_prey_term[et]).rjust(12),
            str(type_pred_ate[et]).rjust(11),
            str(type_prey_ate[et]).rjust(11),
        ]
        print(" | ".join(row))

    # Totals line
    total_events = sum(type_counts.values())
    total_cells = len({cell for s in type_cells.values() for cell in s})
    total_prey_term = sum(type_prey_term.values())
    total_pred_ate = sum(type_pred_ate.values())
    total_prey_ate = sum(type_prey_ate.values())
    print("-" * len(head_line))
    print(
        " | ".join(
            [
                "TOTAL".ljust(32),
                str(total_events).rjust(8),
                str(total_cells).rjust(12),
                str(total_prey_term).rjust(12),
                str(total_pred_ate).rjust(11),
                str(total_prey_ate).rjust(11),
            ]
        )
    )
    # Print deduplicated totals (unique by step+agent id across all event types)
    uniq_prey_term = len(unique_prey_term_pairs)
    uniq_pred_ate = len(unique_pred_ate_pairs)
    uniq_prey_ate = len(unique_prey_ate_pairs)
    print(
        " | ".join(
            [
                "UNIQUE_TOTALS".ljust(32),
                "-".rjust(8),
                "-".rjust(12),
                str(uniq_prey_term).rjust(12),
                str(uniq_pred_ate).rjust(11),
                "-".rjust(11),
            ]
        )
    )
    print(
        " | ".join(
            [
                "UNIQUE_TOTALS_PREY_ATE".ljust(32),
                "-".rjust(8),
                "-".rjust(12),
                "-".rjust(12),
                "-".rjust(11),
                str(uniq_prey_ate).rjust(11),
            ]
        )
    )


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
        # Post-run encounter summary
        try:
            summarize_encounter_events(eval_output_dir)
        except Exception as e:
            print(f"[WARN] Encounter summary failed: {e}")
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
