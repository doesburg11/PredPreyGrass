"""
This script loads (pre) trained PPO policy modules (RLModules) directly from a checkpoint
and runs them in the PredPreyGrass environment for interactive debugging.

The simulation can be controlled in real-time using a graphical interface.
- [Space] Pause/Unpause
- [->] Step Forward
- [<-] Step Backward
- Tooltips are available to inspect agent IDs, positions, energies.

The environment is rendered using PyGame, and the simulation can be recorded as a video.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pygame
import ray
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env
import torch

SEED = 2
SAVE_EVAL_RESULTS = True
SAVE_MOVIE = True
MOVIE_FILENAME = "cooperative_hunting.mp4"
MOVIE_FPS = 10
DISPLAY_SCALE = 0.95
TRAINED_EXAMPLE_DIR = os.getenv("TRAINED_EXAMPLE_DIR")


def _prepend_snapshot_source() -> None:
    script_path = Path(__file__).resolve()
    try:
        if script_path.parents[2].name == "predpreygrass" and script_path.parents[1].name == "rllib":
            source_root = script_path.parents[3]
            if source_root.name in {"REPRODUCE_CODE", "SOURCE_CODE"}:
                source_root_str = str(source_root)
                if source_root_str not in sys.path:
                    sys.path.insert(0, source_root_str)
    except IndexError:
        return


_prepend_snapshot_source()

PredPreyGrass = None
config_env = None
CombinedEvolutionVisualizer = None
PreyDeathCauseVisualizer = None
PyGameRenderer = None
ViewerControlHelper = None
LoopControlHelper = None
aggregate_capture_outcomes_from_event_log = None
aggregate_join_choices = None
compute_opportunity_preference_metrics = None


def prepend_example_sources() -> None:
    if not TRAINED_EXAMPLE_DIR:
        return
    example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
    source_dirs = [
        example_dir / "REPRODUCE_CODE",
        example_dir / "SOURCE_CODE",
        example_dir / "eval" / "REPRODUCE_CODE",
        example_dir / "eval" / "SOURCE_CODE",
    ]
    for path in source_dirs:
        if path.is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def load_predpreygrass_modules() -> None:
    global PredPreyGrass
    global config_env
    global CombinedEvolutionVisualizer, PreyDeathCauseVisualizer
    global PyGameRenderer, ViewerControlHelper, LoopControlHelper
    global aggregate_capture_outcomes_from_event_log, aggregate_join_choices, compute_opportunity_preference_metrics

    from predpreygrass.rllib.checkpoint_genomes.predpreygrass_rllib_env import PredPreyGrass as _PredPreyGrass
    from predpreygrass.rllib.checkpoint_genomes.config.config_env_checkpoint_genomes import config_env as _config_env
    from predpreygrass.rllib.checkpoint_genomes.utils.matplot_renderer import (
        CombinedEvolutionVisualizer as _CombinedEvolutionVisualizer,
        PreyDeathCauseVisualizer as _PreyDeathCauseVisualizer,
    )
    from predpreygrass.rllib.checkpoint_genomes.utils.pygame_grid_renderer_rllib import (
        PyGameRenderer as _PyGameRenderer,
        ViewerControlHelper as _ViewerControlHelper,
        LoopControlHelper as _LoopControlHelper,
    )
    from predpreygrass.rllib.checkpoint_genomes.utils.defection_metrics import (
        aggregate_capture_outcomes_from_event_log as _aggregate_capture_outcomes_from_event_log,
        aggregate_join_choices as _aggregate_join_choices,
        compute_opportunity_preference_metrics as _compute_opportunity_preference_metrics,
    )

    PredPreyGrass = _PredPreyGrass
    config_env = _config_env
    CombinedEvolutionVisualizer = _CombinedEvolutionVisualizer
    PreyDeathCauseVisualizer = _PreyDeathCauseVisualizer
    PyGameRenderer = _PyGameRenderer
    ViewerControlHelper = _ViewerControlHelper
    LoopControlHelper = _LoopControlHelper
    aggregate_capture_outcomes_from_event_log = _aggregate_capture_outcomes_from_event_log
    aggregate_join_choices = _aggregate_join_choices
    compute_opportunity_preference_metrics = _compute_opportunity_preference_metrics


def env_creator(config):
    return PredPreyGrass(config)


_SNAPSHOT_EXCLUDE_DIRS = {
    "genomes",
    "ray_results",
    "ray_results_failed",
    "trained_examples",
    "__pycache__",
}


def copy_module_snapshot(source_dir: Path) -> None:
    module_dir = Path(__file__).resolve().parent
    module_name = module_dir.name

    pkg_root = source_dir / "predpreygrass"
    rllib_root = pkg_root / "rllib"
    rllib_root.mkdir(parents=True, exist_ok=True)

    pkg_init = pkg_root / "__init__.py"
    if not pkg_init.exists():
        pkg_init.write_text("")
    rllib_init = rllib_root / "__init__.py"
    if not rllib_init.exists():
        rllib_init.write_text("")

    dest_dir = rllib_root / module_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    def _ignore(path: str, entries):
        ignored = []
        for entry in entries:
            if entry in _SNAPSHOT_EXCLUDE_DIRS:
                ignored.append(entry)
                continue
            if entry.endswith(".pyc"):
                ignored.append(entry)
        return ignored

    shutil.copytree(module_dir, dest_dir, ignore=_ignore)

    assets_src = None
    for parent in module_dir.parents:
        if parent.name == "REPRODUCE_CODE":
            candidate = parent / "assets" / "images" / "icons"
            if candidate.is_dir():
                assets_src = candidate
            break
    if assets_src is None:
        candidate = module_dir.parents[4] / "assets" / "images" / "icons"
        if candidate.is_dir():
            assets_src = candidate
    if assets_src:
        assets_dest = source_dir / "assets" / "images" / "icons"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        assets_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(assets_src, assets_dest)


def write_pip_freeze(output_path: Path) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        output_path.write_text(result.stdout)
        if result.stderr:
            (output_path.parent / "pip_freeze_eval_stderr.txt").write_text(result.stderr)
    except Exception as exc:
        output_path.write_text(f"pip freeze failed: {exc}")


def _find_run_config_path(checkpoint_path: str):
    candidates = [
        Path(checkpoint_path).parent / "run_config.json",
        Path(checkpoint_path).parent.parent / "run_config.json",
    ]
    if TRAINED_EXAMPLE_DIR:
        candidates.append(Path(TRAINED_EXAMPLE_DIR).expanduser().resolve() / "REPRODUCE_CODE" / "CONFIG" / "run_config.json")
        candidates.append(Path(TRAINED_EXAMPLE_DIR).expanduser().resolve() / "CONFIG" / "run_config.json")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def prepare_eval_output_dir(eval_output_dir: Path, env_cfg: dict, checkpoint_path: str) -> None:
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    reproduce_dir = eval_output_dir / "REPRODUCE_CODE"
    reproduce_dir.mkdir(parents=True, exist_ok=True)
    copy_module_snapshot(reproduce_dir)
    write_pip_freeze(reproduce_dir / "pip_freeze_eval.txt")

    config_dir = reproduce_dir / "CONFIG"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config_env.json", "w") as f:
        json.dump(env_cfg, f, indent=4)
    run_config = _find_run_config_path(checkpoint_path)
    if run_config:
        shutil.copy2(run_config, config_dir / "run_config.json")

    with open(eval_output_dir / "config_env.json", "w") as f:
        json.dump(env_cfg, f, indent=4)


def resolve_trained_example_checkpoint(example_dir: Path) -> Path:
    checkpoint_dir = example_dir / "checkpoint"
    if checkpoint_dir.is_dir():
        return checkpoint_dir
    candidates = sorted(example_dir.glob("checkpoint_*"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {example_dir}")
    raise FileExistsError(f"Multiple checkpoints found in {example_dir}; please keep only one.")


def _load_training_env_config_from_run(checkpoint_path, base_cfg):
    """
    Try to locate a run_config.json near the checkpoint and merge its env settings
    into the provided base_cfg. This aligns evaluation observations with the
    shapes the policy network was trained on (avoids matmul shape errors).
    """
    training_env_cfg = None
    cand = _find_run_config_path(checkpoint_path)
    if cand and cand.is_file():
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
        except Exception:
            pass

    if not isinstance(training_env_cfg, dict):
        return base_cfg  # nothing to merge

    # Only override observation-critical keys to avoid changing wall layout requests
    obs_keys = {
        "grid_size",
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
    if logits.dim() == 2 and logits.size(0) == 1:
        logits = logits[0]

    action_space = getattr(policy_module, "action_space", None)
    if hasattr(action_space, "nvec"):
        actions = []
        idx = 0
        for n in list(action_space.nvec):
            segment = logits[idx:idx + n]
            if deterministic:
                act = int(torch.argmax(segment).item())
            else:
                act = int(torch.distributions.Categorical(logits=segment).sample().item())
            actions.append(act)
            idx += n
        return actions

    if deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    dist = torch.distributions.Categorical(logits=logits)
    return int(dist.sample().item())

def setup_environment_and_visualizer(now):
    if TRAINED_EXAMPLE_DIR:
        example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
        checkpoint_path = resolve_trained_example_checkpoint(example_dir)
        eval_root = example_dir / "eval" / "runs"
        eval_output_dir = eval_root / f"eval_{checkpoint_path.name}_{now}"
    else:
        ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/sexual_reproduction/ray_results/"
        checkpoint_root = "SEXUAL_REPRODUCTION_2026-02-10_12-55-20/PPO_PredPreyGrass_601fe_00000_0_2026-02-10_12-55-20/"
        checkpoint_nr = "checkpoint_000009"
        checkpoint_path = Path(ray_results_dir) / checkpoint_root / checkpoint_nr
        eval_output_dir = Path(checkpoint_path) / f"eval_{checkpoint_nr}_{now}"

    rl_module_dir = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    module_paths = {}

    if rl_module_dir.is_dir():
        for pid in os.listdir(rl_module_dir):
            path = rl_module_dir / pid
            if path.is_dir():
                module_paths[pid] = str(path)
    else:
        raise FileNotFoundError(f"RLModule directory not found: {rl_module_dir}")

    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}

    # Build config based on config_env and align observation-related keys with training run config
    cfg = dict(config_env)

    env = env_creator(config=cfg)
    grid_size = (env.grid_size, env.grid_size)
    # Try to configure FOV overlay similar to random policy, but tolerate legacy renderer
    try:
        visualizer = PyGameRenderer(
            grid_size,
            cell_size=32,
            scale=DISPLAY_SCALE,
            enable_speed_slider=True,
            enable_tooltips=True,
            max_steps=cfg.get("max_steps", 1000),
            predator_obs_range=cfg.get("predator_obs_range"),
            prey_obs_range=cfg.get("prey_obs_range"),
            show_fov=True,
            fov_alpha=40,
            fov_agents=["type_1_predator_0", "type_1_prey_0"],
            fov_respect_walls=True,
            n_possible_type_2_predators=cfg.get("n_possible_type_2_predators"),
            n_possible_type_2_prey=cfg.get("n_possible_type_2_prey"),
        )
    except TypeError:
        visualizer = PyGameRenderer(
            grid_size,
            cell_size=32,
            enable_speed_slider=True,
            enable_tooltips=True,
            max_steps=cfg.get("max_steps", 1000),
            n_possible_type_2_predators=cfg.get("n_possible_type_2_predators"),
            n_possible_type_2_prey=cfg.get("n_possible_type_2_prey"),
        )

    if SAVE_EVAL_RESULTS:
        prepare_eval_output_dir(Path(eval_output_dir), cfg, str(checkpoint_path))
        ceviz = CombinedEvolutionVisualizer(
            destination_path=str(eval_output_dir),
            timestamp=now,
            n_possible_type_1_predators=cfg.get("n_possible_type_1_predators"),
            n_possible_type_1_prey=cfg.get("n_possible_type_1_prey"),
            n_possible_type_2_predators=cfg.get("n_possible_type_2_predators"),
            n_possible_type_2_prey=cfg.get("n_possible_type_2_prey"),
        )
        pdviz = PreyDeathCauseVisualizer(destination_path=str(eval_output_dir), timestamp=now)
    else:
        ceviz = CombinedEvolutionVisualizer(
            destination_path=None,
            timestamp=now,
            n_possible_type_1_predators=cfg.get("n_possible_type_1_predators"),
            n_possible_type_1_prey=cfg.get("n_possible_type_1_prey"),
            n_possible_type_2_predators=cfg.get("n_possible_type_2_predators"),
            n_possible_type_2_prey=cfg.get("n_possible_type_2_prey"),
        )
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
                walls=getattr(env, "wall_positions", None),
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
            dead_prey=getattr(env, "dead_prey", None),
            coop_events=getattr(env, "team_capture_events", None),
        )
    except TypeError:
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
            coop_events=getattr(env, "team_capture_events", None),
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
            dead_prey=getattr(env, "dead_prey", None),
            coop_events=getattr(env, "team_capture_events", None),
        )
    except TypeError:
        # Fallback for legacy renderer without `walls` kwarg
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
            coop_events=getattr(env, "team_capture_events", None),
        )

def parse_uid(uid):
    """
    Parse agent id like 'type_1_predator_2#17' into sortable components:
    → ('type_1_predator', 2, 17)
    """
    match = re.match(r"(type_\d+_(?:predator|prey))_(\d+)(?:#(\d+))?", uid)
    if match:
        group, idx, lifetime = match.groups()
        return group, int(idx), int(lifetime) if lifetime is not None else 0
    else:
        return uid, float("inf"), float("inf")  # fallback for malformed ids

def print_ranked_reward_summary(env, total_reward):
    def _get_group_rewards(env):
        group_rewards = defaultdict(list)
        for uid, stats in env.get_all_agent_stats().items():
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


def compute_defection_metrics(env):
    join_stats = aggregate_join_choices(env.per_step_agent_data)
    capture_stats = aggregate_capture_outcomes_from_event_log(env.agent_event_log)
    opportunity_stats = compute_opportunity_preference_metrics(env.per_step_agent_data)
    return {
        "steps": env.current_step,
        "join_defect": join_stats,
        "capture_outcomes": capture_stats,
        "opportunity_preferences": opportunity_stats,
    }

def save_reward_summary_to_file(env, total_reward, output_dir):
    reward_log_path = os.path.join(output_dir, "reward_summary.txt")
    def _get_group_stats(env):
        group_stats = defaultdict(list)
        for uid, stats in env.get_all_agent_stats().items():
            group, index, reuse = parse_uid(uid)
            reward = stats.get("cumulative_reward", 0.0)
            lifetime = (stats.get("death_step") or env.current_step) - stats.get("birth_step", 0)
            offspring = stats.get("offspring_count", 0)
            off_per_step = offspring / lifetime if lifetime > 0 else 0.0
            lineage_bonus = stats.get("lineage_reward_total", 0.0)
            fert_cap = stats.get("max_fertility_age")
            fert_exp = stats.get("fertility_expired_step")
            fert_blocked = stats.get("fertility_blocked_attempts", 0)
            group_stats[group].append({
                "uid": uid,
                "reward": reward,
                "lifetime": lifetime,
                "offspring": offspring,
                "off_per_step": off_per_step,
                "index": index,
                "reuse": reuse,
                "lineage_bonus": lineage_bonus,
                "max_fertility_age": fert_cap,
                "fertility_expired_step": fert_exp,
                "fertility_blocked_attempts": fert_blocked,
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
            lines.append(
                f"{'Agent':25} | {'R':>8} | {'Life':>6} | {'Off':>4} | {'Off/100':>8} | "
                f"{'Lineage':>8} | {'FertCap':>7} | {'FertExp':>7} | {'FBlk':>5}\n"
            )
            for entry in sorted_group:
                cap_val = entry["max_fertility_age"]
                if isinstance(cap_val, (int, float)) and cap_val >= 0:
                    fert_cap_str = str(int(cap_val))
                else:
                    fert_cap_str = "∞"
                fert_exp = entry["fertility_expired_step"]
                fert_exp_str = "--" if fert_exp is None else str(int(fert_exp))
                lines.append(
                    f"{entry['uid']:25} | "
                    f"{entry['reward']:8.2f} | "
                    f"{entry['lifetime']:6} | "
                    f"{entry['offspring']:4} | "
                    f"{100*entry['off_per_step']:8.2f} | "
                    f"{entry['lineage_bonus']:8.2f} | "
                    f"{fert_cap_str:>7} | "
                    f"{fert_exp_str:>7} | "
                    f"{entry['fertility_blocked_attempts']:5}\n"
                )
            # Print averages for this group
            n = len(sorted_group)
            if n > 0:
                avg_reward = sum(e['reward'] for e in sorted_group) / n
                avg_life = sum(e['lifetime'] for e in sorted_group) / n
                avg_offspring = sum(e['offspring'] for e in sorted_group) / n
                avg_off_per_step = sum(e['off_per_step'] for e in sorted_group) / n
                avg_lineage = sum(e['lineage_bonus'] for e in sorted_group) / n
                avg_fert_block = sum(e['fertility_blocked_attempts'] for e in sorted_group) / n
                fert_caps = [e['max_fertility_age'] for e in sorted_group if isinstance(e['max_fertility_age'], (int, float))]
                avg_fert_cap = sum(fert_caps) / len(fert_caps) if fert_caps else None
                fert_exp_vals = [e['fertility_expired_step'] for e in sorted_group if isinstance(e['fertility_expired_step'], (int, float))]
                avg_fert_exp = sum(fert_exp_vals) / len(fert_exp_vals) if fert_exp_vals else None
                cap_str = "--" if avg_fert_cap is None else f"{avg_fert_cap:7.1f}"
                exp_str = "--" if avg_fert_exp is None else f"{avg_fert_exp:7.1f}"
                lines.append(
                    f"{'(Averages)':25} | {avg_reward:8.2f} | {avg_life:6.1f} | {avg_offspring:4.2f} | "
                    f"{100*avg_off_per_step:8.2f} | {avg_lineage:8.2f} | {cap_str} | {exp_str} | {avg_fert_block:5.2f}\n"
                )
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
    for uid, stats in env.get_all_agent_stats().items():
        group, index, reuse = parse_uid(uid)
        lifetime = (stats["death_step"] or env.current_step) - stats["birth_step"]
        group_stats[group].append(
            {
                "uid": uid,
                "reward": stats.get("cumulative_reward", 0.0),
                "lifetime": lifetime,
                "offspring": stats.get("offspring_count", 0),
                "off_per_step": stats.get("offspring_count", 0) / lifetime if lifetime > 0 else 0.0,
                "lineage_bonus": stats.get("lineage_reward_total", 0.0),
                "max_fertility_age": stats.get("max_fertility_age"),
                "fertility_expired_step": stats.get("fertility_expired_step"),
                "fertility_blocked_attempts": stats.get("fertility_blocked_attempts", 0),
            }
        )

    for group in sorted(group_stats.keys()):
        print(f"\n## {group.replace('_', ' ').title()} ##")
        sorted_group = sorted(group_stats[group], key=lambda x: (-x["offspring"], -x["reward"], -x["lifetime"]))
        print(
            f"{'Agent':25} | {'R':>8} | {'Life':>6} | {'Off':>4} | {'Off/100':>8} | "
            f"{'Lineage':>8} | {'FertCap':>7} | {'FertExp':>7} | {'FBlk':>5}"
        )
        for entry in sorted_group[:10]:  # top 10
            cap_val = entry.get("max_fertility_age")
            if isinstance(cap_val, (int, float)) and cap_val >= 0:
                cap_str = str(int(cap_val))
            else:
                cap_str = "∞"
            fert_exp = entry.get("fertility_expired_step")
            fert_exp_str = "--" if fert_exp is None else str(int(fert_exp))
            print(
                f"{entry['uid']:25} | "
                f"{entry['reward']:8.2f} | "
                f"{entry['lifetime']:6} | "
                f"{entry['offspring']:4} | "
                f"{100*entry['off_per_step']:8.2f} | "
                f"{entry.get('lineage_bonus', 0.0):8.2f} | "
                f"{cap_str:>7} | "
                f"{fert_exp_str:>7} | "
                f"{entry.get('fertility_blocked_attempts', 0):5}"
            )
        # Print averages for this group
        n = len(sorted_group)
        if n > 0:
            avg_reward = sum(e['reward'] for e in sorted_group) / n
            avg_life = sum(e['lifetime'] for e in sorted_group) / n
            avg_offspring = sum(e['offspring'] for e in sorted_group) / n
            avg_off_per_step = sum(e['off_per_step'] for e in sorted_group) / n
            avg_lineage = sum(e.get('lineage_bonus', 0.0) for e in sorted_group) / n
            avg_fert_block = sum(e.get('fertility_blocked_attempts', 0) for e in sorted_group) / n
            fert_caps = [e.get('max_fertility_age') for e in sorted_group if isinstance(e.get('max_fertility_age'), (int, float))]
            avg_cap = sum(fert_caps) / len(fert_caps) if fert_caps else None
            fert_exp_vals = [e.get('fertility_expired_step') for e in sorted_group if isinstance(e.get('fertility_expired_step'), (int, float))]
            avg_exp = sum(fert_exp_vals) / len(fert_exp_vals) if fert_exp_vals else None
            cap_str = "∞" if avg_cap is None else f"{avg_cap:.1f}"
            exp_str = "--" if avg_exp is None else f"{avg_exp:.1f}"
            print(
                f"{'(Averages)':25} | {avg_reward:8.2f} | {avg_life:6.1f} | {avg_offspring:4.2f} | "
                f"{100*avg_off_per_step:8.2f} | {avg_lineage:8.2f} | {cap_str:>7} | {exp_str:>7} | {avg_fert_block:5.2f}"
            )

if __name__ == "__main__":
    prepend_example_sources()
    load_predpreygrass_modules()
    seed = SEED
    ray.init(ignore_reinit_error=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    register_env("PredPreyGrass", lambda config: env_creator(config))

    env, visualizer, rl_modules, ceviz, pdviz, eval_output_dir = setup_environment_and_visualizer(now)
    observations, _ = env.reset(seed=seed)

    if SAVE_MOVIE:
        screen_width = visualizer.screen.get_width()
        screen_height = visualizer.screen.get_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        movie_path = Path(eval_output_dir) / "visuals" / MOVIE_FILENAME
        movie_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(movie_path), fourcc, MOVIE_FPS, (screen_width, screen_height))
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

    if hasattr(env, "reproduction_blocked_due_to_fertility_predator"):
        print("\n--- Fertility Blocks ---")
        print(
            f"Predators blocked: {getattr(env, 'reproduction_blocked_due_to_fertility_predator', 0)} | "
            f"Prey blocked: {getattr(env, 'reproduction_blocked_due_to_fertility_prey', 0)}"
        )

    defection_metrics = compute_defection_metrics(env)

    if SAVE_EVAL_RESULTS:
        save_reward_summary_to_file(env, total_reward, eval_output_dir)
        energy_log_path = os.path.join(eval_output_dir, "energy_by_type.json")
        with open(energy_log_path, "w") as f:
            json.dump(energy_by_type_series, f, indent=2)
        # Export per-agent event log for offline analysis
        def _convert(obj):
            if isinstance(obj, (int, float, str)) or obj is None:
                return obj
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            try:
                return obj.item()
            except Exception:
                return str(obj)

        def _augment_eating_events(event_log, step_agent_data):
            """Add energy_before plus consumer/resource positions to eating events without mutating the live log."""
            log = deepcopy(event_log)

            # Drop unwanted categories before augmenting events
            for record in log.values():
                record.pop("diet_events", None)
                record.pop("lifecycle_events", None)

            def _get_position(t, agent_id):
                if not isinstance(step_agent_data, list):
                    return None
                if not isinstance(t, (int, float, np.integer)):
                    return None
                idx = int(t)
                if idx < 0 or idx >= len(step_agent_data):
                    return None
                step_entry = step_agent_data[idx]
                if not isinstance(step_entry, dict):
                    return None
                agent_entry = step_entry.get(agent_id)
                if not agent_entry:
                    return None
                pos = agent_entry.get("position")
                if pos is None:
                    return None
                if hasattr(pos, "tolist"):
                    pos = pos.tolist()
                elif isinstance(pos, tuple):
                    pos = list(pos)
                return pos

            def _get_resource_position(t, resource_id):
                if resource_id is None:
                    return None
                return _get_position(t, resource_id)

            def _augment_single_event(evt, agent_id):
                bite = evt.get("bite_size")
                energy_after = evt.get("energy_after")
                energy_before = evt.get("energy_before")
                if energy_before is None:
                    if isinstance(bite, (int, float)) and isinstance(energy_after, (int, float)):
                        energy_before = float(energy_after - bite)
                energy_resource = evt.get("energy_resource")
                pos = evt.get("position_consumer") or evt.get("position")
                if pos is None:
                    pos = _get_position(evt.get("t"), agent_id)
                if hasattr(pos, "tolist"):
                    pos = pos.tolist()
                elif isinstance(pos, tuple):
                    pos = list(pos)
                resource_id = evt.get("id_resource") or evt.get("id_eaten")
                pos_resource = evt.get("position_resource")
                if pos_resource is None:
                    pos_resource = _get_resource_position(evt.get("t"), resource_id)
                if hasattr(pos_resource, "tolist"):
                    pos_resource = pos_resource.tolist()
                elif isinstance(pos_resource, tuple):
                    pos_resource = list(pos_resource)

                ordered = {}
                for key in ("t",):
                    if key in evt:
                        ordered[key] = evt[key]
                if resource_id is not None:
                    ordered["id_resource"] = resource_id
                if energy_resource is not None:
                    ordered["energy_resource"] = energy_resource
                if pos_resource is not None:
                    ordered["position_resource"] = pos_resource
                if pos is not None:
                    ordered["position_consumer"] = pos
                if energy_before is not None:
                    ordered["energy_before"] = energy_before
                for key in ("bite_size",):
                    if key in evt:
                        ordered[key] = evt[key]
                for key in ("energy_after", "team_capture", "predator_list"):
                    if key in evt:
                        ordered[key] = evt[key]
                # Append any remaining keys in original order
                for k, v in evt.items():
                    if k in ("id_eaten",):  # drop legacy key once id_resource is set
                        continue
                    if k not in ordered:
                        ordered[k] = v
                return ordered

            for agent_id, record in log.items():
                eating_events = record.get("eating_events", [])
                record["eating_events"] = [_augment_single_event(evt, agent_id) for evt in eating_events]
            return log

        event_log_path = os.path.join(eval_output_dir, f"agent_event_log_{now}.json")
        with open(event_log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    aid: _convert(rec)
                    for aid, rec in _augment_eating_events(env.agent_event_log, env.per_step_agent_data).items()
                },
                f,
                indent=2,
            )
        print(f"Agent event log written to: {event_log_path}")
        defection_metrics_path = os.path.join(eval_output_dir, "defection_metrics.json")
        with open(defection_metrics_path, "w") as f:
            json.dump(defection_metrics, f, indent=2)
        print(f"Defection metrics written to: {defection_metrics_path}")
    # Always show plots on screen
    ceviz.plot()
    if SAVE_EVAL_RESULTS:
        # Export all unique agent fitness stats
        agent_fitness_path = os.path.join(eval_output_dir, "agent_fitness_stats.json")
        with open(agent_fitness_path, "w") as f:
            json.dump(env.get_all_agent_stats(), f, indent=2)

    print("Defection metrics:")
    print(json.dumps(defection_metrics, indent=2))

    print_ranked_fitness_summary(env)

    print("\nNumber of steps simulated:", env.current_step)

    pygame.quit()
    ray.shutdown()
