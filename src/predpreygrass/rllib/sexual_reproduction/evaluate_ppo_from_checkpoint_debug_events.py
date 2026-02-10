"""
Minimal debug runner for sexual_reproduction that prints new events each step.
Controls:
- [Space] Pause/Unpause
- [->] Step Forward (single step)
- [ESC] Quit
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pygame
import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env

SEED = 27
DISPLAY_SCALE = 0.95
TRAINED_EXAMPLE_DIR = os.getenv("TRAINED_EXAMPLE_DIR")

RAY_RESULTS_DIR = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/sexual_reproduction/ray_results/"
CHECKPOINT_ROOT = (
    "SEXUAL_REPRODUCTION_2026-02-09_23-24-52/"
    "PPO_PredPreyGrass_27c47_00000_0_2026-02-09_23-24-53"
)
CHECKPOINT_NR = "checkpoint_000039"


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
PyGameRenderer = None
ViewerControlHelper = None
LoopControlHelper = None


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
    global PredPreyGrass, config_env, PyGameRenderer, ViewerControlHelper, LoopControlHelper

    from predpreygrass.rllib.sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass as _PredPreyGrass
    from predpreygrass.rllib.sexual_reproduction.config.config_env_sexual_reproduction import (
        config_env as _config_env,
    )
    from predpreygrass.rllib.sexual_reproduction.utils.pygame_grid_renderer_rllib import (
        PyGameRenderer as _PyGameRenderer,
        ViewerControlHelper as _ViewerControlHelper,
        LoopControlHelper as _LoopControlHelper,
    )

    PredPreyGrass = _PredPreyGrass
    config_env = _config_env
    PyGameRenderer = _PyGameRenderer
    ViewerControlHelper = _ViewerControlHelper
    LoopControlHelper = _LoopControlHelper


def env_creator(config):
    return PredPreyGrass(config)


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


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
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
            segment = logits[idx : idx + n]
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


def setup_environment_and_visualizer(now: str):
    if TRAINED_EXAMPLE_DIR:
        example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
        checkpoint_path = resolve_trained_example_checkpoint(example_dir)
    else:
        checkpoint_path = Path(RAY_RESULTS_DIR) / CHECKPOINT_ROOT / CHECKPOINT_NR

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

    cfg = dict(config_env)
    env = env_creator(config=cfg)
    grid_size = (env.grid_size, env.grid_size)

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

    return env, visualizer, rl_modules


def print_new_events(env, last_counts):
    event_keys = ("reproduction_events",)
    for agent_id, rec in env.agent_event_log.items():
        agent_counts = last_counts.setdefault(agent_id, {})
        for key in event_keys:
            events = rec.get(key, []) or []
            last_idx = int(agent_counts.get(key, 0))
            for evt in events[last_idx:]:
                if "mate_id" not in evt:
                    continue
                print(f"[mating] {agent_id}: {evt}")
            agent_counts[key] = len(events)


def step_forward(env, observations, rl_modules, control, visualizer, clock, last_event_counts):
    action_dict = {}
    for agent_id in env.agents:
        group = policy_mapping_fn(agent_id)
        if group in rl_modules:
            action_dict[agent_id] = policy_pi(observations[agent_id], rl_modules[group], deterministic=True)

    observations, rewards, terminations, truncations, _ = env.step(action_dict)

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
    control.step_once = False
    clock.tick(visualizer.target_fps)

    print_new_events(env, last_event_counts)
    return observations, terminations, truncations


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
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
            coop_events=getattr(env, "team_capture_events", None),
        )


def main() -> None:
    prepend_example_sources()
    load_predpreygrass_modules()

    ray.init(ignore_reinit_error=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    register_env("PredPreyGrass", lambda config: env_creator(config))

    env, visualizer, rl_modules = setup_environment_and_visualizer(now)
    observations, _ = env.reset(seed=SEED)

    control = ViewerControlHelper(initial_paused=True)
    loop_helper = LoopControlHelper()
    control.fps_slider_rect = visualizer.slider_rect
    control.fps_slider_update_fn = lambda new_fps: setattr(visualizer, "target_fps", new_fps)
    control.visualizer = visualizer
    control.step_once = True

    clock = pygame.time.Clock()
    last_event_counts = {}

    while not loop_helper.simulation_terminated:
        control.handle_events()
        if control.step_backward:
            control.step_backward = False

        if loop_helper.should_step(control):
            observations, terminations, truncations = step_forward(
                env,
                observations,
                rl_modules,
                control,
                visualizer,
                clock,
                last_event_counts,
            )
            loop_helper.update_simulation_terminated(terminations, truncations)
        else:
            render_static_if_paused(env, visualizer)
            pygame.time.wait(50)

    print(f"Simulation ended at step {env.current_step}.")


if __name__ == "__main__":
    main()
