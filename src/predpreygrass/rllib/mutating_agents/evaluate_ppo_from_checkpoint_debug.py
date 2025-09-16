"""
This script loads (pre) trained PPO policy modules (RLModules) directly from a checkpoint
and runs them in the PredPreyGrass environment (mutating_agents) for interactive debugging.

This version differs from v1_0 in that it includes two types of predators and two types of prey, 
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
from predpreygrass.rllib.mutating_agents.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.rllib.mutating_agents.config.config_env_eval import config_env
from predpreygrass.rllib.mutating_agents.utils.matplot_renderer import CombinedEvolutionVisualizer, PreyDeathCauseVisualizer
from predpreygrass.rllib.mutating_agents.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper, LoopControlHelper

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

SAVE_EVAL_RESULTS = False  # Save plots of evolution and prey death causes
SAVE_MOVIE = False
MOVIE_FILENAME = "simulation.mp4"
MOVIE_FPS = 10

# --- NumPy checkpoint compatibility shim ------------------------------------
# Some older checkpoints reference 'numpy._core.numeric' during pickle load.
# Provide an alias to 'numpy.core.numeric' if missing to avoid ModuleNotFoundError.
import sys, types, importlib
try:
    if 'numpy._core.numeric' not in sys.modules:
        core_numeric = importlib.import_module('numpy.core.numeric')
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = types.ModuleType('numpy._core')
        sys.modules['numpy._core.numeric'] = core_numeric
except Exception:  # Silent; only affects legacy checkpoints
    pass


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
    # --- Set your checkpoint path (directory that contains 'learner_group/learner/rl_module/...' ) ---
    script_dir = os.path.dirname(__file__)
    checkpoint_dir = "checkpoint_iter_1000"
    checkpoint_path = os.path.join(
        script_dir,
        "trained_policies",
        "incl_speed_2",
        checkpoint_dir,
    )

    training_dir = os.path.dirname(checkpoint_path)
    eval_output_dir = os.path.join(training_dir, f"eval_{checkpoint_dir}_{now}")

    rl_module_dir = os.path.join(checkpoint_path, "learner_group", "learner", "rl_module")
    module_paths = {}

    if os.path.isdir(rl_module_dir):
        for pid in os.listdir(rl_module_dir):
            path = os.path.join(rl_module_dir, pid)
            if os.path.isdir(path):
                module_paths[pid] = path
    else:
        raise FileNotFoundError(f"RLModule directory not found: {rl_module_dir}")

    rl_modules = {}
    for pid, path in module_paths.items():
        try:
            rl_modules[pid] = RLModule.from_checkpoint(path)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Failed to load RLModule '{pid}' from {path}: {e}\n"
                "Likely a dependency version mismatch (e.g., NumPy internal path). "
                "Shim for numpy._core.numeric applied; if this persists, ensure training and eval environments align."
            ) from e

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
            observations = {agent: env._get_observation(agent) for agent in env.agents}
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
    # Print final reward summary after evaluation
    print(f"\nEvaluation complete! Total Reward: {total_reward}")
    print("\n--- Reward Breakdown per Agent ---")
    predator_rewards, prey_rewards = [], []
    for agent_id, reward in env.cumulative_rewards.items():
        print(f"{agent_id:15}: {reward:.2f}")
        if "predator" in agent_id:
            predator_rewards.append(reward)
        elif "prey" in agent_id:
            prey_rewards.append(reward)
    print("\n--- Aggregated Rewards ---")
    print(f"Total number of steps: {env.current_step-1}")
    print(f"Total Predator Reward: {sum(predator_rewards):.2f}")
    print(f"Total Prey Reward:     {sum(prey_rewards):.2f}")
    print(f"Total All-Agent Reward:{total_reward:.2f}")


def print_prey_death_summary(env):
    # Print causes of prey death
    print("\n--- Prey Death Causes ---")
    stats = {"eaten": 0, "starved": 0}
    for internal_id, cause in env.death_cause_prey.items():
        print(f"Prey internal_id {internal_id:4d}: {cause}")
        if cause in stats:
            stats[cause] += 1
    print("\n--- Summary ---")
    print(f"Total prey eaten   : {stats['eaten']}")
    print(f"Total prey starved : {stats['starved']}")


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


def save_prey_death_summary_to_file(env, output_dir):
    death_log_path = os.path.join(output_dir, "prey_death_causes.txt")
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


def run_post_evaluation_plots(ceviz, pdviz):
    if SAVE_EVAL_RESULTS:
        ceviz.plot()
        pdviz.plot()


if __name__ == "__main__":
    seed = 1
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

    # Always show plots on screen
    ceviz.plot()
    pdviz.plot()

    pygame.quit()
    ray.shutdown()
