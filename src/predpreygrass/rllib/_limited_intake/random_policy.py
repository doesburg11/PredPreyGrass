from predpreygrass.rllib.limited_intake.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.limited_intake.config.config_env_limited_intake import config_env
from predpreygrass.rllib.limited_intake.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper
import pygame
import os
import json

import numpy as np

def env_creator(config):
    return PredPreyGrass(config)


# Use a single reproducible RNG for all random actions
def random_policy_pi(agent_id, env, rng):
    space = env.action_spaces[agent_id]
    # Use reproducible sampling for Discrete, fallback to .sample() for others
    if hasattr(space, "n"):  # Discrete
        return rng.integers(space.n)
    else:
        return space.sample()

if __name__ == "__main__":

    # Always inject wall config for visualization parity
    cfg = dict(config_env)
    # Set seeds for reproducibility
    SEED = cfg.get("seed", 42)
    np.random.seed(SEED)  # For any legacy code, but not used for sampling actions
    rng = np.random.default_rng(SEED)
    # Force manual wall layout for this viewer
    cfg["wall_placement_mode"] = "manual"
    cfg["manual_wall_positions"] = config_env["manual_wall_positions"]
    cfg["num_walls"] = 0  # ignored in manual mode
    # Enable visibility (occlusion) channel so observations include LOS mask as final channel
    cfg["include_visibility_channel"] = True
    # Ensure multi-step eating so carcasses can appear under random play
        # Intake caps are set in config_env_limited_intake.py: max_eating_predator and max_eating_prey
    # Keep carcasses persistent for viewing (no decay, unlimited lifetime)
    cfg["carcass_decay_per_step"] = 0.0
    cfg["carcass_max_lifetime"] = None
    env = env_creator(cfg)
    observations, _ = env.reset(seed=cfg.get("seed", 42))


    # (Debug print removed)

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(
        grid_size,
        enable_speed_slider=False,
        enable_tooltips=True,
        predator_obs_range=cfg.get("predator_obs_range"),
        prey_obs_range=cfg.get("prey_obs_range"),
        show_fov=True,
        fov_alpha=40,
        fov_agents=["type_1_predator_0", "type_1_prey_0"],
        fov_respect_walls=True,
    )
    clock = pygame.time.Clock()

    # Run loop until termination
    terminated = False
    truncated = False


    control = ViewerControlHelper(initial_paused=False)


    while not terminated and not truncated:
        control.handle_events()
        if not control.paused or control.step_once:
            # --- Step forward using random actions ---
            action_dict = {agent_id: random_policy_pi(agent_id, env, rng) for agent_id in env.agents}
            print("Actions taken:", action_dict)
            observations, rewards, terminations, truncations, infos = env.step(action_dict)
            control.step_once = False
            terminated = any(terminations.values())
            truncated = any(truncations.values())

        # --- Update visualizer (always, even if paused) ---
        try:
            visualizer.update(
                grass_positions=env.grass_positions,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate,
                per_step_agent_data=env.per_step_agent_data,
                walls=getattr(env, "wall_positions", None),
                carcass_positions=getattr(env, "carcass_positions", None),
                carcass_energies=getattr(env, "carcass_energies", None),
            )
        except TypeError:
            visualizer.update(
                grass_positions=env.grass_positions,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate,
                per_step_agent_data=env.per_step_agent_data,
            )
        # --- Slow down simulation for visibility ---
        clock.tick(10)  # steps per second
    # --- Trajectory logging at episode end ---
    # Save per-agent, per-step data to the same format as the callback (excluding 'position')
    out_path = os.path.join(os.path.dirname(__file__), 'trajectories_output/agent_trajectories.json')
    out_path = os.path.abspath(out_path)
    per_agent_trajectories = []
    for step, step_data in enumerate(env.per_step_agent_data):
        for agent_id, info in step_data.items():
            traj = {
                'unique_id': info.get('unique_id', agent_id),
                'step': step,
                'energy': info.get('energy'),
                'energy_decay': info.get('energy_decay'),
                'energy_movement': info.get('energy_movement'),
                'energy_eating': info.get('energy_eating'),
                'energy_reproduction': info.get('energy_reproduction'),
                'age': info.get('age'),
                'offspring_count': info.get('offspring_count'),
                'offspring_ids': info.get('offspring_ids'),
                'reward': info.get('reward'),
                'parent_id': info.get('parent_id'),
                'cause_of_death': info.get('cause_of_death'),
                'age_of_death': info.get('age_of_death'),
                'consumed_resources': info.get('consumed_resources'),
            }
            # Optionally add event fields if present
            for k in ["movement", "death", "reproduction", "reward", "consumption_log", "move_blocked_reason", "los_rejected"]:
                if k in info:
                    traj[k] = info[k]
            per_agent_trajectories.append(traj)
    # Write to JSON (overwrite, do not append)
    try:
        with open(out_path, 'w') as f:
            json.dump(per_agent_trajectories, f, indent=2)
        print(f"[random_policy] Trajectories written to {out_path}")
    except Exception as e:
        print(f"[random_policy] Failed to write trajectories: {e}")
    visualizer.close()
    env.close()
