"""
Render one headless self-play episode (both predator_policy and prey_policy from
the SAME checkpoint) into an animated GIF, using the same PyGameRenderer as the
interactive evaluate_ppo_from_checkpoint_debug.py -- but driven forward
automatically frame-by-frame instead of via its interactive pause/step/rewind
controls, and captured to frames instead of shown in a window.

PyGameRenderer calls pygame.display.set_mode(), which needs a working video
driver -- run this under `xvfb-run` (or with SDL_VIDEODRIVER=dummy already
exported) rather than relying on the ambient DISPLAY, since that may be a
stale value with nothing actually listening on it:

    xvfb-run -a python -m predpreygrass.non_evolutionary.base_environment.record_episode_gif \\
        --checkpoint-path <checkpoint_dir> --output out.gif

To keep the output a reasonable size regardless of episode length, frames are
subsampled evenly across the whole episode (via --num-frames) rather than
captured every single step, so the GIF still spans the full arc of the episode
(population growth, decline, oscillation) instead of only showing however far
the first N raw steps happen to get.
"""
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.base_environment.config_env import config_env
from predpreygrass.non_evolutionary.base_environment.evaluate_ppo_from_checkpoint_debug import policy_pi
from predpreygrass.non_evolutionary.base_environment.master_tournament_matrix import policy_mapping_fn
from predpreygrass.non_evolutionary.base_environment.utils.pygame_grid_renderer_rllib import PyGameRenderer

# --- External libraries ---
import argparse
from pathlib import Path

import numpy as np
import pygame
from PIL import Image
from ray.rllib.core.rl_module.rl_module import RLModule


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=str, required=True, help="A single checkpoint_XXXXXX directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None, help="Override config_env's max_steps.")
    parser.add_argument("--num-frames", type=int, default=375, help="Target output frame count (evenly subsampled across the episode).")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed of the output GIF.")
    parser.add_argument("--output", type=str, default="episode.gif")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser()

    rl_modules = {
        pid: RLModule.from_checkpoint(checkpoint_path / "learner_group" / "learner" / "rl_module" / pid)
        for pid in ("predator_policy", "prey_policy")
    }

    env_config = dict(config_env)
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    env = PredPreyGrass(env_config)

    observations, _ = env.reset(seed=args.seed)
    active_agents = list(observations.keys())

    visualizer = PyGameRenderer((env.grid_size, env.grid_size), ennable_speed_slider=False)

    def capture_frame() -> Image.Image:
        visualizer.update(
            agent_positions=env.agent_positions,
            grass_positions=env.grass_positions,
            agent_energies=env.agent_energies,
            grass_energies=env.grass_energies,
            agents_just_ate=env.agents_just_ate,
            step=env.current_step,
        )
        pygame.event.pump()  # keep the (headless) window "responsive" -- avoids OS-level not-responding state
        frame = pygame.surfarray.array3d(visualizer.screen)
        frame = np.transpose(frame, (1, 0, 2))  # (width, height, channels) -> (height, width, channels)
        return Image.fromarray(frame)

    frames = [capture_frame()]

    while True:
        action_dict = {
            agent_id: policy_pi(observations[agent_id], rl_modules[policy_mapping_fn(agent_id)], deterministic=True)
            for agent_id in active_agents
        }
        observations, rewards, terminations, truncations, _ = env.step(action_dict)
        frames.append(capture_frame())

        active_agents = [
            agent_id for agent_id in observations
            if not terminations.get(agent_id, False) and not truncations.get(agent_id, False)
        ]
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break

    pygame.quit()

    total_steps = len(frames)
    print(f"Episode finished at step {env.current_step} ({total_steps} raw frames captured); "
          f"predators={env.current_num_predators}, prey={env.current_num_prey}")

    # Evenly subsample down to --num-frames so the GIF spans the whole episode
    # (growth, decline, oscillation) rather than just however far the first
    # --num-frames raw steps happen to reach.
    if total_steps > args.num_frames:
        indices = np.linspace(0, total_steps - 1, args.num_frames).round().astype(int)
        frames = [frames[i] for i in indices]

    duration_ms = int(1000 / args.fps)
    frames[0].save(
        args.output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True,
    )
    print(f"Saved {len(frames)}-frame GIF ({args.fps} fps) to {args.output}")
