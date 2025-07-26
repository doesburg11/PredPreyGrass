import os
from datetime import datetime
import ray
import torch
import pygame
import cv2
import numpy as np

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env

from predpreygrass.rllib.v2_6.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_6.config.config_env_eval import config_env
from predpreygrass.rllib.v2_6.utils.pygame_grid_renderer_rllib import PyGameRenderer

# ==== CONFIG ====
RAY_RESULTS_DIR = "/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results"
CHECKPOINT_PATH = "PPO_2025-07-25_22-58-56/checkpoint_iter_1000"
SAVE_MOVIE = False
MOVIE_FILENAME = "eval_video.mp4"
MOVIE_FPS = 10
SEED = 42
# ================


def policy_mapping_fn(agent_id):
    return "_".join(agent_id.split("_")[:3])


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        output = policy_module._forward_inference({"obs": obs_tensor})
    logits = output.get("action_dist_inputs")
    return torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(logits=logits).sample().item()


def load_rl_modules(checkpoint_path):
    rl_module_dir = os.path.join(checkpoint_path, "learner_group", "learner", "rl_module")
    module_paths = {
        pid: os.path.join(rl_module_dir, pid)
        for pid in os.listdir(rl_module_dir)
        if os.path.isdir(os.path.join(rl_module_dir, pid))
    }
    return {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrass", lambda config: PredPreyGrass(config))
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_root = os.path.join(RAY_RESULTS_DIR, CHECKPOINT_PATH)
    rl_modules = load_rl_modules(checkpoint_root)

    env = PredPreyGrass(config=config_env)
    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, cell_size=32, enable_speed_slider=False)

    observations, _ = env.reset(seed=SEED)

    if SAVE_MOVIE:
        screen_size = visualizer.screen.get_size()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(MOVIE_FILENAME, fourcc, MOVIE_FPS, screen_size)
    else:
        video_writer = None

    total_reward = 0
    clock = pygame.time.Clock()

    terminated, truncated = False, False
    while not terminated and not truncated:
        action_dict = {aid: policy_pi(observations[aid], rl_modules[policy_mapping_fn(aid)]) for aid in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(action_dict)

        visualizer.update(
            agent_positions=env.agent_positions,
            grass_positions=env.grass_positions,
            agent_energies=env.agent_energies,
            grass_energies=env.grass_energies,
            agents_just_ate=env.agents_just_ate,
            step=env.current_step,
        )

        if SAVE_MOVIE:
            frame = pygame.surfarray.array3d(visualizer.screen)
            frame = np.transpose(frame, (1, 0, 2))  # Pygame to OpenCV format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        total_reward += sum(rewards.values())
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        clock.tick(visualizer.target_fps)

    print(f"\n Evaluation complete! Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {env.current_step}")

    if video_writer:
        video_writer.release()

    pygame.quit()
    ray.shutdown()
