"""Train a shared SELFish prey policy with the paper's DQN or DDPG settings.

Run from the repository root:

    python -m predpreygrass.non_evolutionary.continuous_space.selfish_flocking.train \
        --algo dqn --steps 500000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.environment import (
    DISCRETE_TURNS,
    SelfishFlockingEnv,
    Settings,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.rl import (
    Actor,
    Critic,
    QNetwork,
    ReplayBuffer,
    ddpg_update,
    dqn_update,
    frozen_copy,
)


PAPER_SETTINGS = {
    "dqn": {
        "training_steps": 500_000,
        "hidden_layers": 10,
        "neurons_per_layer": 16,
        "gamma": 0.999999,
        "learning_rate": 0.001,
        "replay_capacity": 50_000,
        "batch_size": 64,
        "epsilon": 0.1,
        "neighbors": 5,
    },
    "ddpg": {
        "training_steps": 500_000,
        "hidden_layers": 5,
        "actor_width": 16,
        "critic_width": 32,
        "gamma": 0.999999,
        "learning_rate": 0.001,
        "replay_capacity": 100_000,
        "batch_size": 512,
        "ou_theta": 0.15,
        "ou_mu": 0.0,
        "ou_sigma": 0.3,
        "neighbors": 1,
    },
}


def _peer_action(module, algo, observation, device):
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    single = tensor.ndim == 1
    if single:
        tensor = tensor.unsqueeze(0)
    with torch.inference_mode():
        if algo == "dqn":
            actions = module(tensor).argmax(dim=1).cpu().numpy()
            return int(actions[0]) if single else actions
        actions = module(tensor).squeeze(-1).cpu().numpy()
        return float(actions[0]) if single else actions


def _ou_noise(state, theta, mu, sigma, rng):
    return state + theta * (mu - state) + sigma * rng.normal()


def train(algo: str, steps: int, seed: int, out_dir: Path, device_name: str | None = None):
    if algo not in PAPER_SETTINGS:
        raise ValueError("algo must be 'dqn' or 'ddpg'")
    if steps < 1:
        raise ValueError("steps must be positive")

    settings = dict(PAPER_SETTINGS[algo])
    settings["training_steps"] = steps
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    env_settings = Settings(prey_count=10, neighbors=settings["neighbors"], max_steps=10_000)
    env = SelfishFlockingEnv(env_settings)
    observation_size = 3 * (2 + env_settings.neighbors)

    if algo == "dqn":
        online = QNetwork(observation_size).to(device)
        target = frozen_copy(online).to(device)
        optimizer = torch.optim.Adam(online.parameters(), lr=settings["learning_rate"])
        replay = ReplayBuffer(settings["replay_capacity"])
    else:
        online = Actor(observation_size).to(device)
        target = frozen_copy(online).to(device)
        critic = Critic(observation_size).to(device)
        critic_target = frozen_copy(critic).to(device)
        actor_optimizer = torch.optim.Adam(online.parameters(), lr=settings["learning_rate"])
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=settings["learning_rate"])
        replay = ReplayBuffer(settings["replay_capacity"])
        noise_state = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    episode = 0
    episode_lengths: list[int] = []
    episode_losses: list[float] = []
    environment_steps = 0

    while environment_steps < steps:
        episode += 1
        peer_snapshot = frozen_copy(online).to(device)
        peer_policy = lambda observations: _peer_action(peer_snapshot, algo, observations, device)
        observation = env.reset(seed=seed * 1_000_003 + episode)
        episode_length = 0
        if algo == "ddpg":
            noise_state = 0.0

        while environment_steps < steps:
            if algo == "dqn":
                if np.random.random() < settings["epsilon"]:
                    action = int(np.random.randint(len(DISCRETE_TURNS)))
                else:
                    action = _peer_action(online, algo, observation, device)
                next_observation, reward, done, _ = env.step(action, peer_policy, discrete=True)
                replay.add(observation, action, reward, next_observation, done)
            else:
                with torch.inference_mode():
                    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    action_value = float(online(state).item())
                noise_state = _ou_noise(
                    noise_state,
                    settings["ou_theta"],
                    settings["ou_mu"],
                    settings["ou_sigma"],
                    np.random,
                )
                action_value = float(np.clip(action_value + noise_state, -1.0, 1.0))
                next_observation, reward, done, _ = env.step(
                    action_value, peer_policy, discrete=False
                )
                replay.add(observation, action_value, reward, next_observation, done)

            observation = next_observation
            episode_length += 1
            environment_steps += 1

            if len(replay) >= settings["batch_size"]:
                if algo == "dqn":
                    loss = dqn_update(
                        online,
                        target,
                        optimizer,
                        replay,
                        settings["batch_size"],
                        settings["gamma"],
                        device,
                    )
                    episode_losses.append(loss)
                    if environment_steps % 10_000 == 0:
                        target.load_state_dict(online.state_dict())
                else:
                    critic_loss, actor_loss = ddpg_update(
                        online,
                        target,
                        critic,
                        critic_target,
                        actor_optimizer,
                        critic_optimizer,
                        replay,
                        settings["batch_size"],
                        settings["gamma"],
                        0.001,
                        device,
                    )
                    episode_losses.append(critic_loss + actor_loss)

            if environment_steps % 10_000 == 0:
                elapsed = time.time() - start
                mean_length = float(np.mean(episode_lengths[-100:])) if episode_lengths else 0.0
                mean_loss = float(np.mean(episode_losses[-10_000:])) if episode_losses else float("nan")
                print(
                    f"{algo}: steps={environment_steps}/{steps} episodes={episode} "
                    f"recent_survival={mean_length:.1f} loss={mean_loss:.5f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if done:
                break

        episode_lengths.append(episode_length)
        if episode % 10 == 0 or environment_steps >= steps:
            path = out_dir / f"{algo}_step_{environment_steps:07d}.pt"
            checkpoint = {"algo": algo, "policy": online.state_dict(), "seed": seed}
            if algo == "ddpg":
                checkpoint["critic"] = critic.state_dict()
            torch.save(checkpoint, path)
            (out_dir / f"{algo}_settings.json").write_text(
                json.dumps(
                    {
                        "paper": "Hahn et al. (2019), Emergent Escape-based Flocking Behavior",
                        "algorithm": algo,
                        "seed": seed,
                        "environment": env_settings.__dict__,
                        "hyperparameters": settings,
                        "implementation_assumptions": {
                            "predator_target_radius": env_settings.predator_target_radius,
                            "predator_target_hold": env_settings.predator_target_hold,
                            "predator_leap_interval": env_settings.predator_leap_interval,
                            "predator_leap_duration": env_settings.predator_leap_duration,
                            "predator_leap_speed": env_settings.predator_leap_speed,
                            "observation_normalization": "distance / max arena side; angles / 360 after mapping relative angle to [0, 360)",
                            "target_network": "DQN hard update every 10,000 training steps; DDPG Polyak tau=0.001",
                        },
                        "episodes_finished": episode,
                        "training_steps_completed": environment_steps,
                        "mean_episode_length": float(np.mean(episode_lengths)),
                        "wall_seconds": round(time.time() - start, 1),
                    },
                    indent=2,
                )
                + "\n"
            )

    return online


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=sorted(PAPER_SETTINGS), default="dqn")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.algo, args.steps, args.seed, args.out_dir, args.device)


if __name__ == "__main__":
    main()
