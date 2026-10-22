"""Evaluate SELFish policies and the paper's Random, TurnAway, and Boids controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.environment import (
    DISCRETE_TURNS,
    SelfishFlockingEnv,
    Settings,
    signed_degrees,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.rl import (
    Actor,
    QNetwork,
)


def load_policy(checkpoint: Path, algo: str, observation_size: int, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("algo") != algo:
        raise ValueError(f"checkpoint algorithm is {saved.get('algo')!r}, not {algo!r}")
    policy = QNetwork(observation_size) if algo == "dqn" else Actor(observation_size)
    policy.load_state_dict(saved["policy"])
    policy.to(device).eval()
    return policy


def learned_action(policy, algo, observation, device):
    single = np.asarray(observation).ndim == 1
    batch = np.asarray(observation, dtype=np.float32)
    state = torch.as_tensor(batch[None, :] if single else batch, device=device)
    with torch.inference_mode():
        if algo == "dqn":
            actions = policy(state).argmax(dim=1).cpu().numpy()
        else:
            actions = policy(state).squeeze(-1).cpu().numpy()
    return int(actions[0]) if single and algo == "dqn" else (
        float(actions[0]) if single else actions
    )


def turn_away_action(observation, discrete: bool):
    relative_bearing = observation[1] * 360.0 - 180.0
    desired_turn = np.clip(signed_degrees(relative_bearing + 180.0), -90.0, 90.0)
    if not discrete:
        return float(desired_turn / 45.0)
    return int(np.argmin(np.abs(DISCRETE_TURNS - desired_turn)))


BOIDS_WEIGHTS = {"cohesion": 1.0, "separation": 1.5, "alignment": 1.0, "predator_avoidance": 2.0}
BOIDS_SEPARATION_RADIUS = 3.0
BOIDS_FLEE_RADIUS = 10.0


def _unit(vector):
    norm = float(np.hypot(*vector))
    return vector / norm if norm > 1e-8 else np.zeros(2)


def boids_action(observation, neighbors: int, discrete: bool, arena_side: float = 40.0):
    """Reynolds-style flocking plus predator avoidance, in the prey's own frame.

    Every vector is expressed relative to the prey's heading (x axis = straight
    ahead), so the resulting bearing is directly a turn. Each rule is a unit
    vector, so the weights alone set their balance.
    """
    predator = observation[:3]
    own_heading = float(observation[5]) * 360.0
    rows = observation[6:].reshape(neighbors, 3)
    positions = []
    relative_headings = []
    for distance, relative_bearing, orientation in rows:
        radians = np.deg2rad(relative_bearing * 360.0 - 180.0)
        positions.append(float(distance) * arena_side * np.array([np.cos(radians), np.sin(radians)]))
        relative_headings.append(np.deg2rad(float(orientation) * 360.0 - own_heading))
    positions = np.stack(positions)
    relative_headings = np.asarray(relative_headings)
    cohesion = _unit(positions.mean(axis=0))
    close = positions[np.hypot(positions[:, 0], positions[:, 1]) < BOIDS_SEPARATION_RADIUS]
    separation = _unit(-np.sum([_unit(position) for position in close], axis=0)) if len(close) else np.zeros(2)
    alignment = _unit(np.array([np.cos(relative_headings).mean(), np.sin(relative_headings).mean()]))
    predator_away = np.zeros(2)
    if float(predator[0]) * arena_side < BOIDS_FLEE_RADIUS:
        predator_angle = np.deg2rad(predator[1] * 360.0 - 180.0)
        predator_away = -np.array([np.cos(predator_angle), np.sin(predator_angle)])
    w = BOIDS_WEIGHTS
    force = (
        w["cohesion"] * cohesion
        + w["separation"] * separation
        + w["alignment"] * alignment
        + w["predator_avoidance"] * predator_away
    )
    if np.hypot(*force) < 1e-8:
        desired_turn = 0.0
    else:
        bearing = np.degrees(np.arctan2(force[1], force[0]))
        desired_turn = float(np.clip(signed_degrees(bearing), -90.0, 90.0))
    if not discrete:
        return float(desired_turn / 45.0)
    return int(np.argmin(np.abs(DISCRETE_TURNS - desired_turn)))


def cluster_labels(positions, width, height, radius):
    count = len(positions)
    labels = np.arange(count)

    def root(index):
        while labels[index] != index:
            labels[index] = labels[labels[index]]
            index = labels[index]
        return index

    delta = positions[None, :, :] - positions[:, None, :]
    delta[..., 0] -= width * np.round(delta[..., 0] / width)
    delta[..., 1] -= height * np.round(delta[..., 1] / height)
    distances = np.hypot(delta[..., 0], delta[..., 1])
    for first, second in zip(*np.where(np.triu(distances <= radius, 1))):
        a, b = root(first), root(second)
        if a != b:
            labels[max(a, b)] = min(a, b)
    return np.asarray([root(index) for index in range(count)])


def component_sizes(positions, width, height, radius, min_size=2):
    _, sizes = np.unique(
        cluster_labels(positions, width, height, radius), return_counts=True
    )
    return sizes[sizes >= min_size], int(np.sum(sizes[sizes < min_size]))


def evaluate(policy_name, policy_fn, algo, episodes, settings, seed, cluster_radius):
    env = SelfishFlockingEnv(settings)
    lengths, catches, pair_distances, cluster_counts, noise_counts, angle_deviations = [], [], [], [], [], []
    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        done = False
        focal_steps = settings.max_steps
        while not done:
            action = policy_fn(observation)
            peer_fn = lambda peer_observations: np.asarray(
                [policy_fn(peer_observation) for peer_observation in peer_observations]
            )
            observation, _, done, info = env.step(
                action,
                peer_fn,
                discrete=algo != "ddpg",
                terminate_on_focal_catch=False,
            )
            if info["focal_caught"] and focal_steps == settings.max_steps:
                focal_steps = info["steps"]
            if info["caught"]:
                catches.append(info["caught"])

            if env.steps % 10 == 0:
                delta = env.positions[:, None, :] - env.positions[None, :, :]
                delta[..., 0] -= settings.width * np.round(delta[..., 0] / settings.width)
                delta[..., 1] -= settings.height * np.round(delta[..., 1] / settings.height)
                pairwise = np.hypot(delta[..., 0], delta[..., 1])
                pair_distances.extend(pairwise[np.triu_indices(settings.prey_count, 1)].tolist())
                labels = cluster_labels(
                    env.positions,
                    settings.width,
                    settings.height,
                    cluster_radius,
                )
                unique_labels, sizes = np.unique(labels, return_counts=True)
                groups = sizes[sizes >= 2]
                noise = int(np.sum(sizes[sizes < 2]))
                cluster_counts.append(len(groups))
                noise_counts.append(noise)
                if len(groups):
                    headings = np.deg2rad(env.headings)
                    group_angles = []
                    for label, size in zip(unique_labels, sizes):
                        if size < 2:
                            continue
                        members = np.flatnonzero(labels == label)
                        mean_angle = np.arctan2(
                            np.sin(headings[members]).mean(),
                            np.cos(headings[members]).mean(),
                        )
                        deviation = np.abs(
                            (headings[members] - mean_angle + np.pi) % (2 * np.pi) - np.pi
                        )
                        group_angles.extend(np.degrees(deviation).tolist())
                    angle_deviations.extend(group_angles)
        lengths.append(focal_steps)
    return {
        "policy": policy_name,
        "episodes": episodes,
        "mean_focal_survival_steps": float(np.mean(lengths)),
        "std_focal_survival_steps": float(np.std(lengths)),
        "mean_catches_per_1000_frames": float(1000 * np.sum(catches) / max(episodes * settings.max_steps, 1)),
        "mean_pairwise_distance": float(np.mean(pair_distances)),
        "mean_cluster_count_min_size_2": float(np.mean(cluster_counts)),
        "mean_noise_agents": float(np.mean(noise_counts)),
        "mean_cluster_heading_deviation_degrees": float(np.mean(angle_deviations)) if angle_deviations else None,
        "cluster_radius": cluster_radius,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--algo", choices=("dqn", "ddpg"), default="dqn")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--prey-count", type=int, default=40)
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-radius", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("selfish_flocking_eval.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policies",
        default="random,turn_away,boids,learned",
        help="comma-separated subset of random,turn_away,boids,learned",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    neighbors = args.neighbors or (5 if args.algo == "dqn" else 1)
    settings = Settings(prey_count=args.prey_count, neighbors=neighbors)
    observation_size = 3 * (2 + neighbors)
    learned = None
    if args.checkpoint:
        learned = load_policy(args.checkpoint, args.algo, observation_size, args.device)

    policies = {
        "random": (
            (lambda obs: float(np.random.uniform(-1.0, 1.0)))
            if args.algo == "ddpg"
            else (lambda obs: int(np.random.randint(5)))
        ),
        "turn_away": lambda obs: turn_away_action(obs, discrete=args.algo != "ddpg"),
        "boids": lambda obs: boids_action(
            obs, neighbors, discrete=args.algo != "ddpg", arena_side=max(settings.width, settings.height)
        ),
    }
    if learned is not None:
        policies[f"selfish_{args.algo}"] = lambda obs: learned_action(
            learned, args.algo, obs, args.device
        )
    wanted = set(args.policies.split(","))
    results = [
        evaluate(name, fn, args.algo, args.episodes, settings, args.seed, args.cluster_radius)
        for name, fn in policies.items()
        if name in wanted or (name.startswith("selfish_") and "learned" in wanted)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "paper": "Hahn et al. (2019), Emergent Escape-based Flocking Behavior using Multi-Agent Reinforcement Learning",
                "settings": settings.__dict__,
                "seed": args.seed,
                "results": results,
                "assumptions": {
                    "cluster_radius": args.cluster_radius,
                    "boids_weights": BOIDS_WEIGHTS,
                    "boids_separation_radius": BOIDS_SEPARATION_RADIUS,
                    "boids_flee_radius": BOIDS_FLEE_RADIUS,
                    "evaluation_episodes": args.episodes,
                    "predator_parameters_not_reported_in_paper": {
                        "target_radius": settings.predator_target_radius,
                        "target_hold_steps": settings.predator_target_hold,
                        "leap_interval": settings.predator_leap_interval,
                        "leap_duration": settings.predator_leap_duration,
                        "leap_speed": settings.predator_leap_speed,
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
