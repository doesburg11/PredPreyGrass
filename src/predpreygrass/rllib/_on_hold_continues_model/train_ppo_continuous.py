"""RLlib PPO training script for PredPreyGrassContinuous.

Usage (examples):

  # Vector observations (nearest-k)
  python -m predpreygrass.rllib.continues_model.train_ppo_continuous \
      --obs-mode vector --timesteps 20000

  # Grid observations (channels-first CxHxW)
  python -m predpreygrass.rllib.continues_model.train_ppo_continuous \
      --obs-mode grid --grid-size 15 --vision-radius-pred 6 --vision-radius-prey 6 \
      --n-predators 2 --n-prey 3 --n-grass 10 \
      --num-workers 0 --rollout-fragment-length 40 \
      --train-batch-size 80 --sgd-minibatch-size 32 --num-sgd-iter 1 \
      --timesteps 160 --framework torch \
      --max-agents-per-type 512 \
      --out /home/doesburg/Projects/PredPreyGrass/logs/tmp_cont_train_grid

This script configures two policies (predator, prey) and maps agent IDs by prefix.
For grid mode, model.conv_format=NCHW is set and light conv filters are provided.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import ray

from predpreygrass.rllib._on_hold_continues_model.predpreygrass_continuous_env import PredPreyGrassContinuous


def build_env_config(args: argparse.Namespace) -> Dict:
    cfg: Dict = {
        "seed": args.seed,
        "world_size": args.world_size,
        "n_initial_predators": args.n_predators,
        "n_initial_prey": args.n_prey,
        "n_grass_patches": args.n_grass,
        "nearest_k_agents": args.nearest_k_agents,
        "nearest_k_grass": args.nearest_k_grass,
        # capacity for dynamic agents (newborns)
        "max_agents_per_type": args.max_agents_per_type,
        # observations
        "obs_mode": args.obs_mode,
        "obs_grid_size": args.grid_size,
        "obs_grid_use_energy": args.grid_use_energy,
        "obs_grid_wall_mode": args.grid_wall_mode,
        "obs_grid_circular_mask": args.grid_circular_mask,
        # radii (for grid, must not be None)
        "vision_radius": args.vision_radius,
        "vision_radius_predator": args.vision_radius_pred,
        "vision_radius_prey": args.vision_radius_prey,
        # dynamics
        "max_speed_predator": args.max_speed_predator,
        "max_speed_prey": args.max_speed_prey,
        "catch_radius": args.catch_radius,
        "eat_radius": args.eat_radius,
        # rewards (tunable)
        "reward_predator_catch_prey": args.rew_catch,
        "reward_prey_eat_grass": args.rew_eat,
        "reward_predator_step": args.rew_step_pred,
        "reward_prey_step": args.rew_step_prey,
        "penalty_prey_caught": args.penalty_prey_caught,
        # episode
        "max_steps": args.max_steps,
    }
    if args.obs_mode == "grid":
        # Ensure some radius is set
        if cfg.get("vision_radius_predator") is None and cfg.get("vision_radius") is None:
            cfg["vision_radius_predator"] = 5.0
        if cfg.get("vision_radius_prey") is None and cfg.get("vision_radius") is None:
            cfg["vision_radius_prey"] = 5.0
    return cfg


def make_policies(env: PredPreyGrassContinuous):
    # Build per-policy spaces from the env’s per-agent mappings (new API stack style)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in env.observation_spaces.items():
        pid = "predator" if agent_id.startswith("predator") else "prey"
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = env.action_spaces[agent_id]
    policies = {
        pid: PolicySpec(observation_space=obs_by_policy[pid], action_space=act_by_policy[pid], config={})
        for pid in obs_by_policy
    }
    return policies


def policy_mapping_fn(agent_id: str, *args, **kwargs) -> str:
    return "predator" if agent_id.startswith("predator") else "prey"


def main():
    parser = argparse.ArgumentParser()
    # Env basics
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=float, default=20.0)
    parser.add_argument("--n-predators", type=int, default=6)
    parser.add_argument("--n-prey", type=int, default=10)
    parser.add_argument("--n-grass", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-agents-per-type", type=int, default=256)
    # Observations
    parser.add_argument("--obs-mode", choices=["vector", "grid"], default="vector")
    parser.add_argument("--nearest-k-agents", type=int, default=4)
    parser.add_argument("--nearest-k-grass", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--grid-use-energy", action="store_true")
    parser.add_argument("--grid-wall-mode", choices=["binary", "distance"], default="binary")
    parser.add_argument("--grid-circular-mask", action="store_true")
    parser.add_argument("--vision-radius", type=float, default=None)
    parser.add_argument("--vision-radius-pred", type=float, default=None)
    parser.add_argument("--vision-radius-prey", type=float, default=None)
    # Dynamics
    parser.add_argument("--max-speed-predator", type=float, default=0.9)
    parser.add_argument("--max-speed-prey", type=float, default=1.0)
    parser.add_argument("--catch-radius", type=float, default=0.6)
    parser.add_argument("--eat-radius", type=float, default=0.5)
    # Rewards
    parser.add_argument("--rew-catch", type=float, default=1.0)
    parser.add_argument("--rew-eat", type=float, default=0.2)
    parser.add_argument("--rew-step-pred", type=float, default=0.0)
    parser.add_argument("--rew-step-prey", type=float, default=0.0)
    parser.add_argument("--penalty-prey-caught", type=float, default=-1.0)
    # RLlib runtime
    parser.add_argument("--framework", choices=["torch", "tf2"], default="torch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=0)
    parser.add_argument("--rollout-fragment-length", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=80)
    parser.add_argument("--sgd-minibatch-size", type=int, default=32)
    parser.add_argument("--num-sgd-iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    # Target number of environment steps to sample before stopping.
    # Increased default for longer runs out-of-the-box.
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--out", type=str, default="~/rllib_predpreygrass")

    args = parser.parse_args()

    outdir = os.path.expanduser(args.out)
    os.makedirs(outdir, exist_ok=True)

    env_config = build_env_config(args)

    # Bootstrap spaces from a single local env
    env = PredPreyGrassContinuous(env_config)
    policies = make_policies(env)

    # New API stack end-to-end: env_runners + new training params
    ppo_cfg = PPOConfig().environment(env=PredPreyGrassContinuous, env_config=env_config).framework(args.framework)
    if hasattr(ppo_cfg, "env_runners"):
        ppo_cfg = ppo_cfg.env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=args.rollout_fragment_length,
        )
    else:
        ppo_cfg = ppo_cfg.rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=args.rollout_fragment_length,
        )
    base_train_kwargs = dict(lr=args.lr, train_batch_size=args.train_batch_size)
    try:
        ppo_cfg = ppo_cfg.training(
            **base_train_kwargs,
            minibatch_size=args.sgd_minibatch_size,
            num_epochs=args.num_sgd_iter,
        )
    except TypeError:
        ppo_cfg = ppo_cfg.training(
            **base_train_kwargs,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
        )

    # New API: leave RLModule/Learner defaults (can be tuned later)
    ppo_cfg = ppo_cfg.resources(num_gpus=args.num_gpus).multi_agent(
        policies=policies, policy_mapping_fn=policy_mapping_fn
    )

    # Model specifics by obs mode
    if args.obs_mode == "grid":
        # Channels-first grid -> set NCHW and simple conv stack
        model_cfg = {
            "conv_format": "NCHW",
            # Filters: (out_channels, kernel, stride)
            # Assuming input (C=4, H=W=args.grid_size)
            "conv_filters": [
                [16, [5, 5], 2],
                [32, [3, 3], 2],
            ],
            "conv_activation": "relu",
            "post_fcnet_hiddens": [256, 128],
            "post_fcnet_activation": "relu",
        }
        ppo_cfg = ppo_cfg.training(model=model_cfg)
    else:
        # Vector obs -> MLP
        model_cfg = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        }
        ppo_cfg = ppo_cfg.training(model=model_cfg)

    # Init and train
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    # Prefer new API: build_algo, else fallback to build
    if hasattr(ppo_cfg, "build_algo"):
        algo = ppo_cfg.build_algo()
    else:
        algo = ppo_cfg.build()

    # Train until we've sampled at least `--timesteps` env steps (robust to batch size changes).
    target_ts = int(args.timesteps)
    i = 0
    last_ckpt_it = 0
    while True:
        result = algo.train()
        i += 1
        # Prefer new stack metrics; fallback to legacy
        ep_r = result.get("episode_reward_mean")
        if ep_r is None:
            ep_r = result.get("env_runners", {}).get("episode_return_mean")
        timesteps_total = result.get("num_env_steps_sampled_lifetime") or result.get("timesteps_total") or 0
        try:
            ts_val = int(timesteps_total)
        except Exception:
            ts_val = 0
        print(f"Iter {i} | timesteps={ts_val} | episode_return_mean={ep_r}")
        if args.checkpoint_every and i % args.checkpoint_every == 0 and i != last_ckpt_it:
            last_ckpt_it = i
            ckpt = algo.save(outdir)
            print(f"Checkpoint saved to: {ckpt}")
        if ts_val >= target_ts:
            break

    # Final checkpoint
    ckpt = algo.save(outdir)
    print(f"Final checkpoint saved to: {ckpt}")


if __name__ == "__main__":
    main()
