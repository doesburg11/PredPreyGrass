"""
Retrain one policy against a FROZEN opponent, starting from a mature co-trained
checkpoint -- a follow-up to master_tournament_matrix.py's finding that the
100x100 tournament matrix flattens out past ~iteration 300 (weak/near-zero
correlation between checkpoint iteration and outcome, both in final_num_prey and
in each side's own average reward). That flatness is consistent with either:

  (a) a genuine mutual equilibrium (neither side can improve further), or
  (b) stagnation -- self-play converged to *a* joint local optimum but stopped
      exploring, and a better response might still exist.

This script tells them apart: pick one policy (--frozen-side) and hold it fixed
at its iteration-500 (or whichever --checkpoint-iteration) weights via RLlib's
`policies_to_train` (it still acts via policy_mapping_fn every step, it just
never receives a gradient update). Train the OTHER policy against that fixed
target, either warm-started from its own iteration-500 weights
(--retrain-init warmstart -- the sharper test: can this specific already-converged
policy still climb once the opponent stops moving?) or from scratch
(--retrain-init scratch -- can ANY policy learn to beat this frozen opponent,
without needing to be genuinely the same converged policy?).

If the retrained side's reward/ecology metrics climb well above where it
plateaued during normal co-training, that points to (b) stagnation. If it stays
flat at roughly the same level even against a now-stationary opponent, that
points to (a) a real local equilibrium.

Reuses env_creator, policy_mapping_fn, and the EpisodeReturn callback directly
from tune_ppo_base_environment.py (same PPO hyperparameters, same environment,
same architecture) so this is a controlled comparison against that baseline run
-- and the checkpoint-discovery/iteration-lookup helpers from
master_tournament_matrix.py, rather than duplicating either.
"""
from predpreygrass.non_evolutionary.base_environment.tune_ppo_base_environment import (
    env_creator,
    policy_mapping_fn,
    EpisodeReturn,
)
from predpreygrass.non_evolutionary.base_environment.master_tournament_matrix import (
    discover_checkpoints,
    read_training_iteration,
)
from predpreygrass.non_evolutionary.base_environment.config_env import config_env
from predpreygrass.global_config import RAY_RESULTS_DIR

# --- External libraries ---
import argparse
import json
from datetime import datetime
from pathlib import Path

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--frozen-side", type=str, required=True, choices=["predator", "prey"],
        help="Which policy stays fixed at its --checkpoint-iteration weights and "
             "is excluded from training (still acts via policy_mapping_fn every "
             "step -- just never receives a gradient update).",
    )
    parser.add_argument(
        "--retrain-init", type=str, required=True, choices=["warmstart", "scratch"],
        help="warmstart: the retrained side starts from ITS OWN --checkpoint-iteration "
             "weights too (isolates whether this specific converged policy can still "
             "climb against a now-stationary opponent). scratch: the retrained side "
             "starts from a fresh random policy (tests whether ANY policy can learn "
             "to beat this frozen opponent).",
    )
    parser.add_argument(
        "--checkpoint-run-dir", type=str,
        default=str(Path(RAY_RESULTS_DIR).expanduser() / "master_tournament_2026-09-06"
                    / "PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45"),
        help="Trial directory containing the checkpoint_* dirs to pull starting "
             "weights from (the original base_environment SEED42 run by default).",
    )
    parser.add_argument(
        "--checkpoint-iteration", type=int, default=500,
        help="Training iteration to use as the frozen/warm-start starting point "
             "(matched via each checkpoint's own recorded training_iteration, not "
             "assumed from checkpoint_frequency).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed for this run.")
    parser.add_argument("--name", type=str, default=None, help="Override the auto-generated experiment name.")
    parser.add_argument("--max-iters", type=int, default=300, help="Training-iteration stop condition.")
    return parser.parse_args()


def find_checkpoint_for_iteration(run_dir: Path, target_iteration: int) -> Path:
    ckpts = discover_checkpoints(run_dir)
    for ckpt in ckpts:
        if read_training_iteration(ckpt) == target_iteration:
            return ckpt
    available = sorted(read_training_iteration(c) for c in ckpts)
    raise ValueError(
        f"No checkpoint with training_iteration == {target_iteration} found under {run_dir}. "
        f"Available iterations: {available}"
    )


if __name__ == "__main__":
    args = parse_args()

    checkpoint_run_dir = Path(args.checkpoint_run_dir).expanduser()
    checkpoint_path = find_checkpoint_for_iteration(checkpoint_run_dir, args.checkpoint_iteration)
    print(f"[setup] Using starting weights from {checkpoint_path}")

    frozen_policy_id = "predator_policy" if args.frozen_side == "predator" else "prey_policy"
    retrain_policy_id = "prey_policy" if args.frozen_side == "predator" else "predator_policy"
    print(f"[setup] Frozen policy (fixed, excluded from training): {frozen_policy_id}")
    print(f"[setup] Retrained policy: {retrain_policy_id}, init={args.retrain_init}")

    def module_load_path(policy_id: str):
        module_dir = checkpoint_path / "learner_group" / "learner" / "rl_module" / policy_id
        if not module_dir.is_dir():
            raise FileNotFoundError(f"Expected RLModule directory not found: {module_dir}")
        return str(module_dir)

    # The frozen side always loads its fixed weights. The retrained side loads
    # its own weights too under "warmstart" (same starting point, but now with
    # gradients flowing and a stationary opponent), or is left at RLlib's
    # default random init under "scratch".
    load_paths = {frozen_policy_id: module_load_path(frozen_policy_id)}
    if args.retrain_init == "warmstart":
        load_paths[retrain_policy_id] = module_load_path(retrain_policy_id)

    env_config = dict(config_env)

    register_env("PredPreyGrass", env_creator)
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    ray_results_path = Path(RAY_RESULTS_DIR).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_tag = f"_SEED{args.seed}" if args.seed is not None else ""
    experiment_name = args.name or (
        f"RETRAIN_FROZEN-{args.frozen_side.upper()}_{args.retrain_init.upper()}-"
        f"{('prey' if args.frozen_side == 'predator' else 'predator').upper()}_"
        f"ITER{args.checkpoint_iteration}{seed_tag}_{timestamp}"
    )
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(
            {
                "config_env": env_config,
                "seed": args.seed,
                "frozen_side": args.frozen_side,
                "retrain_init": args.retrain_init,
                "checkpoint_run_dir": str(checkpoint_run_dir),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_iteration": args.checkpoint_iteration,
                "frozen_policy_id": frozen_policy_id,
                "retrain_policy_id": retrain_policy_id,
            },
            f,
            indent=4,
        )

    sample_env = env_creator(env_config)
    if sample_env is None:
        raise RuntimeError("Failed to create sample environment")
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    if observation_spaces is None or action_spaces is None:
        raise RuntimeError("Sample environment did not initialize observation/action spaces")

    obs_space_pred = observation_spaces["predator_0"]
    act_space_pred = action_spaces["predator_0"]
    obs_space_prey = observation_spaces["prey_0"]
    act_space_prey = action_spaces["prey_0"]

    # Same architecture as tune_ppo_base_environment.py -- must match, since
    # we're loading weights saved by that exact architecture.
    shared_model_config = {
        "conv_filters": [
            [16, [3, 3], 1],
            [32, [3, 3], 1],
            [64, [3, 3], 1],
        ],
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }

    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred,
                action_space=act_space_pred,
                inference_only=False,
                model_config=shared_model_config,
                catalog_class=None,
                load_state_path=load_paths.get("predator_policy"),
            ),
            "prey_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_prey,
                action_space=act_space_prey,
                inference_only=False,
                model_config=shared_model_config,
                catalog_class=None,
                load_state_path=load_paths.get("prey_policy"),
            ),
        }
    )

    use_gpu = torch.cuda.is_available()
    num_gpus_per_learner = 1 if use_gpu else 0
    num_env_runners = 8 if use_gpu else 6
    num_envs_per_env_runner = 3 if use_gpu else 1
    num_cpus_per_env_runner = 3 if use_gpu else 1
    num_cpus_for_main_process = 4 if use_gpu else 1

    print(
        f"Starting retrain experiment: {experiment_name}. "
        f"num_gpus_per_learner={num_gpus_per_learner}, num_env_runners={num_env_runners}, "
        f"num_envs_per_env_runner={num_envs_per_env_runner}, "
        f"num_cpus_per_env_runner={num_cpus_per_env_runner}, "
        f"num_cpus_for_main_process={num_cpus_for_main_process}, seed={args.seed}"
    )

    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies={
                "predator_policy": (None, obs_space_pred, act_space_pred, {}),
                "prey_policy": (None, obs_space_prey, act_space_prey, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
            # This is the freeze mechanism: only retrain_policy_id gets gradient
            # updates. frozen_policy_id is still built, still loaded from its
            # checkpoint, and still used for action selection every step via
            # policy_mapping_fn -- it just never appears in the learner's loss.
            policies_to_train=[retrain_policy_id],
        )
        .learners(num_gpus_per_learner=num_gpus_per_learner, num_learners=1)
        .training(
            train_batch_size_per_learner=1024,
            minibatch_size=128,
            num_epochs=30,
            gamma=0.99,
            lr=0.0003,
            entropy_coeff=0.0,
            vf_loss_coeff=1.0,
            clip_param=0.3,
            kl_coeff=0.2,
            kl_target=0.01,
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .resources(num_cpus_for_main_process=num_cpus_for_main_process)
        .callbacks(EpisodeReturn)
    )
    if args.seed is not None:
        ppo = ppo.debugging(seed=args.seed)

    del sample_env

    tuner = Tuner(
        ppo.algo_class,
        param_space=ppo.to_dict(),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": args.max_iters},
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )
    results = tuner.fit()
    ray.shutdown()
