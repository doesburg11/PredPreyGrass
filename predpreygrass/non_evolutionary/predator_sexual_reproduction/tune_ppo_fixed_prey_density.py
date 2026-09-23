"""
Fixed prey-density variant of tune_ppo_predator_sexual_reproduction.py: trains FixedPreyDensityEnv
(see fixed_prey_density_env.py) instead of the base PredPreyGrass, so a high female hunting-success
setting can be tested without the prey population collapsing -- the confound Codex's second opinion
flagged in ABL_SUCCESS_ONLY / ABL_EQUAL_ODDS (see RESULTS.md, Iterations 8-9: those ablations
removed the k=0.5 division of labor, but also collapsed the prey to 0.4-2.9 alive, so it was
unclear whether raising success removes the split or whether the collapsing ecology does). Only the
prey COUNT is held near a floor, not predator abundance or catch throughput -- "fixed prey-density"
is the accurate name; a third Codex review (RESULTS.md, Iteration 11) pointed out that "matched
ecology" overstates this.

Everything else (policies, PPO config, reward/odds/penalty flags, the EpisodeReturn callback and
policy_mapping_fn, imported unchanged from tune_ppo_predator_sexual_reproduction.py) is identical;
only the environment class and the new --prey-density-floor flag differ. A separate script, rather
than a flag on the existing one, so nothing the existing tune script or any run using it depends on
is touched by this experiment.

Checkpoints and a run_config.json snapshot of the env/PPO config are saved
under ~/simulation_results/ray_results/<experiment_name>/ for provenance.
"""
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_prey_density_env import FixedPreyDensityEnv
from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.tune_ppo_predator_sexual_reproduction import (
    EpisodeReturn,
    policy_mapping_fn,
)
from predpreygrass.global_config import RAY_RESULTS_DIR

#  external libraries
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base RNG seed for this run (for multi-seed replication). Tags the "
             "experiment name so runs don't collide.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override the auto-generated experiment name (used as the Tune "
             "RunConfig name and the ray_results subdirectory).",
    )
    parser.add_argument("--max-iters", type=int, default=1000, help="Training-iteration stop condition.")
    parser.add_argument(
        "--gpu-fraction", type=float, default=None,
        help="Override num_gpus_per_learner. Default: 1 if a GPU is available, else 0.",
    )
    parser.add_argument("--num-env-runners", type=int, default=None, help="Default: 8 if a GPU is available, else 6.")
    parser.add_argument(
        "--num-cpus-per-env-runner", type=int, default=None, help="Default: 3 if a GPU is available, else 1."
    )
    parser.add_argument("--num-cpus-main", type=int, default=None, help="Default: 4 if a GPU is available, else 1.")
    parser.add_argument(
        "--mate-search-radius", type=int, default=None,
        help="Override mate_search_radius. Default: config_env.py's shipped value.",
    )
    parser.add_argument("--male-success-prob", type=float, default=None, help="Override prey_vs_predator_male_success_prob.")
    parser.add_argument("--male-death-prob", type=float, default=None, help="Override prey_vs_predator_male_death_prob.")
    parser.add_argument(
        "--female-success-prob", type=float, default=None, help="Override prey_vs_predator_female_success_prob."
    )
    parser.add_argument("--female-death-prob", type=float, default=None, help="Override prey_vs_predator_female_death_prob.")
    parser.add_argument("--reward-catch-prey", type=float, default=None, help="Override reward_predator_catch_prey.")
    parser.add_argument("--reward-gather-fruit", type=float, default=None, help="Override reward_predator_gather_fruit.")
    parser.add_argument(
        "--reward-per-energy", type=float, default=None,
        help="Set reward_predator_per_energy: reward = this * energy gained, for fruit and prey "
             "alike, added to the flat per-event rewards. Default: config_env.py's shipped value (0.0 = off).",
    )
    parser.add_argument(
        "--penalty-combat-death", type=float, default=None,
        help="Magnitude (>= 0) of the penalty a predator receives when it dies in a failed hunt. "
             "Stored as penalty_predator_death_in_combat = -value. Default: shipped value (0.0).",
    )
    parser.add_argument(
        "--prey-density-floor", type=int, default=20,
        help="FixedPreyDensityEnv's prey_density_floor: the minimum prey population maintained by "
             "replacement spawns after every step (see fixed_prey_density_env.py). Default 20, close "
             "to the healthy k=0.5 baseline's typical end-of-episode prey count (18-21).",
    )
    parser.add_argument(
        "--n-possible-prey", type=int, default=None,
        help="Override n_possible_prey: the per-episode budget of prey agent IDs, shared by normal "
             "reproduction and by FixedPreyDensityEnv's replenishment spawns. Default: config_env.py's "
             "shipped value (2000). RESULTS.md Iteration 11 found this exhausted mid-episode under "
             "heavy hunting pressure (up to ~1,900 catches/episode by iteration 300), silently "
             "disabling the floor; pass a larger value (e.g. 50000) to avoid this. Cheap to raise: "
             "observation_spaces/action_spaces reference a few shared space objects per possible "
             "agent, not per-agent copies (predpreygrass_rllib_env.py:257-274).",
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=1024,
        help="PPO minibatch_size. Default 1024 (this script has no minibatch-128 legacy runs to "
             "stay comparable with, so it defaults to the faster setting).",
    )
    parser.add_argument("--num-epochs", type=int, default=30, help="PPO num_epochs (default 30).")
    return parser.parse_args()


def env_creator(config):
    return FixedPreyDensityEnv({**config_env, **(config or {})})


if __name__ == "__main__":
    args = parse_args()

    env_config = dict(config_env)
    if args.mate_search_radius is not None:
        env_config["mate_search_radius"] = args.mate_search_radius
    if args.male_success_prob is not None:
        env_config["prey_vs_predator_male_success_prob"] = args.male_success_prob
    if args.male_death_prob is not None:
        env_config["prey_vs_predator_male_death_prob"] = args.male_death_prob
    if args.female_success_prob is not None:
        env_config["prey_vs_predator_female_success_prob"] = args.female_success_prob
    if args.female_death_prob is not None:
        env_config["prey_vs_predator_female_death_prob"] = args.female_death_prob
    if args.reward_catch_prey is not None:
        env_config["reward_predator_catch_prey"] = args.reward_catch_prey
    if args.reward_gather_fruit is not None:
        env_config["reward_predator_gather_fruit"] = args.reward_gather_fruit
    if args.reward_per_energy is not None:
        env_config["reward_predator_per_energy"] = args.reward_per_energy
    if args.penalty_combat_death is not None:
        if args.penalty_combat_death < 0:
            raise SystemExit("--penalty-combat-death must be >= 0 (it is a magnitude, stored as a negative reward)")
        env_config["penalty_predator_death_in_combat"] = -args.penalty_combat_death
    env_config["prey_density_floor"] = args.prey_density_floor
    if args.n_possible_prey is not None:
        env_config["n_possible_prey"] = args.n_possible_prey

    register_env("FixedPreyDensityPredPreyGrass", env_creator)
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    ray_results_path = Path(RAY_RESULTS_DIR).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_tag = f"_SEED{args.seed}" if args.seed is not None else ""
    experiment_name = args.name or f"PPO_FIXED_PREY_DENSITY{seed_tag}_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump({"config_env": env_config, "seed": args.seed}, f, indent=4)

    sample_env = env_creator(env_config)
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    if observation_spaces is None or action_spaces is None:
        raise RuntimeError("Sample environment did not initialize observation/action spaces")

    obs_space_pred_male = observation_spaces["predator_male_0"]
    act_space_pred_male = action_spaces["predator_male_0"]
    obs_space_pred_female = observation_spaces["predator_female_0"]
    act_space_pred_female = action_spaces["predator_female_0"]
    obs_space_prey = observation_spaces["prey_0"]
    act_space_prey = action_spaces["prey_0"]

    conv_filters = [
        [16, [3, 3], 1],
        [32, [3, 3], 1],
        [64, [3, 3], 1],
    ]
    model_config = {
        "conv_filters": conv_filters,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }

    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_male_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred_male,
                action_space=act_space_pred_male,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
            "predator_female_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred_female,
                action_space=act_space_pred_female,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
            "prey_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_prey,
                action_space=act_space_prey,
                inference_only=False,
                model_config=model_config,
                catalog_class=None,
            ),
        }
    )

    use_gpu = torch.cuda.is_available()
    num_gpus_per_learner = args.gpu_fraction if args.gpu_fraction is not None else (1 if use_gpu else 0)
    num_env_runners = args.num_env_runners if args.num_env_runners is not None else (8 if use_gpu else 6)
    num_envs_per_env_runner = 3 if use_gpu else 1
    num_cpus_per_env_runner = args.num_cpus_per_env_runner if args.num_cpus_per_env_runner is not None else (3 if use_gpu else 1)
    num_cpus_for_main_process = args.num_cpus_main if args.num_cpus_main is not None else (4 if use_gpu else 1)

    print(
        f"Starting new FixedPreyDensity training experiment: {experiment_name}. "
        f"prey_density_floor={args.prey_density_floor}, "
        f"num_gpus_per_learner={num_gpus_per_learner}, num_env_runners={num_env_runners}, "
        f"num_envs_per_env_runner={num_envs_per_env_runner}, num_cpus_per_env_runner={num_cpus_per_env_runner}, "
        f"num_cpus_for_main_process={num_cpus_for_main_process}, seed={args.seed}"
    )

    ppo = (
        PPOConfig()
        .environment(env="FixedPreyDensityPredPreyGrass", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies={
                "predator_male_policy": (None, obs_space_pred_male, act_space_pred_male, {}),
                "predator_female_policy": (None, obs_space_pred_female, act_space_pred_female, {}),
                "prey_policy": (None, obs_space_prey, act_space_prey, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .learners(num_gpus_per_learner=num_gpus_per_learner, num_learners=1)
        .training(
            train_batch_size_per_learner=1024,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
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
            checkpoint_config=CheckpointConfig(num_to_keep=100, checkpoint_frequency=10, checkpoint_at_end=True),
        ),
    )
    results = tuner.fit()
    ray.shutdown()
