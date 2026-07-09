"""
Run a focused PPO sweep over cadence speed-cost settings.

The goal is to find a middle regime where speed is selectable but not trivial:
speed should drift above the founder distribution without saturating near 1.0.

Examples:
    python -m predpreygrass.evolutionary.eco_evolutionary_cadence.tune_cadence_regime_sweep
    CADENCE_SWEEP_ITERS=40 python -m predpreygrass.evolutionary.eco_evolutionary_cadence.tune_cadence_regime_sweep
    python -m predpreygrass.evolutionary.eco_evolutionary_cadence.tune_cadence_regime_sweep --summarize ~/ray_results/PPO_CADENCE_REGIME_SWEEP_...
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import CheckpointConfig, RunConfig, TuneConfig, Tuner
from ray.tune.registry import register_env

from predpreygrass.evolutionary.eco_evolutionary_cadence.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_cadence.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_cadence.tune_ppo import get_config_ppo, policy_mapping_fn
from predpreygrass.evolutionary.eco_evolutionary_cadence.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_cadence.utils.networks import build_multi_module_spec


SOFT_REGIMES: list[dict[str, float | str]] = [
    {"regime_name": "free_0p0_1p0", "metabolic_speed_coeff": 0.00, "movement_speed_cost_exponent": 1.00},
    {"regime_name": "ultra_cheap_0p05_1p0", "metabolic_speed_coeff": 0.05, "movement_speed_cost_exponent": 1.00},
    {"regime_name": "very_cheap_0p1_1p0", "metabolic_speed_coeff": 0.10, "movement_speed_cost_exponent": 1.00},
    {"regime_name": "ultra_cheap_0p05_0p75", "metabolic_speed_coeff": 0.05, "movement_speed_cost_exponent": 0.75},
]

LOW_FOUNDER_STARTS: list[dict[str, float | str]] = [
    {"founder_start_name": "founder_0p05_std_0p05", "founder_speed_mean": 0.05, "founder_speed_std": 0.05},
    {"founder_start_name": "founder_0p10_std_0p05", "founder_speed_mean": 0.10, "founder_speed_std": 0.05},
]


def env_creator(config: dict[str, Any]) -> PredPreyGrass:
    return PredPreyGrass(config)


def _trial_dirname(trial) -> str:
    env_cfg = trial.config.get("env_config", {})
    regime = env_cfg.get("regime_name", trial.trial_id)
    founder = env_cfg.get("founder_start_name")
    if founder:
        regime = f"{regime}_{founder}"
    return f"{trial.trainable_name}_{regime}_{trial.trial_id}"


def _build_policy_spaces() -> tuple[dict[str, Any], dict[str, Any]]:
    sample_env = env_creator(config=config_env)
    if sample_env.observation_spaces is None or sample_env.action_spaces is None:
        raise RuntimeError("PredPreyGrass must define observation_spaces and action_spaces for all policies.")

    obs_by_policy: dict[str, Any] = {}
    act_by_policy: dict[str, Any] = {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        policy_id = policy_mapping_fn(agent_id)
        if policy_id not in obs_by_policy:
            obs_by_policy[policy_id] = obs_space
            act_by_policy[policy_id] = sample_env.action_spaces[agent_id]
    return obs_by_policy, act_by_policy


def _candidate_env_configs(
    regimes: list[dict[str, float | str]],
    founder_starts: list[dict[str, float | str]],
) -> list[dict[str, Any]]:
    configs = []
    for regime in regimes:
        for founder_start in founder_starts:
            candidate = copy.deepcopy(config_env)
            candidate.update(regime)
            founder_speed_mean = float(founder_start["founder_speed_mean"])
            founder_speed_std = float(founder_start["founder_speed_std"])
            candidate["founder_genome"]["predator"]["speed_mean"] = founder_speed_mean
            candidate["founder_genome"]["prey"]["speed_mean"] = founder_speed_mean
            candidate["founder_genome"]["predator"]["speed_std"] = founder_speed_std
            candidate["founder_genome"]["prey"]["speed_std"] = founder_speed_std
            candidate.update(founder_start)
            candidate["regime_name"] = f"{regime['regime_name']}_{founder_start['founder_start_name']}"
            configs.append(candidate)
    return configs


def _write_provenance(
    experiment_path: Path,
    config_ppo: dict[str, Any],
    regimes: list[dict[str, float | str]],
    founder_starts: list[dict[str, float | str]],
) -> None:
    experiment_path.mkdir(parents=True, exist_ok=True)
    source_dir = experiment_path / "SOURCE_CODE"
    source_dir.mkdir(exist_ok=True)
    module_dir = Path(__file__).parent
    shutil.copy2(module_dir / "predpreygrass_rllib_env.py", source_dir / "predpreygrass_rllib_env_CADENCE.py")
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)

    metadata = {
        "purpose": "cadence middle-regime speed-cost sweep",
        "selection_rule": (
            "Prefer regimes where predator/prey speed p50 drift above the low-founder baseline, "
            "fraction_fast_cadence rises without saturating, and fraction_mobile stays modest."
        ),
        "base_config_env": config_env,
        "config_ppo": config_ppo,
        "regimes": regimes,
        "founder_starts": founder_starts,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(metadata, f, indent=4)


def _build_ppo_config(env_configs: list[dict[str, Any]], config_ppo: dict[str, Any]) -> dict[str, Any]:
    obs_by_policy, act_by_policy = _build_policy_spaces()
    policies = {
        policy_id: (None, obs_by_policy[policy_id], act_by_policy[policy_id], {})
        for policy_id in obs_by_policy
    }
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)

    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=tune.grid_search(env_configs), disable_env_checking=True)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
            minibatch_size=config_ppo["minibatch_size"],
            num_epochs=config_ppo["num_epochs"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
            lambda_=config_ppo["lambda_"],
            entropy_coeff=config_ppo["entropy_coeff"],
            vf_loss_coeff=config_ppo["vf_loss_coeff"],
            clip_param=config_ppo["clip_param"],
            kl_coeff=config_ppo["kl_coeff"],
            kl_target=config_ppo["kl_target"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
            num_learners=config_ppo["num_learners"],
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"])
        .callbacks(EpisodeReturn)
    )
    return ppo_config.to_dict()


def run_sweep() -> Path:
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)

    ray_results_path = Path(os.environ.get("RAY_RESULTS_DIR", "~/ray_results")).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_CADENCE_REGIME_SWEEP_{timestamp}"
    experiment_path = ray_results_path / experiment_name

    config_ppo = copy.deepcopy(get_config_ppo())
    config_ppo["max_iters"] = int(os.environ.get("CADENCE_SWEEP_ITERS", min(40, int(config_ppo["max_iters"]))))
    checkpoint_every = int(os.environ.get("CADENCE_SWEEP_CHECKPOINT_EVERY", "20"))

    regimes = SOFT_REGIMES
    founder_starts = LOW_FOUNDER_STARTS
    env_configs = _candidate_env_configs(regimes, founder_starts)
    _write_provenance(experiment_path, config_ppo, regimes, founder_starts)

    tuner = Tuner(
        PPOConfig().algo_class,
        param_space=_build_ppo_config(env_configs, config_ppo),
        tune_config=TuneConfig(trial_dirname_creator=_trial_dirname),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": config_ppo["max_iters"]},
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
        ),
    )
    tuner.fit()
    ray.shutdown()
    summarize_sweep(experiment_path)
    return experiment_path


def _last_window_mean(df: pd.DataFrame, column: str, window: int) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.tail(window).mean())


def _load_trial_env_config(trial_path: Path) -> dict[str, Any]:
    params_path = trial_path / "params.json"
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        params = json.load(f)
    env_config = params.get("env_config", {})
    return env_config if isinstance(env_config, dict) else {}


def _score_trial(row: dict[str, Any]) -> tuple[float, str]:
    predator_p50 = row.get("predator_speed_p50")
    prey_p50 = row.get("prey_speed_p50")
    predator_mobile = row.get("predator_fraction_mobile")
    prey_mobile = row.get("prey_fraction_mobile")
    predator_fast_cadence = row.get("predator_fraction_fast_cadence")
    prey_fast_cadence = row.get("prey_fraction_fast_cadence")
    founder_speed_mean = float(row.get("founder_speed_mean") or 0.50)
    if predator_p50 is None or prey_p50 is None:
        return -999.0, "missing speed metrics"

    mean_p50 = (float(predator_p50) + float(prey_p50)) / 2.0
    mean_mobile = (
        (float(predator_mobile or 0.0) + float(prey_mobile or 0.0)) / 2.0
    )
    mean_fast_cadence = (
        (float(predator_fast_cadence or 0.0) + float(prey_fast_cadence or 0.0)) / 2.0
    )
    drift = max(0.0, mean_p50 - founder_speed_mean)
    saturation_penalty = (
        max(0.0, mean_p50 - 0.82) * 4.0
        + max(0.0, mean_mobile - 0.35) * 2.0
        + max(0.0, mean_fast_cadence - 0.75) * 0.5
    )
    stall_penalty = max(0.0, 0.06 - drift) * 3.0
    asymmetry_penalty = abs(float(predator_p50) - float(prey_p50)) * 0.5
    score = drift - saturation_penalty - stall_penalty - asymmetry_penalty

    if mean_p50 >= 0.90 or mean_mobile >= 0.60:
        label = "too hot/saturating"
    elif drift <= 0.06:
        label = "too cold/stalled"
    else:
        label = "candidate middle regime"
    return score, label


def summarize_sweep(experiment_path: Path, *, window: int = 10) -> list[dict[str, Any]]:
    rows = []
    for progress_path in sorted(experiment_path.glob("*/progress.csv")):
        df = pd.read_csv(progress_path)
        if df.empty:
            continue
        trial_dir = progress_path.parent.name
        env_config = _load_trial_env_config(progress_path.parent)
        row: dict[str, Any] = {"trial_dir": trial_dir}
        for key, column in {
            "predator_speed_p50": "env_runners/eco_evolution/predator_speed_p50",
            "prey_speed_p50": "env_runners/eco_evolution/prey_speed_p50",
            "predator_fraction_mobile": "env_runners/eco_evolution/predator_fraction_mobile",
            "prey_fraction_mobile": "env_runners/eco_evolution/prey_fraction_mobile",
            "predator_fraction_fast_cadence": "env_runners/eco_evolution/predator_fraction_fast_cadence",
            "prey_fraction_fast_cadence": "env_runners/eco_evolution/prey_fraction_fast_cadence",
            "predator_cooldown_mean": "env_runners/eco_evolution/predator_cooldown_mean",
            "prey_cooldown_mean": "env_runners/eco_evolution/prey_cooldown_mean",
            "predator_movement_energy": "env_runners/eco_evolution/predator_movement_energy_spent_mean",
            "prey_movement_energy": "env_runners/eco_evolution/prey_movement_energy_spent_mean",
        }.items():
            row[key] = _last_window_mean(df, column, window)

        for key in (
            "regime_name",
            "founder_start_name",
            "founder_speed_mean",
            "founder_speed_std",
            "metabolic_speed_coeff",
            "movement_speed_cost_exponent",
        ):
            column = f"config/env_config/{key}"
            progress_values = df[column].dropna() if column in df.columns else pd.Series(dtype=object)
            row[key] = progress_values.iloc[-1] if not progress_values.empty else env_config.get(key)

        founder_genome = env_config.get("founder_genome", {})
        predator_founder = founder_genome.get("predator", {}) if isinstance(founder_genome, dict) else {}
        if row["founder_speed_mean"] is None:
            row["founder_speed_mean"] = predator_founder.get("speed_mean")
        if row["founder_speed_std"] is None:
            row["founder_speed_std"] = predator_founder.get("speed_std")

        row["score"], row["classification"] = _score_trial(row)
        rows.append(row)

    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    summary_path = experiment_path / "cadence_regime_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")
    if rows:
        print("Best candidate:")
        print(json.dumps(rows[0], indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", type=Path, help="Summarize an existing sweep directory instead of training.")
    parser.add_argument("--window", type=int, default=10, help="Number of final progress rows to average.")
    args = parser.parse_args()

    if args.summarize:
        summarize_sweep(args.summarize.expanduser(), window=args.window)
    else:
        path = run_sweep()
        print(f"Finished cadence regime sweep: {path}")


if __name__ == "__main__":
    main()
