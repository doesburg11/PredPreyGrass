from predpreygrass.rllib.stag_hunt_defection.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt_defection.config.config_env_stag_hunt_defection import config_env
from predpreygrass.rllib.stag_hunt_defection.utils.matplot_renderer import CombinedEvolutionVisualizer
from predpreygrass.rllib.stag_hunt_defection.utils.defection_metrics import (
    aggregate_capture_outcomes_from_event_log,
    aggregate_join_choices,
)

# external libraries
import os
import json
import math
from datetime import datetime
import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env


SAVE_EVAL_RESULTS = True
N_RUNS = 10  # Number of evaluation runs
SEED = 1
MIN_STEPS_FOR_STATS = 500 # Minimum steps per run to include in aggregate stats


def policy_mapping_fn(agent_id):
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    raise ValueError(f"Invalid agent_id format: {agent_id}")


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("Missing 'action_dist_inputs' in output.")
    if logits.dim() == 2 and logits.size(0) == 1:
        logits = logits[0]

    action_space = getattr(policy_module, "action_space", None)
    if hasattr(action_space, "nvec"):
        actions = []
        idx = 0
        for n in list(action_space.nvec):
            segment = logits[idx:idx + n]
            if deterministic:
                act = int(torch.argmax(segment).item())
            else:
                act = int(torch.distributions.Categorical(logits=segment).sample().item())
            actions.append(act)
            idx += n
        return actions

    if deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    return int(torch.distributions.Categorical(logits=logits).sample().item())

def _resolve_checkpoint_path(ray_results_dir, checkpoint_root, checkpoint_nr):
    base = os.path.normpath(os.path.join(ray_results_dir, checkpoint_root))
    if os.path.basename(base).startswith("checkpoint_"):
        return base
    return os.path.normpath(os.path.join(base, checkpoint_nr))


def setup_modules():
    # STAG_HUNT_DEFECT_PRED_LOSS_0_08_2026-01-05_01-26-09/PPO_PredPreyGrass_225cc_00000_0_2026-01-05_01-26-10/checkpoint_000029
    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_defection/ray_results/"
    checkpoint_root = "STAG_HUNT_DEFECT_PRED_LOSS_0_08_2026-01-05_01-26-09/PPO_PredPreyGrass_225cc_00000_0_2026-01-05_01-26-10/"
    checkpoint_nr = "checkpoint_000029"
    checkpoint_path = _resolve_checkpoint_path(ray_results_dir, checkpoint_root, checkpoint_nr)
    rl_module_dir = os.path.join(checkpoint_path, "learner_group", "learner", "rl_module")
    module_paths = {
        pid: os.path.join(rl_module_dir, pid)
        for pid in os.listdir(rl_module_dir)
        if os.path.isdir(os.path.join(rl_module_dir, pid))
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}
    return rl_modules, checkpoint_path


def compute_defection_metrics(env):
    join_stats = aggregate_join_choices(env.per_step_agent_data)
    capture_stats = aggregate_capture_outcomes_from_event_log(env.agent_event_log)
    metrics = {
        "steps": env.current_step,
        "join_defect": join_stats,
        "capture_outcomes": capture_stats,
    }
    return metrics


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrass", lambda config: PredPreyGrass(config))
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    per_run_metrics = []

    for run in range(N_RUNS):
        seed = SEED + run
        print(f"\n=== Evaluation Run {run + 1} / {N_RUNS} ===")
        print(f"Using seed: {seed}")
        rl_modules, checkpoint_path = setup_modules()
        env = PredPreyGrass(config=config_env)
        observations, _ = env.reset(seed=SEED + run)  # Use different seed per run
        if SAVE_EVAL_RESULTS:
            eval_output_dir = os.path.join(checkpoint_path, f"eval_multiple_runs_STAG_HUNT_DEFECT_{now}")
            os.makedirs(eval_output_dir, exist_ok=True)
            visualizer = CombinedEvolutionVisualizer(destination_path=eval_output_dir, timestamp=now, run_nr=run + 1)
        else:
            visualizer = None

        total_reward = 0
        terminated = False
        truncated = False 

        while not terminated and not truncated:
            action_dict = {aid: policy_pi(observations[aid], rl_modules[policy_mapping_fn(aid)]) for aid in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            if visualizer:
                visualizer.record(
                    agent_ids=env.agents,
                )

            total_reward += sum(rewards.values())
            # print(f"Step {i} Total Reward so far: {total_reward:.2f}")
            terminated = terminations.get("__all__", False)
            truncated = truncations.get("__all__", False)

        print(f"Evaluation complete! Total Reward: {total_reward:.2f}")
        print(f"Total Steps: {env.current_step}")
        agent_stats = env.get_all_agent_stats()
        defection_metrics = compute_defection_metrics(env)
        print("Defection metrics:")
        print(json.dumps(defection_metrics, indent=2))
        per_run_metrics.append(
            {
                "steps": defection_metrics.get("steps", 0),
                **defection_metrics.get("join_defect", {}),
                **defection_metrics.get("capture_outcomes", {}),
            }
        )
        if SAVE_EVAL_RESULTS:
            visualizer.plot()
            config_env_dir = os.path.join(eval_output_dir, "config_env")
            os.makedirs(config_env_dir, exist_ok=True)
            summary_data_dir = os.path.join(eval_output_dir, "summary_data")
            os.makedirs(summary_data_dir, exist_ok=True)
            with open(os.path.join(config_env_dir, "config_env_" + str(run + 1) + ".json"), "w") as f:
                json.dump(config_env, f, indent=4)
            with open(os.path.join(summary_data_dir, "reward_summary_" + str(run + 1) + ".txt"), "w") as f:
                f.write(f"Total Reward: {total_reward:.2f}\n")
                for aid, rec in agent_stats.items():
                    r = rec.get("cumulative_reward", 0.0)
                    f.write(f"{aid:20}: {r:.2f}\n")
            with open(os.path.join(summary_data_dir, "defection_metrics_" + str(run + 1) + ".json"), "w") as f:
                json.dump(defection_metrics, f, indent=2)

    filtered_runs = [m for m in per_run_metrics if m.get("steps", 0) >= MIN_STEPS_FOR_STATS]
    if filtered_runs:
        def _aggregate_runs(runs):
            total_steps = sum(m.get("steps", 0) for m in runs)
            join_steps = sum(m.get("join_steps", 0) for m in runs)
            defect_steps = sum(m.get("defect_steps", 0) for m in runs)
            total_pred_steps = sum(m.get("total_predator_steps", 0) for m in runs)
            captures_successful = sum(m.get("captures_successful", 0) for m in runs)
            solo_captures = sum(m.get("solo_captures", 0) for m in runs)
            coop_captures = sum(m.get("coop_captures", 0) for m in runs)
            joiners_total = sum(m.get("joiners_total", 0) for m in runs)
            free_riders_total = sum(m.get("free_riders_total", 0) for m in runs)

            join_rate = join_steps / total_pred_steps if total_pred_steps else 0.0
            defect_rate = defect_steps / total_pred_steps if total_pred_steps else 0.0
            captures_total = solo_captures + coop_captures
            solo_rate = solo_captures / captures_total if captures_total else 0.0
            coop_rate = coop_captures / captures_total if captures_total else 0.0
            fr_total = joiners_total + free_riders_total
            free_rider_rate = free_riders_total / fr_total if fr_total else 0.0

            return {
                "n_runs": len(runs),
                "min_steps": MIN_STEPS_FOR_STATS,
                "steps": total_steps,
                "join_defect": {
                    "join_steps": join_steps,
                    "defect_steps": defect_steps,
                    "total_predator_steps": total_pred_steps,
                    "join_rate": join_rate,
                    "defect_rate": defect_rate,
                },
                "capture_outcomes": {
                    "captures_successful": captures_successful,
                    "solo_captures": solo_captures,
                    "coop_captures": coop_captures,
                    "solo_rate": solo_rate,
                    "coop_rate": coop_rate,
                    "joiners_total": joiners_total,
                    "free_riders_total": free_riders_total,
                    "free_rider_rate": free_rider_rate,
                },
            }

        aggregate_metrics = _aggregate_runs(filtered_runs)
        print("\n=== Aggregate defection metrics across runs ===")
        print(f"Filter: steps >= {MIN_STEPS_FOR_STATS} (kept {len(filtered_runs)} of {len(per_run_metrics)})")
        if filtered_runs:
            def _mean(values):
                return sum(values) / len(values) if values else 0.0

            def _std(values):
                if len(values) < 2:
                    return 0.0
                avg = _mean(values)
                return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))

            def _percentile(values, pct):
                if not values:
                    return 0.0
                sorted_vals = sorted(values)
                if len(sorted_vals) == 1:
                    return sorted_vals[0]
                k = (len(sorted_vals) - 1) * (pct / 100.0)
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return sorted_vals[int(k)]
                return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

            stat_keys = [
                "steps",
                "join_steps",
                "defect_steps",
                "total_predator_steps",
                "join_rate",
                "defect_rate",
                "captures_successful",
                "solo_captures",
                "coop_captures",
                "solo_rate",
                "coop_rate",
                "joiners_total",
                "free_riders_total",
                "free_rider_rate",
            ]
            stats = {
                "n_runs": len(filtered_runs),
                "min_steps": MIN_STEPS_FOR_STATS,
                "metrics": {},
                "per_run": filtered_runs,
                "excluded_runs": [m for m in per_run_metrics if m.get("steps", 0) < MIN_STEPS_FOR_STATS],
            }
            for key in stat_keys:
                values = [m.get(key, 0.0) for m in filtered_runs]
                stats["metrics"][key] = {
                    "mean": _mean(values),
                    "std": _std(values),
                    "percentiles": {
                        "p25": _percentile(values, 25),
                        "p50": _percentile(values, 50),
                        "p75": _percentile(values, 75),
                    },
                }
            aggregate_metrics["per_run_stats"] = stats

        print(json.dumps(aggregate_metrics, indent=2))
        if SAVE_EVAL_RESULTS:
            aggregate_path = os.path.join(
                checkpoint_path, f"eval_multiple_runs_PRED_DECAY_0_20_PRED_OBS_RANGE_9_GRID_30_INITS_15_{now}"
            )
            summary_dir = os.path.join(aggregate_path, "summary_data")
            os.makedirs(summary_dir, exist_ok=True)
            with open(os.path.join(summary_dir, "defection_metrics_aggregate.json"), "w") as f:
                json.dump(aggregate_metrics, f, indent=2)
    else:
        print("\n=== Aggregate defection metrics across runs ===")
        print(f"No runs met MIN_STEPS_FOR_STATS={MIN_STEPS_FOR_STATS}")

    ray.shutdown()
