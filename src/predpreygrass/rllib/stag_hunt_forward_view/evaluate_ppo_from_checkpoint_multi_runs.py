import ast
import importlib.util
import os
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env


TRAINED_EXAMPLE_DIR = os.getenv("TRAINED_EXAMPLE_DIR")

PredPreyGrass = None
config_env = None
CombinedEvolutionVisualizer = None
aggregate_capture_outcomes_from_event_log = None
aggregate_join_choices = None


SAVE_EVAL_RESULTS = True
N_RUNS = 20  # Number of evaluation runs
SEED = 1
MIN_STEPS_FOR_STATS = 500 # Minimum steps per run to include in aggregate stats


def prepend_example_sources() -> None:
    if not TRAINED_EXAMPLE_DIR:
        return
    example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
    source_dirs = [
        example_dir / "SOURCE_CODE",
        example_dir / "eval" / "SOURCE_CODE",
    ]
    for path in source_dirs:
        if path.is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def load_predpreygrass_modules() -> None:
    global PredPreyGrass, config_env, CombinedEvolutionVisualizer
    global aggregate_capture_outcomes_from_event_log, aggregate_join_choices

    from predpreygrass.rllib.stag_hunt_forward_view.predpreygrass_rllib_env import PredPreyGrass as _PredPreyGrass
    from predpreygrass.rllib.stag_hunt_forward_view.config.config_env_stag_hunt_forward_view import config_env as _config_env
    from predpreygrass.rllib.stag_hunt_forward_view.utils.matplot_renderer import (
        CombinedEvolutionVisualizer as _CombinedEvolutionVisualizer,
    )
    from predpreygrass.rllib.stag_hunt_forward_view.utils.defection_metrics import (
        aggregate_capture_outcomes_from_event_log as _aggregate_capture_outcomes_from_event_log,
        aggregate_join_choices as _aggregate_join_choices,
    )

    PredPreyGrass = _PredPreyGrass
    config_env = _config_env
    CombinedEvolutionVisualizer = _CombinedEvolutionVisualizer
    aggregate_capture_outcomes_from_event_log = _aggregate_capture_outcomes_from_event_log
    aggregate_join_choices = _aggregate_join_choices


def _copy_init_files(origin: Path, source_dir: Path, root_name: str) -> None:
    parts = origin.parts
    if root_name not in parts:
        return
    idx = parts.index(root_name)
    for parent in origin.parents:
        if len(parent.parts) <= idx:
            break
        if parent.parts[idx] != root_name:
            continue
        init_path = parent / "__init__.py"
        if init_path.is_file():
            rel_init = Path(*init_path.parts[idx:])
            dest_init = source_dir / rel_init
            dest_init.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(init_path, dest_init)


def copy_imported_modules(source_dir: Path) -> None:
    script_path = Path(__file__).resolve()
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    for name in sorted(modules):
        if not name.startswith("predpreygrass."):
            continue
        spec = importlib.util.find_spec(name)
        if not spec or not spec.origin or not spec.origin.endswith(".py"):
            continue
        origin = Path(spec.origin).resolve()
        parts = origin.parts
        if "predpreygrass" in parts:
            idx = parts.index("predpreygrass")
            rel = Path(*parts[idx:])
            dest = source_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, dest)
            _copy_init_files(origin, source_dir, "predpreygrass")
        else:
            shutil.copy2(origin, source_dir / origin.name)

    shutil.copy2(script_path, source_dir / script_path.name)


def write_pip_freeze(output_path: Path) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        output_path.write_text(result.stdout)
        if result.stderr:
            (output_path.parent / "pip_freeze_eval_stderr.txt").write_text(result.stderr)
    except Exception as exc:
        output_path.write_text(f"pip freeze failed: {exc}")


def prepare_eval_output_dir(eval_output_dir: Path, env_cfg: dict) -> None:
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = eval_output_dir / "CONFIG"
    if not config_dir.exists():
        config_dir.mkdir(exist_ok=True)
        write_pip_freeze(config_dir / "pip_freeze_eval.txt")
        with open(config_dir / "config_env.json", "w") as f:
            json.dump(env_cfg, f, indent=4)

    with open(eval_output_dir / "config_env.json", "w") as f:
        json.dump(env_cfg, f, indent=4)

    source_dir = eval_output_dir / "SOURCE_CODE"
    if not source_dir.exists():
        source_dir.mkdir(exist_ok=True)
        copy_imported_modules(source_dir)


def resolve_trained_example_checkpoint(example_dir: Path) -> Path:
    checkpoint_dir = example_dir / "checkpoint"
    if checkpoint_dir.is_dir():
        return checkpoint_dir
    candidates = sorted(example_dir.glob("checkpoint_*"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {example_dir}")
    raise FileExistsError(f"Multiple checkpoints found in {example_dir}; please keep only one.")


def get_eval_output_dir(checkpoint_path: Path, now: str) -> Path:
    if TRAINED_EXAMPLE_DIR:
        example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
        return example_dir / "eval" / "runs" / f"eval_multiple_runs_STAG_HUNT_FORWARD_VIEW_{now}"
    return checkpoint_path / f"eval_multiple_runs_STAG_HUNT_FORWARD_VIEW_{now}"


def get_aggregate_output_dir(checkpoint_path: Path, now: str) -> Path:
    return get_eval_output_dir(checkpoint_path, now)


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
    base = Path(ray_results_dir) / checkpoint_root
    if base.name.startswith("checkpoint_"):
        return base
    return base / checkpoint_nr


def setup_modules():
    if TRAINED_EXAMPLE_DIR:
        example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
        checkpoint_path = resolve_trained_example_checkpoint(example_dir)
    else:
        # STAG_HUNT_FORWARD_VIEW_2026-01-24_20-13-33/PPO_PredPreyGrass_c6e2d_00000_0_2026-01-24_20-13-33
        ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_forward_view/ray_results/"
        checkpoint_root = "STAG_HUNT_FORWARD_VIEW_JOIN_COST_0.02_SCAVENGER_0.1_2026-01-25_14-20-20/PPO_PredPreyGrass_99161_00000_0_2026-01-25_14-20-20/"
        checkpoint_nr = "checkpoint_000099"
        checkpoint_path = _resolve_checkpoint_path(ray_results_dir, checkpoint_root, checkpoint_nr)

    rl_module_dir = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    module_paths = {
        pid: str(rl_module_dir / pid)
        for pid in os.listdir(rl_module_dir)
        if (rl_module_dir / pid).is_dir()
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}
    return rl_modules, Path(checkpoint_path)


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
    prepend_example_sources()
    load_predpreygrass_modules()
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
            eval_output_dir = get_eval_output_dir(checkpoint_path, now)
            if run == 0:
                prepare_eval_output_dir(eval_output_dir, config_env)
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            visualizer = CombinedEvolutionVisualizer(
                destination_path=str(eval_output_dir),
                timestamp=now,
                destination_filename="visuals",
                run_nr=run + 1,
            )
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
            config_env_dir = eval_output_dir / "config_env"
            config_env_dir.mkdir(exist_ok=True)
            summary_data_dir = eval_output_dir / "summary_data"
            summary_data_dir.mkdir(exist_ok=True)
            with open(config_env_dir / f"config_env_{run + 1}.json", "w") as f:
                json.dump(config_env, f, indent=4)
            with open(summary_data_dir / f"reward_summary_{run + 1}.txt", "w") as f:
                f.write(f"Total Reward: {total_reward:.2f}\n")
                for aid, rec in agent_stats.items():
                    r = rec.get("cumulative_reward", 0.0)
                    f.write(f"{aid:20}: {r:.2f}\n")
            with open(summary_data_dir / f"defection_metrics_{run + 1}.json", "w") as f:
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
            coop_participants_total = sum(m.get("coop_participants_total", 0) for m in runs)
            coop_free_riders_total = sum(m.get("coop_free_riders_total", 0) for m in runs)
            coop_captures_with_free_riders = sum(m.get("coop_captures_with_free_riders", 0) for m in runs)

            join_rate = join_steps / total_pred_steps if total_pred_steps else 0.0
            defect_rate = defect_steps / total_pred_steps if total_pred_steps else 0.0
            captures_total = solo_captures + coop_captures
            solo_rate = solo_captures / captures_total if captures_total else 0.0
            coop_rate = coop_captures / captures_total if captures_total else 0.0
            fr_total = joiners_total + free_riders_total
            free_rider_rate = free_riders_total / fr_total if fr_total else 0.0
            coop_defection_rate = (
                coop_free_riders_total / coop_participants_total if coop_participants_total else 0.0
            )
            coop_free_rider_presence_rate = (
                coop_captures_with_free_riders / coop_captures if coop_captures else 0.0
            )

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
                    "coop_participants_total": coop_participants_total,
                    "coop_free_riders_total": coop_free_riders_total,
                    "coop_defection_rate": coop_defection_rate,
                    "coop_captures_with_free_riders": coop_captures_with_free_riders,
                    "coop_free_rider_presence_rate": coop_free_rider_presence_rate,
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
                "coop_participants_total",
                "coop_free_riders_total",
                "coop_defection_rate",
                "coop_captures_with_free_riders",
                "coop_free_rider_presence_rate",
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
            aggregate_path = get_aggregate_output_dir(checkpoint_path, now)
            summary_dir = aggregate_path / "summary_data"
            summary_dir.mkdir(parents=True, exist_ok=True)
            with open(summary_dir / "defection_metrics_aggregate.json", "w") as f:
                json.dump(aggregate_metrics, f, indent=2)
    else:
        print("\n=== Aggregate defection metrics across runs ===")
        print(f"No runs met MIN_STEPS_FOR_STATS={MIN_STEPS_FOR_STATS}")

    ray.shutdown()
