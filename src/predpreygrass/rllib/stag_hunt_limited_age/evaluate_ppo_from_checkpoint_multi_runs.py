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


def _prepend_snapshot_source() -> None:
    script_path = Path(__file__).resolve()
    try:
        if script_path.parents[2].name == "predpreygrass" and script_path.parents[1].name == "rllib":
            source_root = script_path.parents[3]
            if source_root.name in {"REPRODUCE_CODE", "SOURCE_CODE"}:
                source_root_str = str(source_root)
                if source_root_str not in sys.path:
                    sys.path.insert(0, source_root_str)
    except IndexError:
        return


_prepend_snapshot_source()

PredPreyGrass = None
config_env = None
CombinedEvolutionVisualizer = None
aggregate_capture_outcomes_from_event_log = None
aggregate_join_choices = None


SAVE_EVAL_RESULTS = True
N_RUNS = 10  # Number of evaluation runs
SEED = 1
MIN_STEPS_FOR_STATS = 500 # Minimum steps per run to include in aggregate stats
SURVIVAL_MIN_STEP = 1000
SURVIVAL_WINDOW = 1
SURVIVAL_REQUIRED_TYPES = ("type_1_predator", "type_1_prey", "type_2_prey")


def prepend_example_sources() -> None:
    if not TRAINED_EXAMPLE_DIR:
        return
    example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
    source_dirs = [
        example_dir / "REPRODUCE_CODE",
        example_dir / "SOURCE_CODE",
        example_dir / "eval" / "REPRODUCE_CODE",
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

    from predpreygrass.rllib.stag_hunt_limited_age.predpreygrass_rllib_env import PredPreyGrass as _PredPreyGrass
    from predpreygrass.rllib.stag_hunt_limited_age.config.config_env_stag_hunt_limited_age import config_env as _config_env
    from predpreygrass.rllib.stag_hunt_limited_age.utils.matplot_renderer import (
        CombinedEvolutionVisualizer as _CombinedEvolutionVisualizer,
    )
    from predpreygrass.rllib.stag_hunt_limited_age.utils.defection_metrics import (
        aggregate_capture_outcomes_from_event_log as _aggregate_capture_outcomes_from_event_log,
        aggregate_join_choices as _aggregate_join_choices,
    )

    PredPreyGrass = _PredPreyGrass
    config_env = _config_env
    CombinedEvolutionVisualizer = _CombinedEvolutionVisualizer
    aggregate_capture_outcomes_from_event_log = _aggregate_capture_outcomes_from_event_log
    aggregate_join_choices = _aggregate_join_choices


_SNAPSHOT_EXCLUDE_DIRS = {
    "ray_results",
    "ray_results_failed",
    "trained_examples",
    "__pycache__",
}


def copy_module_snapshot(source_dir: Path) -> None:
    module_dir = Path(__file__).resolve().parent
    module_name = module_dir.name

    pkg_root = source_dir / "predpreygrass"
    rllib_root = pkg_root / "rllib"
    rllib_root.mkdir(parents=True, exist_ok=True)

    pkg_init = pkg_root / "__init__.py"
    if not pkg_init.exists():
        pkg_init.write_text("")
    rllib_init = rllib_root / "__init__.py"
    if not rllib_init.exists():
        rllib_init.write_text("")

    dest_dir = rllib_root / module_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    def _ignore(path: str, entries):
        ignored = []
        for entry in entries:
            if entry in _SNAPSHOT_EXCLUDE_DIRS:
                ignored.append(entry)
                continue
            if entry.endswith(".pyc"):
                ignored.append(entry)
        return ignored

    shutil.copytree(module_dir, dest_dir, ignore=_ignore)

    assets_src = None
    for parent in module_dir.parents:
        if parent.name == "REPRODUCE_CODE":
            candidate = parent / "assets" / "images" / "icons"
            if candidate.is_dir():
                assets_src = candidate
            break
    if assets_src is None:
        candidate = module_dir.parents[4] / "assets" / "images" / "icons"
        if candidate.is_dir():
            assets_src = candidate
    if assets_src:
        assets_dest = source_dir / "assets" / "images" / "icons"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        assets_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(assets_src, assets_dest)


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

    reproduce_dir = eval_output_dir / "REPRODUCE_CODE"
    reproduce_dir.mkdir(parents=True, exist_ok=True)
    copy_module_snapshot(reproduce_dir)
    write_pip_freeze(reproduce_dir / "pip_freeze_eval.txt")

    config_dir = reproduce_dir / "CONFIG"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config_env.json", "w") as f:
        json.dump(env_cfg, f, indent=4)

    with open(eval_output_dir / "config_env.json", "w") as f:
        json.dump(env_cfg, f, indent=4)


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
        return example_dir / "eval" / "runs" / f"eval_multiple_runs_STAG_HUNT_LIMITED_AGE_{now}"
    return checkpoint_path / f"eval_multiple_runs_STAG_HUNT_LIMITED_AGE_{now}"


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


def _count_required_types(agent_ids):
    counts = {k: 0 for k in SURVIVAL_REQUIRED_TYPES}
    for aid in agent_ids:
        if "type_1_predator" in aid:
            counts["type_1_predator"] += 1
        elif "type_1_prey" in aid:
            counts["type_1_prey"] += 1
        elif "type_2_prey" in aid:
            counts["type_2_prey"] += 1
    return counts

def _resolve_checkpoint_path(ray_results_dir, checkpoint_root, checkpoint_nr):
    base = Path(ray_results_dir) / checkpoint_root
    if base.name.startswith("checkpoint_"):
        return base
    return base / checkpoint_nr


def write_survival_summary(output_dir: Path, survivor_runs: list[dict], now: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_txt = output_dir / "survival_summary.txt"
    summary_json = output_dir / "survival_summary.json"

    header = (
        "Runs with all required types > 0 at final step\n"
        f"Timestamp: {now}\n"
        f"Final step only (min step {SURVIVAL_MIN_STEP})\n"
        f"Required types: {', '.join(SURVIVAL_REQUIRED_TYPES)}\n"
    )
    if survivor_runs:
        runs_line = ", ".join(f"run {m['run']} (seed {m['seed']})" for m in survivor_runs)
    else:
        runs_line = "None found."
    summary_txt.write_text(header + runs_line + "\n")

    payload = {
        "timestamp": now,
        "window": SURVIVAL_WINDOW,
        "min_step": SURVIVAL_MIN_STEP,
        "required_types": list(SURVIVAL_REQUIRED_TYPES),
        "n_runs": len(survivor_runs),
        "runs": [
            {
                "run": m.get("run"),
                "seed": m.get("seed"),
                "survival_last_counts": m.get("survival_last_counts", {}),
            }
            for m in survivor_runs
        ],
    }
    summary_json.write_text(json.dumps(payload, indent=2))


def setup_modules():
    if TRAINED_EXAMPLE_DIR:
        example_dir = Path(TRAINED_EXAMPLE_DIR).expanduser().resolve()
        checkpoint_path = resolve_trained_example_checkpoint(example_dir)
    else:
        # STAG_HUNT_LIMITED_AGE_JOIN_COST_0.02_SCAVENGER_0.1_2026-01-25_14-20-20/PPO_PredPreyGrass_99161_00000_0_2026-01-25_14-20-20
        ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_limited_age/ray_results/"
        checkpoint_root = "STAG_HUNT_LIMITED_AGE_JOIN_COST_0_02_SCAVENGER_0_1_2026-01-25_14-20-20/PPO_PredPreyGrass_99161_00000_0_2026-01-25_14-20-20/"
        checkpoint_nr = "checkpoint_000009"
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
                n_possible_type_1_predators=config_env.get("n_possible_type_1_predators"),
                n_possible_type_1_prey=config_env.get("n_possible_type_1_prey"),
                n_possible_type_2_predators=config_env.get("n_possible_type_2_predators"),
                n_possible_type_2_prey=config_env.get("n_possible_type_2_prey"),
            )
        else:
            visualizer = None

        total_reward = 0
        terminated = False
        truncated = False 

        counts_history = []
        while not terminated and not truncated:
            action_dict = {aid: policy_pi(observations[aid], rl_modules[policy_mapping_fn(aid)]) for aid in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            if visualizer:
                visualizer.record(
                    agent_ids=env.agents,
                )

            counts_history.append(_count_required_types(env.agents))
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
        survival_ok = False
        survival_window = []
        if env.current_step >= SURVIVAL_MIN_STEP and counts_history:
            survival_window = counts_history[-SURVIVAL_WINDOW:]
            survival_ok = bool(survival_window) and all(
                all(c[k] > 0 for k in SURVIVAL_REQUIRED_TYPES) for c in survival_window
            )

        per_run_metrics.append(
            {
                "run": run + 1,
                "seed": seed,
                "steps": defection_metrics.get("steps", 0),
                "survival_ok": survival_ok,
                "survival_last_counts": counts_history[-1] if counts_history else {},
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
    survivor_runs = [m for m in per_run_metrics if m.get("survival_ok")]
    if survivor_runs:
        print(
            "\n=== Runs with all required types > 0 at final step ===\n"
            f"Final step only (min step {SURVIVAL_MIN_STEP})\n"
            + ", ".join(f"run {m['run']} (seed {m['seed']})" for m in survivor_runs)
        )
    else:
        print(
            "\n=== Runs with all required types > 0 at final step ===\n"
            "None found."
        )
    if SAVE_EVAL_RESULTS:
        aggregate_output_dir = get_aggregate_output_dir(checkpoint_path, now)
        summary_dir = aggregate_output_dir / "summary_data"
        write_survival_summary(summary_dir, survivor_runs, now)
        print(f"SURVIVAL_RUNS -> {summary_dir / 'survival_summary.txt'}")
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

            join_decision_rate = join_steps / total_pred_steps if total_pred_steps else 0.0
            defect_decision_rate = defect_steps / total_pred_steps if total_pred_steps else 0.0
            captures_total = solo_captures + coop_captures
            solo_capture_rate = solo_captures / captures_total if captures_total else 0.0
            coop_capture_rate = coop_captures / captures_total if captures_total else 0.0
            fr_total = joiners_total + free_riders_total
            free_rider_share = free_riders_total / fr_total if fr_total else 0.0
            coop_free_rider_rate = (
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
                    "join_decision_rate": join_decision_rate,
                    "defect_decision_rate": defect_decision_rate,
                },
                "capture_outcomes": {
                    "captures_successful": captures_successful,
                    "solo_captures": solo_captures,
                    "coop_captures": coop_captures,
                    "solo_capture_rate": solo_capture_rate,
                    "coop_capture_rate": coop_capture_rate,
                    "joiners_total": joiners_total,
                    "free_riders_total": free_riders_total,
                    "free_rider_share": free_rider_share,
                    "coop_participants_total": coop_participants_total,
                    "coop_free_riders_total": coop_free_riders_total,
                    "coop_free_rider_rate": coop_free_rider_rate,
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
                "join_decision_rate",
                "defect_decision_rate",
                "captures_successful",
                "solo_captures",
                "coop_captures",
                "solo_capture_rate",
                "coop_capture_rate",
                "joiners_total",
                "free_riders_total",
                "free_rider_share",
                "coop_participants_total",
                "coop_free_riders_total",
                "coop_free_rider_rate",
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
