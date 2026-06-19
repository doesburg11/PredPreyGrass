"""
Paper-like evaluator for exact Malthusian reproduction runs.

Reads Ray Tune `progress.csv` files, derives the island-level metrics used in
Leibo et al. Figure 3 where possible, writes plots, and produces an acceptance
report against the frozen mapped-protocol bands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from predpreygrass.rllib.malthusian_rl.config.config_article_protocol import (
    ARTICLE_EXACT_BLOCKERS,
    ARTICLE_EXPERIMENT_CONDITIONS,
)
from predpreygrass.rllib.malthusian_rl.config.config_paper_protocol import acceptance_bands
from predpreygrass.rllib.malthusian_rl.utils.reproduction_metadata import (
    verify_run_config_metadata,
)

MALTHUSIAN_COLUMN_RE = re.compile(
    r"(?:^|/)malthusian/(?P<kind>mu|phi|count)/(?P<species>[^/]+)/island_(?P<island>\d+)$"
)
SWITCHING_COST_COLUMN_RE = re.compile(
    r"(?:^|/)malthusian/switching_cost/island_(?P<island>\d+)$"
)
SOLITARY_RETURN_COLUMN_RE = re.compile(
    r"(?:^|/)malthusian/solitary_return/(?P<species>[^/]+)$"
)


def _float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int_or_none(value: Any) -> int | None:
    number = _float_or_nan(value)
    if math.isnan(number):
        return None
    return int(number)


def _load_run_config(progress_csv: Path) -> dict[str, Any]:
    for parent in [progress_csv.parent, *progress_csv.parents]:
        config_path = parent / "run_config.json"
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    return {}


def find_progress_files(ray_results_dir: Path, experiment_glob: str) -> list[Path]:
    if ray_results_dir.is_file() and ray_results_dir.name == "progress.csv":
        return [ray_results_dir]

    experiment_dirs = [path for path in ray_results_dir.glob(experiment_glob) if path.is_dir()]
    if not experiment_dirs:
        experiment_dirs = [ray_results_dir]

    progress_files: list[Path] = []
    for experiment_dir in experiment_dirs:
        progress_files.extend(sorted(experiment_dir.rglob("progress.csv")))
    return sorted(set(progress_files))


def _malthusian_columns(fieldnames: list[str]) -> dict[str, dict[tuple[str, int], str]]:
    grouped: dict[str, dict[tuple[str, int], str]] = {
        "mu": {},
        "phi": {},
        "count": {},
    }
    for name in fieldnames:
        match = MALTHUSIAN_COLUMN_RE.search(name)
        if not match:
            continue
        grouped[match.group("kind")][(match.group("species"), int(match.group("island")))] = name
    return grouped


def _switching_cost_columns(fieldnames: list[str]) -> dict[int, str]:
    columns = {}
    for name in fieldnames:
        match = SWITCHING_COST_COLUMN_RE.search(name)
        if match:
            columns[int(match.group("island"))] = name
    return columns


def _solitary_return_columns(fieldnames: list[str]) -> dict[str, str]:
    columns = {}
    for name in fieldnames:
        match = SOLITARY_RETURN_COLUMN_RE.search(name)
        if match:
            columns[match.group("species")] = name
    return columns


def derive_paper_like_rows(progress_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_config = _load_run_config(progress_csv)
    metadata_integrity = verify_run_config_metadata(run_config)
    env_config = run_config.get("config_env", {})
    appo_config = run_config.get("config_appo_exact", {})
    seed = env_config.get("seed")
    experiment_dir = progress_csv.parent.parent if progress_csv.parent.parent.exists() else progress_csv.parent

    with progress_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        columns = _malthusian_columns(fieldnames)
        switching_cost_columns = _switching_cost_columns(fieldnames)
        solitary_return_columns = _solitary_return_columns(fieldnames)
        rows = []
        for row in reader:
            count_by_island: dict[int, float] = defaultdict(float)
            collective_by_island: dict[int, float] = defaultdict(float)
            per_capita_by_island: dict[int, float] = {}

            for key, count_column in columns["count"].items():
                species, island = key
                count = _float_or_nan(row.get(count_column))
                if math.isnan(count):
                    continue
                count_by_island[island] += count
                phi_column = columns["phi"].get((species, island))
                phi = _float_or_nan(row.get(phi_column)) if phi_column else math.nan
                if not math.isnan(phi):
                    collective_by_island[island] += count * phi

            for island, collective_return in collective_by_island.items():
                population = count_by_island.get(island, 0.0)
                if population > 0:
                    per_capita_by_island[island] = collective_return / population

            max_collective = max(collective_by_island.values(), default=math.nan)
            max_per_capita = max(per_capita_by_island.values(), default=math.nan)
            max_population = max(count_by_island.values(), default=math.nan)
            switching_cost_values = [
                _float_or_nan(row.get(column))
                for column in switching_cost_columns.values()
            ]
            finite_switching_costs = [
                value for value in switching_cost_values
                if not math.isnan(value)
            ]
            min_switching_cost = min(finite_switching_costs, default=math.nan)
            solitary_return_values = [
                _float_or_nan(row.get(column))
                for column in solitary_return_columns.values()
            ]
            finite_solitary_returns = [
                value for value in solitary_return_values
                if not math.isnan(value)
            ]
            solitary_return_mean = (
                sum(finite_solitary_returns) / len(finite_solitary_returns)
                if finite_solitary_returns
                else math.nan
            )

            rows.append(
                {
                    "progress_csv": str(progress_csv),
                    "experiment_dir": str(experiment_dir),
                    "trial_dir": str(progress_csv.parent),
                    "seed": seed,
                    "task": env_config.get("task", ""),
                    "variant": env_config.get("variant", ""),
                    "experiment_condition": env_config.get("experiment_condition", ""),
                    "condition_key": env_config.get("condition_key", ""),
                    "report_results_from": env_config.get("report_results_from", ""),
                    "paper_protocol_variant": env_config.get("paper_protocol_variant", ""),
                    "training_iteration": _int_or_none(row.get("training_iteration")),
                    "num_env_steps_sampled_lifetime": _float_or_nan(row.get("num_env_steps_sampled_lifetime")),
                    "episode_return_mean": _float_or_nan(row.get("env_runners/episode_return_mean")),
                    "episode_len_mean": _float_or_nan(row.get("env_runners/episode_len_mean")),
                    "max_collective_return_over_islands": max_collective,
                    "max_per_capita_collective_return_over_islands": max_per_capita,
                    "max_island_population_size": max_population,
                    "min_switching_cost_over_islands": min_switching_cost,
                    "solitary_return_mean": solitary_return_mean,
                }
            )

    metadata = {
        "progress_csv": str(progress_csv),
        "seed": seed,
        "task": env_config.get("task"),
        "variant": env_config.get("variant"),
        "experiment_condition": env_config.get("experiment_condition"),
        "condition_key": env_config.get("condition_key"),
        "report_results_from": env_config.get("report_results_from"),
        "expected_max_iters": appo_config.get("max_iters"),
        "malthusian_metric_columns": sum(len(v) for v in columns.values()) + len(switching_cost_columns) + len(solitary_return_columns),
        "run_config_found": bool(run_config),
        "metadata_integrity": metadata_integrity,
        "metadata_integrity_passed": all(
            bool(metadata_integrity.get(key))
            for key in (
                "checksum_present",
                "checksum_valid",
                "environment_snapshot_present",
                "package_versions_present",
                "git_commit_present",
            )
        ),
    }
    return rows, metadata


def evaluate_metadata_integrity(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    by_progress_csv: dict[str, dict[str, Any]] = {}
    for item in metadata:
        progress_csv = str(item.get("progress_csv", ""))
        integrity = item.get("metadata_integrity", {})
        by_progress_csv[progress_csv] = {
            "run_config_found": bool(item.get("run_config_found")),
            "metadata_integrity_passed": bool(item.get("metadata_integrity_passed")),
            "checksum_present": bool(integrity.get("checksum_present")),
            "checksum_valid": bool(integrity.get("checksum_valid")),
            "environment_snapshot_present": bool(integrity.get("environment_snapshot_present")),
            "package_versions_present": bool(integrity.get("package_versions_present")),
            "git_commit_present": bool(integrity.get("git_commit_present")),
            "metadata_schema": integrity.get("metadata_schema"),
        }

    failed_progress_csvs = [
        progress_csv
        for progress_csv, values in by_progress_csv.items()
        if not (
            values["run_config_found"]
            and values["metadata_integrity_passed"]
        )
    ]
    return {
        "passed": bool(by_progress_csv) and not failed_progress_csvs,
        "failed_progress_csvs": failed_progress_csvs,
        "by_progress_csv": by_progress_csv,
    }


def _write_metrics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "progress_csv",
        "experiment_dir",
        "trial_dir",
        "seed",
        "task",
        "variant",
        "experiment_condition",
        "condition_key",
        "report_results_from",
        "paper_protocol_variant",
        "training_iteration",
        "num_env_steps_sampled_lifetime",
        "episode_return_mean",
        "episode_len_mean",
        "max_collective_return_over_islands",
        "max_per_capita_collective_return_over_islands",
        "max_island_population_size",
        "min_switching_cost_over_islands",
        "solitary_return_mean",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _metric_max(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return max(finite, default=math.nan)


def summarize_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("task", "")),
                str(row.get("variant", "")),
                str(row.get("experiment_condition", "")),
                str(row.get("condition_key", "")),
                row.get("seed"),
            )
        ].append(row)

    summaries = []
    for (task, variant, condition, condition_key, seed), group_rows in sorted(grouped.items()):
        if not group_rows:
            continue
        final_rows = [
            row for row in group_rows
            if row.get("training_iteration") is not None
        ]
        final_row = (
            max(final_rows, key=lambda item: int(item["training_iteration"]))
            if final_rows
            else group_rows[-1]
        )
        summary = {
            "task": task,
            "variant": variant,
            "experiment_condition": condition,
            "condition_key": condition_key,
            "seed": seed,
            "final_training_iteration": final_row.get("training_iteration"),
            "max_collective_return": _metric_max([
                _float_or_nan(row.get("max_collective_return_over_islands"))
                for row in group_rows
            ]),
            "max_per_capita_collective_return": _metric_max([
                _float_or_nan(row.get("max_per_capita_collective_return_over_islands"))
                for row in group_rows
            ]),
            "max_island_population": _metric_max([
                _float_or_nan(row.get("max_island_population_size"))
                for row in group_rows
            ]),
            "min_switching_cost": min(
                (
                    value for value in [
                        _float_or_nan(row.get("min_switching_cost_over_islands"))
                        for row in group_rows
                    ]
                    if not math.isnan(value)
                ),
                default=math.nan,
            ),
            "final_solitary_return": _float_or_nan(final_row.get("solitary_return_mean")),
            "mean_solitary_return": _metric_mean([
                _float_or_nan(row.get("solitary_return_mean"))
                for row in group_rows
            ]),
        }
        summaries.append(summary)
    return summaries


def evaluate_article_condition_coverage(
    metadata: list[dict[str, Any]],
    *,
    min_seeds_per_condition: int,
) -> dict[str, Any]:
    seed_sets: dict[str, set[Any]] = {
        condition: set()
        for condition in sorted(ARTICLE_EXPERIMENT_CONDITIONS)
    }
    unknown_conditions: dict[str, set[Any]] = defaultdict(set)

    for item in metadata:
        if not item.get("task"):
            continue
        condition = item.get("condition_key") or item.get("experiment_condition")
        if not condition:
            unknown_conditions[""].add(item.get("seed"))
            continue
        if condition in seed_sets:
            seed_sets[condition].add(item.get("seed"))
        else:
            unknown_conditions[str(condition)].add(item.get("seed"))

    seed_count_by_condition = {
        condition: len({seed for seed in seeds if seed is not None})
        for condition, seeds in seed_sets.items()
    }
    missing_conditions = [
        condition
        for condition, count in seed_count_by_condition.items()
        if count == 0
    ]
    conditions_with_insufficient_seeds = [
        condition
        for condition, count in seed_count_by_condition.items()
        if 0 < count < min_seeds_per_condition
    ]
    unknown_condition_seed_counts = {
        condition: len({seed for seed in seeds if seed is not None})
        for condition, seeds in unknown_conditions.items()
    }
    passed = (
        not missing_conditions
        and not conditions_with_insufficient_seeds
        and not unknown_condition_seed_counts
    )

    return {
        "passed": passed,
        "min_seeds_per_condition": min_seeds_per_condition,
        "required_conditions": sorted(ARTICLE_EXPERIMENT_CONDITIONS),
        "observed_conditions": [
            condition
            for condition, count in seed_count_by_condition.items()
            if count > 0
        ],
        "missing_conditions": missing_conditions,
        "conditions_with_insufficient_seeds": conditions_with_insufficient_seeds,
        "seed_count_by_condition": seed_count_by_condition,
        "unknown_condition_seed_counts": unknown_condition_seed_counts,
    }


def _write_condition_summaries(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_conditions(rows)
    fieldnames = [
        "task",
        "variant",
        "experiment_condition",
        "condition_key",
        "seed",
        "final_training_iteration",
        "max_collective_return",
        "max_per_capita_collective_return",
        "max_island_population",
        "min_switching_cost",
        "final_solitary_return",
        "mean_solitary_return",
    ]
    written = []

    def write_csv(filename: str, selected: list[dict[str, Any]]) -> None:
        with (output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow(row)
        written.append(filename)

    write_csv("condition_summary.csv", summaries)
    write_csv(
        "figure2_clamity_summary.csv",
        [row for row in summaries if row["task"] == "clamity"],
    )
    write_csv(
        "figure3_allelopathy_summary.csv",
        [row for row in summaries if row["task"] == "allelopathy"],
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "condition_count": len(summaries),
        "figure2_conditions": sorted({
            row["condition_key"] or row["experiment_condition"]
            for row in summaries
            if row["task"] == "clamity"
        }),
        "figure3_conditions": sorted({
            row["condition_key"] or row["experiment_condition"]
            for row in summaries
            if row["task"] == "allelopathy"
        }),
        "known_article_conditions": sorted(ARTICLE_EXPERIMENT_CONDITIONS),
        "article_exact_status": ARTICLE_EXACT_BLOCKERS["status"],
    }
    with (output_dir / "paper_figure_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, allow_nan=True)
    written.append("paper_figure_manifest.json")
    return written


def _mean_by_condition(summaries: list[dict[str, Any]], *, task: str, metric: str) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in summaries:
        if row["task"] != task:
            continue
        condition = row["condition_key"] or row["experiment_condition"] or "unknown_condition"
        value = _float_or_nan(row.get(metric))
        if not math.isnan(value):
            grouped[condition].append(value)
    labels = sorted(grouped)
    values = [_metric_mean(grouped[label]) for label in labels]
    return labels, values


def _plot_condition_summary(
    summaries: list[dict[str, Any]],
    *,
    task: str,
    plot_specs: list[tuple[str, str]],
    title: str,
    output_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    task_summaries = [row for row in summaries if row["task"] == task]
    if not task_summaries:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = list(axes.ravel())
    for ax, (metric, ylabel) in zip(axes_flat, plot_specs, strict=False):
        labels, values = _mean_by_condition(task_summaries, task=task, metric=metric)
        x_values = list(range(len(labels)))
        ax.bar(x_values, values, color="#2f6f73", alpha=0.86)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes_flat[len(plot_specs):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def write_condition_figure_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    summaries = summarize_conditions(rows)
    written = []
    if _plot_condition_summary(
        summaries,
        task="clamity",
        title="Figure 2 family: Clamity condition summary",
        output_path=output_dir / "figure2_clamity_summary.png",
        plot_specs=[
            ("final_solitary_return", "Final solitary return"),
            ("mean_solitary_return", "Mean solitary return"),
            ("max_island_population", "Max island population"),
            ("max_collective_return", "Max collective return"),
        ],
    ):
        written.append("figure2_clamity_summary.png")

    if _plot_condition_summary(
        summaries,
        task="allelopathy",
        title="Figure 3 family: Allelopathy condition summary",
        output_path=output_dir / "figure3_allelopathy_summary.png",
        plot_specs=[
            ("max_collective_return", "Max collective return"),
            ("max_per_capita_collective_return", "Max per-capita return"),
            ("max_island_population", "Max island population"),
            ("min_switching_cost", "Min switching cost"),
        ],
    ):
        written.append("figure3_allelopathy_summary.png")
    return written


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    for idx in range(len(values)):
        span = values[max(0, idx - window + 1) : idx + 1]
        finite = [value for value in span if not math.isnan(value)]
        smoothed.append(sum(finite) / len(finite) if finite else math.nan)
    return smoothed


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["training_iteration"] is not None:
            by_trial[str(row["trial_dir"])].append(row)

    if not by_trial:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    aggregate: dict[int, list[float]] = defaultdict(list)
    for trial_rows in by_trial.values():
        trial_rows = sorted(trial_rows, key=lambda item: item["training_iteration"])
        x_values = [int(row["training_iteration"]) for row in trial_rows]
        y_values = [_float_or_nan(row[metric]) for row in trial_rows]
        y_smooth = _rolling_mean(y_values, smooth_window)
        ax.plot(x_values, y_smooth, color="#8f8f8f", linewidth=1.0, alpha=0.45)
        for x_value, y_value in zip(x_values, y_smooth, strict=False):
            if not math.isnan(y_value):
                aggregate[x_value].append(y_value)

    if aggregate:
        xs = sorted(aggregate)
        ys = [sum(aggregate[x]) / len(aggregate[x]) for x in xs]
        ax.plot(xs, ys, color="#0b3d91", linewidth=2.4, label="seed mean")
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Ecological step / training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def write_plots(rows: list[dict[str, Any]], output_dir: Path, smooth_window: int) -> list[str]:
    written = []
    plot_specs = [
        (
            "max_collective_return_over_islands",
            "Maximum collective return over islands",
            "Max collective return",
            "paper_like_collective_return.png",
        ),
        (
            "max_per_capita_collective_return_over_islands",
            "Maximum per-capita collective return over islands",
            "Max per-capita collective return",
            "paper_like_per_capita_return.png",
        ),
        (
            "max_island_population_size",
            "Maximum island population size over islands",
            "Max island population",
            "paper_like_population.png",
        ),
        (
            "min_switching_cost_over_islands",
            "Minimum switching cost over islands",
            "Min switching cost",
            "paper_like_switching_cost.png",
        ),
        (
            "solitary_return_mean",
            "Mean solitary-island return",
            "Mean solitary return",
            "paper_like_solitary_return.png",
        ),
    ]
    for metric, title, ylabel, filename in plot_specs:
        path = output_dir / filename
        if _plot_metric(rows, metric=metric, title=title, ylabel=ylabel, output_path=path, smooth_window=smooth_window):
            written.append(filename)
    return written


def evaluate_acceptance(rows: list[dict[str, Any]], metadata: list[dict[str, Any]], output_files: list[str]) -> dict[str, Any]:
    bands = acceptance_bands["mapped_protocol"]
    tasks = {item.get("task") for item in metadata if item.get("task")}
    is_article_task = bool(tasks)
    seeds = {item["seed"] for item in metadata if item.get("seed") is not None}
    max_iters_by_trial = {
        item["progress_csv"]: int(item["expected_max_iters"])
        for item in metadata
        if item.get("expected_max_iters") is not None
    }

    final_iteration_fraction = 0.0
    if max_iters_by_trial:
        fractions = []
        for progress_csv, expected in max_iters_by_trial.items():
            trial_rows = [row for row in rows if row["progress_csv"] == progress_csv and row["training_iteration"] is not None]
            if trial_rows and expected > 0:
                fractions.append(max(int(row["training_iteration"]) for row in trial_rows) / expected)
        final_iteration_fraction = min(fractions) if fractions else 0.0

    core_values = []
    for row in rows:
        for metric in (
            "episode_return_mean",
            "max_collective_return_over_islands",
            "max_per_capita_collective_return_over_islands",
            "max_island_population_size",
        ):
            core_values.append(_float_or_nan(row[metric]))
    nan_fraction = sum(math.isnan(value) for value in core_values) / len(core_values) if core_values else 1.0

    final_population_values = []
    for progress_csv in {row["progress_csv"] for row in rows}:
        trial_rows = [row for row in rows if row["progress_csv"] == progress_csv and row["training_iteration"] is not None]
        if not trial_rows:
            continue
        final_row = max(trial_rows, key=lambda item: int(item["training_iteration"]))
        final_population_values.append(_float_or_nan(final_row["max_island_population_size"]))
    min_final_population = min((value for value in final_population_values if not math.isnan(value)), default=math.nan)

    min_malthusian_columns = min((int(item.get("malthusian_metric_columns", 0)) for item in metadata), default=0)
    metadata_integrity = evaluate_metadata_integrity(metadata)
    required_outputs = set(bands["required_outputs"])
    if "allelopathy" in tasks:
        required_outputs.add("paper_like_switching_cost.png")
        required_outputs.add("figure3_allelopathy_summary.png")
    if "clamity" in tasks:
        required_outputs.add("paper_like_solitary_return.png")
        required_outputs.add("figure2_clamity_summary.png")
    missing_outputs = sorted(required_outputs - set(output_files))

    checks = {
        "completed_seed_count": len(seeds) >= bands["min_completed_seeds"],
        "final_iteration_fraction": final_iteration_fraction >= bands["min_final_training_iteration_fraction"],
        "core_metric_nan_fraction": nan_fraction <= bands["max_nan_fraction_core_metrics"],
        "malthusian_metric_columns": min_malthusian_columns >= bands["min_malthusian_metric_columns"],
        "metadata_integrity": metadata_integrity["passed"],
        "final_max_island_population": (
            not math.isnan(min_final_population)
            and min_final_population >= bands["min_final_max_island_population"]
        ),
        "required_outputs": not missing_outputs,
    }

    run_quality_passed = all(checks.values())
    article_exact_passed = False if is_article_task else None
    article_condition_coverage = (
        evaluate_article_condition_coverage(
            metadata,
            min_seeds_per_condition=bands["min_completed_seeds"],
        )
        if is_article_task
        else None
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "claim": (
            "Article task reconstruction evaluated; article-exact reproduction remains blocked"
            if is_article_task
            else bands["claim"]
        ),
        "passed": run_quality_passed if not is_article_task else False,
        "run_quality_passed": run_quality_passed,
        "article_exact_passed": article_exact_passed,
        "article_condition_coverage_passed": (
            article_condition_coverage["passed"]
            if isinstance(article_condition_coverage, dict)
            else None
        ),
        "metadata_integrity_passed": metadata_integrity["passed"],
        "checks": checks,
        "observed": {
            "seed_count": len(seeds),
            "seeds": sorted(seeds),
            "final_iteration_fraction": final_iteration_fraction,
            "core_metric_nan_fraction": nan_fraction,
            "min_malthusian_metric_columns": min_malthusian_columns,
            "min_final_max_island_population": min_final_population,
            "missing_outputs": missing_outputs,
        },
        "article_exact": (
            ARTICLE_EXACT_BLOCKERS
            if is_article_task
            else acceptance_bands["article_exact"]
        ),
        "article_condition_coverage": article_condition_coverage,
        "metadata_integrity": metadata_integrity,
    }


def _write_acceptance_reports(report: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "acceptance_report.json").open("w") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    lines = [
        "# Exact Reproduction Acceptance Report",
        "",
        f"Claim: {report['claim']}",
        f"Passed: {report['passed']}",
        "",
        "## Checks",
    ]
    for key, passed in report["checks"].items():
        lines.append(f"- {key}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Observed",
            f"- seeds: {report['observed']['seeds']}",
            f"- final_iteration_fraction: {report['observed']['final_iteration_fraction']}",
            f"- core_metric_nan_fraction: {report['observed']['core_metric_nan_fraction']}",
            f"- min_malthusian_metric_columns: {report['observed']['min_malthusian_metric_columns']}",
            f"- min_final_max_island_population: {report['observed']['min_final_max_island_population']}",
            f"- run_quality_passed: {report['run_quality_passed']}",
            f"- article_exact_passed: {report['article_exact_passed']}",
            f"- article_condition_coverage_passed: {report['article_condition_coverage_passed']}",
            f"- metadata_integrity_passed: {report['metadata_integrity_passed']}",
            "",
            "## Article Exact Boundary",
            f"- status: {report['article_exact']['status']}",
        ]
    )
    metadata_integrity = report.get("metadata_integrity")
    if isinstance(metadata_integrity, dict):
        lines.extend(
            [
                "",
                "## Metadata Integrity",
                f"- failed_progress_csvs: {metadata_integrity['failed_progress_csvs']}",
            ]
        )
    coverage = report.get("article_condition_coverage")
    if isinstance(coverage, dict):
        lines.extend(
            [
                "",
                "## Article Condition Coverage",
                f"- required_conditions: {len(coverage['required_conditions'])}",
                f"- observed_conditions: {coverage['observed_conditions']}",
                f"- missing_conditions: {coverage['missing_conditions']}",
                f"- conditions_with_insufficient_seeds: {coverage['conditions_with_insufficient_seeds']}",
            ]
        )
    if "required_unimplemented_items" in report["article_exact"]:
        for item in report["article_exact"]["required_unimplemented_items"]:
            lines.append(f"- blocked item: {item}")
    for task, items in report["article_exact"].get("missing_published_constants", {}).items():
        lines.append(f"- missing constants for {task}:")
        for item in items:
            lines.append(f"  - {item}")
    (output_dir / "acceptance_report.md").write_text("\n".join(lines) + "\n")


def run_evaluation(ray_results_dir: Path, experiment_glob: str, output_dir: Path, smooth_window: int) -> dict[str, Any]:
    progress_files = find_progress_files(ray_results_dir, experiment_glob)
    all_rows: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    for progress_csv in progress_files:
        rows, metadata = derive_paper_like_rows(progress_csv)
        all_rows.extend(rows)
        all_metadata.append(metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(all_rows, output_dir / "paper_like_metrics.csv")
    output_files = ["paper_like_metrics.csv"]
    output_files.extend(_write_condition_summaries(all_rows, output_dir))
    output_files.extend(write_condition_figure_plots(all_rows, output_dir))
    output_files.extend(write_plots(all_rows, output_dir, smooth_window=smooth_window))
    output_files.extend(["acceptance_report.json", "acceptance_report.md"])
    report = evaluate_acceptance(all_rows, all_metadata, output_files)
    _write_acceptance_reports(report, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ray-results-dir",
        type=Path,
        default=Path("~/Dropbox/02_marl_results/predpreygrass_results/ray_results/").expanduser(),
    )
    parser.add_argument("--experiment-glob", default="APPO_MALTHUSIAN_EXACT_*")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.ray_results_dir / "paper_like_evaluation"

    report = run_evaluation(
        ray_results_dir=args.ray_results_dir,
        experiment_glob=args.experiment_glob,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
