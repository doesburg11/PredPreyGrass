#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise SystemExit(
        "TensorBoard is required. Install it in your env (pip/conda) and rerun."
    ) from exc


DEFAULT_LOGDIR = (
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "stag_hunt_vectorized/ray_results"
)
DEFAULT_TAGS = [
    "ray/tune/timing/iter_minutes",
    "ray/tune/timers/env_runner_sampling_timer",
    "ray/tune/env_runners/num_env_steps_sampled_lifetime_throughput/throughput_since_last_reduce",
]


def find_trial_dir(event_path: Path) -> Path:
    for parent in event_path.parents:
        if (parent / "params.json").exists() or (parent / "result.json").exists():
            return parent
    return event_path.parent


def load_scalars(event_path: Path, tags: list[str]) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    results: dict[str, list[tuple[int, float]]] = {tag: [] for tag in tags}
    for tag in tags:
        if tag not in available:
            continue
        for ev in acc.Scalars(tag):
            results[tag].append((int(ev.step), float(ev.value)))
    return results


def format_float(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute median scalars over an iteration window from TensorBoard logs."
    )
    parser.add_argument("--logdir", default=DEFAULT_LOGDIR, help="Ray results directory")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=DEFAULT_TAGS,
        help="TensorBoard scalar tags to summarize",
    )
    parser.add_argument("--min-step", type=int, default=50)
    parser.add_argument("--max-step", type=int, default=275)
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--baseline-regex", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        raise SystemExit(f"logdir not found: {logdir}")

    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No event files found under {logdir}")

    trials: dict[Path, list[Path]] = {}
    for event_path in event_files:
        trial_dir = find_trial_dir(event_path)
        trials.setdefault(trial_dir, []).append(event_path)

    include_re = re.compile(args.include_regex) if args.include_regex else None
    baseline_re = re.compile(args.baseline_regex) if args.baseline_regex else None

    rows = []
    for trial_dir, files in sorted(trials.items(), key=lambda item: str(item[0])):
        run_name = str(trial_dir)
        try:
            run_name = str(trial_dir.relative_to(logdir))
        except ValueError:
            pass
        if include_re and not include_re.search(run_name):
            continue

        tag_samples: dict[str, list[tuple[int, float]]] = {tag: [] for tag in args.tags}
        for event_path in files:
            scalars = load_scalars(event_path, args.tags)
            for tag, samples in scalars.items():
                tag_samples[tag].extend(samples)

        for tag, samples in tag_samples.items():
            if not samples:
                continue
            values = [v for step, v in samples if args.min_step <= step <= args.max_step]
            if not values:
                continue
            steps = [step for step, _ in samples]
            rows.append(
                {
                    "run": run_name,
                    "tag": tag,
                    "median": statistics.median(values),
                    "count": len(values),
                    "min_step": min(steps),
                    "max_step": max(steps),
                }
            )

    if not rows:
        raise SystemExit("No matching scalar data found for the requested window/tags.")

    baseline_by_tag: dict[str, dict] = {}
    if baseline_re:
        baseline_candidates = [row for row in rows if baseline_re.search(row["run"])]
        for row in baseline_candidates:
            baseline_by_tag.setdefault(row["tag"], row)

    output_rows = []
    for row in rows:
        output = dict(row)
        baseline = baseline_by_tag.get(row["tag"])
        if baseline:
            delta = row["median"] - baseline["median"]
            pct = delta / baseline["median"] * 100.0 if baseline["median"] else 0.0
            output["delta_vs_baseline"] = delta
            output["pct_vs_baseline"] = pct
        output_rows.append(output)

    header = [
        "run",
        "tag",
        "median",
        "count",
        "min_step",
        "max_step",
        "delta_vs_baseline",
        "pct_vs_baseline",
    ]
    print(",".join(header))
    for row in output_rows:
        print(
            ",".join(
                [
                    row.get("run", ""),
                    row.get("tag", ""),
                    format_float(row.get("median", 0.0)),
                    str(row.get("count", "")),
                    str(row.get("min_step", "")),
                    str(row.get("max_step", "")),
                    format_float(row.get("delta_vs_baseline", 0.0)),
                    format_float(row.get("pct_vs_baseline", 0.0)),
                ]
            )
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
