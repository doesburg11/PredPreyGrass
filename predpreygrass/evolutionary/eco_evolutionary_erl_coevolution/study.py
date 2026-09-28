"""Batch runner + analysis for eco_evolutionary_erl_coevolution.

One tool for both the step-0 regression check and step-1 calibration/
comparison: runs (strategy x seed) jobs in parallel worker processes, one
JSON line per finished run in `<out-dir>/results.jsonl` (resume-safe: a
rerun skips every (tag, strategy, seed) already recorded), plus a per-run
population time series in `<out-dir>/timeseries/`.

Usage:
    # step 0 regression check: 20 seeds x 5 strategies, 50k-step budget
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.study run \\
        --preset step0 --seeds 1-20 --steps 50000 --workers 20 --out-dir ~/simulation_results/erl_results/coevo_step0

    # ...then compare against the erl_baldwin §9 logs, run-for-run
    python -m ...study analyze --out-dir ~/simulation_results/erl_results/coevo_step0 --compare-study

    # step 1 calibration: one tag per parameter setting, ERL only
    python -m ...study run --preset step1 --strategies ERL --seeds 1-8 --steps 30000 \\
        --tag thr14 --set carnivore_reproduction_energy_threshold=14 --out-dir .../coevo_step1_sweep

`end_reason` per run: "agent_extinct", "carnivore_extinct" (only possible
when immigration is off; the run then stops if `end_on_carnivore_extinction`
is set), or "budget". Survival time (§9's metric) is `agent_extinction_step`
or the budget; coexistence time (step 1's metric) is `end_step`.
"""

import argparse
import fcntl
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world import STRATEGIES

STUDY_LOGS = Path.home() / "simulation_results" / "erl_results" / "erl_full_study_logs"
STUDY_BUDGET = 1_000_000


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return list(dict.fromkeys(seeds))  # dedupe, keep order


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def run_one(job: dict) -> dict:
    """Run a single world to extinction or budget. Top-level so worker processes can pickle it."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import PRESETS
    from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world import ErlWorld

    cfg = {**PRESETS[job["preset"]], **job["overrides"], "strategy": job["strategy"], "seed": job["seed"]}
    for key in job["overrides"]:
        if key not in PRESETS[job["preset"]]:
            raise KeyError(f"--set {key}: not a config key")
    world = ErlWorld(cfg, np.random.default_rng(cfg["seed"]))
    end_on_carnivore = cfg.get("end_on_carnivore_extinction", False)
    sample_every = job["sample_every"]

    start = time.time()
    agent_ext = carnivore_ext = None
    series = []
    while world.current_step < job["steps"]:
        world.step()
        step = world.current_step
        counts = world.population_counts()
        if carnivore_ext is None and counts["carnivore"] == 0 and not world.immigration_active():
            carnivore_ext = step  # permanent: no immigration left to bring them back
        if step % sample_every == 0:
            series.append((step, counts["agent"], counts["carnivore"]))
        if counts["agent"] == 0:
            agent_ext = step
            break
        if carnivore_ext is not None and end_on_carnivore:
            break
    if not series or series[-1][0] != world.current_step:
        final = world.population_counts()
        series.append((world.current_step, final["agent"], final["carnivore"]))

    if agent_ext is not None:
        end_reason = "agent_extinct"
    elif carnivore_ext is not None and end_on_carnivore:
        end_reason = "carnivore_extinct"
    else:
        end_reason = "budget"

    ts_dir = Path(job["out_dir"]) / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    with open(ts_dir / f"{job['tag']}_{job['strategy']}_seed{job['seed']}.csv", "w") as f:
        f.write("step,agent_count,carnivore_count\n")
        f.writelines(f"{s},{a},{c}\n" for s, a, c in series)

    return {
        "tag": job["tag"],
        "preset": job["preset"],
        "overrides": job["overrides"],
        "strategy": job["strategy"],
        "seed": job["seed"],
        "budget": job["steps"],
        "end_step": world.current_step,
        "end_reason": end_reason,
        "agent_extinction_step": agent_ext,
        "carnivore_extinction_step": carnivore_ext,
        "final_agents": world.population_counts()["agent"],
        "final_carnivores": world.population_counts()["carnivore"],
        # Over the sampled series including the terminal row (so never empty).
        "mean_agents": float(np.mean([a for _, a, _ in series])),
        "mean_carnivores": float(np.mean([c for _, _, c in series])),
        "deaths": {species: dict(counter) for species, counter in world.deaths.items()},
        "carnivore_births": world.carnivore_births,
        "carnivore_immigrants": world.carnivore_immigrants,
        "wall_seconds": round(time.time() - start, 1),
    }


def run_identity(r: dict) -> tuple:
    """Everything that defines a tag's experiment besides strategy/seed. All
    rows under one tag must share it -- resume and analysis both check this."""
    return (r["preset"], json.dumps(r["overrides"], sort_keys=True), r["budget"])


def append_result(path: Path, result: dict):
    """One whole line per call under an exclusive lock, so several concurrent
    `study run` processes (e.g. one per sweep tag) can share results.jsonl."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(result) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_results(out_dir: Path) -> list[dict]:
    path = out_dir / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_run(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from multiprocessing import Pool

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.set)
    existing = [r for r in load_results(out_dir) if r["tag"] == args.tag]
    identity = (args.preset, json.dumps(overrides, sort_keys=True), args.steps)
    if existing and run_identity(existing[0]) != identity:
        raise SystemExit(
            f"tag {args.tag!r} already has results with a different setup "
            f"{run_identity(existing[0])} vs. now {identity} -- use a new --tag."
        )
    done = {(r["tag"], r["strategy"], r["seed"]) for r in existing}
    jobs = [
        {
            "tag": args.tag, "preset": args.preset, "overrides": overrides,
            "strategy": strategy, "seed": seed, "steps": args.steps,
            "sample_every": args.sample_every, "out_dir": str(out_dir),
        }
        for strategy in args.strategies.split(",")
        for seed in parse_seeds(args.seeds)
        if (args.tag, strategy, seed) not in done
    ]
    print(f"{len(jobs)} jobs to run ({len(done)} already done) -> {out_dir}", flush=True)
    if not jobs:
        return
    # Longest-expected first (ERL survives longest) so the tail isn't one straggler.
    order = {s: i for i, s in enumerate(["ERL", "L", "E", "B", "F"])}
    jobs.sort(key=lambda j: order.get(j["strategy"], 9))
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_one, jobs), 1):
            append_result(out_dir / "results.jsonl", result)
            print(
                f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {result['tag']} {result['strategy']} "
                f"seed {result['seed']}: {result['end_reason']} at {result['end_step']} "
                f"(agents {result['final_agents']}, carnivores {result['final_carnivores']}, "
                f"{result['wall_seconds']}s)",
                flush=True,
            )


def study_survival(strategy: str, seed: int) -> int | None:
    """Agent extinction step from erl_baldwin's §9 log, or STUDY_BUDGET if it survived."""
    path = STUDY_LOGS / f"{strategy}_seed{seed}.log"
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"extinction at step (\d+)", text)
    if m:
        return int(m.group(1))
    return STUDY_BUDGET if "Reached step limit" in text else None


def cmd_analyze(args):
    from scipy.stats import mannwhitneyu

    out_dir = Path(args.out_dir).expanduser()
    results = load_results(out_dir)
    if not results:
        print("no results")
        return
    for tag in sorted({r["tag"] for r in results}):
        rows = [r for r in results if r["tag"] == tag]
        if len({run_identity(r) for r in rows}) > 1:
            print(f"\n=== tag {tag!r}: SKIPPED -- rows with different setups: {sorted({run_identity(r) for r in rows})}")
            continue
        budget = rows[0]["budget"]
        print(f"\n=== tag {tag!r}  preset={rows[0]['preset']}  overrides={rows[0]['overrides']}  budget={budget:,}")
        print(f"{'strat':5s} {'n':>3s} {'median':>9s} {'mean':>9s} {'full':>5s}  {'agent-ext':>9s} {'carn-ext':>8s}"
              f" {'mean_agents':>11s} {'mean_carn':>9s}  deaths (agent: carnivore/starve/other)")
        by_strategy = {}
        for strategy in STRATEGIES:
            rs = [r for r in rows if r["strategy"] == strategy]
            if not rs:
                continue
            ends = np.array([r["end_step"] for r in rs])
            by_strategy[strategy] = ends
            full = np.mean([r["end_reason"] == "budget" for r in rs])
            n_agent_ext = sum(r["end_reason"] == "agent_extinct" for r in rs)
            n_carn_ext = sum(r["carnivore_extinction_step"] is not None for r in rs)
            d = {k: sum(r["deaths"]["agent"].get(k, 0) for r in rs) for k in ("carnivore", "starvation")}
            total = sum(sum(r["deaths"]["agent"].values()) for r in rs)
            other = total - d["carnivore"] - d["starvation"]
            share = (lambda x: f"{100 * x / total:.0f}%") if total else (lambda x: "-")
            mean_a = np.mean([r["mean_agents"] for r in rs if r["mean_agents"] is not None] or [np.nan])
            mean_c = np.mean([r["mean_carnivores"] for r in rs if r["mean_carnivores"] is not None] or [np.nan])
            print(f"{strategy:5s} {len(rs):3d} {np.median(ends):9,.0f} {ends.mean():9,.0f} {full:5.0%}  "
                  f"{n_agent_ext:9d} {n_carn_ext:8d} {mean_a:11.1f} {mean_c:9.1f}  "
                  f"{share(d['carnivore'])}/{share(d['starvation'])}/{share(other)}")
        # Runs that reach the budget are right-censored; with one common budget
        # per tag they are ties at `budget`, i.e. this tests the ordinal endpoint
        # "time to end, capped at budget" (the same test erl_baldwin §9 used).
        # `end_step` is agent survival under step 0 and coexistence time (either
        # species' extinction) under step 1.
        if "ERL" in by_strategy and len(by_strategy) > 1:
            for strategy, ends in by_strategy.items():
                if strategy == "ERL":
                    continue
                p = mannwhitneyu(by_strategy["ERL"], ends, alternative="two-sided").pvalue
                print(f"  ERL vs {strategy}: Mann-Whitney p={p:.2g}")

        if args.compare_study:
            # Step-0 regression check: every run must match §9 exactly, truncated to this budget.
            mismatches, compared = [], 0
            for r in rows:
                expected = study_survival(r["strategy"], r["seed"])
                if expected is None:
                    continue
                compared += 1
                got = r["agent_extinction_step"] if r["agent_extinction_step"] is not None else budget
                if min(expected, budget) != got:
                    mismatches.append((r["strategy"], r["seed"], min(expected, budget), got))
            print(f"  vs. erl_baldwin §9 logs: {compared - len(mismatches)}/{compared} runs match exactly")
            for m in mismatches:
                print(f"    MISMATCH {m[0]} seed {m[1]}: §9={m[2]} now={m[3]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--preset", choices=["step0", "step1"], default="step0")
    run.add_argument("--strategies", default=",".join(STRATEGIES))
    run.add_argument("--seeds", default="1-20", help="e.g. 1-20 or 1,5,9")
    run.add_argument("--steps", type=int, default=50_000, help="Step budget per run.")
    run.add_argument("--workers", type=int, default=20)
    run.add_argument("--sample-every", type=int, default=500, help="Population time-series resolution.")
    run.add_argument("--tag", default="default", help="Label for this parameter setting (sweeps).")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="Config override on top of the preset; repeatable.")
    run.add_argument("--out-dir", required=True)
    run.set_defaults(func=cmd_run)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--out-dir", required=True)
    analyze.add_argument("--compare-study", action="store_true",
                         help="Check every run against erl_baldwin's §9 logs (step-0 regression check).")
    analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
