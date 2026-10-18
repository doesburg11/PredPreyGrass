"""Cross-play tournament: every predator checkpoint against every prey checkpoint.

Evidence for co-adaptation (an arms race between the species) is in the
pattern of the catch-risk matrix (rows: predator checkpoints, columns: prey
checkpoints), not in either species' own training curve. Co-adaptation needs
both of these, each measured against a fixed opponent:

- prey improve: along each row, newer prey have a lower catch risk;
- predators improve: down each column, newer predators cause a higher one.

Either alone is one-sided learning. If both hold while the diagonal (each
checkpoint against its own training partner) stays roughly level, each side
keeps up with the other, the Red Queen pattern.

Both policies come from the same training run; the environment settings are
read from that run's checkpoint, so they match what the policies learned in.
Episode k uses the same seed in every matchup (common random numbers), so
matchups differ by policy, not by starting layout. Actions are sampled like
in training.

Usage (from foraging_swarms/):

    python tournament.py runs/<run>
    python tournament.py runs/<run> --every 100 --episodes 20
    python tournament.py runs/<run> --iterations 50,200,380
    python tournament.py runs/<run> --prey-iteration 290  # fixed prey
    python tournament.py runs/<run1> runs/<run2> --every 300  # continued runs

Several run directories are treated as one continuous training, for runs
started from the previous run's final checkpoint (init_checkpoint): their
checkpoints get global iteration numbers that continue where the previous
run ended, and all runs must share the same environment settings.

--prey-iteration plays every predator checkpoint against one prey checkpoint,
e.g. for a run trained against frozen prey (where all prey checkpoints are
the same policy), and prints one row per predator checkpoint.

Writes one CSV row per episode (default: <run>/tournament.csv) and prints
the summary matrices.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.checkpoints import checkpoint_env_config, load_module

SPECIES = ("predator", "prey")
CATCH_RISK = "prey caught per prey per 1000 physics steps"
OUTCOMES = ("time_limit", "prey_extinct", "predators_extinct", "both_extinct")
FIELDS = (
    "predator_iter",
    "prey_iter",
    "episode",
    "seed",
    "physics_steps",
    "decisions",
    "outcome",
    "prey_caught",
    "predator_births",
    "prey_births",
    "predators_starved",
    "prey_starved",
    "mean_predators",
    "mean_prey",
    "prey_exposure",
)


def checkpoint_iterations(run_dir):
    """The iteration numbers of a run's checkpoint/iter_* directories."""
    found = []
    for path in (Path(run_dir) / "checkpoint").glob("iter_*"):
        try:
            found.append(int(path.name.removeprefix("iter_")))
        except ValueError:
            continue
    return sorted(found)


def combined_checkpoints(run_dirs):
    """{global iteration: (run directory, its own iteration)} for runs that
    continue one another. A run's length is its last checkpoint plus the
    checkpoint spacing (990 + 10 = 1000), so the next run's iteration 10 is
    global iteration 1010."""
    combined = {}
    offset = 0
    for run_dir in run_dirs:
        iterations = checkpoint_iterations(run_dir)
        if not iterations:
            raise ValueError(f"{run_dir} has no checkpoints")
        for iteration in iterations:
            combined[offset + iteration] = (str(run_dir), iteration)
        spacing = iterations[-1] - iterations[-2] if len(iterations) > 1 else 0
        offset += iterations[-1] + spacing
    return combined


def select_iterations(available, every=None, iterations=None):
    """The requested iterations (all must exist), or every `every`-th one,
    always including the newest checkpoint."""
    if iterations:
        missing = sorted(set(iterations) - set(available))
        if missing:
            raise ValueError(f"No checkpoint for iterations {missing}")
        return sorted(set(iterations))
    if not available:
        raise ValueError("The run has no checkpoints")
    chosen = [i for i in available if i % every == 0] if every else list(available)
    if available[-1] not in chosen:
        chosen.append(available[-1])
    return chosen


def checkpoint_dir(run_dir, iteration):
    return Path(run_dir) / "checkpoint" / f"iter_{iteration:06d}"


def run_env_config(run_dir, iteration):
    """The env_config a checkpoint was trained with, read without starting
    Ray."""
    return checkpoint_env_config(checkpoint_dir(run_dir, iteration))


# Per-process state for the worker pool.
_worker = {}


def _init_worker(locations, env_config):
    import torch

    torch.set_num_threads(1)
    _worker.update(locations=locations, env_config=env_config, modules={}, env=None)


def _module(species, iteration):
    key = (species, iteration)
    if key not in _worker["modules"]:
        run_dir, own_iteration = _worker["locations"][iteration]
        _worker["modules"][key] = load_module(checkpoint_dir(run_dir, own_iteration), species)
    return _worker["modules"][key]


def _env():
    from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import PredPreyGrassContinuous

    if _worker["env"] is None:
        _worker["env"] = PredPreyGrassContinuous(dict(_worker["env_config"]))
    return _worker["env"]


def sample_actions(modules, obs, generator, raw_env=None):
    """Sample every agent's action, one batched forward pass per species. A
    species' entry can also be a scripted policy with an act(raw_env, agents)
    method returning {agent: action}."""
    import torch
    from ray.rllib.core.columns import Columns

    actions = {}
    for species in SPECIES:
        agents = [agent for agent in obs if agent.startswith(species)]
        if not agents:
            continue
        if hasattr(modules[species], "act"):
            actions.update(modules[species].act(raw_env, agents))
            continue
        batch = torch.from_numpy(np.stack([np.asarray(obs[agent], dtype=np.float32) for agent in agents]))
        with torch.inference_mode():
            logits = modules[species].forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        sampled = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
        actions.update(zip(agents, sampled.squeeze(1).tolist()))
    return actions


def play_episode(env, modules, seed):
    """Play one episode; returns its outcome and counts."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    raw = env
    obs, _ = env.reset(seed=seed)
    counts: dict[str, int] = dict.fromkeys(
        (
            "prey_caught",
            "predator_births",
            "prey_births",
            "predators_starved",
            "prey_starved",
        ),
        0,
    )
    # Animal-steps alive, per species: a decision's length in physics steps
    # varies (a birth ends it early), so weight each decision by its length.
    exposure = {"predator": 0.0, "prey": 0.0}
    decisions = 0
    while True:
        before = (raw.time_step, len(raw.predators), len(raw.prey))
        obs, _, terms, truncs, infos = env.step(sample_actions(modules, obs, generator, raw))
        decisions += 1
        steps = raw.time_step - before[0]
        # Mean of the counts before and after: animals born or killed during
        # a decision count for about half of it.
        exposure["predator"] += steps * (before[1] + len(raw.predators)) / 2
        exposure["prey"] += steps * (before[2] + len(raw.prey)) / 2
        for agent, info in infos.items():
            species = "predator" if agent.startswith("predator") else "prey"
            if info.get("born"):
                counts[f"{species}_births"] += 1
            if info.get("starved"):
                counts["predators_starved" if species == "predator" else "prey_starved"] += 1
            elif species == "prey" and terms.get(agent):
                counts["prey_caught"] += 1
        if terms["__all__"] or truncs["__all__"]:
            break
    steps = max(raw.time_step, 1)
    return dict(
        physics_steps=raw.time_step,
        decisions=decisions,
        outcome=classify(len(raw.predators), len(raw.prey)),
        mean_predators=round(exposure["predator"] / steps, 2),
        mean_prey=round(exposure["prey"] / steps, 2),
        prey_exposure=exposure["prey"],
        **counts,
    )


def classify(predators, prey):
    """How an episode ended, from the survivors (energy can starve the last
    animals of both species in the same step)."""
    if not predators and not prey:
        return "both_extinct"
    if not prey:
        return "prey_extinct"
    if not predators:
        return "predators_extinct"
    return "time_limit"


def _play(task):
    predator_iter, prey_iter, episode, seed = task
    modules = {
        "predator": _module("predator", predator_iter),
        "prey": _module("prey", prey_iter),
    }
    result = play_episode(_env(), modules, seed)
    return dict(
        predator_iter=predator_iter,
        prey_iter=prey_iter,
        episode=episode,
        seed=seed,
        **result,
    )


def summarize(rows, iterations):
    """Matrices indexed [predator checkpoint, prey checkpoint]; NaN where a
    matchup has no episodes."""
    index = {iteration: i for i, iteration in enumerate(iterations)}
    n = len(iterations)
    sums = {name: np.zeros((n, n)) for name in ("episodes", "steps", "caught", "prey_exposure", *OUTCOMES)}
    for row in rows:
        i, j = index[row["predator_iter"]], index[row["prey_iter"]]
        sums["episodes"][i, j] += 1
        sums["steps"][i, j] += row["physics_steps"]
        sums["caught"][i, j] += row["prey_caught"]
        sums["prey_exposure"][i, j] += row["prey_exposure"]
        sums[row["outcome"]][i, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        episodes = np.where(sums["episodes"] > 0, sums["episodes"], np.nan)
        matrices = {
            "episode length (physics steps)": sums["steps"] / episodes,
            # Per prey alive, so it measures how catchable prey are, not how
            # many there happen to be.
            CATCH_RISK: 1000 * sums["caught"] / sums["prey_exposure"],
        }
        for outcome in OUTCOMES[1:]:
            matrices[f"{outcome.replace('_', ' ')} (fraction)"] = sums[outcome] / episodes
    matrices[CATCH_RISK][np.isnan(episodes)] = np.nan
    return matrices


def adaptation_trends(risk):
    """Each species' improvement against fixed opponents, from the catch-risk
    matrix [predator checkpoint, prey checkpoint]:

    - prey: the share of same-row pairs (same predator) in which the newer
      prey has the lower catch risk;
    - predator: the share of same-column pairs (same prey) in which the newer
      predator causes the higher catch risk.

    0.5 means no trend, 1.0 a consistent improvement. Pairs with a missing
    or tied value are skipped. Returns None with fewer than 2 checkpoints."""
    n = risk.shape[0]
    if n < 2:
        return None

    def share(pairs):
        wins = [a > b for a, b in pairs if np.isfinite(a) and np.isfinite(b) and a != b]
        return float(np.mean(wins)) if wins else float("nan")

    older_newer = [(k, m) for k in range(n) for m in range(k + 1, n)]
    prey = share((risk[i, k], risk[i, m]) for i in range(n) for k, m in older_newer)
    predator = share((risk[m, j], risk[k, j]) for j in range(n) for k, m in older_newer)
    return {"prey": prey, "predator": predator}


def format_matrix(name, matrix, iterations):
    width = max(8, len(str(iterations[-1])) + 2)
    decimals = 0 if "length" in name else 2
    lines = [name, " pred \\ prey" + "".join(f"{i:>{width}}" for i in iterations)]
    for iteration, row in zip(iterations, matrix):
        cells = "".join(f"{value:>{width}.{decimals}f}" for value in row)
        lines.append(f" {iteration:>10}" + cells)
    return "\n".join(lines)


def predator_table(rows, iterations):
    """Per predator checkpoint, pooled over its episodes: catch risk per prey
    and catches per predator (per 1000 physics steps), the share of episodes
    that reach the time limit, and mean episode length and populations."""
    table = []
    for iteration in iterations:
        episodes = [row for row in rows if row["predator_iter"] == iteration]
        if not episodes:
            continue
        caught = sum(row["prey_caught"] for row in episodes)
        steps = [row["physics_steps"] for row in episodes]
        predator_exposure = sum(row["mean_predators"] * row["physics_steps"] for row in episodes)
        table.append(
            {
                "predator_iter": iteration,
                "episodes": len(episodes),
                "catch_risk": 1000 * caught / sum(r["prey_exposure"] for r in episodes),
                "catches_per_predator": 1000 * caught / max(predator_exposure, 1),
                "coexist": float(np.mean([row["outcome"] == "time_limit" for row in episodes])),
                "length": float(np.mean(steps)),
                "mean_predators": float(np.mean([r["mean_predators"] for r in episodes])),
                "mean_prey": float(np.mean([r["mean_prey"] for r in episodes])),
            }
        )
    return table


def format_predator_table(rows, iterations):
    lines = [
        (" predator  episodes  catch risk  catches/predator  coexist  " "length  predators  prey"),
        "                     (per prey and 1000 physics steps)",
    ]
    for t in predator_table(rows, iterations):
        lines.append(
            f" {t['predator_iter']:>8}  {t['episodes']:>8}  {t['catch_risk']:>10.2f}"
            f"  {t['catches_per_predator']:>16.2f}  {t['coexist']:>7.0%}"
            f"  {t['length']:>6.0f}  {t['mean_predators']:>9.1f}"
            f"  {t['mean_prey']:>4.1f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "runs",
        nargs="+",
        help="Training run directory (contains checkpoint/); several directories "
        "for runs that continue one another, oldest first",
    )
    parser.add_argument("--every", type=int, default=50, help="Checkpoint spacing")
    parser.add_argument(
        "--iterations",
        default=None,
        help="Comma-separated checkpoint iterations (overrides --every)",
    )
    parser.add_argument(
        "--prey-iteration",
        type=int,
        default=None,
        help="Play every predator checkpoint against only this prey checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Per matchup")
    parser.add_argument("--seed", type=int, default=0, help="Seed of episode 0")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument(
        "--out",
        default=None,
        help="CSV (default: tournament.csv in the (last) run directory)",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.every < 1:
        raise SystemExit("--episodes, --workers and --every must be positive")

    run_dirs = [Path(run).resolve() for run in args.runs]
    locations = combined_checkpoints(run_dirs)
    available = sorted(locations)
    requested = [int(i) for i in args.iterations.split(",")] if args.iterations else None
    iterations = select_iterations(available, every=args.every, iterations=requested)
    env_config = run_env_config(*locations[iterations[-1]])
    for run_dir in run_dirs[:-1]:
        other = run_env_config(run_dir, checkpoint_iterations(run_dir)[-1])
        differing = sorted(key for key in set(other) | set(env_config) if other.get(key) != env_config.get(key))
        if differing:
            raise SystemExit(f"{run_dir.name} has other env settings: {differing}")
    prey_iterations = iterations
    if args.prey_iteration is not None:
        prey_iterations = select_iterations(available, iterations=[args.prey_iteration])
    tasks = [
        (predator_iter, prey_iter, episode, args.seed + episode)
        for predator_iter in iterations
        for prey_iter in prey_iterations
        for episode in range(args.episodes)
    ]
    out = Path(args.out) if args.out else run_dirs[-1] / "tournament.csv"
    print(
        f"{len(iterations)} checkpoints {iterations}: "
        f"{len(iterations) * len(prey_iterations)} "
        f"matchups x {args.episodes} episodes = {len(tasks)} episodes "
        f"on {args.workers} workers",
        flush=True,
    )

    rows = []
    start = time.time()
    with (
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(locations, env_config),
        ) as pool,
        open(out, "w", encoding="utf-8") as csv,
    ):
        csv.write(",".join(FIELDS) + "\n")
        for done, row in enumerate(pool.map(_play, tasks, chunksize=1), 1):
            rows.append(row)
            csv.write(",".join(str(row[field]) for field in FIELDS) + "\n")
            csv.flush()
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                elapsed = time.time() - start
                left = elapsed / done * (len(tasks) - done)
                print(
                    f"{done}/{len(tasks)} episodes, {elapsed / 60:.1f} min, " f"~{left / 60:.1f} min left",
                    flush=True,
                )

    print(f"\nWrote {out}\nRows: predator checkpoint; columns: prey checkpoint.\n")
    if args.prey_iteration is not None:
        print(f"Every predator checkpoint against prey {args.prey_iteration}:\n")
        print(format_predator_table(rows, iterations))
        return
    matrices = summarize(rows, iterations)
    for name, matrix in matrices.items():
        print(format_matrix(name, matrix, iterations) + "\n")
    trends = adaptation_trends(matrices[CATCH_RISK])
    if trends is not None:
        print(
            "Against a fixed opponent (0.5 = no trend, 1.0 = always):\n"
            f"  newer prey are harder to catch in {trends['prey']:.0%} of pairs\n"
            f"  newer predators catch more in {trends['predator']:.0%} of pairs\n"
            "Co-adaptation needs both clearly above 50%. Diagonal catch risk "
            "(each checkpoint against its training partner): "
            + ", ".join(f"{value:.2f}" for value in np.diag(matrices[CATCH_RISK]))
        )


if __name__ == "__main__":
    main()
