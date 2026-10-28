"""Two descriptive checks of the pilots' prey, at checkpoints over the run
(README TODO; design note, related work).

1. Do prey react to a predator they see (Buason and Ziemke 2003: their
   prey evolved to circle whatever the predator did, and a prey that
   ignores predators cannot coevolve its eyes with them)? At every decision
   of a prey with a predator in its observation, the prey policy's action
   distribution is computed twice: from the observation as it is, and with
   the predator slot blanked (zeros, as when no predator is in view).
   Read-outs: the total-variation distance between the two distributions,
   and the expected velocity away from that predator (true offset, in units
   of the prey's top speed) under each. Placebo: the same for a prey with
   another prey in its first prey slot, blanking that slot instead.
2. Prey grouping (Olson et al. 2016: predators' view angle and prey
   swarming cycled together). At every decision, the prey's mean distance to
   their nearest other prey (wrapped), and the Clark-Evans ratio R, that
   mean over its expectation for the same number of prey placed uniformly
   at random on the same torus (Monte Carlo, so exact for small numbers,
   where 0.5 / sqrt(density) is not); R < 1 is grouping. Correlated over
   checkpoints with the predators' mean field-of-view gene from
   analyze_pilot's snapshots, as levels and as changes between successive
   checkpoints (levels share trends; neither is a test, one run per arm).

Episodes are played as in the final check: the checkpoint's policies
(actions sampled from the unmasked distributions), founders from its gene
pool held fixed, seeds 0, 1, ... common to all checkpoints.

Writes results/prey_behaviour_summary.json and assets/prey_behaviour.png.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.prey_behaviour [--arms tradeoff control] [--every 100] [--episodes 4] [--workers 8]
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import SELF_SIZE, SLOT_SIZE, SPEEDS, DIRECTIONS
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.geometry import direction_vector

MODULE = Path(__file__).resolve().parent
RUNS = MODULE / "runs"
PREDATOR_SLOT = SELF_SIZE  # a prey's first (nearest) predator slot
# Unit velocity of every action, (actions, 2), in units of the top speed.
ACTION_VELOCITY = np.array(
    [[direction_vector(a % DIRECTIONS, DIRECTIONS).x * SPEEDS[a // DIRECTIONS], direction_vector(a % DIRECTIONS, DIRECTIONS).y * SPEEDS[a // DIRECTIONS]] for a in range(DIRECTIONS * len(SPEEDS))]
)

RANDOM_DRAWS = 400  # random placements per prey count for R's baseline

_cache = {}
_random_nn = {}


def checkpoint_path(arm, iteration, final):
    root = RUNS / f"pilot_{arm}" / "checkpoint"
    return root if iteration == final else root / f"iter_{iteration:06d}"


def _load(path):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import (
        checkpoint_env_config,
        load_gene_pool,
        load_modules,
        use_gene_pool,
    )
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import PredPreyGrassContinuous

    if path not in _cache:
        _cache.clear()  # one checkpoint at a time per process
        env = PredPreyGrassContinuous(checkpoint_env_config(path))
        use_gene_pool(env, load_gene_pool(path))
        _cache[path] = (env, load_modules(path))
    return _cache[path]


def _probs(module, batch):
    import torch
    from ray.rllib.core.columns import Columns

    with torch.inference_mode():
        logits = module.forward_inference({Columns.OBS: torch.from_numpy(np.asarray(batch, dtype=np.float32))})[Columns.ACTION_DIST_INPUTS]
    return torch.softmax(logits, dim=-1).double().numpy()


def nearest_in_view(env, observer, animals):
    """True wrapped offset to the nearest of animals in observer's view, as
    _slots picks it (None if none)."""
    best = None
    for animal in animals:
        if animal is observer or not env.torus.in_view(observer, animal.position, observer.view_distance, observer.fov):
            continue
        dx, dy = env.torus.offset(observer.position, animal.position)
        d = math.hypot(dx, dy)
        if best is None or d < best[0]:
            best = (d, dx, dy)
    return best


def mean_nearest(pos, size):
    """Mean wrapped distance from each point to its nearest other point."""
    diff = np.abs(pos[:, None, :] - pos[None, :, :])
    diff = np.minimum(diff, size - diff)
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1).mean()


def random_nearest(n, width, height):
    """Expected mean_nearest of n points placed uniformly at random on the
    torus (Monte Carlo, fixed seed, cached)."""
    if (n, width, height) not in _random_nn:
        rng = np.random.default_rng(n)
        size = np.array([width, height], dtype=float)
        _random_nn[n, width, height] = float(np.mean([mean_nearest(rng.uniform(0, 1, (n, 2)) * size, size) for _ in range(RANDOM_DRAWS)]))
    return _random_nn[n, width, height]


def grouping(env):
    """(mean nearest-neighbour distance among prey, Clark-Evans R), or None
    with fewer than 2 prey."""
    n = len(env.prey)
    if n < 2:
        return None
    pos = np.array([[p.position.x, p.position.y] for p in env.prey])
    mean_nn = mean_nearest(pos, np.array([env.width, env.height], dtype=float))
    return mean_nn, mean_nn / random_nearest(n, env.width, env.height)


def _play(task):
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.tournament import sample_actions

    arm, iteration, final, seed = task
    torch.set_num_threads(1)
    env, modules = _load(str(checkpoint_path(arm, iteration, final)))
    prey_slot = PREDATOR_SLOT + SLOT_SIZE * env.observe_count["predator"]
    generator = torch.Generator().manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    tv, away_real, away_masked, placebo_tv = [], [], [], []
    nn, ratio = [], []
    while True:
        prey = {a.id: a for a in env.prey}
        ids = [agent for agent in obs if agent in prey]
        seen = [agent for agent in ids if obs[agent][PREDATOR_SLOT] == 1.0]
        if seen:
            real = np.stack([obs[a] for a in seen])
            masked = real.copy()
            masked[:, PREDATOR_SLOT : PREDATOR_SLOT + SLOT_SIZE] = 0.0
            p_real, p_masked = _probs(modules["prey"], real), _probs(modules["prey"], masked)
            change = 0.5 * np.abs(p_real - p_masked).sum(axis=1)
            for k, agent in enumerate(seen):
                d, dx, dy = nearest_in_view(env, prey[agent], env.predators)
                if d == 0:
                    continue  # no direction away; left out of all three read-outs
                away = -np.array([dx, dy]) / math.hypot(dx, dy)
                along = ACTION_VELOCITY @ away
                tv.append(float(change[k]))
                away_real.append(float(p_real[k] @ along))
                away_masked.append(float(p_masked[k] @ along))
        others = [agent for agent in ids if obs[agent][prey_slot] == 1.0]
        if others:
            real = np.stack([obs[a] for a in others])
            masked = real.copy()
            masked[:, prey_slot : prey_slot + SLOT_SIZE] = 0.0
            placebo_tv += (0.5 * np.abs(_probs(modules["prey"], real) - _probs(modules["prey"], masked)).sum(axis=1)).tolist()
        g = grouping(env)
        if g is not None:
            nn.append(g[0])
            ratio.append(g[1])
        obs, _, terms, truncs, _ = env.step(sample_actions(modules, obs, generator, env))
        if terms["__all__"] or truncs["__all__"]:
            break
    return dict(
        arm=arm,
        iteration=iteration,
        seed=seed,
        sums=dict(tv=sum(tv), away_real=sum(away_real), away_masked=sum(away_masked), placebo_tv=sum(placebo_tv), nn=sum(nn), ratio=sum(ratio)),
        counts=dict(seen=len(tv), placebo=len(placebo_tv), grouping=len(nn)),
    )


def summarize(rows, arms, iterations):
    out = {}
    for arm in arms:
        series = []
        for iteration in iterations[arm]:
            mine = [r for r in rows if r["arm"] == arm and r["iteration"] == iteration]
            s = {k: sum(r["sums"][k] for r in mine) for k in mine[0]["sums"]}
            c = {k: sum(r["counts"][k] for r in mine) for k in mine[0]["counts"]}
            per = lambda key, n: s[key] / c[n] if c[n] else float("nan")  # noqa: E731
            series.append(
                dict(
                    iteration=iteration,
                    seen_decisions=c["seen"],
                    tv=per("tv", "seen"),
                    placebo_tv=per("placebo_tv", "placebo"),
                    away_real=per("away_real", "seen"),
                    away_masked=per("away_masked", "seen"),
                    away_gain=per("away_real", "seen") - per("away_masked", "seen"),
                    nearest_neighbour=per("nn", "grouping"),
                    clark_evans=per("ratio", "grouping"),
                )
            )
        out[arm] = {"series": series}
        summary = MODULE / "results" / "pilot_summary.json"
        if summary.is_file():
            pilot = json.loads(summary.read_text()).get(arm)
            if pilot:
                fov = dict(zip(pilot["iterations"], pilot["species"]["predator"]["trajectory"]["fov"]))
                x = np.array([fov[p["iteration"]] for p in series if p["iteration"] in fov])
                y = np.array([p["clark_evans"] for p in series if p["iteration"] in fov])
                out[arm]["r_predator_fov_vs_clark_evans"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None
                out[arm]["r_changes"] = float(np.corrcoef(np.diff(x), np.diff(y))[0, 1]) if len(x) > 3 else None
    return out


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for arm, color in zip(summary, ("#c0392b", "#2471a3", "#7d3c98", "#1e8449")):
        s = summary[arm]["series"]
        x = [p["iteration"] for p in s]
        axes[0].plot(x, [p["tv"] for p in s], color=color, label=f"{arm}: predator blanked")
        axes[0].plot(x, [p["placebo_tv"] for p in s], color=color, ls=":", label=f"{arm}: other prey blanked")
        axes[1].plot(x, [p["away_real"] for p in s], color=color, label=f"{arm}: as seen")
        axes[1].plot(x, [p["away_masked"] for p in s], color=color, ls=":", label=f"{arm}: predator blanked")
        axes[2].plot(x, [p["clark_evans"] for p in s], color=color, label=arm)
    axes[0].set_ylabel("change in prey action\n(total variation)")
    axes[1].set_ylabel("expected velocity away\nfrom predator (x top speed)")
    axes[2].set_ylabel("prey Clark-Evans R\n(<1: grouped)")
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[2].axhline(1, color="grey", lw=0.6)
    axes[2].set_xlabel("checkpoint iteration")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["tradeoff", "control"])
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="prey_behaviour")
    args = parser.parse_args()
    iterations, tasks = {}, []
    for arm in args.arms:
        settings = json.loads((RUNS / f"pilot_{arm}" / "source_code" / "settings.json").read_text())
        final = settings["config_ppo"]["iterations"]
        iterations[arm] = list(range(args.every, final + 1, args.every))
        tasks += [(arm, i, final, seed) for i in iterations[arm] for seed in range(args.episodes)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(_play, tasks, chunksize=args.episodes))
    summary = summarize(rows, args.arms, iterations)
    (MODULE / "results" / f"{args.out}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    plot(summary, MODULE / "assets" / f"{args.out}.png")
    for arm in args.arms:
        print(arm, "r(predator fov gene, Clark-Evans R) =", summary[arm].get("r_predator_fov_vs_clark_evans"), "changes:", summary[arm].get("r_changes"))
        for p in summary[arm]["series"][:: max(1, len(summary[arm]["series"]) // 10)] + summary[arm]["series"][-1:]:
            print(
                f"  iter {p['iteration']:5d}  seen {p['seen_decisions']:6d}  TV {p['tv']:.3f} (placebo {p['placebo_tv']:.3f})"
                f"  away {p['away_real']:+.3f} vs blanked {p['away_masked']:+.3f}  NN {p['nearest_neighbour']:.1f}  R {p['clark_evans']:.2f}"
            )


if __name__ == "__main__":
    main()
