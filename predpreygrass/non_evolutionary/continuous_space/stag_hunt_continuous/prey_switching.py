"""Do hunters leave scarce rabbits alone? Prey switching in seeded evaluation
episodes, played as in evaluate.py (same seeds, same action sampling, whole
episodes). Descriptive: the policies are fixed, and rabbit numbers are not
set by the experimenter, so low-rabbit decisions differ in other ways too
(when in the episode they fall, which rabbits are left).

Three measures, per decision, by the number of rabbits at its start:

- catch rate: rabbits caught per rabbit x hunter x physics step, counted at
  each physics step (mass-action availability, not observed encounters).
  A rise with rabbit numbers would be the pattern of scarce rabbits being
  safer, a fall that of saturating hunters; rabbit numbers go with episode
  stage, hunter condition and clustering, so this does not identify a
  functional response (Holling type).
- pursuit: the net displacement over the decision of a hunter that lived
  through it, along the direction to the nearest rabbit in view at its
  start (that rabbit's position then), as a share of its top speed (-1 to
  1). Directional movement, not intent: a mammoth or another rabbit can lie
  the same way.
- catches per sighting: all rabbits caught in the decision per hunter with
  a rabbit in view at its start. A population ratio: catches are not linked
  to the hunters or rabbits of the sightings.

And the diet: for the prey taken while both species live, the slope b of
log(rabbits caught / mammoths killed) on log(rabbits / mammoths) (Murdoch
1969, as a logistic regression on the captures). b = 1 is proportional to a
fixed preference, b > 1 more of the commoner prey, b < 1 the reverse. It
describes the resulting diet: catches and group kills are different
processes, so it is not an individual's choice between prey.

Intervals resample whole episodes; compare checkpoints by pairing seeds, not
by these separate intervals.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.prey_switching \\
        <checkpoint dir | untrained> <episodes> [out.json] [--workers N]
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import PARTY_SIZES

RABBIT_BINS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 70), (71, None))
BOOTSTRAPS = 2000
COUNTERS = ("decisions", "physics_steps", "rabbit_hunter_steps", "catches", "kills", "sightings", "pursuit", "sightings_at_start")


def rabbit_bin(rabbits):
    """Index of the rabbit-count bin; None without rabbits."""
    for i, (low, high) in enumerate(RABBIT_BINS):
        if rabbits >= low and (high is None or rabbits <= high):
            return i
    return None


def bin_label(i):
    low, high = RABBIT_BINS[i]
    return f"{low}+" if high is None else f"{low}-{high}"


def rabbits_visible_to_hunters(env):
    """The rabbits, without those hidden inside burrows (burrow_hides_rabbits)."""
    if env.burrow_count and env.burrow_hides:
        return [rabbit for rabbit, inside in zip(env.rabbits, env._inside(env.rabbits)) if not inside]
    return env.rabbits


def nearest_in_view(env, hunter, animals):
    """The nearest animal in the hunter's view, or None."""
    best, best_distance = None, math.inf
    for animal in animals:
        if env.torus.in_view(hunter, animal.position, hunter.view_distance, hunter.fov):
            distance = env.torus.distance(hunter.position, animal.position)
            if distance < best_distance:
                best, best_distance = animal, distance
    return best


def pursuit_along(env, before, after, target, speed, steps):
    """Displacement from before to after along the direction to target, as
    a share of the distance the hunter could cover in steps."""
    dx, dy = env.torus.offset(before, target)
    length = math.hypot(dx, dy)
    if not length or not steps:
        return 0.0
    mx, my = env.torus.offset(before, after)
    return (mx * dx + my * dy) / length / (speed * steps)


def new_bins():
    return [{key: 0.0 for key in COUNTERS} for _ in RABBIT_BINS]


def play_episode(env, choose, seed):
    """Plays an episode with choose(obs) -> actions. Returns the bin counters
    and, per decision with prey taken while both prey live, (rabbits,
    mammoths, caught, killed)."""
    bins = new_bins()
    captures = []
    exposure = 0
    physics_step = env._physics_step

    def counting(moves):
        nonlocal exposure
        exposure += len(env.rabbits) * len(env.predators)
        return physics_step(moves)

    env._physics_step = counting
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            rabbits, mammoths = len(env.rabbits), len(env.mammoths)
            index = rabbit_bin(rabbits)
            sightings = []
            if index is not None:
                visible = rabbits_visible_to_hunters(env)
                for hunter in env.predators:
                    target = nearest_in_view(env, hunter, visible)
                    if target is not None:
                        sightings.append((hunter.id, hunter.position.copy(), target.position.copy(), hunter.max_speed))
            time_before = env.time_step
            exposure = 0
            stats = env.hunt_stats
            caught_before = stats["rabbits_caught"]
            kills_before = sum(stats[f"kills_party_{size}"] for size in range(1, PARTY_SIZES + 1))
            obs, _, terms, truncs, _ = env.step(choose({a: obs[a] for a in sorted(obs) if a in env.active}))
            steps = env.time_step - time_before
            caught = stats["rabbits_caught"] - caught_before
            killed = sum(stats[f"kills_party_{size}"] for size in range(1, PARTY_SIZES + 1)) - kills_before
            if index is not None:
                row = bins[index]
                row["decisions"] += 1
                row["physics_steps"] += steps
                row["rabbit_hunter_steps"] += exposure
                row["catches"] += caught
                row["kills"] += killed
                alive = {hunter.id: hunter for hunter in env.predators}
                row["sightings_at_start"] += len(sightings)
                for hunter_id, before, target, speed in sightings:
                    if hunter_id in alive:
                        row["sightings"] += 1
                        row["pursuit"] += pursuit_along(env, before, alive[hunter_id].position, target, speed, steps)
            if rabbits and mammoths and (caught or killed):
                captures.append((rabbits, mammoths, caught, killed))
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        del env._physics_step
    return {"seed": seed, "outcome": evaluate.outcome(env), "bins": bins, "captures": captures, "hunt_stats": env.episode_hunt_stats}


def diet_slope(captures):
    """Intercept and slope of logit(share of rabbits among captures) on
    log(rabbits / mammoths), fitted by Newton's method; (nan, nan) when the
    captures do not identify them or the fit does not converge (separation)."""
    data = np.asarray(captures, dtype=float).reshape(-1, 4)
    if not len(data) or not data[:, 2].sum() or not data[:, 3].sum():
        return math.nan, math.nan
    x = np.log(data[:, 0] / data[:, 1])
    if np.ptp(x) == 0:
        return math.nan, math.nan
    total = data[:, 2] + data[:, 3]
    design = np.column_stack((np.ones_like(x), x))
    beta = np.zeros(2)
    for _ in range(50):
        p = np.clip(1 / (1 + np.exp(-np.clip(design @ beta, -30, 30))), 1e-9, 1 - 1e-9)
        hessian = (design * (total * p * (1 - p))[:, None]).T @ design
        try:
            change = np.linalg.solve(hessian, design.T @ (data[:, 2] - total * p))
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        beta = beta + change
        if np.abs(change).max() < 1e-8:
            return (float(beta[0]), float(beta[1])) if np.abs(beta).max() < 20 else (math.nan, math.nan)
    return math.nan, math.nan


def bin_rates(row):
    """The rates of one bin's counters; None where there is no denominator."""
    return {
        "catch_rate_x1e5": 1e5 * row["catches"] / row["rabbit_hunter_steps"] if row["rabbit_hunter_steps"] else None,
        "pursuit": row["pursuit"] / row["sightings"] if row["sightings"] else None,
        "catches_per_sighting": row["catches"] / row["sightings_at_start"] if row["sightings_at_start"] else None,
    }


def pooled(rows, weights=None):
    """The bin counters over episodes, each counted weights[i] times."""
    bins = new_bins()
    for i, row in enumerate(rows):
        weight = 1 if weights is None else weights[i]
        if weight:
            for into, values in zip(bins, row["bins"]):
                for key in COUNTERS:
                    into[key] += weight * values[key]
    return bins


def summarise(rows, seed=0):
    """Rates by bin and the diet slope, with 95% intervals over episodes."""
    rng = np.random.default_rng(seed)
    point = [bin_rates(row) for row in pooled(rows)]
    samples = [{key: [] for key in point[0]} for _ in RABBIT_BINS]
    slopes = []
    for _ in range(BOOTSTRAPS):
        weights = rng.multinomial(len(rows), [1 / len(rows)] * len(rows))
        for into, row in zip(samples, pooled(rows, weights)):
            for key, value in bin_rates(row).items():
                if value is not None:
                    into[key].append(value)
        slopes.append(diet_slope([capture for i, row in enumerate(rows) for capture in row["captures"] * int(weights[i])])[1])
    table = []
    totals = pooled(rows)
    for i, rates in enumerate(point):
        entry = {"rabbits": bin_label(i), **{key: totals[i][key] for key in COUNTERS}, **rates}
        for key in rates:
            entry[key + "_ci"] = [float(q) for q in np.quantile(samples[i][key], (0.025, 0.975))] if samples[i][key] else None
        table.append(entry)
    slopes = np.array(slopes)
    unidentified = int((~np.isfinite(slopes)).sum())
    slopes = slopes[np.isfinite(slopes)]
    captures = [capture for row in rows for capture in row["captures"]]
    return {
        "episodes": len(rows),
        "bins": table,
        "diet_slope": diet_slope(captures)[1],
        "diet_slope_ci": [float(q) for q in np.quantile(slopes, (0.025, 0.975))] if len(slopes) else None,
        "diet_slope_resamples_unidentified": unidentified,
        "diet_captures": {"rabbits": int(sum(c[2] for c in captures)), "mammoths": int(sum(c[3] for c in captures))},
    }


def _play(seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    modules = evaluate._worker["modules"]
    return play_episode(evaluate._worker["env"], lambda obs: evaluate.sample_actions(modules, obs, generator), seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=evaluate._init, initargs=(args.checkpoint,)) as pool:
        rows = sorted(pool.map(_play, range(args.episodes)), key=lambda row: row["seed"])
    summary = {"checkpoint": str(args.checkpoint), **summarise(rows)}

    def shown(value, interval):
        return "-" if value is None else f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"

    print(f"{'rabbits':8}{'decisions':>10}{'catches':>9}  {'catch rate x1e5':28}{'pursuit':28}catches per sighting")
    for row in summary["bins"]:
        print(f"{row['rabbits']:8}{row['decisions']:>10.0f}{row['catches']:>9.0f}  {shown(row['catch_rate_x1e5'], row['catch_rate_x1e5_ci']):28}{shown(row['pursuit'], row['pursuit_ci']):28}{shown(row['catches_per_sighting'], row['catches_per_sighting_ci'])}")
    print("diet slope b:", summary["diet_slope"], summary["diet_slope_ci"], summary["diet_captures"], "unidentified resamples:", summary["diet_slope_resamples_unidentified"])
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": [{key: row[key] for key in ("seed", "outcome", "bins", "captures")} for row in rows]}, indent=1))


if __name__ == "__main__":
    main()
