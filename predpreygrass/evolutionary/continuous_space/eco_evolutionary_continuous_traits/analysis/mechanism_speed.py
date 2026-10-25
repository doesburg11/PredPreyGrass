"""Analysis of the mechanism test (mechanism_speed.py; RESULTS.md section 6).

The analysis pre-specified in mechanism_speed.py's docstring: per condition
and measure, the least-squares slope of relative lineage fitness (per
founder, divided by the episode's mean over founders; episodes with a mean
of zero left out) on founder speed. Contrasts, with 95% intervals from a
bootstrap over episode seeds, resampled jointly for all conditions so that
the pairing by seed is kept:

- primary (same prey, measure "alive"): mean slope over the 6 learned runs
  minus the frozen slope, and each learned run minus frozen;
- own prey: per run, its learned predators minus frozen predators, both
  facing that run's prey, and the mean over runs;
- the same for "births" and "catches" (secondary).

The bootstrap interval is over evaluation episodes for these checkpoints,
not over training runs; a run-level 95% t interval over the 6 learned runs'
contrasts is reported next to it. Also the mean relative fitness at each
speed value, frozen and the mean over learned runs, and the episodes kept
per condition.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.mechanism_speed
"""

import json

import numpy as np
from scipy import stats

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.pilot import MODULE
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.mechanism_speed import LEARNED, OUT, SPEEDS

MEASURES = ("alive", "births", "catches")
BOOTSTRAP = 20000


def load(name):
    """{measure: (episodes, founders) array}, plus "speed", ordered by seed."""
    rows = [json.loads(line) for line in (OUT / f"{name}.jsonl").read_text().splitlines()]
    out = {key: np.array([r[key] for r in rows], dtype=float) for key in MEASURES + ("speed",)}
    out["seed"] = np.array([r["seed"] for r in rows])
    out["prey_alive"] = np.array([r["prey_alive"] for r in rows], dtype=float)
    return out


def slope(data, measure, index):
    """Slope of relative fitness on founder speed over the episodes in index."""
    w, s = data[measure][index], data["speed"][index]
    mean = w.mean(axis=1, keepdims=True)
    keep = mean[:, 0] > 0
    relative, speed = (w[keep] / mean[keep]).ravel(), s[keep].ravel()
    speed = speed - speed.mean()
    return float((speed * (relative - relative.mean())).sum() / (speed**2).sum())


def by_speed(data, measure):
    """Mean relative fitness at each speed value."""
    w = data[measure]
    mean = w.mean(axis=1, keepdims=True)
    keep = mean[:, 0] > 0
    relative, speed = w[keep] / mean[keep], data["speed"][keep]
    return {str(v): float(relative[np.isclose(speed, v)].mean()) for v in SPEEDS}


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def run_level(values):
    """Mean over runs, 95% t interval, and how many runs are above zero."""
    values = np.array(values)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(values.mean()), "ci95": [float(values.mean() - half), float(values.mean() + half)], "positive": int((values > 0).sum()), "runs": len(values)}


def main():
    runs = list(LEARNED)
    data = {"frozen__same_prey": load("frozen__same_prey")}
    for run in runs:
        for name in (f"{run}__same_prey", f"{run}__own_prey", f"frozen__prey_of_{run}"):
            data[name] = load(name)
    seeds = data["frozen__same_prey"]["seed"]
    for name, d in data.items():
        if not np.array_equal(d["seed"], seeds) or not np.allclose(d["speed"], data["frozen__same_prey"]["speed"]):
            raise ValueError(f"{name}: seeds or speed assignments differ from frozen__same_prey")
    rng = np.random.default_rng(0)
    samples = [rng.integers(0, len(seeds), len(seeds)) for _ in range(BOOTSTRAP)]
    everything = np.arange(len(seeds))

    def contrasts(index, measure):
        frozen = slope(data["frozen__same_prey"], measure, index)
        same = {run: slope(data[f"{run}__same_prey"], measure, index) - frozen for run in runs}
        own = {
            run: slope(data[f"{run}__own_prey"], measure, index) - slope(data[f"frozen__prey_of_{run}"], measure, index) for run in runs
        }
        return {
            "same_prey_mean": float(np.mean(list(same.values()))),
            "own_prey_mean": float(np.mean(list(own.values()))),
            **{f"same_prey_{run}": v for run, v in same.items()},
            **{f"own_prey_{run}": v for run, v in own.items()},
        }

    output = {"episodes": int(len(seeds)), "speeds": list(SPEEDS), "bootstrap": BOOTSTRAP, "measures": {}}
    for measure in MEASURES:
        point = contrasts(everything, measure)
        boot = [contrasts(index, measure) for index in samples]
        output["measures"][measure] = {
            "slopes": {name: slope(d, measure, everything) for name, d in data.items()},
            "contrasts": {key: {"estimate": value, "ci95": interval([b[key] for b in boot])} for key, value in point.items()},
            "run_level": {
                prey: run_level([point[f"{prey}_{run}"] for run in runs]) for prey in ("same_prey", "own_prey")
            },
            "episodes_kept": {name: int((d[measure].mean(axis=1) > 0).sum()) for name, d in data.items()},
            "by_speed": {
                "frozen__same_prey": by_speed(data["frozen__same_prey"], measure),
                "learned__same_prey_mean_of_runs": {
                    str(v): float(np.mean([by_speed(data[f"{run}__same_prey"], measure)[str(v)] for run in runs])) for v in SPEEDS
                },
            },
        }
    output["prey_alive_at_end"] = {name: float(d["prey_alive"].mean()) for name, d in data.items()}
    output["predators_alive_at_end"] = {name: float(d["alive"].sum(axis=1).mean()) for name, d in data.items()}
    path = MODULE / "analysis" / "results" / "mechanism_speed.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written {path}")
    for measure in MEASURES:
        m = output["measures"][measure]
        print(f"\n{measure}: frozen slope {m['slopes']['frozen__same_prey']:+.3f}")
        for key, c in m["contrasts"].items():
            print(f"  {key:28s} {c['estimate']:+.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
        for prey, r in m["run_level"].items():
            print(f"  run level, {prey}: {r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], {r['positive']}/{r['runs']} runs > 0")
        print("  by speed, frozen :", {k: round(v, 2) for k, v in m["by_speed"]["frozen__same_prey"].items()})
        print("  by speed, learned:", {k: round(v, 2) for k, v in m["by_speed"]["learned__same_prey_mean_of_runs"].items()})


if __name__ == "__main__":
    main()
