"""RESULTS.md section 1's tests and verdict, from results/<run>_iter<mark>_{eval,coop}.json
(launchers/evaluate_run.py).

Run value of a measure: its mean over checkpoints 500, 750 and 1000 (S:
pooled over them).
- A: adults caught per episode (50 evaluated episodes);
- S: adult attempt success, adults caught / (adults caught + failed adult
  attempts), pooled over all 150 episodes of the three checkpoints (0 if
  there were no attempts);
- C (supporting): adult approaches with a partner near / chance (coop_fair, 20 episodes);
- coexisting episodes (of 50).

Tests (one-sided exact Mann-Whitney on run values, p < 0.05):
1. B > L on A and on S;  2. B > E on A and on S;
3. in at least 4 of 5 B runs, coexisting averaged over the checkpoints >= 45.

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/verdict.py [--prefix ""] [--runs 5]
Runs are named <prefix><arm>_<k> (arm E, L or B; k = 1..runs).

Also printed, not tests: per arm, the mean genes of the final checkpoint's
gene pool, the mean selection differential per gene over each run's
episodes (from the checkpoints' budget.json; L's unexpressed genes are the
drift reference), and the evolutionary budget (physics steps, episodes,
hunter births, generations) at the final checkpoint.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
ARMS = ("E", "L", "B")
CHECKPOINTS = (500, 750, 1000)
ALPHA = 0.05
COEXIST, COEXIST_RUNS = 45, 4
BOOTSTRAP = 10000


def checkpoint_values(run, mark):
    evaluation = json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())["summary"]
    coop = json.loads((RESULTS / f"{run}_iter{mark}_coop.json").read_text())["summary"]
    if evaluation["episodes"] != 50 or coop["episodes"] != 20:
        raise ValueError(f"{run} iter {mark}: expected 50 and 20 episodes")
    t = evaluation["totals"]
    return {
        "A": t.get("adults_caught", 0) / evaluation["episodes"],
        "caught": t.get("adults_caught", 0),
        "failures": t.get("adult_failures", 0),
        "C": coop["adult_over_chance"],
        "coexisting": evaluation["coexisting"],
    }


def run_value(run, marks=CHECKPOINTS):
    """A, C and coexisting: means over the checkpoints (C over those with a
    value); S: pooled over them."""
    per = [checkpoint_values(run, mark) for mark in marks]
    value = {}
    for key in ("A", "C", "coexisting"):
        values = [p[key] for p in per if p[key] is not None]
        value[key] = sum(values) / len(values) if values else None
    caught, attempts = sum(p["caught"] for p in per), sum(p["caught"] + p["failures"] for p in per)
    value["S"] = caught / attempts if attempts else 0.0
    return value


def evolution(run, final):
    """Mean genes of the final gene pool, mean selection differentials over
    the run, and the final budget."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import budget, checkpoint_at, gene_means, selection_between
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR

    checkpoint = checkpoint_at(RUNS_DIR / run, final)
    end = budget(checkpoint)
    return {"genes": gene_means(checkpoint), "selection": selection_between(None, end), "budget": end}


def mann_whitney_greater(xs, ys):
    """U = #(x > y) + ties / 2 and the exact one-sided p of U this large
    under random assignment of the pooled values to the two groups."""

    def u(a, b):
        return sum((x > y) + 0.5 * (x == y) for x in a for y in b)

    observed = u(xs, ys)
    pooled = list(xs) + list(ys)
    n = len(xs)
    count = total = 0
    for chosen in itertools.combinations(range(len(pooled)), n):
        a = [pooled[i] for i in chosen]
        b = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        count += u(a, b) >= observed - 1e-12
    return observed, count / total


def bootstrap_difference(xs, ys, seed=0):
    """95% interval of mean(xs) - mean(ys), resampling runs within arms."""
    rng = random.Random(seed)
    diffs = sorted(
        sum(rng.choice(xs) for _ in xs) / len(xs) - sum(rng.choice(ys) for _ in ys) / len(ys) for _ in range(BOOTSTRAP)
    )
    return [diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP) - 1]]


def compare(values, a, b, key):
    xs, ys = [v[key] for v in values[a]], [v[key] for v in values[b]]
    u, p = mann_whitney_greater(xs, ys)
    return {
        "U": u,
        "p": p,
        "met": p < ALPHA,
        "difference": sum(xs) / len(xs) - sum(ys) / len(ys),
        "interval": bootstrap_difference(xs, ys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--marks", type=int, nargs="+", default=list(CHECKPOINTS))
    args = parser.parse_args()
    values = {arm: [run_value(f"{args.prefix}{arm}_{k}", args.marks) for k in range(1, args.runs + 1)] for arm in ARMS}
    for arm in ARMS:
        for k, v in enumerate(values[arm], 1):
            print(f"{arm}_{k}: " + ", ".join(f"{key} {value:.3f}" if value is not None else f"{key} -" for key, value in v.items()))
    tests = {}
    for name, (a, b) in (("1_B_over_L", ("B", "L")), ("2_B_over_E", ("B", "E"))):
        result = {key: compare(values, a, b, key) for key in ("A", "S")}
        tests[name] = all(r["met"] for r in result.values())
        for key, r in result.items():
            print(f"{name} {key}: U {r['U']}, p {r['p']:.4f}, difference {r['difference']:+.3f}, 95% interval {[round(x, 3) for x in r['interval']]}")
    coexisting = sum(v["coexisting"] >= COEXIST for v in values["B"])
    tests["3_B_coexists"] = coexisting >= COEXIST_RUNS
    print(f"3_B_coexists: {coexisting} of {len(values['B'])} B runs at >= {COEXIST}")
    supporting = compare(values, "B", "L", "C") if all(v["C"] is not None for arm in ("B", "L") for v in values[arm]) else None
    print(f"supporting C, B over L: {supporting}; B runs with C > 1: {sum((v['C'] or 0) > 1 for v in values['B'])} of {len(values['B'])}")
    print("not tests: final gene pool means, mean selection differentials over the run, budget")
    for arm in ARMS:
        for k in range(1, args.runs + 1):
            e = evolution(f"{args.prefix}{arm}_{k}", max(args.marks))
            sel = e["selection"] or {}
            b = e["budget"]
            print(
                f"  {arm}_{k}: genes " + " ".join(f"{g} {v:+.3f}" for g, v in e["genes"].items())
                + "; selection " + " ".join(f"{g} {v:+.4f}" for g, v in sel.items())
                + f"; physics {b['physics_steps']}, episodes {b['episodes']}, births {b['predator_births']}, generations max {b['max_generation']} mean {b['mean_max_generation']:.1f}"
            )
    print(f"tests: {tests}")
    print("verdict: met" if all(tests.values()) else "verdict: not met")


if __name__ == "__main__":
    main()
