"""RESULTS.md section 2's tests and verdict, from
results/<run>_iter<mark>_{eval,coop}.json (launchers/evaluate_run.py; each
checkpoint evaluated against the deer it was saved with).

Run values as section 1 (launchers/verdict.py): A, C and coexisting are
means over checkpoints 500, 750 and 1000, S is pooled over them.

1. B beats L on A and on S, paired by run: d_k = B_k - L_k (k = 1..6), a
   one-sided sign-flip permutation test on the mean of d over all 2^6
   arrangements of signs, p = share of arrangements with a mean at least
   the observed one; p < 0.05 for A and for S.
2. In at least 5 of 6 B runs, coexisting averaged over the checkpoints is
   at least 45.

Also printed, not tests: B - L per deer schedule, E's run values, the
fixed-deer panel (results/<run>_final_vs_<touch_j>_eval.json) and the
evolution summary of verdict.py.

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/verdict_2.py [--prefix ""] [--runs 6]
"""

import argparse
import itertools
import json

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict import CHECKPOINTS, RESULTS, bootstrap_difference, evolution, run_value

ALPHA = 0.05
COEXIST, COEXIST_RUNS = 45, 5
SCHEDULES = ("touch_1", "touch_2", "touch_3")


def check_provenance(run, mark, coop=True):
    """Refuse a result file that is not this run's checkpoint at this mark,
    evaluated against D(mark), the deer the checkpoint was saved with."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import deer_schedule
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR

    checkpoint = checkpoint_at(RUNS_DIR / run, mark)
    active = deer_schedule.active(checkpoint)
    if active is None or active["source_iteration"] != deer_schedule.source_iteration(mark):
        raise ValueError(f"{checkpoint}: saved with deer {active}, not D({mark})")
    deer = str(checkpoint_deer(checkpoint))
    files = [RESULTS / f"{run}_iter{mark}_eval.json"] + ([RESULTS / f"{run}_iter{mark}_coop.json"] if coop else [])
    for path in files:
        data = json.loads(path.read_text())
        found = (data.get("checkpoint") or data["summary"].get("checkpoint"), data.get("deer") or data["summary"].get("deer"))
        if found != (str(checkpoint), deer):
            raise ValueError(f"{path}: evaluated {found}, expected {(str(checkpoint), deer)}")


def sign_flip_p(differences):
    """One-sided p of the mean of paired differences under random signs."""
    observed = sum(differences) / len(differences)
    count = 0
    arrangements = list(itertools.product((1, -1), repeat=len(differences)))
    for signs in arrangements:
        count += sum(s * d for s, d in zip(signs, differences)) / len(differences) >= observed - 1e-12
    return observed, count / len(arrangements)


def paired(values, key):
    d = [b[key] - l[key] for b, l in zip(values["B"], values["L"])]
    mean, p = sign_flip_p(d)
    return {"differences": d, "mean": mean, "p": p, "met": p < ALPHA}


def judge(values):
    """(tests, details) from {arm: [run values]} for arms L and B."""
    details = {key: paired(values, key) for key in ("A", "S")}
    coexisting = sum(v["coexisting"] >= COEXIST for v in values["B"])
    tests = {"1_B_over_L": all(r["met"] for r in details.values()), "2_B_coexists": coexisting >= COEXIST_RUNS}
    details["coexisting_B_runs"] = coexisting
    return tests, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()
    if args.runs != 6:
        raise SystemExit("section 2's verdict is on 6 runs per arm")
    for arm in ("L", "B", "E"):
        for k in range(1, args.runs + 1):
            for mark in CHECKPOINTS:
                check_provenance(f"{args.prefix}{arm}_{k}", mark)
    values = {arm: [run_value(f"{args.prefix}{arm}_{k}", CHECKPOINTS) for k in range(1, args.runs + 1)] for arm in ("L", "B", "E")}
    for arm in ("L", "B", "E"):
        for k, v in enumerate(values[arm], 1):
            print(f"{arm}_{k} ({SCHEDULES[(k - 1) % 3]}): " + ", ".join(f"{key} {value:.3f}" if value is not None else f"{key} -" for key, value in v.items()))
    tests, details = judge(values)
    for key in ("A", "S"):
        r = details[key]
        interval = bootstrap_difference([v[key] for v in values["B"]], [v[key] for v in values["L"]])
        print(f"test 1, {key}: B - L per run {[round(x, 3) for x in r['differences']]}, mean {r['mean']:+.3f}, sign-flip p {r['p']:.4f} ({'met' if r['met'] else 'not met'}); bootstrap 95% interval (unpaired, reported) {[round(x, 3) for x in interval]}")
    print(f"test 2: {details['coexisting_B_runs']} of 6 B runs coexisting >= {COEXIST} ({'met' if tests['2_B_coexists'] else 'not met'})")
    print("not tests:")
    for j, schedule in enumerate(SCHEDULES):
        pairs = [k for k in range(args.runs) if k % 3 == j]
        print(f"  {schedule}: B - L, A " + " ".join(f"{values['B'][k]['A'] - values['L'][k]['A']:+.2f}" for k in pairs) + ", S " + " ".join(f"{values['B'][k]['S'] - values['L'][k]['S']:+.3f}" for k in pairs))
    print("  fixed-deer panel (final checkpoint against each final touch deer): adults per episode, coexisting")
    for arm in ("L", "B"):
        for k in range(1, args.runs + 1):
            row = []
            for schedule in SCHEDULES:
                path = RESULTS / f"{args.prefix}{arm}_{k}_final_vs_{schedule}_eval.json"
                if path.is_file():
                    s = json.loads(path.read_text())["summary"]
                    row.append(f"{schedule} {s['totals'].get('adults_caught', 0) / s['episodes']:.2f}/{s['coexisting']}")
                else:
                    row.append(f"{schedule} -")
            print(f"    {arm}_{k}: " + ", ".join(row))
    print("  final gene pool means, mean selection differentials over the run, budget")
    for arm in ("E", "L", "B"):
        for k in range(1, args.runs + 1):
            e = evolution(f"{args.prefix}{arm}_{k}", max(CHECKPOINTS))
            sel = e["selection"] or {}
            b = e["budget"]
            print(
                f"    {arm}_{k}: genes " + " ".join(f"{g} {v:+.3f}" for g, v in e["genes"].items())
                + "; selection " + " ".join(f"{g} {v:+.4f}" for g, v in sel.items())
                + f"; physics {b['physics_steps']}, episodes {b['episodes']}, births {b['predator_births']}, generations max {b['max_generation']} mean {b['mean_max_generation']:.1f}"
            )
    print(f"tests: {tests}")
    print("verdict: met" if all(tests.values()) else "verdict: not met")


if __name__ == "__main__":
    main()
