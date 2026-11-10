"""RESULTS.md section 3's tests and verdict (DESIGN_3.md), from
results/<run>_iter<mark>_{eval,coop}.json (own world,
launchers/evaluate_run.py) and results/<hunters>_iter<mark>_vs_<deer>_eval.json
(cross-play panel, launchers/cross_play.py). Runs main3_L1..6, main3_B1..6.

Test statistic everywhere: mean(B) - mean(L) of run values; one-sided exact
permutation test over all C(12, 6) = 924 labellings of 6 of the 12 runs as
B, p = share of labellings with a difference >= the observed one.

1. Primary: panel A and panel S, p < 0.05 each, both. A hunter run's panel
   A is the mean of adults per episode over its 33 off-diagonal cells (11
   other runs' deer x checkpoints 500, 750, 1000); panel S is pooled over
   those cells' episodes.
2. Condition: in at least 5 of 6 B runs, own-world coexisting counts at
   500, 750 and 1000 sum to at least 135.
3. Secondary, per gene (a_hunter, a_adult), p < 0.025 each: run value = mean
   of the per-episode selection differential over own-world evaluation
   episodes at 500, 750, 1000 with a defined value (undefined counted).

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/verdict_3.py
"""

import itertools
import json

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict import RESULTS, bootstrap_difference, run_value

CHECKPOINTS = (500, 750, 1000)
RUNS = {arm: [f"main3_{arm}{k}" for k in range(1, 7)] for arm in ("L", "B")}
ALPHA, ALPHA_GENES = 0.05, 0.025
COEXIST_SUM, COEXIST_RUNS = 135, 5
SELECTED = ("a_hunter", "a_adult")
REPORTED_GENES = ("a_hunter", "a_young", "a_adult", "neutral")


def permutation_p(bs, ls):
    """(mean(bs) - mean(ls), one-sided p over every relabelling of the
    pooled runs into groups of the same sizes)."""
    pooled, n = list(bs) + list(ls), len(bs)
    observed = sum(bs) / n - sum(ls) / len(ls)
    count = total = 0
    for chosen in itertools.combinations(range(len(pooled)), n):
        a = [pooled[i] for i in chosen]
        b = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        count += sum(a) / len(a) - sum(b) / len(b) >= observed - 1e-12
    return observed, count / total


def check_own(run, mark):
    """Refuse an own-world result file that is not this run's checkpoint at
    this mark against its own deer."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR

    checkpoint = checkpoint_at(RUNS_DIR / run, mark)
    expected = (str(checkpoint), str(checkpoint_deer(checkpoint)))
    for kind in ("eval", "coop"):
        data = json.loads((RESULTS / f"{run}_iter{mark}_{kind}.json").read_text())
        found = (data.get("checkpoint") or data["summary"].get("checkpoint"), data.get("deer") or data["summary"].get("deer"))
        if found != expected:
            raise ValueError(f"{run} iter {mark} {kind}: evaluated {found}, expected {expected}")


def check_cell(data, hunter_run, deer_run, mark):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.cross_play import provenance

    expected = provenance(checkpoint_at(RUNS_DIR / hunter_run, mark), checkpoint_deer(checkpoint_at(RUNS_DIR / deer_run, mark)))
    if data.get("provenance") != expected or data["summary"]["episodes"] != 50:
        raise ValueError(f"panel cell {hunter_run} vs {deer_run} at {mark}: provenance or episodes differ")


def load_panel(runs, marks=CHECKPOINTS, check=True):
    """{(hunter_run, deer_run, mark): summary} for every off-diagonal cell."""
    panel = {}
    for mark in marks:
        for h in runs:
            for d in runs:
                if h != d:
                    data = json.loads((RESULTS / f"{h}_iter{mark}_vs_{d}_eval.json").read_text())
                    if check:
                        check_cell(data, h, d, mark)
                    panel[(h, d, mark)] = data["summary"]
    return panel


def panel_values(panel, run, side="hunters"):
    """A and pooled S over a run's off-diagonal cells, as hunters (its row)
    or as deer (its column: adults lost per episode, attempts survived)."""
    cells = [s for (h, d, _), s in panel.items() if (h if side == "hunters" else d) == run]
    if not cells:
        raise ValueError(f"no panel cells for {run}")
    a = sum(s["totals"].get("adults_caught", 0) / s["episodes"] for s in cells) / len(cells)
    caught = sum(s["totals"].get("adults_caught", 0) for s in cells)
    attempts = caught + sum(s["totals"].get("adult_failures", 0) for s in cells)
    return {"A": a, "S": caught / attempts if attempts else 0.0, "cells": len(cells)}


def selection_value(rows_by_mark, gene):
    """(mean per-episode selection differential over episodes with a
    defined value, defined episodes, undefined episodes)."""
    key = f"gene_{gene}_selection"
    values, undefined = [], 0
    for rows in rows_by_mark:
        for row in rows:
            stats = row.get("stats") or {}
            if key in stats:
                values.append(stats[key])
            else:
                undefined += 1
    return (sum(values) / len(values) if values else None), len(values), undefined


def own_rows(run, marks=CHECKPOINTS):
    return [json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())["episodes"] for mark in marks]


def own_coexisting_sum(run, marks=CHECKPOINTS):
    return sum(json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())["summary"]["coexisting"] for mark in marks)


def judge(panel_by_arm, coexist_by_arm, selection_by_arm):
    """(tests, details) from per-arm lists: panel values {A, S}, own-world
    coexisting sums, and {gene: selection run value}."""
    details = {}
    for key in ("A", "S"):
        diff, p = permutation_p([v[key] for v in panel_by_arm["B"]], [v[key] for v in panel_by_arm["L"]])
        details[f"panel_{key}"] = {"difference": diff, "p": p, "met": p < ALPHA}
    details["coexisting_B_runs"] = sum(c >= COEXIST_SUM for c in coexist_by_arm["B"])
    tests = {
        "1_panel_B_over_L": details["panel_A"]["met"] and details["panel_S"]["met"],
        "2_B_coexists": details["coexisting_B_runs"] >= COEXIST_RUNS,
    }
    for gene in SELECTED:
        bs, ls = [v[gene] for v in selection_by_arm["B"]], [v[gene] for v in selection_by_arm["L"]]
        if any(x is None for x in bs + ls):
            details[f"selection_{gene}"] = {"difference": None, "p": None, "met": False, "note": "a run without any defined value"}
        else:
            diff, p = permutation_p(bs, ls)
            details[f"selection_{gene}"] = {"difference": diff, "p": p, "met": p < ALPHA_GENES}
        tests[f"3_selection_{gene}"] = details[f"selection_{gene}"]["met"]
    return tests, details


def main():
    runs = RUNS["L"] + RUNS["B"]
    for run in runs:
        for mark in (250,) + CHECKPOINTS:
            check_own(run, mark)
    panel = load_panel(runs)
    panel_by_arm = {arm: [panel_values(panel, r) for r in RUNS[arm]] for arm in RUNS}
    deer_by_arm = {arm: [panel_values(panel, r, "deer") for r in RUNS[arm]] for arm in RUNS}
    coexist_by_arm = {arm: [own_coexisting_sum(r) for r in RUNS[arm]] for arm in RUNS}
    selection_full = {arm: [{g: selection_value(own_rows(r), g) for g in REPORTED_GENES} for r in RUNS[arm]] for arm in RUNS}
    selection_by_arm = {arm: [{g: v[g][0] for g in SELECTED} for v in selection_full[arm]] for arm in RUNS}
    own_by_arm = {arm: [run_value(r, CHECKPOINTS) for r in RUNS[arm]] for arm in RUNS}
    tests, details = judge(panel_by_arm, coexist_by_arm, selection_by_arm)

    print("per run: panel A, panel S (hunters); deer side A lost, S; own world A, S, C; coexisting sum of 150")
    for arm in ("L", "B"):
        for r, pv, dv, ov, c in zip(RUNS[arm], panel_by_arm[arm], deer_by_arm[arm], own_by_arm[arm], coexist_by_arm[arm]):
            C = f"{ov['C']:.3f}" if ov["C"] is not None else "-"
            print(f"  {r}: panel A {pv['A']:.2f} S {pv['S']:.4f} ({pv['cells']} cells); deer A {dv['A']:.2f} S {dv['S']:.4f}; own A {ov['A']:.2f} S {ov['S']:.4f} C {C}; coexisting {c}")
    for key in ("A", "S"):
        d = details[f"panel_{key}"]
        interval = bootstrap_difference([v[key] for v in panel_by_arm["B"]], [v[key] for v in panel_by_arm["L"]])
        print(f"test 1, panel {key}: B - L {d['difference']:+.4f}, permutation p {d['p']:.4f} ({'met' if d['met'] else 'not met'}); bootstrap 95% {[round(x, 4) for x in interval]}")
    print(f"test 2: {details['coexisting_B_runs']} of 6 B runs with coexisting sum >= {COEXIST_SUM} ({'met' if tests['2_B_coexists'] else 'not met'})")
    for gene in SELECTED:
        d = details[f"selection_{gene}"]
        text = f"B - L {d['difference']:+.5f}, permutation p {d['p']:.4f}" if d["p"] is not None else d["note"]
        print(f"test 3, {gene}: {text} ({'met' if d['met'] else 'not met'})")
    print("not tests:")
    for key in ("A", "S"):
        diff, p = permutation_p([v[key] for v in own_by_arm["B"]], [v[key] for v in own_by_arm["L"]])
        print(f"  own world {key}: B - L {diff:+.4f} (p {p:.4f}, reported)")
        diff, p = permutation_p([v[key] for v in deer_by_arm["B"]], [v[key] for v in deer_by_arm["L"]])
        print(f"  deer side {key} (adults lost / attempts against them): B's deer - L's {diff:+.4f} (p {p:.4f}, reported)")
    for arm in ("L", "B"):
        for r, sel in zip(RUNS[arm], selection_full[arm]):
            print(f"  {r} selection: " + "; ".join(f"{g} {v[0]:+.5f} ({v[1]} defined, {v[2]} undefined)" if v[0] is not None else f"{g} - ({v[2]} undefined)" for g, v in sel.items()))
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict import evolution

    for arm in ("L", "B"):
        for r in RUNS[arm]:
            e = evolution(r, 1000)
            print(f"  {r} final gene pool means: " + " ".join(f"{g} {v:+.3f}" for g, v in e["genes"].items()))
    print(f"tests: {tests}")
    print("verdict (tests 1 and 2): met" if tests["1_panel_B_over_L"] and tests["2_B_coexists"] else "verdict (tests 1 and 2): not met")
    for gene in SELECTED:
        print(f"secondary claim, {gene}: {'met' if tests[f'3_selection_{gene}'] else 'not met'}")


if __name__ == "__main__":
    main()
