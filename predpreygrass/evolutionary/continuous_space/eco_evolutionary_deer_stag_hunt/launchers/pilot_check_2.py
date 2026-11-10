"""RESULTS.md section 2.2: does the pilot pass? From
results/<run>_iter<250|500>_eval.json of pilot_L1, pilot_L2, pilot_L3 and
pilot_B1: in every one, hunters and deer both alive at the end of at least
40 of 50 episodes at iterations 250 and 500, and at least 5 hunter births
per evaluated episode at iteration 500. (That nothing crashed is the
launcher's: these files exist only after every run and evaluation
finished.) pilot_E1 is printed, not judged.

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/pilot_check_2.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = ("pilot_L1", "pilot_L2", "pilot_L3", "pilot_B1")
COEXIST, BIRTHS = 40, 5.0


def summary(run, mark):
    s = json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())["summary"]
    if s["episodes"] != 50:
        raise ValueError(f"{run} iter {mark}: {s['episodes']} episodes, not 50")
    return s


def judge(summaries):
    """{run: (passed, text)} from {(run, mark): evaluation summary}."""
    out = {}
    for run in sorted({run for run, _ in summaries}):
        s250, s500 = summaries[(run, 250)], summaries[(run, 500)]
        births = s500["totals"].get("births_predator", 0) / s500["episodes"]
        ok = s250["coexisting"] >= COEXIST and s500["coexisting"] >= COEXIST and births >= BIRTHS
        out[run] = (ok, f"coexisting {s250['coexisting']}/50 at 250, {s500['coexisting']}/50 at 500; hunter births per episode at 500: {births:.2f}")
    return out


def main():
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict_2 import check_provenance

    for run in RUNS:
        for mark in (250, 500):
            check_provenance(run, mark)
    results = judge({(run, mark): summary(run, mark) for run in RUNS for mark in (250, 500)})
    for run, (ok, text) in results.items():
        print(f"{run}: {text} -> {'pass' if ok else 'FAIL'}")
    for mark in (250, 500):
        path = RESULTS / f"pilot_E1_iter{mark}_eval.json"
        if path.is_file():
            s = json.loads(path.read_text())["summary"]
            print(f"pilot_E1 (reported only) iter {mark}: coexisting {s['coexisting']}/50, hunter births per episode {s['totals'].get('births_predator', 0) / s['episodes']:.2f}, adults per episode {s['totals'].get('adults_caught', 0) / s['episodes']:.2f}")
    print("pilot: PASS" if all(ok for ok, _ in results.values()) else "pilot: FAIL")


if __name__ == "__main__":
    main()
