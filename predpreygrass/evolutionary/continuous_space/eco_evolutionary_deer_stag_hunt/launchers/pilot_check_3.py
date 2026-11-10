"""RESULTS.md section 3.2: does the pilot pass? From
results/<run>_iter<250|500>_eval.json of pilot3_L1 and pilot3_B1 (own world,
each checkpoint against its own deer): in both, at least 40 of 50 episodes
coexisting at iteration 250 and, separately, at 500, and at least 5 hunter
births per evaluated episode at 500. (That nothing crashed is the
launcher's: these files exist only after every run and evaluation
finished.)

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/pilot_check_3.py
"""

import json

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.pilot_check_2 import judge
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict import RESULTS

RUNS = ("pilot3_L1", "pilot3_B1")


def summary(run, mark):
    s = json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())["summary"]
    if s["episodes"] != 50:
        raise ValueError(f"{run} iter {mark}: {s['episodes']} episodes, not 50")
    return s


def main():
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict_3 import check_own

    for run in RUNS:
        for mark in (250, 500):
            check_own(run, mark)
    results = judge({(run, mark): summary(run, mark) for run in RUNS for mark in (250, 500)})
    for run, (ok, text) in results.items():
        print(f"{run}: {text} -> {'pass' if ok else 'FAIL'}")
    for run in RUNS:
        s = summary(run, 500)
        print(f"{run} (reported) iter 500: adults per episode {s['adults_caught_per_episode']:.2f}, attempt success {s.get('adult_attempt_success')}")
    print("pilot: PASS" if len(results) == len(RUNS) and all(ok for ok, _ in results.values()) else "pilot: FAIL")


if __name__ == "__main__":
    main()
