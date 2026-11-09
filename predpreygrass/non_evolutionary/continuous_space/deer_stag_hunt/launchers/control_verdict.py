"""RESULTS.md section 5's verdict: the control world (young deer caught alone
for certain), from results/<RUN>_eval.json and results/<RUN>_energy.json.

A run is valid if its final checkpoint coexists in at least 45 of 50
episodes. Of the valid runs (at least 2 needed, else inconclusive):
"payoff supported" if in at least 2, test B is not met and C's 95% interval
lies entirely below 0.5; "payoff not supported" if in at least 2, A, B and C
are all met; otherwise inconclusive. Tests A, B and C are section 2.1's
(energy_verdict.py).

Run from the repository root:
    python .../launchers/control_verdict.py control_1 control_2 control_3
"""

import json
import sys

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.launchers.energy_verdict import C_MIN, RESULTS, line, load, met

COEXIST = 45


def classify(coexisting, s):
    """(valid, all_met, absent, text) for one run's coexistence and section
    2.1 summary."""
    text, all_met = line(s.get("run", ""), s)
    c_iv = s["joint_ratio_interval"]
    absent = not met(s["top_minus_chance_interval"]) and c_iv is not None and c_iv[1] < C_MIN
    return coexisting >= COEXIST, all_met, absent, text


def verdict_of(rows):
    """Section 5's verdict from (valid, all_met, absent) per run."""
    valid = [r for r in rows if r[0]]
    if len(valid) < 2:
        return "inconclusive (fewer than 2 valid runs)"
    if sum(r[2] for r in valid) >= 2:
        return "payoff supported"
    if sum(r[1] for r in valid) >= 2:
        return "payoff not supported"
    return "inconclusive"


def judge(runs):
    """(verdict, rows): rows are (run, coexisting, valid, all_met, absent, text)."""
    rows = []
    for run in runs:
        coexisting = json.loads((RESULTS / f"{run}_eval.json").read_text())["summary"]["coexisting"]
        valid, all_met, absent, text = classify(coexisting, dict(load(run), run=run))
        rows.append((run, coexisting, valid, all_met, absent, text))
    return verdict_of([r[2:5] for r in rows]), rows


def main():
    runs = sys.argv[1:]
    if len(runs) != 3:
        raise SystemExit("section 5's verdict needs exactly 3 runs")
    verdict, rows = judge(runs)
    print(f"Section 5 (valid: coexisting >= {COEXIST}/50; absent: B not met and C upper bound < {C_MIN})")
    for run, coexisting, valid, all_met, absent, text in rows:
        print(f"{text}\n    coexisting {coexisting}/50 {'valid' if valid else 'NOT valid'}; A,B,C all met: {all_met}; pattern absent: {absent}")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
