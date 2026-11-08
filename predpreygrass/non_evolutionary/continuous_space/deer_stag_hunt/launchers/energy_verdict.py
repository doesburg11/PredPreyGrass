"""RESULTS.md section 2.1's tests and verdict: partnering by the deer's
energy under the touch rule, from launchers/coop_by_energy.py's results
(results/<RUN>_energy.json).

Per run: test A, near share of middle deer in (4, 5) minus that in (2, 3.5],
95% interval above 0; test B, near share of middle deer in (4.5, 5) minus
chance, 95% interval above 0; test C, joint share of middle deer in (4, 5)
divided by joint share of adults, 95% interval above C_MIN. The pattern is
shown if all three are met in at least 2 of the runs. Comparison runs (--compare, e.g. section 1's) are
printed but enter no test.

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/energy_verdict.py \\
        touch_1 touch_2 touch_3 [--compare run_1 run_2 run_3]
"""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(run):
    return json.loads((RESULTS / f"{run}_energy.json").read_text())["summary"]


C_MIN = 0.5


def met(interval, threshold=0.0):
    return interval is not None and interval[0] > threshold


def line(run, s):
    def show(key):
        value, iv = s[key], s[f"{key}_interval"]
        text = "-" if value is None else f"{value:+.3f}"
        return text + (" [-]" if iv is None else f" [{iv[0]:+.3f}, {iv[1]:+.3f}]")
    a, b = met(s["trend_interval"]), met(s["top_minus_chance_interval"])
    c = met(s["joint_ratio_interval"], C_MIN)
    word = lambda ok: "met" if ok else "not met"
    return (f"  {run:16s} episodes {s['episodes']:3d}  A {show('trend')} {word(a)}  B {show('top_minus_chance')} {word(b)}"
            f"  C {show('joint_ratio')} {word(c)}"), a and b and c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--compare", nargs="*", default=[])
    args = parser.parse_args()
    if len(args.runs) != 3:
        raise SystemExit("section 2.1's verdict needs exactly 3 runs (at least 2 of 3)")
    print(f"Section 2.1 tests (A: near (4,5) - near (2,3.5] > 0; B: near (4.5,5) - chance > 0; C: joint (4,5) / joint adult > {C_MIN}; 95% intervals over episodes)")
    passed = 0
    for run in args.runs:
        text, ok = line(run, load(run))
        passed += ok
        print(text)
    need = 2
    print(f"runs meeting A, B and C: {passed} of {len(args.runs)} (needed: {need})")
    print("verdict:", "pattern shown" if passed >= need else "pattern not shown")
    if args.compare:
        print("comparison (no test):")
        for run in args.compare:
            print(line(run, load(run))[0])


if __name__ == "__main__":
    main()
