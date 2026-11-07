"""The tests of RESULTS.md section 1, applied to each run's final checkpoint,
and the replication verdict (all three tests met in at least 2 of the 3
runs). Reads results/<run>_eval.json (evaluate.py, 50 episodes) and
results/<run>_coop.json (launchers/coop_fair.py, 20 episodes).

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/verdict.py [RUN ...]
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = ("run_1", "run_2", "run_3")
COEXIST = 45  # of 50 episodes
EPISODES = {"eval": 50, "coop": 20}
SECTION_7 = 0.90  # stag_hunt_continuous RESULTS.md section 7.1: fat share / chance


def tests(evaluation, coop):
    """{test: met} for one run, from its evaluation and coop_fair summaries."""
    if evaluation["episodes"] != EPISODES["eval"] or coop["episodes"] != EPISODES["coop"]:
        raise ValueError(f"expected {EPISODES} episodes, got eval {evaluation['episodes']}, coop {coop['episodes']}")
    share, chance = coop["share"], coop["chance"]
    cooperation = None not in (share["adult"], share["fawn"], chance) and share["adult"] > share["fawn"] and share["adult"] > chance
    ratio = share["adult"] / chance if share["adult"] is not None and chance else None
    return {
        "1_coexistence": evaluation["coexisting"] >= COEXIST,
        "2_cooperation": cooperation,
        "3_above_section_7": ratio is not None and ratio > SECTION_7,
    }


def verdict(per_run):
    """Replicated if every test is met in at least 2 of the 3 runs."""
    if len(per_run) != len(RUNS):
        raise ValueError(f"the verdict needs {len(RUNS)} runs, got {len(per_run)}")
    return sum(all(met.values()) for met in per_run.values()) >= 2


def main():
    runs = sys.argv[1:] or RUNS
    per_run = {}
    for run in runs:
        evaluation = json.loads((RESULTS / f"{run}_eval.json").read_text())["summary"]
        coop = json.loads((RESULTS / f"{run}_coop.json").read_text())["summary"]
        per_run[run] = tests(evaluation, coop)
        print(
            f"{run}: coexisting {evaluation['coexisting']}/{evaluation['episodes']}; chance {coop['chance']:.3f}, "
            f"fawn {coop['share']['fawn']}, adult {coop['share']['adult']}, adult/chance {coop.get('adult_over_chance')}, "
            f"interval {coop.get('adult_minus_chance_interval')}; tests {per_run[run]}"
        )
    if len(runs) == len(RUNS):
        print("replicated" if verdict(per_run) else "not replicated")


if __name__ == "__main__":
    main()
