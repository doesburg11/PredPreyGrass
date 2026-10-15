"""Arms-race test with and without Aquarium's quirks: BE-01, BE-02 and BE-03
(exact environment) against BE-04 and BE-05 (every quirk fixed,
environment.FIXES), with the same tournament design and statistics as
arms_race_three_runs.py. Writes analysis/results/arms_race_quirks_fixed.json
and the README figure.

    python -m predpreygrass.non_evolutionary.base_environment_continuous.analysis.arms_race_quirks_fixed
"""

import json

import numpy as np

from predpreygrass.non_evolutionary.base_environment_continuous.analysis.arms_race_three_runs import (
    BOOTSTRAP_SEED,
    EPISODES,
    FIRST_SEED,
    ITERATIONS,
    LATE_FROM,
    MODULE,
    REPLICATES,
    REPO,
    RUNS,
    analyse,
    load,
    plot,
)

# The three exact runs first, so their bootstrap draws match arms_race_three_runs.json.
FIXED_RUNS = {
    "BE-04": "base_environment_continuous_2026-10-14_21-35-19_937331",
    "BE-05": "base_environment_continuous_2026-10-14_21-35-40_020808",
}
ALL_RUNS = {**RUNS, **FIXED_RUNS}
LABELS = {run: f"{run} · exact" for run in RUNS} | {run: f"{run} · fixed" for run in FIXED_RUNS}


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    risks, results, outcomes = {}, {}, {}
    for run, folder in ALL_RUNS.items():
        caught, exposure, outcomes[run] = load(MODULE / "runs" / folder)
        risks[run] = 1000 * caught.sum(2) / exposure.sum(2)
        results[run] = analyse(caught, exposure, rng)
    output = {
        "exact_runs": list(RUNS),
        "fixed_runs": list(FIXED_RUNS),
        "checkpoints": ITERATIONS,
        "late_checkpoints": [i for i in ITERATIONS if i >= LATE_FROM],
        "episodes_per_matchup": EPISODES,
        "seeds": [FIRST_SEED, FIRST_SEED + EPISODES - 1],
        "bootstrap_replicates": REPLICATES,
        "outcomes_over_grid": outcomes,
        "catch_risk": {run: np.round(risk, 4).tolist() for run, risk in risks.items()},
        "results": results,
    }
    out = MODULE / "analysis" / "results" / "arms_race_quirks_fixed.json"
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    figure = REPO / "assets" / "images" / "readme" / "arms_race_quirks_fixed.png"
    plot(
        risks,
        results,
        figure,
        title="Arms race with and without Aquarium's quirks",
        subtitle="Three runs in the exact environment, two with every quirk fixed · checkpoints 300–2,990 (every 300) "
        "against each other · 100 episodes per matchup · 50,000 episodes",
        labels=LABELS,
        figsize=(17, 12),
        caption_y=0.505,
    )
    print(out)
    print(figure)
    for run in ALL_RUNS:
        for window in ("all", "late"):
            print(run, window, {k: round(v["estimate"], 3) for k, v in results[run][window].items()})


if __name__ == "__main__":
    main()
