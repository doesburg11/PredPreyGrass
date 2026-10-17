"""v2 round: cheaper capacity with fixed observation scales. CT-03 (neutral
control), CT-04 (costs 0.5x the pilot's) and CT-05 (0.25x), 10 simulation
workers each. Gene means and selection differentials over training, final
checks, and tournaments with paired-seed bootstrap intervals (as in
base_environment_continuous). Writes analysis/results/v2_costs.json and the
README figure.

    python -m predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.v2_costs
"""

import json

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.pilot import (
    MODULE,
    REPO,
    SPECIES,
    TRAITS,
    plot,
    scalars,
    tournament,
)
from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.analysis.arms_race_three_runs import (
    BOOTSTRAP_SEED,
    analyse,
    load,
)

RUNS = {
    "CT-03": "eco_evolutionary_continuous_traits_2026-10-16_20-52-50_092027",
    "CT-04": "eco_evolutionary_continuous_traits_2026-10-16_20-53-10_085977",
    "CT-05": "eco_evolutionary_continuous_traits_2026-10-16_20-53-30_161103",
}
LABELS = {"CT-03": "CT-03 · neutral control", "CT-04": "CT-04 · costs ×0.5", "CT-05": "CT-05 · costs ×0.25"}
COLORS = {"CT-03": "#eb6834", "CT-04": "#2a78d6", "CT-05": "#1baf7a"}


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    series, output = {}, {"runs": RUNS, "training": {}, "final_check": {}, "tournament": {}}
    for run, folder in RUNS.items():
        run_dir = MODULE / "runs" / folder
        values = series[run] = scalars(run_dir)
        entry = {}
        for species in SPECIES:
            for trait in TRAITS:
                mean = values[f"{species}_{trait}_mean"]
                selection = values[f"{species}_{trait}_selection"]
                entry[f"{species}_{trait}"] = {
                    "mean_at": {str(i): float(mean[min(i, len(mean)) - 1]) for i in (500, 1000, 1500, 2000, 2500, 3000)},
                    "mean_max_abs": float(np.abs(mean).max()),
                    "selection_mean": float(selection.mean()),
                    "selection_share_positive": float(np.mean(selection > 0)),
                }
            entry[f"{species}_alive_last500"] = float(values[f"{species}_alive"][-500:].mean())
            entry[f"{species}_cost_multiplier_end"] = float(values[f"{species}_cost_multiplier_mean"][-1])
        output["training"][run] = entry
        output["final_check"][run] = json.loads((MODULE / "runs" / f"final_check_{run.lower().replace('-', '')}.json").read_text())["summary"]
        risk, outcomes = tournament(run_dir)
        caught, exposure, _ = load(run_dir)
        output["tournament"][run] = {"catch_risk": np.round(risk, 4).tolist(), "outcomes": outcomes, **analyse(caught, exposure, rng)}
    out = MODULE / "analysis" / "results" / "v2_costs.json"
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    figure = REPO / "assets" / "images" / "readme" / "continuous_traits_v2_costs.png"
    plot(series, figure, labels=LABELS, colors=COLORS, title="Heritable traits with cheaper capacity: v2 round")
    print(out)
    print(figure)


if __name__ == "__main__":
    main()
