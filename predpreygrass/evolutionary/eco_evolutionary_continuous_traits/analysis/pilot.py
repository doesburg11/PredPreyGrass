"""Pilot CT-01 (traits expressed) against CT-02 (neutral control): gene means
and selection differentials over training (from TensorBoard), final checks
and tournaments. Writes analysis/results/pilot.json and the README figure.

    python -m predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.pilot
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parents[2]
RUNS = {
    "CT-01": "eco_evolutionary_continuous_traits_2026-10-15_21-31-22_924979",
    "CT-02": "eco_evolutionary_continuous_traits_2026-10-15_21-31-42_894279",
}
LABELS = {"CT-01": "CT-01 · traits expressed", "CT-02": "CT-02 · neutral control"}
SPECIES = ("predator", "prey")
TRAITS = ("speed", "vision", "fov")
TRAIT_NAMES = {"speed": "Top speed", "vision": "View distance", "fov": "Field of view"}
BOUND = 0.7
ITERATIONS = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 2990]
SMOOTH = 50  # iterations, for the selection-differential curves

# Reference palette of the dataviz method (light mode).
COLORS = {"CT-01": "#2a78d6", "CT-02": "#eb6834"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def scalars(run_dir):
    """{tag: values per iteration} of the trait metrics."""
    events = EventAccumulator(str(run_dir / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    out = {}
    for tag in events.Tags()["scalars"]:
        if tag.startswith("env_runners/traits/"):
            out[tag[len("env_runners/traits/") :]] = np.array([e.value for e in events.Scalars(tag)])
    return out


def tournament(run_dir):
    df = pd.read_csv(run_dir / "tournament_grid300_100ep.csv")
    totals = df.groupby(["predator_iter", "prey_iter"])[["prey_caught", "prey_exposure"]].sum()
    risk = (1000 * totals.prey_caught / totals.prey_exposure).unstack()
    return risk.loc[ITERATIONS, ITERATIONS].to_numpy(), df.outcome.value_counts().to_dict()


def smooth(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot(series, path, labels=LABELS, colors=COLORS, title="Heritable traits in the continuous world: pilot"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(4, 3, figsize=(13, 12.5), sharex=True, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.885, bottom=0.06, hspace=0.35, wspace=0.22)
    fig.text(0.08, 0.965, title, fontsize=15, weight="bold")
    fig.text(0.08, 0.937, "Mean gene of the living animals at episode ends (rows 1–2) and selection differential "
             f"cov(relative offspring, gene), {SMOOTH}-iteration mean (rows 3–4)", fontsize=10.5, color=INK_2)
    for row, (kind, species) in enumerate([("mean", "predator"), ("mean", "prey"), ("selection", "predator"), ("selection", "prey")]):
        for col, trait in enumerate(TRAITS):
            ax = axes[row, col]
            for run, values in series.items():
                y = values[f"{species}_{trait}_{kind}"]
                if kind == "selection":
                    y = smooth(y, SMOOTH)
                    x = np.arange(len(y)) + SMOOTH
                else:
                    x = np.arange(1, len(y) + 1)
                ax.plot(x, y, color=colors[run], lw=2, label=labels[run])
            ax.axhline(0, color=AXIS, lw=1, zorder=0)
            if kind == "mean":
                for bound in (-BOUND, BOUND):
                    ax.axhline(bound, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
                ax.set_ylim(-0.8, 0.8)
            ax.set_title(f"{species.capitalize()} · {TRAIT_NAMES[trait]}", fontsize=10.5, weight="bold", loc="left")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, length=0)
            if col == 0:
                ax.set_ylabel("Mean gene (log-ratio)" if kind == "mean" else "Selection differential")
            if row == 3:
                ax.set_xlabel("Training iteration")
    axes[0, 0].text(60, -0.66, "gene limits ±0.7", fontsize=8.5, color=MUTED)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.075, 0.93), ncol=len(series), frameon=False, fontsize=10)
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    series, final, tournaments = {}, {}, {}
    for run, folder in RUNS.items():
        run_dir = MODULE / "runs" / folder
        series[run] = scalars(run_dir)
        final[run] = json.loads((MODULE / "runs" / f"final_check_{run.lower().replace('-', '')}.json").read_text())["summary"]
        risk, outcomes = tournament(run_dir)
        tournaments[run] = {"catch_risk": np.round(risk, 4).tolist(), "outcomes": outcomes}
    summary = {}
    for run, values in series.items():
        entry = {}
        for species in SPECIES:
            for trait in TRAITS:
                mean = values[f"{species}_{trait}_mean"]
                selection = values[f"{species}_{trait}_selection"]
                sixths = np.array_split(selection, 6)
                entry[f"{species}_{trait}"] = {
                    "mean_at": {str(i): float(mean[i - 1]) for i in (1, 500, 1000, 1500, 2000, 2500, len(mean))},
                    "selection_mean": float(selection.mean()),
                    "selection_by_sixth": [float(s.mean()) for s in sixths],
                    "selection_share_positive": float(np.mean(selection > 0)),
                }
            alive = values[f"{species}_alive"]
            entry[f"{species}_alive_last500"] = float(alive[-500:].mean())
            entry[f"{species}_cost_multiplier_end"] = float(values[f"{species}_cost_multiplier_mean"][-1])
        summary[run] = entry
    output = {"runs": RUNS, "training": summary, "final_check": final, "tournament_checkpoints": ITERATIONS, "tournament": tournaments}
    out = MODULE / "analysis" / "results" / "pilot.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    figure = REPO / "assets" / "images" / "readme" / "continuous_traits_pilot.png"
    plot(series, figure)
    print(out)
    print(figure)


if __name__ == "__main__":
    main()
