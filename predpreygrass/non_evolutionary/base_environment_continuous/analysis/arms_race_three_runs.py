"""Arms-race test repeated over BE-01, BE-02 and BE-03: catch-risk matrices
and improvement shares with paired-seed bootstrap intervals.

Each run's tournament crosses the predator and prey checkpoints 300, 600, ...,
2,700 and 2,990, 100 episodes per matchup with seeds 10,000-10,099
(tournament_grid300_100ep.csv in its run folder). Writes
analysis/results/arms_race_three_runs.json and the README figure.

    python -m predpreygrass.non_evolutionary.base_environment_continuous.analysis.arms_race_three_runs
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parents[2]
RUNS = {
    "BE-01": "base_environment_continuous_2026-10-13_11-25-07_420466",
    "BE-02": "base_environment_continuous_2026-10-13_11-25-27_332055",
    "BE-03": "base_environment_continuous_2026-10-14_00-27-11_018144",
}
CSV = "tournament_grid300_100ep.csv"
ITERATIONS = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 2990]
LATE_FROM = 2100
EPISODES = 100
FIRST_SEED = 10_000
REPLICATES = 10_000
BOOTSTRAP_SEED = 20261014
NAMES = ["prey_improvement_share", "predator_improvement_share", "prey_mean_risk_reduction", "predator_mean_risk_increase"]

# Reference palette of the dataviz method (light mode).
PREY, PREDATOR = "#1baf7a", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load(run_dir):
    """Prey caught and prey exposure as (predator, prey, seed) arrays."""
    df = pd.read_csv(run_dir / CSV)
    seeds = sorted(df.seed.unique())
    assert sorted(df.predator_iter.unique()) == ITERATIONS == sorted(df.prey_iter.unique())
    assert seeds == list(range(FIRST_SEED, FIRST_SEED + EPISODES)) and len(df) == len(ITERATIONS) ** 2 * EPISODES
    assert not df.duplicated(["predator_iter", "prey_iter", "seed"]).any()
    index = pd.MultiIndex.from_product([ITERATIONS, ITERATIONS, seeds])
    table = df.set_index(["predator_iter", "prey_iter", "seed"]).reindex(index)
    assert table.prey_exposure.notna().all() and (table.prey_exposure > 0).all()
    shape = (len(ITERATIONS), len(ITERATIONS), EPISODES)
    outcomes = {str(k): int(v) for k, v in df.outcome.value_counts().items()}
    return table.prey_caught.to_numpy().reshape(shape), table.prey_exposure.to_numpy().reshape(shape), outcomes


def statistics(risk):
    """Improvement shares (ties count one half) and mean catch-risk changes,
    averaged over checkpoint pairs and fixed opponents, as in BE-03's test."""
    pairs = [(a, b) for a in range(len(risk)) for b in range(a + 1, len(risk))]
    prey = np.array([risk[:, a] - risk[:, b] for a, b in pairs]).ravel()
    pred = np.array([risk[b, :] - risk[a, :] for a, b in pairs]).ravel()
    return np.array(
        [np.mean((prey > 0) + 0.5 * (prey == 0)), np.mean((pred > 0) + 0.5 * (pred == 0)), prey.mean(), pred.mean()]
    )


def analyse(caught, exposure, rng):
    """Estimates and paired-seed bootstrap 95% intervals, all and late checkpoints."""
    out = {}
    for label, keep in [("all", np.arange(len(ITERATIONS))), ("late", np.flatnonzero(np.array(ITERATIONS) >= LATE_FROM))]:
        c, e = caught[keep][:, keep], exposure[keep][:, keep]
        estimate = statistics(1000 * c.sum(2) / e.sum(2))
        boot = np.empty((REPLICATES, len(NAMES)))
        for b in range(REPLICATES):
            sample = rng.integers(0, EPISODES, EPISODES)
            boot[b] = statistics(1000 * c[:, :, sample].sum(2) / e[:, :, sample].sum(2))
        out[label] = {
            name: {"estimate": float(estimate[k]), "ci95": np.quantile(boot[:, k], [0.025, 0.975]).tolist()}
            for k, name in enumerate(NAMES)
        }
    return out


def plot(risks, results, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK_2})
    fig = plt.figure(figsize=(12, 10.6), facecolor="white")
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1.02], hspace=0.42, wspace=0.12,
                            left=0.08, right=0.94, top=0.88, bottom=0.08)
    fig.text(0.08, 0.965, "Predator–prey arms race in three independent training runs", fontsize=15, weight="bold")
    fig.text(0.08, 0.937, "Each run's checkpoints 300–2,990 (every 300) played against each other · "
             "100 episodes per matchup · 30,000 episodes", fontsize=10.5, color=INK_2)

    # Top: catch risk of every predator checkpoint against every prey checkpoint.
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    vmax = max(r.max() for r in risks.values())
    ticks = range(len(ITERATIONS))
    labels = [f"{i:,}" if k % 3 == 0 or i == ITERATIONS[-1] else "" for k, i in enumerate(ITERATIONS)]
    for k, (run, risk) in enumerate(risks.items()):
        ax = fig.add_subplot(grid[0, k])
        image = ax.imshow(risk, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        ax.set_title(run, fontsize=12, weight="bold", loc="left")
        ax.set_xticks(ticks, labels, fontsize=9, color=MUTED)
        ax.set_yticks(ticks, labels if k == 0 else [], fontsize=9, color=MUTED)
        ax.tick_params(length=0)
        ax.set_xlabel("Prey checkpoint (iteration)")
        if k == 0:
            ax.set_ylabel("Predator checkpoint (iteration)")
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 3]))
    bar.set_label("Catch risk", color=INK_2)
    bar.ax.tick_params(labelsize=9, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    fig.text(0.08, 0.475, "Catch risk: prey caught per prey alive per 1,000 physics steps. Darker downwards: newer predators "
             "catch more. Lighter to the right: newer prey escape more.", fontsize=9.5, color=INK_2)

    # Bottom: improvement shares with 95% intervals.
    ax = fig.add_subplot(grid[1, :3])
    rows, headers, y = [], [], 0.0
    for window, title in [("all", "All checkpoints"), ("late", "Late training (2,100–2,990)")]:
        for species, color, key in [("Prey", PREY, "prey_improvement_share"), ("Predators", PREDATOR, "predator_improvement_share")]:
            headers.append((y - 0.75, f"{title} · {species}"))
            for run in RUNS:
                rows.append((y, run, results[run][window][key], color))
                y += 1
            y += 0.9
        y += 0.4
    for y, _, result, color in rows:
        estimate, (low, high) = 100 * result["estimate"], [100 * v for v in result["ci95"]]
        ax.plot([low, high], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([estimate], [y], s=55, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(103, y, f"{estimate:.0f}%  [{low:.0f}–{high:.0f}]", va="center", fontsize=9.5, color=INK)
    for y, header in headers:
        ax.text(35.5, y, header, fontsize=9.5, weight="bold", color=INK, va="center")
    ax.axvline(50, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(50.6, -1.6, "50% = no trend", fontsize=9, color=MUTED, va="center")
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows], fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.set_ylim(rows[-1][0] + 0.8, -2.1)
    ax.set_xlim(35, 117)
    ax.set_xticks(range(40, 101, 10))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_xlabel("Comparisons in which the newer checkpoint improves against a fixed opponent (%)")
    for y, label, color in [(0, "Prey", PREY), (1, "Predators", PREDATOR)]:
        ax.scatter([], [], s=55, color=color, label=label)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9.5, handletextpad=0.3)
    fig.text(0.08, 0.018, "Paired-seed bootstrap 95% intervals (10,000 replicates, seeds resampled jointly across matchups); "
             "pointwise, no multiplicity correction. Ties count one half.", fontsize=9, color=INK_2)
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    risks, results, outcomes = {}, {}, {}
    for run, folder in RUNS.items():
        caught, exposure, outcomes[run] = load(MODULE / "runs" / folder)
        risks[run] = 1000 * caught.sum(2) / exposure.sum(2)
        results[run] = analyse(caught, exposure, rng)
    output = {
        "checkpoints": ITERATIONS,
        "late_checkpoints": [i for i in ITERATIONS if i >= LATE_FROM],
        "episodes_per_matchup": EPISODES,
        "seeds": [FIRST_SEED, FIRST_SEED + EPISODES - 1],
        "bootstrap_replicates": REPLICATES,
        "outcomes_over_grid": outcomes,
        "catch_risk": {run: np.round(risk, 4).tolist() for run, risk in risks.items()},
        "results": results,
    }
    out = MODULE / "analysis" / "results" / "arms_race_three_runs.json"
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    figure = REPO / "assets" / "images" / "readme" / "arms_race_three_runs.png"
    plot(risks, results, figure)
    print(out)
    print(figure)
    for run in RUNS:
        for window in ("all", "late"):
            print(run, window, {k: round(v["estimate"], 3) for k, v in results[run][window].items()})


if __name__ == "__main__":
    main()
