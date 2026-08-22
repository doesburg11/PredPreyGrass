"""Visualize the nature/nurture interaction from the 5x100-seed comparative
study (RESULTS.md §9): per-seed survival time (extinction step, or the
1,000,000-step ceiling) for each of Ackley & Littman's five conditions --
ERL (genome + lifetime learning), L (learning alone), E (evolution alone),
F (neither), B (luck alone, action selection ignores genome/network).

ERL clearly beating both E and L alone is the "nature and nurture combine"
result (RESULTS.md's Mann-Whitney tests, p<0.00001 vs. all four). This
script re-derives the per-seed numbers straight from the run logs -- the
same source RESULTS.md's table was computed from -- and plots the full
distribution (not just median/mean) so the "83% of ERL runs hit the
ceiling" shape is visible, not just its summary statistic.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.plot_full_study_survival \\
        --out /tmp/erl_survival.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from predpreygrass.global_config import ERL_RESULTS_DIR

STRATEGIES = ["ERL", "L", "E", "F", "B"]  # RESULTS.md §9 order, strongest first
STEP_LIMIT = 1_000_000

# Palette slots 1-5 from the dataviz skill's validated categorical order
# (blue, orange, aqua, yellow, magenta) -- fixed per strategy, not by rank.
_COLORS = {
    "ERL": "#2a78d6",
    "L": "#eb6834",
    "E": "#1baf7a",
    "F": "#eda100",
    "B": "#e87ba4",
}

_EXTINCTION_RE = re.compile(r"extinction at step (\d+)")


def parse_log(path: Path) -> int | None:
    text = path.read_text()
    m = _EXTINCTION_RE.search(text)
    if m:
        return int(m.group(1))
    if "Reached step limit" in text:
        return STEP_LIMIT
    return None  # run incomplete/crashed -- excluded, not zero


def load_survival_times(logs_dir: Path) -> dict[str, list[int]]:
    data = {s: [] for s in STRATEGIES}
    for strategy in STRATEGIES:
        for path in sorted(logs_dir.glob(f"{strategy}_seed*.log")):
            step = parse_log(path)
            if step is not None:
                data[strategy].append(step)
    return data


def plot(data: dict[str, list[int]], out_path: Path, web: bool = False):
    """`web=True` drops the in-figure title and ceiling annotation text (the
    embedding page's own heading/figcaption carries that instead) and scales
    fonts/markers up, since the site displays this at a fixed ~700-800px
    width rather than at native pixel size like the RESULTS.md copy.
    """
    scale = 1.3 if web else 1.0
    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(0)

    for i, strategy in enumerate(STRATEGIES):
        values = np.array(data[strategy])
        color = _COLORS[strategy]

        box = ax.boxplot(
            values, positions=[i], widths=0.5, vert=True, patch_artist=True,
            showfliers=False, zorder=2,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.12)
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
        for key in ("whiskers", "caps"):
            for line in box[key]:
                line.set_color(color)
                line.set_linewidth(1.5)
        for line in box["medians"]:
            line.set_color(color)
            line.set_linewidth(2)

        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter, values,
            s=22 * scale, color=color, alpha=0.5, edgecolors="#fcfcfb", linewidths=0.5, zorder=3,
        )

        median = int(np.median(values))
        ax.annotate(
            f"median {median:,}", xy=(i, median), xytext=(0, 10),
            textcoords="offset points", ha="center", fontsize=9 * scale, color="#0b0b0b",
        )

    ax.axhline(STEP_LIMIT, color="#898781", linewidth=1, zorder=1)
    if not web:
        ax.annotate(
            "1,000,000-step ceiling (survived to the study's cutoff)",
            xy=(len(STRATEGIES) - 1, STEP_LIMIT), xytext=(6, 4),
            textcoords="offset points", ha="left", fontsize=8, color="#52514e",
        )

    ax.set_yscale("log")
    all_values = np.concatenate([np.array(v) for v in data.values()])
    ax.set_ylim(all_values.min() / 1.6, STEP_LIMIT * 2.4)
    ax.set_ylabel("survival time (steps, log scale)", color="#52514e", fontsize=11 * scale)
    ax.tick_params(axis="both", labelsize=10 * scale)
    ax.set_xticks(range(len(STRATEGIES)))
    n_per = [len(data[s]) for s in STRATEGIES]
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(STRATEGIES, n_per)], fontsize=10 * scale)
    if not web:
        ax.set_title(
            "Nature + nurture combined (ERL) survives far longer than either alone\n"
            "Ackley & Littman comparative study -- 100 seeds/condition, World AL",
            fontsize=12,
        )
    ax.grid(True, axis="y", which="major", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir", type=str, default=str(Path(ERL_RESULTS_DIR) / "erl_full_study_logs"),
        help="Directory of {strategy}_seed{N}.log files from the comparative study.",
    )
    parser.add_argument("--out", type=str, default="erl_survival_by_strategy.png")
    parser.add_argument(
        "--web", action="store_true",
        help="Drop the in-figure title/ceiling caption and scale up fonts/markers, "
             "for embedding on a page that supplies its own heading and figcaption.",
    )
    args = parser.parse_args()

    data = load_survival_times(Path(args.logs_dir))
    for strategy in STRATEGIES:
        n = len(data[strategy])
        if n == 0:
            raise SystemExit(f"No logs found for strategy {strategy} in {args.logs_dir}")
        print(f"{strategy}: n={n}, median={int(np.median(data[strategy])):,}")

    plot(data, Path(args.out), web=args.web)


if __name__ == "__main__":
    main()
