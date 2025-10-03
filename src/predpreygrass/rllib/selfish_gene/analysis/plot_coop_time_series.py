import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot AI/KPA vs training iteration from sweeper CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV produced by eval_checkpoints_coop.py")
    parser.add_argument("--out", default="output/coop_time_series.png", help="Output image path (PNG)")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA smoothing factor in [0,1), e.g., 0.8 (0 to disable)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "iteration" not in df.columns:
        raise ValueError("CSV must contain 'iteration' column")

    df = df.sort_values("iteration")

    def ema(series, alpha):
        if alpha <= 0.0:
            return series
        s = []
        m = None
        for x in series:
            m = x if m is None else alpha * m + (1 - alpha) * x
            s.append(m)
        return s

    ai = df.get("ai")
    kpa = df.get("kpa")

    if ai is None or kpa is None:
        raise ValueError("CSV must contain 'ai' and 'kpa' columns")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(df["iteration"], ema(ai, args.ema), label="AI", color="#1f77b4")
    if {"ai_lo", "ai_hi"}.issubset(df.columns):
        # Plot CI band without EMA (to avoid confusion); shaded lightly
        ax[0].fill_between(
            df["iteration"],
            df["ai_lo"],
            df["ai_hi"],
            color="#1f77b4",
            alpha=0.15,
            linewidth=0,
            label="AI 95% CI" if "ai" not in ax[0].get_legend_handles_labels()[1] else None,
        )
    ax[0].set_ylabel("Assortment Index (AI)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(df["iteration"], ema(kpa, args.ema), label="KPA", color="#d62728")
    if {"kpa_lo", "kpa_hi"}.issubset(df.columns):
        ax[1].fill_between(
            df["iteration"],
            df["kpa_lo"],
            df["kpa_hi"],
            color="#d62728",
            alpha=0.15,
            linewidth=0,
            label="KPA 95% CI" if "KPA" not in ax[1].get_legend_handles_labels()[1] else None,
        )
    ax[1].set_ylabel("Kin Proximity Advantage (KPA)")
    ax[1].set_xlabel("Training iteration (checkpoint)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[PLOT] Wrote {out_path}")


if __name__ == "__main__":
    main()
