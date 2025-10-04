import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot helping rate and type frequencies over time.")
    p.add_argument("--csv", type=str, required=True, help="CSV produced by eval_checkpoints_helping.py")
    p.add_argument("--out", type=str, default=None, help="Output image path (PNG). If omitted, shows interactively.")
    p.add_argument("--ema", type=int, default=0, help="EMA window for smoothing (0=off)")
    return p.parse_args()


def ema(series: pd.Series, span: int) -> pd.Series:
    if span and span > 1:
        return series.ewm(span=span, adjust=False).mean()
    return series


def main():
    args = parse_args()
    df = pd.read_csv(Path(args.csv).expanduser())
    df = df.sort_values("iteration")

    # Helping rate with CI band
    x = df["iteration"]
    y = df["helping_rate"]
    lo = df["helping_rate_lo"]
    hi = df["helping_rate_hi"]

    if args.ema and args.ema > 1:
        y = ema(y, args.ema)
        lo = ema(lo, args.ema)
        hi = ema(hi, args.ema)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(x, y, label="helping_rate", color="tab:blue")
    ax[0].fill_between(x, lo, hi, color="tab:blue", alpha=0.2, label="CI")
    ax[0].set_ylabel("Helping rate")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)

    # Type counts
    for col, color in [
        ("count_type_1_prey", "tab:green"),
        ("count_type_2_prey", "tab:olive"),
        ("count_type_1_predator", "tab:red"),
        ("count_type_2_predator", "tab:orange"),
    ]:
        ax[1].plot(x, df[col], label=col, color=color)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean agents per ep")
    ax[1].legend(loc="best", ncols=2)
    ax[1].grid(True, alpha=0.3)

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print("Saved:", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
