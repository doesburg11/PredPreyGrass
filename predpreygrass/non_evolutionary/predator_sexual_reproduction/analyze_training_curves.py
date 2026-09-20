"""
Summarize and plot the training curves of the predator_sexual_reproduction runs from their
TensorBoard event files (ecology/* metrics are only stored there, not in progress.csv).

Writes results_figures/population_over_training.png and prints a block-averaged table per run.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.expanduser("~/simulation_results/ray_results")
RUNS = {
    "REALISTIC (sparse reward)": "PPO_PREDATOR_SEXUAL_REPRODUCTION_REALISTIC_SEED42",
    "FORAGING (foraging reward)": "PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_CHECK_SEED42",
}
P = "ray/tune/env_runners/"
TAGS = {
    "len": P + "episode_len_mean",
    "male": P + "ecology/final_num_predator_male",
    "female": P + "ecology/final_num_predator_female",
    "prey": P + "ecology/final_num_prey",
    "att_male": P + "ecology/hunting_attempts_predator_male",
    "att_female": P + "ecology/hunting_attempts_predator_female",
    "births_male": P + "ecology/births_predator_male",
    "births_female": P + "ecology/births_predator_female",
}


def load(run):
    path = glob.glob(os.path.join(ROOT, run, "PPO_PredPreyGrass_*", "events.out*"))[0]
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    return {k: {e.step: e.value for e in ea.Scalars(t)} for k, t in TAGS.items() if t in have}


def smooth(steps, values, window=10):
    v = np.asarray(values, float)
    k = min(window, len(v))
    return steps[k - 1:], np.convolve(v, np.ones(k) / k, mode="valid")


def series(data, key):
    steps = sorted(data[key])
    return np.array(steps), np.array([data[key][s] for s in steps])


def main():
    runs = {label: load(run) for label, run in RUNS.items()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = {"male": "tab:red", "female": "tab:purple", "prey": "tab:blue"}
    for row, (label, data) in enumerate(runs.items()):
        ax = axes[row, 0]
        for key in ("male", "female", "prey"):
            if key in data:
                s, v = series(data, key)
                x, y = smooth(s, v)
                ax.plot(x, y, color=colors[key], label={"male": "predator male", "female": "predator female", "prey": "prey"}[key])
        ax.set_title(f"{label}: alive at episode end (10-iter mean)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax = axes[row, 1]
        s, v = series(data, "len")
        x, y = smooth(s, v)
        ax.plot(x, y, color="black")
        ax.axhline(1000, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{label}: episode length (cap 1000)")
        ax.set_ylim(0, 1050)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("training iteration")
    axes[0, 0].set_xlim(0, 300)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "results_figures", "population_over_training.png")
    fig.savefig(out, dpi=110)
    print("saved", out)

    for label, data in runs.items():
        print(f"\n== {label}")
        keys = [k for k in TAGS if k in data]
        print("iters      " + " ".join(f"{k:>13}" for k in keys))
        n = max(data["len"]) 
        edges = [(1, 10), (41, 50), (91, 100)] + ([(141, 150), (191, 200), (241, 250), (291, 300)] if n >= 290 else [])
        for lo, hi in edges:
            vals = []
            for k in keys:
                v = [x for s, x in data[k].items() if lo <= s <= hi]
                vals.append(f"{np.mean(v):13.2f}" if v else f"{'n/a':>13}")
            print(f"{lo:>3}-{hi:<3}    " + " ".join(vals))


if __name__ == "__main__":
    main()
