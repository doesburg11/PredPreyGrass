"""
Population dynamics of trained predator_bands policies (no training): is there predator-prey coupling or a cycle?

Rolls out a checkpoint for several episodes (deterministic actions, the env from run_config.json, optionally with overrides) and records the
number of males, females and prey at every step. After a burn-in (--burn-in steps) it reports, per episode: means and coefficients of
variation; the correlation between prey and total predators at lags from -60 to +60 steps (a Lotka-Volterra-like cycle would show a consistent
sign at a consistent positive lag, prey leading predators) and the dominant period of the detrended predator series from its power spectrum with
the share of variance in that peak (a cycle would give a sharp peak at a period of tens to hundreds of steps). Summary lines give the fraction of
episodes with the same sign of best-lag correlation and the median dominant period. Descriptive; a handful of episodes, no significance test.

Example:
  python -m predpreygrass.non_evolutionary.predator_bands.analyze_population_dynamics \
      --run MEAT060=~/simulation_results/ray_results/PPO_PREDATOR_BANDS_CALIB_F008_MEAT060_SEED42 --checkpoint 19 --episodes 6
"""
import argparse
import glob
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

from predpreygrass.non_evolutionary.predator_bands.analyze_band_behavior import InstrumentedBandsEnv, load_modules, policy_of
from predpreygrass.non_evolutionary.predator_bands.evaluate_ppo_from_checkpoint_debug import policy_pi


def lag_corr(x, y, lag):
    """corr(x_t, y_{t+lag})."""
    a, b = (x[: len(x) - lag], y[lag:]) if lag >= 0 else (x[-lag:], y[: len(y) + lag])
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def dominant_period(series, min_period=20):
    """(period, share of variance in that spectral peak) of the detrended series; None if too short or constant."""
    x = np.asarray(series, float)
    if len(x) < 100 or x.std() == 0:
        return None
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    ok = (freqs > 0) & (1.0 / np.maximum(freqs, 1e-12) >= min_period)
    if not ok.any():
        return None
    i = np.flatnonzero(ok)[int(np.argmax(power[ok]))]
    return float(1.0 / freqs[i]), float(power[i] / power[freqs > 0].sum())


def rollout_series(config, modules, n_episodes, seed0):
    out = []
    for ep in range(n_episodes):
        env = InstrumentedBandsEnv(config)
        obs, _ = env.reset(seed=seed0 + ep)
        active = list(obs)
        rows = []
        while True:
            actions = {a: policy_pi(obs[a], modules[policy_of(a)], True) for a in active}
            obs, _, term, trunc, _ = env.step(actions)
            active = [a for a in obs if not term.get(a, False) and not trunc.get(a, False)]
            rows.append((env.current_num_predator_male, env.current_num_predator_female, env.current_num_prey))
            if term.get("__all__") or trunc.get("__all__"):
                break
        own_m, own_f = sum(env.own["meat"].values()), sum(env.own["fruit"].values())
        out.append((np.array(rows, float), own_m / max(own_m + own_f, 1e-9)))
    return out


def summarize(label, series, burn_in):
    lines = [f"== {label}: episode lengths {[len(s) for s, _ in series]}, prey share of own foraging energy {100 * np.mean([p for _, p in series]):.1f}%"]
    signs, periods = [], []
    for i, (s, _) in enumerate(series):
        b = s[burn_in:] if len(s) > burn_in + 100 else None
        if b is None:
            lines.append(f"   ep{i}: only {len(s)} steps, skipped (burn-in {burn_in})")
            continue
        males, females, prey = b[:, 0], b[:, 1], b[:, 2]
        pred = males + females
        cv = lambda x: x.std() / max(x.mean(), 1e-9)
        lags = list(range(-60, 61, 5))
        cors = [lag_corr(prey, pred, L) for L in lags]
        k = int(np.nanargmax(np.abs(cors))) if not np.all(np.isnan(cors)) else None
        dp = dominant_period(pred)
        best = "n/a" if k is None else f"lag {lags[k]:+d} -> {cors[k]:+.2f}"
        signs.append(np.sign(cors[k]) if k is not None else 0)
        if dp:
            periods.append(dp[0])
        lines.append(
            f"   ep{i}: {len(s)} steps | predators {pred.mean():.1f} (cv {cv(pred):.2f}), prey {prey.mean():.1f} (cv {cv(prey):.2f}) | "
            f"corr(prey, predators) best {best}; lag0 {lag_corr(prey, pred, 0):+.2f}"
            + (f" | dominant period {dp[0]:.0f} steps ({100 * dp[1]:.0f}% of variance)" if dp else "")
        )
    if signs:
        lines.append(
            f"   summary: best-lag correlation positive in {sum(1 for x in signs if x > 0)} / negative in {sum(1 for x in signs if x < 0)} episodes; "
            + (f"median dominant period {np.median(periods):.0f} steps" if periods else "no period")
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=14)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--override", action="append", default=[], help="KEY=VALUE env config override (numbers only), repeatable")
    args = parser.parse_args()
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        for kv in args.override:
            k, v = kv.split("=", 1)
            config[k] = float(v) if "." in v else int(v)
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        for line in summarize(f"{label} ckpt{args.checkpoint}", rollout_series(config, modules, args.episodes, args.seed), args.burn_in):
            print(line, flush=True)


if __name__ == "__main__":
    main()
