"""
Mutual-benefit / fitness test: does the division of labor function as a partnership benefit, or is
it just two individuals independently doing what's locally best for themselves?

For every predator_male/predator_female pair that reproduces at least once in an episode (identified
via `env.agent_parents`, keyed by child -> (male, female); reset fresh every episode, so pairs are
tallied per-episode, not across episodes where agent names would collide), this computes:
  - offspring_count: how many children that specific pair produced together, this episode.
  - differentiation: |male's own prey-share - female's own prey-share| over each parent's OWN
    whole-episode foraging energy (energy_from_prey / (energy_from_prey + energy_from_fruit), the
    same per-agent bookkeeping `analyze_energy_sources.py`'s `InstrumentedEnv` already tracks).
    0 = identical foraging mix between the two parents, 1 = perfectly specialized apart.

Then asks: among pairs that reproduced at least once, do more-differentiated ones produce MORE
offspring than less-differentiated ones (a within-episode median split on differentiation, plus a
tie-aware Spearman rank correlation as a second, non-dichotomized view)? Both are evaluated with an
EPISODE-cluster bootstrap: whole episodes are resampled with replacement (not individual pairs), all
of a resampled episode's pairs travel together, and the median split (or the correlation) is
recomputed fresh inside every replicate -- a Codex review caught that an earlier version resampled
pairs directly, which treats pairs sharing one episode's ecology (population size, resource levels,
policy) as independent when they are not, and would understate the true uncertainty.

Two things this is NOT:
- A general fitness/reproductive-propensity test. Every included pair has already reproduced at
  least once (that's how a "pair" is defined here) -- this asks whether differentiation predicts
  ADDITIONAL offspring among already-successful pairs, not whether it predicts reproducing at all.
  Conditioning on reproduction like this can itself introduce selection bias.
- A causal or temporally-controlled test. Each parent's differentiation score is computed over its
  WHOLE episode's foraging, not just the time before its first reproduction with this partner -- a
  pair that reproduces early simply has more remaining lifetime for its foraging mix to diverge (or
  not) for reasons unrelated to that reproduction event, and reproducing itself changes subsequent
  foraging via birth costs and parental care. An association here would show differentiation and
  reproductive output covary among successfully-reproducing pairs, not that one causes the other.

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_pair_fitness \
      --run CONTROL_s42=~/simulation_results/ray_results/PPO_FIXED_PREY_DENSITY_CONTROL_SEED42 \
      --checkpoint 29 --episodes 30
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_energy_sources import make_instrumented_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    load_modules,
    module_probs,
    policy_of,
)


def _prey_share(env, agent):
    prey_e = env.energy_from_prey[agent]
    fruit_e = env.energy_from_fruit[agent]
    total = prey_e + fruit_e
    if total <= 0:
        return None  # no own-forage energy all episode -- can't score this parent
    return prey_e / total


def run_episodes(env_config, modules, n_episodes, seed0):
    env = make_instrumented_env(env_config)
    rng = np.random.default_rng(seed0)
    pairs = []  # one dict per (reproducing pair, episode) -- NOT one per episode
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        while True:
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                rows = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    actions[agent] = int(rng.choice(env.num_actions, p=rows[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break

        # Tally offspring per reproducing pair THIS episode -- agent_parents is reset fresh at the
        # next reset(), and agent names are reused episode to episode, so this must be done now.
        offspring_counts = {}
        for child, parents in env.agent_parents.items():
            offspring_counts[parents] = offspring_counts.get(parents, 0) + 1
        for (male, female), count in offspring_counts.items():
            male_share = _prey_share(env, male)
            female_share = _prey_share(env, female)
            if male_share is None or female_share is None:
                continue
            pairs.append(
                {
                    "episode": ep,  # needed for the episode-cluster bootstrap in summarize()
                    "offspring_count": count,
                    "differentiation": abs(male_share - female_share),
                    "male_share": male_share,
                    "female_share": female_share,
                }
            )
    return pairs


def _median_split_diff(sample_pairs):
    """Point statistic: mean-offspring difference between the >=median and <median differentiation
    groups, with the median (and so the split itself) computed fresh from whichever pairs are passed
    in -- called once on the real sample, then again inside every bootstrap replicate, so the
    uncertainty of the cutoff itself is captured, not just the uncertainty of two fixed groups."""
    diffs = np.array([p["differentiation"] for p in sample_pairs])
    counts = np.array([p["offspring_count"] for p in sample_pairs], dtype=float)
    median_diff = float(np.median(diffs))
    high = counts[diffs >= median_diff]
    low = counts[diffs < median_diff]
    if len(high) == 0 or len(low) == 0:
        return None, median_diff, len(high), len(low)
    return float(high.mean() - low.mean()), median_diff, len(high), len(low)


def _spearman(sample_pairs):
    diffs = [p["differentiation"] for p in sample_pairs]
    counts = [p["offspring_count"] for p in sample_pairs]
    if len(set(diffs)) < 2 or len(set(counts)) < 2:
        return None  # no variation in one of the two variables -- correlation is undefined
    rho, _ = spearmanr(diffs, counts)  # scipy handles ties via average ranks; a naive
    # double-argsort (an earlier version's approach, caught by Codex review) assigns ties distinct
    # arbitrary ranks instead, which is wrong whenever offspring_count repeats -- as it does here,
    # almost always (small integer counts).
    return float(rho)


def _episode_samples(pairs, rng, n_boot=2000):
    """Resamples whole EPISODES with replacement (not individual pairs), pooling all of a resampled
    episode's pairs together each replicate -- the fix for pair-level resampling (Codex review):
    pairs from the same episode share ecology, population size, and policy, so they are not
    independent draws, and resampling pairs directly understates the true uncertainty."""
    episodes = sorted(set(p["episode"] for p in pairs))
    by_episode = {ep: [p for p in pairs if p["episode"] == ep] for ep in episodes}
    for _ in range(n_boot):
        sampled_episodes = rng.choice(episodes, size=len(episodes), replace=True)
        yield [p for ep in sampled_episodes for p in by_episode[ep]]


def _episode_cluster_bootstrap(pairs, statistic_fn, rng, n_boot=2000):
    """Recomputes statistic_fn fresh on each episode-cluster-resampled pooled sample from
    _episode_samples. Replicates where statistic_fn returns None (e.g. a degenerate median split
    with an empty side, or no variation to correlate) are discarded."""
    boots = []
    for sample in _episode_samples(pairs, rng, n_boot):
        if not sample:
            continue
        value = statistic_fn(sample)
        if value is not None:
            boots.append(value)
    return boots


def summarize(label, pairs, rng):
    lines = []
    n = len(pairs)
    if n < 4:
        lines.append(f"{label}: only {n} scoreable reproducing pairs (that reproduced >= once) -- too few to compare")
        return lines

    point_diff, median_diff, n_high, n_low = _median_split_diff(pairs)
    lines.append(
        f"{label}: n_pairs={n} (each reproduced >= 1 time), median differentiation={median_diff:.3f}, "
        f"n_high={n_high}, n_low={n_low}"
    )
    if point_diff is not None:
        boots = _episode_cluster_bootstrap(
            pairs, lambda s: _median_split_diff(s)[0], rng
        )
        if len(boots) >= 100:  # enough surviving replicates for a meaningful percentile CI
            lo, hi = np.percentile(boots, [2.5, 97.5])
            lines.append(
                f"    high-minus-low mean-offspring diff = {point_diff:+.3f} [{lo:+.3f},{hi:+.3f}] "
                f"(episode-cluster bootstrap, {len(boots)}/{2000} replicates usable)"
            )
        else:
            lines.append(f"    high-minus-low mean-offspring diff = {point_diff:+.3f} (too few usable bootstrap replicates for a CI)")
    else:
        lines.append("    high-minus-low diff = n/a (one side of the median split is empty)")

    rho = _spearman(pairs)
    if rho is not None:
        rho_boots = _episode_cluster_bootstrap(pairs, _spearman, rng)
        if len(rho_boots) >= 100:
            lo, hi = np.percentile(rho_boots, [2.5, 97.5])
            lines.append(
                f"    Spearman rank correlation (differentiation vs offspring count) = {rho:+.3f} "
                f"[{lo:+.3f},{hi:+.3f}] (episode-cluster bootstrap, tie-aware, {len(rho_boots)}/2000 replicates usable)"
            )
        else:
            lines.append(f"    Spearman rank correlation = {rho:+.3f} (too few usable bootstrap replicates for a CI)")
    else:
        lines.append("    Spearman rank correlation = n/a (no variation in differentiation or offspring count)")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=29, help="checkpoint index (10 iterations per index step +1)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--out", type=str, default=None, help="optional path for a JSON dump of the summary lines")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    runs = [(r.split("=", 1)[0], os.path.expanduser(r.split("=", 1)[1])) for r in args.run]
    output = []

    def emit(line):
        print(line, flush=True)
        output.append(line)

    for label, path in runs:
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        checkpoint = os.path.join(trial, f"checkpoint_{args.checkpoint:06d}")
        modules = load_modules(checkpoint)
        pairs = run_episodes(config, modules, args.episodes, args.seed)
        for line in summarize(f"{label} ckpt{args.checkpoint}", pairs, rng):
            emit(line)
        emit("")
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(output, f, indent=1)


if __name__ == "__main__":
    main()
