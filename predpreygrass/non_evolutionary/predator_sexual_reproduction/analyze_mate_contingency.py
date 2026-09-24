"""
Behavioral-contingency test for coordination (as opposed to parallel, fixed individual
specialization): does a predator's prey/fruit approach behavior depend on where its recorded mate
currently is, rather than being a fixed per-sex tendency?

A "mate" here is `env.agent_mate` -- the reproductive-partner bond recorded the first time two
specific predators successfully reproduce together (predpreygrass_rllib_env.py's Step 5b; serial
monogamy, so it can be overwritten by a later re-mating). It is NOT the same thing as
mate_search_radius, which only governs *eligibility* to reproduce in the first place -- an agent
that has never reproduced has no recorded mate and is excluded from the near/away comparison below
(counted separately, under "none", for transparency).

At every step, a live predator's state is bucketed by its recorded mate's status as of the START of
that step (a second Codex review noted "real-time" slightly overstates this: the environment defers
removing a just-terminated agent from agent_positions until the START of the following step(), so a
mate that died on the PREVIOUS step can still be classified "near"/"far" for one extra step before
this bucketing catches up and reclassifies it "dead" -- not a bucket conflation, just a one-step lag):
  near      -- mate alive, within --near-radius (Chebyshev) of the agent
  far       -- mate alive, farther than --near-radius
  dead      -- mate was recorded but is no longer alive
  abandoned -- agent has reproduced before (is in env.has_reproduced) but has no CURRENT agent_mate
               entry -- its former partner re-mated with someone else, which severs the abandoned
               side's entry too (serial monogamy's cleanup in Step 5b). Conceptually the same "no
               partner currently present" situation as far/dead, just via reassignment rather than
               distance or death -- included in "away" below, not a separate excluded population.
               (An earlier version of this script conflated this with "virgin" under a single "none"
               bucket, which a Codex review caught: that silently dropped real behavioral data from
               the near-vs-away contrast and mislabeled it as "never reproduced".)
  virgin    -- agent has never reproduced at all (not in env.has_reproduced); reported for
               transparency, excluded from the contrast (no baseline partnered behavior to compare)
  away      -- far + dead + abandoned combined: the primary contrast against "near"

The same approach_bias / adjacent-step-onto metrics as analyze_prey_approach_from_checkpoint.py are
computed per bucket (imported from there, unchanged), then a near-vs-away DIFFERENCE is bootstrapped
per episode (paired: the same resampled episodes feed both ratios), which is the actual test of
contingency -- two similar-looking point estimates are not evidence on their own without it.

This tests behavioral contingency on a partner's PROXIMITY only, not on what the partner is
currently doing (e.g. "is he mid-hunt") -- a coarser, cheaper first pass. It also cannot show
communicated intent: as far as this environment's action/observation space goes, there is no
explicit signaling channel between agents (movement and foraging actions only; the grid exposes
nearby predator occupancy/energy, but nothing that identifies WHICH nearby predator is the recorded
mate specifically). A near-vs-away difference, if found, is evidence that approach behavior is
ASSOCIATED with the recorded mate's proximity -- an observational correlation, not an experimental
manipulation, and confoundable by location, target configuration, own energy/history, and survival
selection. It does not by itself establish mate recognition, causality, coordination, or
communication (a Codex review flagged the original wording here, "evidence of a partner-contingent
policy," as too strong; this paragraph is the corrected version).

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_mate_contingency \
      --run CONTROL_s42=~/simulation_results/ray_results/PPO_FIXED_PREY_DENSITY_CONTROL_SEED42 \
      --checkpoints 29 --episodes 30
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    action_geometry,
    load_modules,
    module_probs,
    new_stats,
    policy_of,
    record,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analysis_env import make_env

SEXES = ("predator_male", "predator_female")
TARGETS = ("prey", "fruit")
RAW_BUCKETS = ("near", "far", "dead", "abandoned", "virgin")
AWAY_BUCKETS = ("far", "dead", "abandoned")


def mate_bucket(env, agent, near_radius):
    mate = env.agent_mate.get(agent)
    if mate is None:
        # No CURRENT recorded mate -- but this is two very different populations (Codex review):
        # never having reproduced at all, versus having reproduced before and been left without a
        # mate entry because the former partner re-mated with someone else (see the module
        # docstring's "abandoned" bucket). Distinguish them via has_reproduced rather than lumping
        # both into one bucket.
        return "virgin" if agent not in env.has_reproduced else "abandoned"
    if mate not in env.agent_positions:
        return "dead"
    pos = env.agent_positions[agent]
    mate_pos = env.agent_positions[mate]
    dist = max(abs(pos[0] - mate_pos[0]), abs(pos[1] - mate_pos[1]))
    return "near" if dist <= near_radius else "far"


def run_episodes(env_config, modules, n_episodes, seed0, near_radius):
    env = make_env(env_config)
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    sample_rng = np.random.default_rng(seed0)
    per_episode = []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        # Five keys per (sex, target): four raw buckets plus "away" (far+dead combined), which is
        # the primary near-vs-away contrast computed in summarize().
        stats = {(sex, tgt, bucket): new_stats() for sex in SEXES for tgt in TARGETS for bucket in (*RAW_BUCKETS, "away")}
        while True:
            targets = {
                "prey": np.array(list(env.prey_positions.values()), dtype=int).reshape(-1, 2),
                "fruit": np.array(
                    [p for f, p in env.fruit_positions.items() if env.fruit_energies[f] > 0], dtype=int
                ).reshape(-1, 2),
            }
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                rows = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    is_predator = pid != "prey_policy"
                    probs = rows[i]
                    if is_predator:
                        pos = np.array(env.agent_positions[agent], dtype=int)
                        geo = {t: action_geometry(pos, targets[t], moves, env.grid_size, offset) for t in TARGETS}
                        sex = "predator_male" if pid == "predator_male_policy" else "predator_female"
                        bucket = mate_bucket(env, agent, near_radius)
                        for t in TARGETS:
                            if geo[t] is not None:
                                record(stats[(sex, t, bucket)], geo[t], probs)
                                if bucket in AWAY_BUCKETS:
                                    record(stats[(sex, t, "away")], geo[t], probs)
                    actions[agent] = int(sample_rng.choice(env.num_actions, p=probs))
            observations, _, terminations, truncations, _ = env.step(actions)
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        per_episode.append(stats)
    return per_episode


def bootstrap_ratio(numerators, denominators, rng, n_boot=2000):
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = num.sum() / den.sum()
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(1)
    ok = d > 0
    boots = num[idx].sum(1)[ok] / d[ok]
    return point, *np.percentile(boots, [2.5, 97.5])


def bootstrap_diff_of_ratios(num_a, den_a, num_b, den_b, rng, n_boot=2000):
    """Paired episode-level bootstrap of ratio_a - ratio_b (the SAME resampled episode indices
    feed both ratios each replicate, since near/away/etc. are measured within the same episodes)."""
    num_a, den_a = np.asarray(num_a, float), np.asarray(den_a, float)
    num_b, den_b = np.asarray(num_b, float), np.asarray(den_b, float)
    if den_a.sum() == 0 or den_b.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    idx = rng.integers(0, len(num_a), size=(n_boot, len(num_a)))
    da, db = den_a[idx].sum(1), den_b[idx].sum(1)
    ok = (da > 0) & (db > 0)
    boots = num_a[idx].sum(1)[ok] / da[ok] - num_b[idx].sum(1)[ok] / db[ok]
    if len(boots) == 0:
        return point, float("nan"), float("nan")
    return point, *np.percentile(boots, [2.5, 97.5])


def _diff_contrast(label_prefix, per_episode, sex, tgt, bucket_a, bucket_b, rng):
    """near-minus-bucket_b diff, with the episode-overlap count Codex asked be reported alongside
    every such contrast (the bootstrap's real sample size is informative episodes, not decisions)."""
    rows_a = [ep[(sex, tgt, bucket_a)] for ep in per_episode]
    rows_b = [ep[(sex, tgt, bucket_b)] for ep in per_episode]
    n_a, n_b = sum(r["n"] for r in rows_a), sum(r["n"] for r in rows_b)
    n_both = sum(1 for ra, rb in zip(rows_a, rows_b) if ra["n"] > 0 and rb["n"] > 0)
    if not (n_a and n_b):
        return f" | {label_prefix} diff=n/a (one side empty)"
    diff, lo, hi = bootstrap_diff_of_ratios(
        [r["bias"] for r in rows_a], [r["n"] for r in rows_a],
        [r["bias"] for r in rows_b], [r["n"] for r in rows_b],
        rng,
    )
    return f" | {label_prefix} diff={diff:+.3f} [{lo:+.3f},{hi:+.3f}] ({n_both}/{len(per_episode)} episodes have both)"


def summarize(label, per_episode, rng):
    lines = []
    for sex in SEXES:
        for tgt in TARGETS:
            n_by_bucket = {b: sum(ep[(sex, tgt, b)]["n"] for ep in per_episode) for b in (*RAW_BUCKETS, "away")}
            if sum(n_by_bucket[b] for b in RAW_BUCKETS) == 0:
                lines.append(f"{label:22} {sex[9:]:6} {tgt:5} no visible-target decisions")
                continue
            parts = []
            for b in RAW_BUCKETS:
                rows = [ep[(sex, tgt, b)] for ep in per_episode]
                n = sum(r["n"] for r in rows)
                if n == 0:
                    parts.append(f"{b}=n/a")
                    continue
                bias, lo, hi = bootstrap_ratio([r["bias"] for r in rows], [r["n"] for r in rows], rng)
                parts.append(f"{b} n={n:5d} bias={bias:+.3f} [{lo:+.3f},{hi:+.3f}]")
            contrast = _diff_contrast("near-away", per_episode, sex, tgt, "near", "away", rng)
            # Compensation test: "away" pools far/dead/abandoned together (Codex flagged this doesn't
            # isolate bereavement specifically). A dedicated near-vs-dead contrast asks the narrower
            # question directly -- does she forage more once her mate has actually died, not just
            # wandered off or been reassigned -- using the same paired-bootstrap machinery.
            contrast += _diff_contrast("near-dead", per_episode, sex, tgt, "near", "dead", rng)
            lines.append(f"{label:22} {sex[9:]:6} {tgt:5} " + " | ".join(parts) + contrast)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[29], help="checkpoint indices")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--near-radius", type=int, default=3,
        help="Chebyshev distance at or below which the mate counts as 'near' (default 3, matching "
             "this module's default mate_search_radius -- not the same field, just the same scale).",
    )
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
        for k in args.checkpoints:
            checkpoint = os.path.join(trial, f"checkpoint_{k:06d}")
            modules = load_modules(checkpoint)
            per_episode = run_episodes(config, modules, args.episodes, args.seed, args.near_radius)
            n_virgin = sum(ep[(sex, "prey", "virgin")]["n"] for ep in per_episode for sex in SEXES)
            n_abandoned = sum(ep[(sex, "prey", "abandoned")]["n"] for ep in per_episode for sex in SEXES)
            n_total = sum(
                ep[(sex, "prey", b)]["n"] for ep in per_episode for sex in SEXES for b in RAW_BUCKETS
            )
            emit(
                f"# {label} ckpt{k}: {n_virgin}/{n_total} prey-decisions from never-reproduced agents "
                f"(bucket 'virgin', excluded from near-away); {n_abandoned}/{n_total} from agents whose "
                f"former mate has since re-mated elsewhere (bucket 'abandoned', included IN 'away')"
            )
            for line in summarize(f"{label} ckpt{k}", per_episode, rng):
                emit(line)
            emit("")
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(output, f, indent=1)


if __name__ == "__main__":
    main()
