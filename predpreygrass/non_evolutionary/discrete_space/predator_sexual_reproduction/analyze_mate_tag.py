"""
Cooperation test with the mate observation channel (RESULTS.md, Iteration 18).

Iteration 17 ended on a design limit: predators saw one anonymous predator layer, so no analysis could separate a
response to "my mate" from a response to "a predator nearby". Runs trained with `mate_obs_channel=True` (config_env.py)
give predators a sixth channel that marks the cell of their recorded mate (env.agent_mate) when it is alive and in view.
This script has two parts.

PART A -- tag edits (MATE runs only; policy-only, no training). On-policy states are sampled from rollouts in the env
the run was trained in (analysis_env.py), every --every steps, for predators whose mate is marked in their observation.
For each state the policy is queried on edited copies of the SAME observation (the predator layer is never touched; the
mate stays visible there as an anonymous predator in every condition):
  base     the observation as it was (mate marked)
  untag    the mate mark removed
  decoy    the mark moved from the mate to ONE other visible predator (one query per other visible predator, averaged
           per state; only states with at least one other predator in view)
Approach bias toward a point = E_uniform[d_after] - E_policy[d_after], d_after the Euclidean distance from each action's
destination (clipped at the grid edge, collisions ignored: this is the policy's preference, not the realised move) to
that point -- the same convention as the other scripts. Effects, paired per state:
  mate-tag effect   bias_to_mate(base) - bias_to_mate(untag): does marking the real mate change approach to it?
  decoy-tag effect  bias_to_decoy(decoy) - bias_to_decoy(untag): does the mark alone pull (or push) the policy, whoever
                    carries it? Same sign as the mate-tag effect = the policy responds to the mark, not to other cues
                    that happen to come with the real mate.
  food-tag effect   bias_to_target(base) - bias_to_target(untag) for prey and fruit (states with that target in view):
                    does the mate's marked presence change foraging?
  TV                mean total-variation distance between the base and untag action distributions (size of the
                    change, any direction).
Estimands: "pooled" weighs every tagged state equally (a random tagged decision; episodes with many tagged states
weigh more), "per-episode" averages within each episode first. 95% intervals: episode-cluster percentile bootstrap
(whole episodes resampled) for both.

PART B -- pair outcomes (any run, MATE or NOMATE, so both arms are measured identically). Per episode, from
stochastic rollouts in the trained env:
  post_birth_near_frac  over the --window (50) steps after each birth, the fraction in which the two parents are
                   still bonded, both alive and within predator_gift_range (Chebyshev 3 = the window offset, so near =
                   visible = gift possible); death or a severed bond counts as not near (event-aligned, primary);
                   births less than --window steps before the episode end are left out (censored, count reported)
  births_per_1000_pred_steps, gifts_per_1000_male_steps   rates, not raw counts (episode length and sex ratio differ)
  pair_near_frac / pair_dist   over predator-steps with a LIVING recorded mate (descriptive: post-treatment selection)
  paired_step_frac  coverage: predator-steps with a living mate / all predator-steps
  repeat_birth_frac  fraction of predator births whose two parents already had a child together earlier this episode
  births, steps, gifts per 1000 predator-steps, mean predator population   context
Bookkeeping is in PairTracker (unit-tested).
Reported as the mean over episodes with a 95% episode bootstrap interval. The cross-arm comparison (MATE vs NOMATE,
3 seeds each) is done on the per-run means, see RESULTS.md Iteration 18 for the pre-registered rules.

What this can and cannot show. Part A shows whether a trained policy USES the mark and in which direction (mate
recognition at the policy level). Part B shows whether pairs stay together, re-mate, provision and reproduce more when
the mark is available. Neither is cooperation on its own: in this env the energy transfers (male gift, parental care)
are mechanical, so the only learnable cooperative act is staying near (or seeking) the partner, and its benefit runs
through those fixed transfers and repeated mating. A positive Part A with a NEGATIVE pair_near_frac shift (e.g. males
avoiding a mate who costs them 30% of each nearby hunt) would be recognition used for avoidance, not cooperation.

Examples:
  python -m predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analyze_mate_tag \\
      --run MATE_s42=predpreygrass/non_evolutionary/discrete_space/predator_sexual_reproduction/runs/PPO_PREDATOR_SEXUAL_REPRODUCTION_MATE_K05_MB1024_SEED42
  python -m ...analyze_mate_tag --part B --run NOMATE_s42=.../PPO_PREDATOR_SEXUAL_REPRODUCTION_NOMATE_K05_MB1024_SEED42
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analysis_env import make_env
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    action_geometry,
    load_modules,
    module_probs,
    policy_of,
)

PRED_CH, MATE_CH = 1, 5
TARGETS = ("prey", "fruit")
SEXES = ("male", "female")
MIN_EPISODES = 5


# ---------------------------------------------------------------- geometry helpers

def d_after_to(point, pos, moves, grid_size):
    new = np.clip(pos + moves, 0, grid_size - 1)
    return np.sqrt(((new - point) ** 2).sum(1))


def approach_bias(probs, d_after):
    return float(d_after.mean() - probs @ d_after)


def untag(obs):
    out = obs.copy()
    out[MATE_CH] = 0.0
    return out


def retag(obs, cell):
    out = untag(obs)
    out[MATE_CH, cell[0], cell[1]] = 1.0
    return out


# ---------------------------------------------------------------- Part B bookkeeping

class PairTracker:
    """Per-episode pair bookkeeping. Call on_decision(env) once per env step BEFORE env.step (the state the policies
    act on) and after_step(env) right AFTER it. Uses only env.predator_positions, env.agent_mate, env.agent_parents
    and env.mate_gift_events, so it is identical for MATE, blank and off runs.

    Denominators (Codex review): pair_near_frac/pair_dist are over predator-steps with a LIVING recorded mate, a
    post-treatment population (who has a living mate depends on the policies), so they are descriptive;
    post_birth_near_frac is event-aligned and counts every one of the K steps after a birth, with a dead parent (or a
    severed bond) counted as not near, which avoids that selection; births whose window is cut off by the episode end
    are left out (censored; their number is reported); paired_step_frac reports the coverage. Rates use
    predator-steps (births) or male-steps (gifts, which only males give)."""

    def __init__(self, gift_range, window):
        self.gift_range, self.window = gift_range, window
        self.seen_children = None
        self.pair_counts = {}
        self.births = self.repeat_births = 0
        self.pred_steps = self.male_steps = 0
        self.pair_steps = self.pair_near = 0
        self.pair_dist_sum = 0.0
        self.open = []  # [male, female, steps_left, near_so_far]
        self.pb_seen = self.pb_near = self.pb_complete = 0
        self.steps = 0
        self.gifts_end = 0

    @staticmethod
    def _cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def on_decision(self, env):
        if self.seen_children is None:
            self.seen_children = set(env.agent_parents)
        pp = env.predator_positions
        for agent, pos in pp.items():
            self.pred_steps += 1
            self.male_steps += "predator_male" in agent
            mate = env.agent_mate.get(agent)
            if mate is not None and mate in pp:
                d = self._cheb(pos, pp[mate])
                self.pair_steps += 1
                self.pair_dist_sum += d
                self.pair_near += d <= self.gift_range
        still_open = []
        for male, female, left, near in self.open:
            bonded = env.agent_mate.get(male) == female and env.agent_mate.get(female) == male
            if bonded and male in pp and female in pp and self._cheb(pp[male], pp[female]) <= self.gift_range:
                near += 1
            if left > 1:
                still_open.append([male, female, left - 1, near])
            else:  # window complete: only complete windows count (a window cut off by the episode end is censored)
                self.pb_seen += self.window
                self.pb_near += near
                self.pb_complete += 1
        self.open = still_open

    def after_step(self, env):
        self.steps += 1
        for child in sorted(set(env.agent_parents) - self.seen_children):
            parents = tuple(env.agent_parents[child])
            self.births += 1
            self.repeat_births += self.pair_counts.get(parents, 0) > 0
            self.pair_counts[parents] = self.pair_counts.get(parents, 0) + 1
            self.seen_children.add(child)
            if self.window > 0:
                self.open.append([parents[0], parents[1], self.window, 0])
        self.gifts_end = env.mate_gift_events

    def result(self):
        def ratio(a, b, scale=1.0):
            return scale * a / b if b else np.nan

        return {
            "steps": self.steps,
            "births": self.births,
            "births_per_1000_pred_steps": ratio(self.births, self.pred_steps, 1000.0),
            "repeat_birth_frac": ratio(self.repeat_births, self.births),
            "pair_near_frac": ratio(self.pair_near, self.pair_steps),
            "pair_dist": ratio(self.pair_dist_sum, self.pair_steps),
            "paired_step_frac": ratio(self.pair_steps, self.pred_steps),
            "post_birth_near_frac": ratio(self.pb_near, self.pb_seen),
            "post_birth_windows": self.pb_complete,
            "post_birth_windows_censored": len(self.open),
            "gifts_per_1000_male_steps": ratio(self.gifts_end, self.male_steps, 1000.0),
            "gifts_per_1000_pred_steps": ratio(self.gifts_end, self.pred_steps, 1000.0),
            "mean_predators": ratio(self.pred_steps, self.steps),
        }


# ---------------------------------------------------------------- rollouts

def rollout(env_config, modules, n_episodes, seed0, every, max_states, collect_bank, window=50):
    """One pass that collects Part B per-episode outcomes and (if collect_bank) Part A states."""
    env = make_env(env_config)
    if collect_bank and not env.mate_obs_channel:
        raise SystemExit("Part A needs a run trained with mate_obs_channel=True")
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    rng = np.random.default_rng(seed0)
    bank = []
    outcomes = []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        tracker = PairTracker(env.predator_gift_range, window)
        step = 0
        while True:
            tracker.on_decision(env)
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
                    if collect_bank and step % every == 0 and pid != "prey_policy" and agent in env.predator_positions:
                        obs = observations[agent]
                        tag = np.argwhere(obs[MATE_CH] > 0)
                        if len(tag) == 1:
                            pos = np.array(env.agent_positions[agent], dtype=int)
                            mate_cell = tuple(int(v) for v in tag[0])
                            others = [
                                tuple(int(v) for v in c)
                                for c in np.argwhere(obs[PRED_CH] > 0)
                                if tuple(c) != (offset, offset) and tuple(c) != mate_cell
                            ]
                            to_grid = lambda c: pos + np.array(c) - offset  # noqa: E731
                            geo = {}
                            for t in TARGETS:
                                g = action_geometry(pos, targets[t], moves, env.grid_size, offset)
                                geo[t] = None if g is None else g[0]
                            bank.append(
                                {
                                    "obs": np.array(obs, dtype=np.float64),
                                    "episode": ep,
                                    "sex": "male" if "predator_male" in agent else "female",
                                    "d_mate": d_after_to(to_grid(mate_cell), pos, moves, env.grid_size),
                                    "decoys": [(c, d_after_to(to_grid(c), pos, moves, env.grid_size)) for c in others],
                                    "geo": geo,
                                }
                            )
                    actions[agent] = int(rng.choice(env.num_actions, p=rows[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            step += 1
            tracker.after_step(env)
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        outcomes.append(tracker.result())
    if len(bank) > max_states:
        keep = np.sort(rng.choice(len(bank), size=max_states, replace=False))
        bank = [bank[i] for i in keep]
    return bank, outcomes


# ---------------------------------------------------------------- Part A

def tag_effects(bank, module):
    """Per-state effects (NaN where not defined) for one sex's bank."""
    N = len(bank)
    base = module_probs(module, [b["obs"] for b in bank])
    unt = module_probs(module, [untag(b["obs"]) for b in bank])
    mate = np.array([approach_bias(base[i], b["d_mate"]) - approach_bias(unt[i], b["d_mate"]) for i, b in enumerate(bank)])
    tv = 0.5 * np.abs(base - unt).sum(1)
    food = {t: np.full(N, np.nan) for t in TARGETS}
    for i, b in enumerate(bank):
        for t in TARGETS:
            if b["geo"][t] is not None:
                food[t][i] = approach_bias(base[i], b["geo"][t]) - approach_bias(unt[i], b["geo"][t])
    decoy = np.full(N, np.nan)
    flat, owner = [], []
    for i, b in enumerate(bank):
        for c, d in b["decoys"]:
            flat.append(retag(b["obs"], c))
            owner.append((i, d))
    if flat:
        probs = module_probs(module, flat)
        acc, cnt = np.zeros(N), np.zeros(N)
        for k, (i, d) in enumerate(owner):
            acc[i] += approach_bias(probs[k], d) - approach_bias(unt[i], d)
            cnt[i] += 1
        ok = cnt > 0
        decoy[ok] = acc[ok] / cnt[ok]
    return {"mate": mate, "decoy": decoy, "tv": tv, **{f"food_{t}": food[t] for t in TARGETS}}


def cluster_mean(values, episodes, rng, n_boot=2000, per_episode=False):
    ok = ~np.isnan(values)
    x, ep = values[ok], episodes[ok]
    eps, inv = np.unique(ep, return_inverse=True)
    if len(eps) < MIN_EPISODES:
        return None
    sums = np.bincount(inv, weights=x, minlength=len(eps))
    cnts = np.bincount(inv, minlength=len(eps)).astype(float)
    idx = rng.integers(0, len(eps), size=(n_boot, len(eps)))
    if per_episode:  # every episode weighs the same, whatever its number of tagged states
        means = sums / cnts
        point, boots = means.mean(), means[idx].mean(1)
    else:  # every tagged state weighs the same (pooled; episodes with many tagged states weigh more)
        point, boots = sums.sum() / cnts.sum(), sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi), int(ok.sum()), len(eps)


def fmt(r):
    return "n/a (too few episodes)" if r is None else f"{r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}] (states {r[3]}, episodes {r[4]})"


def summarize_a(label, bank, modules, rng):
    lines, result = [], {}
    for sex in SEXES:
        sb = [b for b in bank if b["sex"] == sex]
        if len(sb) < 30:
            lines.append(f"{label} A {sex}: only {len(sb)} tagged states, skipped")
            continue
        eff = tag_effects(sb, modules[f"predator_{sex}_policy"])
        ep = np.array([b["episode"] for b in sb])
        result[sex] = {k: cluster_mean(v, ep, rng) for k, v in eff.items()}
        result[sex].update({f"{k}_episode_mean": cluster_mean(v, ep, rng, per_episode=True) for k, v in eff.items()})
        lines.append(f"{label} A {sex}: {len(sb)} tagged states from {len(np.unique(ep))} episodes")
        for k in ("mate", "decoy", "food_prey", "food_fruit", "tv"):
            name = k + "-tag effect" if k != "tv" else "TV(base, untag)"
            lines.append(f"    {name:22}: pooled {fmt(result[sex][k])}")
            lines.append(f"    {'':22}  per-episode {fmt(result[sex][k + '_episode_mean'])}")
    return lines, result


# ---------------------------------------------------------------- Part B

B_KEYS = (
    "post_birth_near_frac",
    "post_birth_windows",
    "post_birth_windows_censored",
    "births_per_1000_pred_steps",
    "gifts_per_1000_male_steps",
    "pair_near_frac",
    "pair_dist",
    "paired_step_frac",
    "repeat_birth_frac",
    "births",
    "steps",
    "gifts_per_1000_pred_steps",
    "mean_predators",
)


def summarize_b(label, outcomes, rng, n_boot=2000):
    lines, result = [f"{label} B: {len(outcomes)} episodes"], {}
    for k in B_KEYS:
        x = np.array([o[k] for o in outcomes], dtype=float)
        x = x[~np.isnan(x)]  # ratio undefined in an episode (e.g. no births): that episode is left out, count shown
        if len(x) == 0:
            result[k] = None
            lines.append(f"    {k:26}: n/a")
            continue
        boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        result[k] = (float(x.mean()), float(lo), float(hi), len(x))
        lines.append(f"    {k:26}: {x.mean():.4f} [{lo:.4f},{hi:.4f}] (episodes {len(x)})")
    return lines, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a run dir (has run_config.json)")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--every", type=int, default=2, help="Part A: sample every k-th step")
    parser.add_argument("--max-states", type=int, default=8000, help="Part A: cap on sampled states (both sexes)")
    parser.add_argument("--part", choices=["A", "B", "AB"], default=None, help="default: AB for MATE runs, B otherwise")
    parser.add_argument("--window", type=int, default=50, help="Part B: steps after each birth for post_birth_near_frac")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--out", default=None, help="optional JSON file for the summaries")
    args = parser.parse_args()
    if args.every < 1:
        raise SystemExit("--every must be >= 1")
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    dump = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        part = args.part or ("AB" if config.get("mate_obs_channel", False) in (True, "mate") else "B")
        trial = sorted(glob.glob(os.path.join(path, "*PPO_*/")))[0]  # PPO_* (older runs) or LocalPPO_*
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        bank, outcomes = rollout(
            config, modules, args.episodes, args.seed, args.every, args.max_states, "A" in part, args.window
        )
        entry = {"mate_obs_channel": config.get("mate_obs_channel", False), "checkpoint": args.checkpoint}
        if "A" in part:
            lines, entry["A"] = summarize_a(f"{label} ckpt{args.checkpoint}", bank, modules, rng)
            print("\n".join(lines), flush=True)
        lines, entry["B"] = summarize_b(f"{label} ckpt{args.checkpoint}", outcomes, rng)
        print("\n".join(lines) + "\n", flush=True)
        dump[label] = entry
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(dump, f, indent=1)


if __name__ == "__main__":
    main()
