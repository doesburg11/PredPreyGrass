"""
Step-0 screen for Iteration 19 (learned gift): does a scripted "give" rule pay off for the GIVER?

PPO only learns an action whose expected return for the actor is higher. Before training a learned gift,
this checks on already-trained policies (Iteration 7, PROP k = 0.5) whether giving raises the giver's own
return. The trained run's environment is rebuilt with learned_gift="on" and the mechanical male gift switched
off (male_gift_donation_rate = 0); the trained policies choose all actions except when the scripted rule fires.

Rule (males only): replace the policy's action with "give" when the male's recorded mate is alive within
predator_gift_range, his energy is >= --min-own, and her energy is < predator_creation_energy_threshold.

Arms, on the same episode seeds:
  NONE  nobody gives (rule never fires);
  HALF  a random half of males (assigned per male per episode, label = hash of episode seed and id) give,
        the rest never do (within-episode assigned-giver vs assigned-non-giver contrast);
  ALL   every male gives.
The same labels are computed in NONE and ALL, where they change nothing; the label contrast there is a
placebo for the label assignment.

Reported:
  HALF: assigned giver minus assigned non-giver, per male-step: own return, own births (as father);
        episode-cluster bootstrap. This is an intention-to-treat contrast under interference (gifts change the
        ecology non-givers live in), not a clean individual counterfactual; successful transfers per group are
        reported as the dose. The rule is checked before movement and the transfer after it, so some firings
        transfer nothing.
  Caveat: the policies were trained WITH the mechanical male gift, which is switched off here (in all arms).
  ALL vs NONE: births per 1000 predator-steps, predator return per predator-step, mean predators (group effect).

Usage:
  python -m predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.screen_gift_payoff \
      --run runs/PPO_PREDATOR_SEXUAL_REPRODUCTION_PROP_K05_MB1024_SEED42 --episodes 10 --out screen_s42.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analysis_env import make_env
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    load_modules,
    module_probs,
    policy_of,
)

ARMS = ("NONE", "HALF", "ALL")


def label(agent, episode_seed):
    """Random 50/50 assignment, fixed per (episode seed, male id) so it is identical across arms."""
    return bool(np.random.default_rng([episode_seed, int(agent.rsplit("_", 1)[1])]).random() < 0.5)


def is_giver(agent, arm, episode_seed):
    if arm == "NONE" or "predator_male" not in agent:
        return False
    if arm == "ALL":
        return True
    return label(agent, episode_seed)


def rule_fires(env, agent, min_own):
    mate = env.agent_mate.get(agent)
    if mate is None or mate not in env.predator_positions:
        return False
    if env.agent_energies[agent] < min_own or env.agent_energies[mate] >= env.predator_creation_energy_threshold:
        return False
    p, q = env.agent_positions[agent], env.agent_positions[mate]
    return max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= env.predator_gift_range


def run_episode(env, modules, arm, seed, min_own):
    rng = np.random.default_rng(seed)
    observations, _ = env.reset(seed=seed)
    active = list(observations.keys())
    ret = defaultdict(float)
    steps = defaultdict(int)
    fathered = defaultdict(int)
    seen_children = set(env.agent_parents)
    fires = 0
    transfers = defaultdict(int)
    pred_steps = 0
    births = 0
    while True:
        actions = {}
        for pid in POLICY_IDS:
            agents = [a for a in active if policy_of(a) == pid]
            if not agents:
                continue
            rows = module_probs(modules[pid], [observations[a] for a in agents])
            for i, agent in enumerate(agents):
                actions[agent] = int(rng.choice(rows.shape[1], p=rows[i]))
                if "predator" in agent:
                    steps[agent] += 1
                    pred_steps += 1
                if is_giver(agent, arm, seed) and rule_fires(env, agent, min_own):
                    actions[agent] = env.give_action_id
                    fires += 1
        observations, rewards, terminations, truncations, _ = env.step(actions)
        for g in env.learned_gifts_this_step:
            transfers[g] += 1
        for a, r in rewards.items():
            if "predator" in a:
                ret[a] += r
        for child in set(env.agent_parents) - seen_children:
            seen_children.add(child)
            fathered[env.agent_parents[child][0]] += 1
            births += 1
        active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
        if terminations.get("__all__") or truncations.get("__all__"):
            break
    males = [a for a in steps if "predator_male" in a]

    def group(sel):
        s = sum(steps[a] for a in sel)
        return {
            "male_steps": s,
            "return": sum(ret[a] for a in sel),
            "births": sum(fathered[a] for a in sel),
            "transfers": sum(transfers[a] for a in sel),
            "n_males": len(sel),
        }

    return {
        "givers": group([a for a in males if label(a, seed)]),
        "nongivers": group([a for a in males if not label(a, seed)]),
        "pred_steps": pred_steps,
        "pred_return": sum(ret.values()),
        "births": births,
        "gift_events": env.learned_gift_events,
        "gift_attempts": env.learned_gift_attempts,
        "rule_fires": fires,
        "mean_predators": pred_steps / max(1, env.current_step),
    }


def boot(fn, episodes, rng, n_boot=2000):
    point = fn(episodes)
    idx = np.arange(len(episodes))
    draws = [fn([episodes[i] for i in rng.choice(idx, size=len(idx))]) for _ in range(n_boot)]
    draws = [d for d in draws if np.isfinite(d)]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rate(eps, grp, key, per=1.0):
    s = sum(e[grp]["male_steps"] for e in eps)
    return per * sum(e[grp][key] for e in eps) / s if s else float("nan")


def summarize(results, rng):
    out = {}
    half = results["HALF"]
    out["HALF_return_diff"] = boot(lambda e: rate(e, "givers", "return") - rate(e, "nongivers", "return"), half, rng)
    out["HALF_births_diff_per1000"] = boot(
        lambda e: rate(e, "givers", "births", 1000) - rate(e, "nongivers", "births", 1000), half, rng
    )
    for arm in ("NONE", "ALL"):
        out[f"{arm}_placebo_return_diff"] = boot(
            lambda e: rate(e, "givers", "return") - rate(e, "nongivers", "return"), results[arm], rng
        )
    out["HALF_giver_transfers_per1000"] = rate(half, "givers", "transfers", 1000)
    out["HALF_nongiver_transfers_per1000"] = rate(half, "nongivers", "transfers", 1000)
    out["HALF_giver_return"] = rate(half, "givers", "return")
    out["HALF_nongiver_return"] = rate(half, "nongivers", "return")
    out["HALF_giver_births_per1000"] = rate(half, "givers", "births", 1000)
    out["HALF_nongiver_births_per1000"] = rate(half, "nongivers", "births", 1000)
    for arm in ARMS:
        eps = results[arm]
        ps = sum(e["pred_steps"] for e in eps)
        out[f"{arm}_births_per1000_pred_steps"] = 1000 * sum(e["births"] for e in eps) / ps if ps else float("nan")
        out[f"{arm}_pred_return_per_step"] = sum(e["pred_return"] for e in eps) / ps if ps else float("nan")
        out[f"{arm}_mean_predators"] = float(np.mean([e["mean_predators"] for e in eps]))
        out[f"{arm}_gifts_per_episode"] = float(np.mean([e["gift_events"] for e in eps]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run dir with run_config.json (5-channel, 9-action policies)")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--min-own", type=float, default=6.0, help="male gives only with energy >= this")
    parser.add_argument("--amount", type=float, default=2.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = json.load(open(os.path.join(args.run, "run_config.json")))["config_env"]
    config = dict(config, learned_gift="on", learned_gift_amount=args.amount, male_gift_donation_rate=0.0)
    trials = sorted(glob.glob(os.path.join(args.run, "*PPO_*/")))
    if len(trials) != 1:
        raise SystemExit(f"expected exactly one trial dir in {args.run}, found {len(trials)}")
    modules = load_modules(os.path.join(trials[0], f"checkpoint_{args.checkpoint:06d}"))
    env = make_env(config)
    if config.get("mate_obs_channel") not in (None, False, "off"):
        raise SystemExit("screen expects 5-channel policies (mate_obs_channel off)")
    probe = module_probs(modules["predator_male_policy"], [env.observation_spaces["predator_male_0"].sample() * 0])
    if probe.shape[1] != env.num_actions:
        raise SystemExit(f"screen expects {env.num_actions}-action policies, got {probe.shape[1]}")
    results = {arm: [run_episode(env, modules, arm, args.seed + ep, args.min_own) for ep in range(args.episodes)]
               for arm in ARMS}
    summary = summarize(results, np.random.default_rng(0))
    print(f"run {args.run} ckpt {args.checkpoint} episodes {args.episodes} min_own {args.min_own} amount {args.amount}")
    for k, v in summary.items():
        if isinstance(v, tuple):
            print(f"  {k:32s}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
        else:
            print(f"  {k:32s}: {v:.4f}")
    if args.out:
        json.dump({"args": vars(args), "summary": summary, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
