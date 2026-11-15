"""Measure whether learned-gift policies select ``give`` where it can help.

For each predator decision this reports the policy probability of action 9 and,
in stochastic rollouts, actual successful transfers.  An *eligible* state means
the recorded mate is alive, within gift range, positive-energy, below the
predator reproduction threshold, and the giver can retain positive energy.
The comparison state is every other predator decision.  Thus this does not
mistake a policy's general preference for the extra no-move action for
mate-directed provisioning.
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
    load_modules,
    module_probs,
    policy_of,
)

SEXES = ("predator_male", "predator_female")


def sex_of(agent):
    return "predator_male" if "predator_male" in agent else "predator_female"


def eligible(env, agent):
    """Whether give would transfer if selected now (pre-movement positions)."""
    mate = env.agent_mate.get(agent)
    if mate is None or mate not in env.predator_positions:
        return False
    if env.agent_energies[mate] <= 0 or env.agent_energies[mate] >= env.predator_creation_energy_threshold:
        return False
    if env.agent_energies[agent] - env.learned_gift_amount <= 0:
        return False
    a, b = env.predator_positions[agent], env.predator_positions[mate]
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= env.predator_gift_range


def summarize(values, rng, n_boot=2000):
    """Episode means then an episode bootstrap; undefined episodes are omitted."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), int(len(x))


def rollout(config, modules, episodes, seed0):
    env = make_env(config)
    if env.give_action_id is None:
        raise ValueError("run has no learned-gift action")
    rng = np.random.default_rng(seed0)
    per_episode = []
    for ep in range(episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations)
        # sums/counts are separately held for eligible/ineligible decisions.
        stats = {sex: {k: [0.0, 0] for k in ("eligible", "other")} for sex in SEXES}
        pred_steps = successful = attempted = 0
        while True:
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                probs = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    if "predator" in agent:
                        bucket = "eligible" if eligible(env, agent) else "other"
                        slot = stats[sex_of(agent)][bucket]
                        slot[0] += float(probs[i, env.give_action_id])
                        slot[1] += 1
                        pred_steps += 1
                    actions[agent] = int(rng.choice(len(probs[i]), p=probs[i]))
            observations, _, term, trunc, _ = env.step(actions)
            attempted += len([a for a in actions if actions[a] == env.give_action_id])
            successful += len(env.learned_gifts_this_step)
            active = [a for a in observations if not term.get(a, False) and not trunc.get(a, False)]
            if term.get("__all__") or trunc.get("__all__"):
                break
        row = {"gift_events_per1000_pred_steps": 1000 * successful / pred_steps if pred_steps else np.nan,
               "gift_attempts_per1000_pred_steps": 1000 * attempted / pred_steps if pred_steps else np.nan}
        for sex in SEXES:
            e_sum, e_n = stats[sex]["eligible"]
            o_sum, o_n = stats[sex]["other"]
            row[f"{sex}_p_eligible"] = e_sum / e_n if e_n else np.nan
            row[f"{sex}_p_other"] = o_sum / o_n if o_n else np.nan
            row[f"{sex}_gap"] = row[f"{sex}_p_eligible"] - row[f"{sex}_p_other"] if e_n and o_n else np.nan
            row[f"{sex}_eligible_states"] = e_n
        per_episode.append(row)
    return per_episode


def render(label, rows, rng):
    keys = ["gift_events_per1000_pred_steps", "gift_attempts_per1000_pred_steps"]
    for sex in SEXES:
        keys += [f"{sex}_p_eligible", f"{sex}_p_other", f"{sex}_gap", f"{sex}_eligible_states"]
    result, lines = {}, [f"{label}: {len(rows)} episodes"]
    for key in keys:
        val = summarize([r[key] for r in rows], rng)
        result[key] = val
        if val is None:
            lines.append(f"    {key:42}: n/a")
        else:
            mean, lo, hi, n = val
            lines.append(f"    {key:42}: {mean:.4f} [{lo:.4f},{hi:.4f}] (episodes {n})")
    return lines, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=run-dir")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out, rng = {}, np.random.default_rng(0)
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trials = sorted(glob.glob(os.path.join(path, "*PPO_*/")))
        if len(trials) != 1:
            raise SystemExit(f"expected one trial under {path}, found {len(trials)}")
        modules = load_modules(os.path.join(trials[0], f"checkpoint_{args.checkpoint:06d}"))
        rows = rollout(config, modules, args.episodes, args.seed)
        lines, out[label] = render(label, rows, rng)
        print("\n".join(lines), flush=True)
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
