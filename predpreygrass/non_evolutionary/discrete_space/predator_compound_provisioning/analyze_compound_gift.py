"""Evaluate learned transfer-bit choice in the 18-action compound-gift environment (RESULTS.md Iteration 20).

Per episode: realised successful gifts and gift-bit requests per 1000 predator-steps (also per sex); the policy
probability of the gift bit (sum over actions 9..17) in eligible states minus all other predator states, by sex;
and the pair outcomes of Iteration 18's PairTracker (births per 1000 predator-steps, post-birth proximity).
An eligible state: recorded mate alive within predator_gift_range, mate energy in (0, reproduction threshold)
and still positive after its homeostatic cost, and the giver would keep positive energy after its homeostatic cost
and the transfer when using the stay-put gift variant (Iteration 21 amendment, from the Iteration 20 Codex review:
the environment charges these costs before it validates a gift). Eligibility is read before the step (the state the
policy acts on); the transfer is resolved after movement, as in the environment."""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
)
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analyze_mate_tag import PairTracker
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
    mate = env.agent_mate.get(agent)
    if mate is None or mate not in env.predator_positions:
        return False
    cost = env.homeostatic_energy_cost_per_step_predator
    if env.agent_energies[mate] - cost <= 0 or env.agent_energies[mate] >= env.predator_creation_energy_threshold:
        return False
    if env.agent_energies[agent] - cost - env.learned_gift_amount <= 0:
        return False
    a, b = env.predator_positions[agent], env.predator_positions[mate]
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= env.predator_gift_range


def summarize(values, rng, n_boot=2000):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), int(len(x))


def rollout(config, modules, episodes, seed0, window=50):
    if config.get("compound_gift") not in ("on", "inert"):
        raise SystemExit("not a compound-gift run (config_env.compound_gift must be 'on' or 'inert')")
    env = CompoundGiftPredPreyGrass(config)
    rng, rows = np.random.default_rng(seed0), []
    for episode in range(episodes):
        observations, _ = env.reset(seed=seed0 + episode)
        active = list(observations)
        stats = {sex: {key: [0.0, 0] for key in ("eligible", "other")} for sex in SEXES}
        pred_steps = successful = attempts = 0
        sex_steps = {sex: 0 for sex in SEXES}
        sex_gifts = {sex: 0 for sex in SEXES}
        sex_requests = {sex: 0 for sex in SEXES}
        tracker = PairTracker(env.predator_gift_range, window)
        while True:
            tracker.on_decision(env)
            actions = {}
            for policy_id in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == policy_id]
                if not agents:
                    continue
                probs = module_probs(modules[policy_id], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    if "predator" in agent:
                        bucket = "eligible" if eligible(env, agent) else "other"
                        stats[sex_of(agent)][bucket][0] += float(probs[i, env.num_actions :].sum())
                        stats[sex_of(agent)][bucket][1] += 1
                        pred_steps += 1
                        sex_steps[sex_of(agent)] += 1
                    actions[agent] = int(rng.choice(len(probs[i]), p=probs[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            tracker.after_step(env)
            for a, act in actions.items():
                if "predator" in a and act >= env.num_actions:
                    attempts += 1
                    sex_requests[sex_of(a)] += 1
            successful += len(env.learned_gifts_this_step)
            for giver in env.learned_gifts_this_step:
                sex_gifts[sex_of(giver)] += 1
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        pairs = tracker.result()
        row = {
            "gift_events_per1000_pred_steps": 1000 * successful / pred_steps if pred_steps else np.nan,
            "gift_bit_attempts_per1000_pred_steps": 1000 * attempts / pred_steps if pred_steps else np.nan,
            "births_per_1000_pred_steps": pairs["births_per_1000_pred_steps"],
            "post_birth_near_frac": pairs["post_birth_near_frac"],
            "post_birth_windows_censored": pairs["post_birth_windows_censored"],
            "mean_predators": pairs["mean_predators"],
        }
        for sex in SEXES:
            row[f"{sex}_gifts_per1000_sex_steps"] = 1000 * sex_gifts[sex] / sex_steps[sex] if sex_steps[sex] else np.nan
            row[f"{sex}_requests_per1000_sex_steps"] = (
                1000 * sex_requests[sex] / sex_steps[sex] if sex_steps[sex] else np.nan
            )
        for sex in SEXES:
            e_sum, e_n = stats[sex]["eligible"]
            o_sum, o_n = stats[sex]["other"]
            row[f"{sex}_p_eligible"] = e_sum / e_n if e_n else np.nan
            row[f"{sex}_p_other"] = o_sum / o_n if o_n else np.nan
            row[f"{sex}_gap"] = row[f"{sex}_p_eligible"] - row[f"{sex}_p_other"] if e_n and o_n else np.nan
        rows.append(row)
    return rows


def render(label, rows, rng):
    keys = ["gift_events_per1000_pred_steps", "gift_bit_attempts_per1000_pred_steps", "births_per_1000_pred_steps",
            "post_birth_near_frac", "post_birth_windows_censored", "mean_predators"]
    for sex in SEXES:
        keys += [f"{sex}_p_eligible", f"{sex}_p_other", f"{sex}_gap", f"{sex}_gifts_per1000_sex_steps",
                 f"{sex}_requests_per1000_sex_steps"]
    out, lines = {}, [f"{label}: {len(rows)} episodes"]
    for key in keys:
        out[key] = summarize([r[key] for r in rows], rng)
        if out[key] is None:
            lines.append(f"    {key:46}: n/a")
        else:
            m, lo, hi, n = out[key]
            lines.append(f"    {key:46}: {m:.4f} [{lo:.4f},{hi:.4f}] (episodes {n})")
    return lines, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=run-dir")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(2)
    dump, rng = {}, np.random.default_rng(0)
    for spec in args.run:
        label, path = spec.split("=", 1)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trials = sorted(glob.glob(os.path.join(path, "*PPO_*/")))
        if len(trials) != 1:
            raise SystemExit(f"expected one trial under {path}, found {len(trials)}")
        modules = load_modules(os.path.join(trials[0], f"checkpoint_{args.checkpoint:06d}"))
        lines, dump[label] = render(label, rollout(config, modules, args.episodes, args.seed), rng)
        print("\n".join(lines), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dump, f, indent=1)


if __name__ == "__main__":
    main()
