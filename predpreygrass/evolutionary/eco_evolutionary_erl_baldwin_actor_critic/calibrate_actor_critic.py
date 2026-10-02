"""Matched-seed calibration and stability runner for linear actor-critic."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl as baseline_config
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld as ReinforceWorld
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.config import config_actor_critic
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.world import ActorCriticWorld


def _float_list(text): return tuple(float(x) for x in text.split(","))
def _int_list(text): return tuple(int(x) for x in text.split(","))


def treatment_specs(alphas, beta_multipliers, gammas):
    yield {"name": "reinforce", "algorithm": "reinforce", "strategy": "ERL"}
    yield {"name": "evolution_only", "algorithm": "reinforce", "strategy": "E"}
    for alpha in alphas:
        for multiplier in beta_multipliers:
            for gamma in gammas:
                yield {"name": f"ac_a{alpha:g}_b{alpha*multiplier:g}_g{gamma:g}",
                       "algorithm": "actor_critic", "strategy": "ERL", "alpha": alpha,
                       "beta": alpha * multiplier, "gamma": gamma}


def run_treatment(spec, seed, steps, diagnostic_every=50):
    if spec["algorithm"] == "actor_critic":
        cfg = dict(config_actor_critic)
        cfg.update(seed=seed, strategy=spec["strategy"], actor_alpha=spec["alpha"],
                   critic_beta=spec["beta"], actor_critic_gamma=spec["gamma"])
        world = ActorCriticWorld(cfg, np.random.default_rng(seed))
    else:
        cfg = dict(baseline_config); cfg.update(seed=seed, strategy=spec["strategy"])
        world = ReinforceWorld(cfg, np.random.default_rng(seed))
    populations = [world.population_counts()["agent"]]
    carnivores = [world.population_counts()["carnivore"]]
    extinction_step = None; failure = ""
    td_max = actor_max = critic_max = critic_value_max = float("nan")
    try:
        for completed in range(1, steps + 1):
            world.step(); counts = world.population_counts()
            populations.append(counts["agent"]); carnivores.append(counts["carnivore"])
            if isinstance(world, ActorCriticWorld) and (completed % diagnostic_every == 0 or counts["agent"] == 0):
                alive = [a for a in world.agents if a.alive]
                if alive:
                    td_max = float(np.nanmax([td_max, max(abs(a.last_td_error) for a in alive)]))
                    actor_max = float(np.nanmax([actor_max, max(a.last_actor_update_norm for a in alive)]))
                    critic_max = float(np.nanmax([critic_max, max(a.last_critic_update_norm for a in alive)]))
                    critic_value_max = float(np.nanmax([critic_value_max, max(max(np.max(np.abs(a.critic_weights)), abs(a.critic_bias[0])) for a in alive)]))
            if counts["agent"] == 0:
                extinction_step = world.current_step; break
    except (FloatingPointError, OverflowError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if extinction_step is not None:
        populations += [0] * (steps + 1 - len(populations))
    valid = not failure
    alive = [a for a in world.agents if a.alive]
    displacement = float(np.mean([np.linalg.norm(np.r_[
        (a.action_weights-a.genome.action_weights).ravel(), a.action_bias-a.genome.action_bias])
        for a in alive])) if valid and alive else float("nan")
    return {"treatment": spec["name"], "algorithm": spec["algorithm"],
            "alpha": spec.get("alpha", float("nan")), "beta": spec.get("beta", float("nan")),
            "gamma": spec.get("gamma", float("nan")), "seed": seed, "requested_steps": steps,
            "completed_steps": min(steps, world.current_step),
            "extinction_step": extinction_step if extinction_step is not None else "",
            "population_end": populations[-1] if valid else float("nan"),
            "population_mean": float(np.mean(populations)) if valid else float("nan"),
            "population_peak": max(populations) if valid else float("nan"),
            "carnivore_peak": max(carnivores) if valid else float("nan"),
            "births": world._next_agent_id - cfg["n_initial_agents"],
            "learning_displacement": displacement, "td_absmax_seen": td_max,
            "actor_update_max_seen": actor_max, "critic_update_max_seen": critic_max,
            "critic_parameter_absmax_seen": critic_value_max, "numerical_failure": failure}


def summarize(rows):
    output = []
    for treatment in dict.fromkeys(row["treatment"] for row in rows):
        group = [r for r in rows if r["treatment"] == treatment and not r["numerical_failure"]]
        output.append({"treatment": treatment, "n": len(group),
                       "extinctions": sum(r["extinction_step"] != "" for r in group),
                       "population_end_mean": float(np.mean([r["population_end"] for r in group])),
                       "population_mean_mean": float(np.mean([r["population_mean"] for r in group])),
                       "population_peak_mean": float(np.mean([r["population_peak"] for r in group])),
                       "carnivore_peak_mean": float(np.mean([r["carnivore_peak"] for r in group]))})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=_int_list, default=(41, 42, 43))
    parser.add_argument("--alphas", type=_float_list, default=(0.002, 0.01, 0.05))
    parser.add_argument("--beta-multipliers", type=_float_list, default=(2.0, 5.0))
    parser.add_argument("--gammas", type=_float_list, default=(0.0, 0.9))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for spec in treatment_specs(args.alphas, args.beta_multipliers, args.gammas):
            row = run_treatment(spec, seed, args.steps); rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__": main()
