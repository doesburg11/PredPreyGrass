"""Matched-seed development calibration for the SARSA Baldwin variant.

This is deliberately a short population screen, not the confirmatory study.
It compares the validated immediate policy-gradient learner, evolution-only,
and a small SARSA(alpha, lambda) grid under identical ecology configuration.

Example:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.calibrate_sarsa \
        --steps 300 --seeds 41,42,43 --out calibration.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import (
    config_erl as baseline_config,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import (
    ErlWorld as ReinforceWorld,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.config import (
    config_sarsa,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.world import SarsaWorld


DEFAULT_ALPHAS = (0.001, 0.01, 0.05)
DEFAULT_LAMBDAS = (0.0, 0.5, 0.9)


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def treatment_specs(alphas=DEFAULT_ALPHAS, lambdas=DEFAULT_LAMBDAS):
    yield {"name": "reinforce", "algorithm": "reinforce", "strategy": "ERL"}
    yield {"name": "evolution_only", "algorithm": "reinforce", "strategy": "E"}
    for alpha in alphas:
        for trace_lambda in lambdas:
            yield {
                "name": f"sarsa_a{alpha:g}_l{trace_lambda:g}",
                "algorithm": "sarsa",
                "strategy": "ERL",
                "alpha": alpha,
                "lambda": trace_lambda,
            }


def _learning_displacement(world) -> float:
    agents = [agent for agent in world.agents if agent.alive]
    if not agents:
        return float("nan")
    return float(
        np.mean(
            [
                np.linalg.norm(
                    np.concatenate(
                        [
                            (agent.action_weights - agent.genome.action_weights).ravel(),
                            agent.action_bias - agent.genome.action_bias,
                        ]
                    )
                )
                for agent in agents
            ]
        )
    )


def _sarsa_diagnostics(world) -> tuple[float, float, float]:
    if not isinstance(world, SarsaWorld) or not world.agents:
        return float("nan"), float("nan"), float("nan")
    agents = [agent for agent in world.agents if agent.alive]
    if not agents:
        return float("nan"), float("nan"), float("nan")
    td_absmean = float(np.mean([abs(agent.last_td_error) for agent in agents]))
    q_absmax = float(
        max(
            max(np.max(np.abs(agent.action_weights)), np.max(np.abs(agent.action_bias)))
            for agent in agents
        )
    )
    trace_absmax = float(
        max(
            max(
                np.max(np.abs(agent.eligibility_weights)),
                np.max(np.abs(agent.eligibility_bias)),
            )
            for agent in agents
        )
    )
    return td_absmean, q_absmax, trace_absmax


def _trajectory_mean(populations: list[int], requested_steps: int, extinct: bool) -> float:
    """Mean over the common step-0..requested_steps horizon.

    Extinction is absorbing, so missing post-extinction populations are zero.
    Numerical failures are handled separately and never call this helper.
    """
    if extinct:
        populations = populations + [0] * (requested_steps + 1 - len(populations))
    if len(populations) != requested_steps + 1:
        raise ValueError("non-extinct trajectory does not cover the requested horizon")
    return float(np.mean(populations))


def run_treatment(spec: dict, seed: int, steps: int, diagnostic_every: int = 50) -> dict:
    if spec["algorithm"] == "sarsa":
        cfg = dict(config_sarsa)
        cfg.update(
            seed=seed,
            strategy=spec["strategy"],
            sarsa_alpha=spec["alpha"],
            sarsa_lambda=spec["lambda"],
        )
        world = SarsaWorld(cfg, np.random.default_rng(seed))
    else:
        cfg = dict(baseline_config)
        cfg.update(seed=seed, strategy=spec["strategy"])
        world = ReinforceWorld(cfg, np.random.default_rng(seed))

    initial_counts = world.population_counts()
    populations = [initial_counts["agent"]]
    carnivore_populations = [initial_counts["carnivore"]]
    extinction_step = None
    numerical_failure = None
    completed_steps = 0
    q_absmax_seen = trace_absmax_seen = td_absmax_seen = float("nan")
    try:
        for _ in range(steps):
            world.step()
            completed_steps += 1
            counts = world.population_counts()
            population = counts["agent"]
            populations.append(population)
            carnivore_populations.append(counts["carnivore"])
            if isinstance(world, SarsaWorld) and (
                completed_steps % diagnostic_every == 0 or population == 0
            ):
                td_now, q_now, trace_now = _sarsa_diagnostics(world)
                td_absmax_seen = float(np.nanmax([td_absmax_seen, td_now]))
                q_absmax_seen = float(np.nanmax([q_absmax_seen, q_now]))
                trace_absmax_seen = float(np.nanmax([trace_absmax_seen, trace_now]))
            if population == 0:
                extinction_step = world.current_step
                break
    except (FloatingPointError, OverflowError) as exc:
        numerical_failure = f"{type(exc).__name__}: {exc}"

    valid = numerical_failure is None
    if valid:
        td_absmean, q_absmax, trace_absmax = _sarsa_diagnostics(world)
        population_end = populations[-1]
        population_mean = _trajectory_mean(populations, steps, extinction_step is not None)
        population_min = min(populations)
        learning_displacement = _learning_displacement(world)
        peak_population = max(populations)
        peak_step = populations.index(peak_population)
        post_peak_min = min(populations[peak_step:])
        carnivore_end = carnivore_populations[-1]
        # Unlike the adaptive-agent population trajectory, this series is not
        # zero-padded after extinction: post-extinction predator abundance is
        # undefined because the simulation stops. Treat this as an
        # observed-horizon diagnostic, not a cross-treatment endpoint.
        carnivore_mean = float(np.mean(carnivore_populations))
        carnivore_peak = max(carnivore_populations)
    else:
        td_absmean = q_absmax = trace_absmax = float("nan")
        population_end = population_mean = population_min = float("nan")
        learning_displacement = float("nan")
        peak_population = peak_step = post_peak_min = float("nan")
        carnivore_end = carnivore_mean = carnivore_peak = float("nan")
    births = world._next_agent_id - cfg["n_initial_agents"]
    deaths = cfg["n_initial_agents"] + births - populations[-1]
    return {
        "treatment": spec["name"],
        "algorithm": spec["algorithm"],
        "strategy": spec["strategy"],
        "alpha": spec.get("alpha", float("nan")),
        "lambda": spec.get("lambda", float("nan")),
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "extinction_step": extinction_step if extinction_step is not None else "",
        "population_end": population_end,
        "population_mean": population_mean,
        "population_min": population_min,
        "population_peak": peak_population,
        "population_peak_step": peak_step,
        "population_post_peak_min": post_peak_min,
        "population_cap_steps": sum(
            population >= cfg["max_population_cap"] for population in populations
        ),
        "births": births,
        "deaths": deaths,
        "carnivore_end": carnivore_end,
        "carnivore_mean": carnivore_mean,
        "carnivore_peak": carnivore_peak,
        "learning_displacement": learning_displacement,
        "td_absmean": td_absmean,
        "q_absmax": q_absmax,
        "trace_absmax": trace_absmax,
        "td_absmax_seen": td_absmax_seen,
        "q_absmax_seen": q_absmax_seen,
        "trace_absmax_seen": trace_absmax_seen,
        "numerical_failure": numerical_failure or "",
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for treatment in dict.fromkeys(row["treatment"] for row in rows):
        group = [row for row in rows if row["treatment"] == treatment]
        valid = [row for row in group if not row["numerical_failure"]]
        summaries.append(
            {
                "treatment": treatment,
                "n": len(group),
                "failures": sum(bool(row["numerical_failure"]) for row in group),
                "extinctions": sum(row["extinction_step"] != "" for row in group),
                "population_end_mean": float(np.mean([row["population_end"] for row in valid]))
                if valid else float("nan"),
                "population_mean_mean": float(np.mean([row["population_mean"] for row in valid]))
                if valid else float("nan"),
                "births_mean": float(np.mean([row["births"] for row in valid]))
                if valid else float("nan"),
                "deaths_mean": float(np.mean([row["deaths"] for row in valid]))
                if valid else float("nan"),
                "population_peak_mean": float(
                    np.mean([row["population_peak"] for row in valid])
                ) if valid else float("nan"),
                "carnivore_peak_mean": float(
                    np.mean([row["carnivore_peak"] for row in valid])
                ) if valid else float("nan"),
                "learning_displacement_mean": float(
                    np.nanmean([row["learning_displacement"] for row in valid])
                ) if valid else float("nan"),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=_parse_int_list, default=(41, 42, 43))
    parser.add_argument("--alphas", type=_parse_float_list, default=DEFAULT_ALPHAS)
    parser.add_argument("--lambdas", type=_parse_float_list, default=DEFAULT_LAMBDAS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.steps <= 0 or not args.seeds or not args.alphas or not args.lambdas:
        parser.error("steps, seeds, alphas, and lambdas must be non-empty/positive")

    rows = []
    specs = list(treatment_specs(args.alphas, args.lambdas))
    for seed in args.seeds:
        for spec in specs:
            row = run_treatment(spec, seed, args.steps)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = summarize(rows)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"summary": summaries, "csv": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
