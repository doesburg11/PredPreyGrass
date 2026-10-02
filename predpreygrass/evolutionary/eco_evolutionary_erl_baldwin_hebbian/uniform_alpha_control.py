"""Standardized lifetime control of genomic versus uniform-positive alpha.

Both modes receive the identical sequence of cardinal threat observations and
identical pre-generated action-sampling uniforms. Each trial has a defined
next-state visual consequence: moving directly away clears the threat signal,
moving toward leaves it unchanged, and a perpendicular move halves it. A
fixed evaluator values lower visual obstruction. This counterfactual assay
isolates alpha routing without treatment-dependent encounter denominators or
diverging ecology trajectories.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .genome import founder_genome
from .networks import action_probs, effective_action_weights, evaluate, hebbian_trace_update, routed_action_alpha
from .world import N_ACTIONS, OBS_DIM


def _sample_with_uniform(probs: np.ndarray, uniform: float) -> int:
    return min(int(np.searchsorted(np.cumsum(probs), uniform, side="right")), len(probs) - 1)


def _opposite(direction: int) -> int:
    return (1, 0, 3, 2)[direction]


def run_probe(seed: int, mode: str, steps: int, eta: float = 0.1, trace_clip: float = 1.0) -> dict:
    genome = founder_genome(OBS_DIM, N_ACTIONS, np.random.default_rng(seed), init_std=0.5)
    alpha_weights, alpha_bias = routed_action_alpha(genome.action_alpha_weights, genome.action_alpha_bias, mode)
    trace = np.zeros_like(genome.action_weights)
    bias_trace = np.zeros_like(genome.action_bias)
    evaluator_weights = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    stimulus_rng = np.random.default_rng(seed + 1_000_003)
    threat_directions = stimulus_rng.integers(0, N_ACTIONS, size=steps)
    action_uniforms = stimulus_rng.random(steps)
    away_actions = 0
    argmax_different = 0
    initial_argmax_by_threat = {}

    for threat_direction, uniform in zip(threat_directions, action_uniforms):
        threat_direction = int(threat_direction)
        obs = np.zeros(OBS_DIM)
        obs[threat_direction] = 1.0
        obs[5:] = 1.0
        weights, bias = effective_action_weights(
            genome.action_weights, genome.action_bias, alpha_weights, alpha_bias, trace, bias_trace
        )
        probs = action_probs(obs, weights, bias)
        action = _sample_with_uniform(probs, float(uniform))
        argmax = int(np.argmax(probs))
        initial_argmax_by_threat.setdefault(threat_direction, argmax)
        argmax_different += argmax != initial_argmax_by_threat[threat_direction]

        away = _opposite(threat_direction)
        away_actions += action == away
        next_obs = obs.copy()
        if action == away:
            next_obs[threat_direction] = 0.0
        elif action != threat_direction:
            next_obs[threat_direction] = 0.5
        reinforcement = evaluate(next_obs, evaluator_weights, 0.0) - evaluate(obs, evaluator_weights, 0.0)
        hebbian_trace_update(trace, bias_trace, obs, probs.copy(), action, reinforcement, eta, trace_clip)

    return {
        "seed": seed,
        "mode": mode,
        "steps": steps,
        "away_actions": away_actions,
        "away_fraction": away_actions / steps,
        "fraction_different_from_initial_argmax": argmax_different / steps,
        "trace_norm": float(np.linalg.norm(trace) + np.linalg.norm(bias_trace)),
        "effective_displacement": float(
            np.linalg.norm(alpha_weights * trace) + np.linalg.norm(alpha_bias * bias_trace)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", default="41,42,43,44,45,46,47,48,49,50")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        for mode in ("genomic", "uniform_positive"):
            row = run_probe(seed, mode, args.steps)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
