"""Follow-up to positive_control.py: that script found reward-genome content has NO
detectable effect on realized fitness in this ecology, even between maximally
opposed fixed genomes (avoider vs. anti_adaptive). Two very different mechanisms
could explain a null that clean:

  (a) agents DO behave differently according to their reward genome (flee vs.
      approach predators, seek vs. ignore food), but it just doesn't move the
      needle on survival/reproduction here -- an ecology-design fact; or
  (b) the 1-step-REINFORCE/linear-policy action network never learns to express
      the genome behaviorally at all within an agent's lifetime, regardless of
      what its eval_weights say -- a learning-mechanism weakness.

This script answers that directly and cheaply, without touching fitness: for each
fixed genome, run it live and measure, every step, which direction the agent's
CHOSEN ACTION points relative to the nearest visible predator/food at the exact
moment the action was picked (`PreyGenomeState.prev_obs`/`prev_action`, set by
driver.py's `_select_prey_action`), via the dot product of the action's move
vector (`env.action_to_move_tuple`) with the threat/food offset. This is the same
"measure behavior directly instead of assuming it" method that root-caused the
earlier predator-transfer puzzle (README.md's Status section) via action-
distribution entropy.

Deliberately NOT measuring realized post-step distance (an earlier version of
this script did, and a Codex review caught two real problems with that
approach, fixed here): (1) survivorship bias -- a prey that dies this step
(e.g. BECAUSE it walked toward a predator) is removed from `driver.registry`
before any post-step measurement, silently excluding exactly the outcome most
relevant to the predator statistic; (2) confounding -- the predator itself also
moves during the same `env.step()`, and the nearest-predator/nearest-food
identity can change, so a realized-distance delta reflects joint dynamics, not
the prey's own choice. Measuring the chosen action's direction against the
PRE-step observation avoids both: it needs no post-step state at all (so it
works identically for agents that die this step), and it isolates the one
thing under the genome's control -- which way the agent decided to move.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.behavior_diagnostic \\
        --steps 5000 --seeds 5
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_predator import CentralizedPredatorPolicy
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import (
    N_ACTIONS,
    PREDATOR_OBS_DIM,
    config_env_flagship,
    config_erl_flagship,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.positive_control import GENOMES
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass

PRED_DX = FEATURE_NAMES.index("predator_dx")
PRED_DY = FEATURE_NAMES.index("predator_dy")
PRED_PROX = FEATURE_NAMES.index("predator_proximity")
FOOD_DX = FEATURE_NAMES.index("food_dx")
FOOD_DY = FEATURE_NAMES.index("food_dy")
FOOD_PROX = FEATURE_NAMES.index("food_proximity")
EPS = 1e-9


def _classify(obs: np.ndarray, action: int, move_tuple: dict, dx_i: int, dy_i: int, prox_i: int, counts: dict, prefix: str):
    """Dot product of the chosen action's move vector with the (dx, dy) offset toward
    the target (predator or food) at decision time. Positive = moved toward it,
    negative = moved away, zero = orthogonal or stayed (action 4, (0, 0))."""
    if obs[prox_i] <= 0:
        return  # target not visible when this action was chosen -- not an encounter
    move_row, move_col = move_tuple[action]
    dot = move_row * obs[dx_i] + move_col * obs[dy_i]
    counts[f"{prefix}_encounters"] += 1
    counts[f"{prefix}_dot_sum"] += dot
    if dot > EPS:
        counts[f"{prefix}_toward"] += 1
    elif dot < -EPS:
        counts[f"{prefix}_away"] += 1
    else:
        counts[f"{prefix}_same"] += 1


def run_one(genome_name: str, seed: int, steps: int) -> dict:
    cfg = dict(config_erl_flagship)
    cfg["seed"] = seed
    cfg["fixed_eval_weights"] = GENOMES[genome_name]
    cfg["mutation_rate"] = 0.0

    config_env = dict(config_env_flagship)
    rng = np.random.default_rng(seed)
    env = PredPreyGrass(config_env)
    predator_policy = CentralizedPredatorPolicy(
        PREDATOR_OBS_DIM, N_ACTIONS, rng,
        init_std=cfg["predator_founder_weight_std"],
        lr_positive=cfg["predator_lr_positive"],
        lr_negative=cfg["predator_lr_negative"],
    )
    driver = Trial13Driver(env, predator_policy, cfg, rng)
    driver.reset()

    counts = defaultdict(float)
    # Captures (obs, action) for prey that die THIS step too -- state.prev_obs/prev_action
    # were already set by _select_prey_action before env.step() detected the death, so this
    # is real decision-time data, not a survivorship-biased sample (see module docstring).
    died_this_step = []
    driver.on_agent_death = lambda state, death_step: (
        died_this_step.append((state.prev_obs, state.prev_action)) if state.prev_obs is not None else None
    )

    for _ in range(steps):
        living_before = set(driver.registry.keys())
        died_this_step.clear()
        driver.step()
        if driver.population_counts()["prey"] == 0:
            break

        for obs, action in died_this_step:
            _classify(obs, action, env.action_to_move_tuple, PRED_DX, PRED_DY, PRED_PROX, counts, "pred")
            _classify(obs, action, env.action_to_move_tuple, FOOD_DX, FOOD_DY, FOOD_PROX, counts, "food")
        for agent_id in living_before:
            state = driver.registry.get(agent_id)
            if state is None or state.prev_obs is None:
                continue
            _classify(state.prev_obs, state.prev_action, env.action_to_move_tuple, PRED_DX, PRED_DY, PRED_PROX, counts, "pred")
            _classify(state.prev_obs, state.prev_action, env.action_to_move_tuple, FOOD_DX, FOOD_DY, FOOD_PROX, counts, "food")

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--genomes", type=str, default="avoider,anti_adaptive,forager,inert",
                         help="Comma-separated genome names from positive_control.GENOMES.")
    args = parser.parse_args()
    genome_names = args.genomes.split(",")
    jobs = [(name, seed) for name in genome_names for seed in range(1, args.seeds + 1)]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Running {len(jobs)} runs ({len(genome_names)} genomes x {args.seeds} seeds) on {workers} workers...")

    agg = {name: defaultdict(float) for name in genome_names}
    per_seed_frac_pred_toward = defaultdict(list)  # for a significance test across seeds, not just pooled counts
    per_seed_frac_food_toward = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, name, seed, args.steps): (name, seed) for name, seed in jobs}
        for fut in as_completed(futures):
            name, seed = futures[fut]
            counts = fut.result()
            for k, v in counts.items():
                agg[name][k] += v
            if counts.get("pred_encounters", 0) > 0:
                per_seed_frac_pred_toward[name].append(counts["pred_toward"] / counts["pred_encounters"])
            if counts.get("food_encounters", 0) > 0:
                per_seed_frac_food_toward[name].append(counts["food_toward"] / counts["food_encounters"])

    print("\n=== Predator direction (which way does the CHOSEN action point relative to a visible predator?) ===")
    print(f"{'genome':<16}{'n_encounters':<14}{'%fled':<10}{'%approached':<14}{'%orthogonal':<12}{'mean_dot'}")
    for name in genome_names:
        c = agg[name]
        n = c.get("pred_encounters", 0)
        if n == 0:
            print(f"{name:<16} no predator encounters recorded")
            continue
        print(f"{name:<16}{int(n):<14}{100*c['pred_away']/n:<10.1f}{100*c['pred_toward']/n:<14.1f}"
              f"{100*c['pred_same']/n:<12.1f}{c['pred_dot_sum']/n:+.4f}")

    print("\n=== Food direction (which way does the CHOSEN action point relative to visible food?) ===")
    print(f"{'genome':<16}{'n_encounters':<14}{'%approached':<12}{'%avoided':<11}{'%orthogonal':<12}{'mean_dot'}")
    for name in genome_names:
        c = agg[name]
        n = c.get("food_encounters", 0)
        if n == 0:
            print(f"{name:<16} no food encounters recorded")
            continue
        print(f"{name:<16}{int(n):<14}{100*c['food_toward']/n:<12.1f}{100*c['food_away']/n:<11.1f}"
              f"{100*c['food_same']/n:<12.1f}{c['food_dot_sum']/n:+.4f}")
    print(
        "\nDot product of the chosen action's move vector with the (dx, dy) offset toward the "
        "target at decision time -- positive means the action pointed toward it (approach), "
        "negative means away (flee/avoid), zero means orthogonal or the agent chose to stay put. "
        "Includes agents that died the same step their action was chosen (see module docstring)."
    )

    if len(genome_names) >= 2:
        from scipy import stats
        print("\n=== Significance check (Kruskal-Wallis across genomes, per-seed fractions) ===")
        groups = [per_seed_frac_pred_toward[n] for n in genome_names if per_seed_frac_pred_toward[n]]
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            try:
                h, p = stats.kruskal(*groups)
                print(f"%approached-predator-when-visible across genomes: H={h:.3f}, p={p:.4f}")
            except ValueError as e:
                print(f"%approached-predator-when-visible: Kruskal-Wallis undefined ({e})")
        groups = [per_seed_frac_food_toward[n] for n in genome_names if per_seed_frac_food_toward[n]]
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            try:
                h, p = stats.kruskal(*groups)
                print(f"%approached-food-when-visible across genomes: H={h:.3f}, p={p:.4f}")
            except ValueError as e:
                print(f"%approached-food-when-visible: Kruskal-Wallis undefined ({e})")


if __name__ == "__main__":
    main()
