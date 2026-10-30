"""Coevolution branches, test 2 (DESIGN_fov_tradeoff.md in
eco_evolutionary_continuous_traits, addendum of 2026-10-29): from each
pilot-2 run, three new runs started from its iteration-10 policy weights
(fresh PPO optimizer state) and its iteration-10 gene pool pooled over the
run's workers (fresh episodes), with the run's own settings and seed. They
differ only in which species' gene pool is carried over between episodes.
Policies keep learning in every arm. The source run's saved settings are
checked against this module's config before launch.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.launchers.coevolution [late] SOURCE ARM [ITERATIONS]

SOURCE is a pilot-2 run, tradeoff_m120 (seed 1) or tradeoff_m120_seed2
(seed 2); the branch keeps its seed and settings (noise 0.3, monocular
field 120). ARM is one of
  both          both gene pools evolve
  prey_frozen   the prey's founders come from the branch point's pool
                every episode (offspring still mutate within an episode)
  predator_frozen  the same for the predators
Output: runs/coevolution_<SOURCE>_<ARM>/ (or ..._smoke with ITERATIONS
given). Test 2's pre-specified contrasts: predator field-of-view gene,
both minus prey_frozen; prey expressed field of view, both minus
predator_frozen (mean of the last 300 iterations; the design note's
addendum of 2026-10-29).

With `late` (the follow-up, design note addendum of 2026-10-30): the
branch point is instead the end (iteration 1,000) of test 2's `both` run
from SOURCE, whose settings are checked; output
runs/coevolution_late_<SOURCE>_<ARM>/. Its pre-specified outcomes are
coexistence (final_check.py at iterations 700 to 1,000) of each frozen arm
against `both`, with speed-gene secondaries, not the field-of-view
contrasts above.
"""

import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
RUNS = MODULE / "runs"
SOURCES = {"tradeoff_m120": 1, "tradeoff_m120_seed2": 2}  # pilot-2 run: seed
ARMS = {"both": [], "prey_frozen": ["prey"], "predator_frozen": ["predator"]}
BRANCH_ITERATION = 10
ITERATIONS = 1000
WORKERS = 10
# Settings that differ from the source run by design: outputs and the branch.
OWN_PPO = {"checkpoint_dir", "tensorboard_dir", "iterations", "init_checkpoint"}
OWN_ENV = {"trait_initial_gene_pool", "trait_frozen_species"}


def check_settings(start, config_env, config_ppo):
    """Abort unless every other setting equals those of the run that saved
    the branch point `start`."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import run_settings

    saved = run_settings(start)
    if saved is None:
        raise SystemExit(f"no saved settings for {start}")
    for name, current, own in (("config_env", config_env, OWN_ENV), ("config_ppo", config_ppo, OWN_PPO)):
        keys = (set(saved[name]) | set(current)) - own
        differ = sorted(k for k in keys if saved[name].get(k, "<missing>") != current.get(k, "<missing>"))
        if differ:
            raise SystemExit(f"{name} differs from {start}'s settings: " + ", ".join(f"{k}={current.get(k)!r} (source {saved[name].get(k)!r})" for k in differ))


def branch_point(source, late=False):
    if late:
        return RUNS / f"coevolution_{source}_both" / "checkpoint"
    return RUNS / f"pilot_{source}" / "checkpoint" / f"iter_{BRANCH_ITERATION:06d}"


def run_name(source, arm, late=False):
    return f"coevolution_{'late_' if late else ''}{source}_{arm}"


def main():
    args = sys.argv[1:]
    late = bool(args) and args[0] == "late"
    if late:
        args = args[1:]
    if len(args) not in (2, 3):
        raise SystemExit("usage: coevolution [late] SOURCE ARM [ITERATIONS]")
    source, arm = args[0], args[1]
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}")
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    iterations = int(args[2]) if len(args) > 2 else ITERATIONS
    out = RUNS / (run_name(source, arm, late) + ("" if len(args) <= 2 else "_smoke"))
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")
    start = branch_point(source, late)
    if not (start / "gene_pools.json").is_file():
        raise SystemExit(f"no branch point at {start}")

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov import train
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import GENE_POOLS

    train.config_env.update(
        fov_monocular_distance_noise=0.3,
        fov_monocular_field=120.0,
        trait_initial_gene_pool=str(start / GENE_POOLS),
        trait_frozen_species=ARMS[arm],
    )
    train.config_ppo.update(
        iterations=iterations,
        num_env_runners=WORKERS,
        seed=SOURCES[source],
        init_checkpoint=str(start),
        checkpoint_dir=str(out / "checkpoint"),
        tensorboard_dir=str(out / "tensorboard"),
    )
    check_settings(start, train.config_env, train.config_ppo)
    train.main()


if __name__ == "__main__":
    main()
