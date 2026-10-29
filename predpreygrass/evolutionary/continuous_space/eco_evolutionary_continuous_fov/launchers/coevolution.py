"""Coevolution branches, test 2 (DESIGN_fov_tradeoff.md in
eco_evolutionary_continuous_traits, addendum of 2026-10-29): from each
pilot-2 run, three new runs started from its iteration-10 policy weights
(fresh PPO optimizer state) and its iteration-10 gene pool pooled over the
run's workers (fresh episodes), with the run's own settings and seed. They
differ only in which species' gene pool is carried over between episodes.
Policies keep learning in every arm. The source run's saved settings are
checked against this module's config before launch.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.launchers.coevolution SOURCE ARM [ITERATIONS]

SOURCE is a pilot-2 run, tradeoff_m120 (seed 1) or tradeoff_m120_seed2
(seed 2); the branch keeps its seed and settings (noise 0.3, monocular
field 120). ARM is one of
  both          both gene pools evolve
  prey_frozen   the prey's founders come from the branch point's pool
                every episode (offspring still mutate within an episode)
  predator_frozen  the same for the predators
Output: runs/coevolution_<SOURCE>_<ARM>/ (or ..._smoke with ITERATIONS
given). Pre-specified contrasts: predator field-of-view gene, both minus
prey_frozen; prey expressed field of view, both minus predator_frozen (mean
of the last 300 iterations; the design note's addendum).
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


def check_settings(source, config_env, config_ppo):
    """Abort unless every other setting equals the source run's."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import run_settings

    saved = run_settings(RUNS / f"pilot_{source}" / "checkpoint")
    if saved is None:
        raise SystemExit(f"no saved settings for {source}")
    for name, current, own in (("config_env", config_env, OWN_ENV), ("config_ppo", config_ppo, OWN_PPO)):
        keys = (set(saved[name]) | set(current)) - own
        differ = sorted(k for k in keys if saved[name].get(k, "<missing>") != current.get(k, "<missing>"))
        if differ:
            raise SystemExit(f"{name} differs from {source}'s settings: " + ", ".join(f"{k}={current.get(k)!r} (source {saved[name].get(k)!r})" for k in differ))


def branch_point(source):
    return RUNS / f"pilot_{source}" / "checkpoint" / f"iter_{BRANCH_ITERATION:06d}"


def main():
    source, arm = sys.argv[1], sys.argv[2]
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}")
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else ITERATIONS
    out = RUNS / (f"coevolution_{source}_{arm}" + ("" if len(sys.argv) <= 3 else "_smoke"))
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")
    start = branch_point(source)

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
    check_settings(source, train.config_env, train.config_ppo)
    train.main()


if __name__ == "__main__":
    main()
