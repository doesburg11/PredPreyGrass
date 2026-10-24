"""Ecology check for the Baldwin-style replication (RESULTS.md section 5.1).

Do the learning and frozen arms differ in predator and prey numbers, and
does selection on predator speed line up with predator learning or with
prey numbers? Freezing a policy is an intervention on behaviour, but it can
also change demography, which could shift selection on speed by itself.

The arms run through two code paths, so each measure is taken in the same
way for both:

- Numbers alive: the genomes alive in every worker or island at each saved
  gene pool (every 10 iterations, or the same number of steps in the frozen
  arm; the same environment method in both paths), averaged over workers.
- Births per episode and selection differentials (cov(offspring / mean
  offspring, gene)): the RLlib arms only log these through
  `metrics_logger.log_value` in callbacks.py, which by default keeps an
  exponential moving average (coefficient 0.01) over episodes. The frozen
  arm's per-episode island logs are passed through the same moving average
  per island, read at every iteration boundary (TRAIN_BATCH / islands steps
  per island) and averaged over islands. How RLlib merges the workers'
  averages is not reproduced exactly, so this is an approximation. Episodes
  without a value (a species died out, or too few animals for a
  covariance) leave the average unchanged in both paths.

The late window is the last LATE iterations, as in baldwin_replication.py;
with the moving average, late values still weight in episodes from before
it.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.baldwin_ecology
"""

import json
import warnings

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.baldwin_branches import (
    ITERATIONS,
    LATE,
    SNAPSHOT,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.baldwin_replication import (
    ARMS,
    RUNS,
    SEEDS,
    finished,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.pilot import MODULE, SPECIES

EMA_COEFF = 0.01  # RLlib's default for MetricsLogger.log_value
TRAIN_BATCH = 4800  # environment steps per iteration (config_ppo.py, evolve_frozen.py)
LOGGED = ("predator_births", "prey_births", "predator_speed_selection", "prey_speed_selection")
ALIVE = tuple(f"{species}_alive" for species in SPECIES)


def alive(run_dir):
    """{species_alive: per-snapshot mean number alive per worker}, snapshots in order."""
    out = {key: [] for key in ALIVE}
    for n in range(SNAPSHOT, ITERATIONS + 1, SNAPSHOT):
        path = run_dir / "checkpoint" / f"iter_{n:06d}" / "gene_pools.json"
        if not path.is_file():  # the last snapshot is the final checkpoint
            path = run_dir / "checkpoint" / "gene_pools.json"
        workers = json.loads(path.read_text())
        for species in SPECIES:
            out[f"{species}_alive"].append(np.mean([len(w[species]["alive"]) for w in workers]))
    return {key: np.array(values) for key, values in out.items()}


def rllib_logged(run_dir):
    """{key: per-iteration moving average} from the TensorBoard scalars."""
    events = sorted((run_dir / "tensorboard").glob("events*"))
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {key: np.array([s.value for s in accumulator.Scalars(f"env_runners/traits/{key}")]) for key in LOGGED}


def frozen_logged(run_dir):
    """{key: per-iteration moving average}, RLlib's average rebuilt from the island logs."""
    paths = sorted((run_dir / "islands").glob("island_*.jsonl"))
    per_island_steps = TRAIN_BATCH / len(paths)
    series = {key: [] for key in LOGGED}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        boundaries = np.arange(1, ITERATIONS + 1) * per_island_steps
        for key in LOGGED:
            species, stat = key.split("_", 1)
            ema, values, i = np.nan, [], 0
            for boundary in boundaries:
                while i < len(rows) and rows[i]["steps"] <= boundary:
                    x = rows[i]["stats"][species].get(stat)
                    if x is not None:
                        ema = x if np.isnan(ema) else (1 - EMA_COEFF) * ema + EMA_COEFF * x
                    i += 1
                values.append(ema)
            series[key].append(values)
    with warnings.catch_warnings():  # iterations before any island's first episode
        warnings.simplefilter("ignore", RuntimeWarning)
        return {key: np.nanmean(np.array(values), axis=0) for key, values in series.items()}


def summary(values, late):
    return {"whole_run": float(np.nanmean(values)), "late": float(np.nanmean(values[-late:]))}


def main():
    runs, missing = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = RUNS / f"blrep_{arm}_s{seed}"
            if not finished(run_dir):
                missing.append(run_dir.name)
                continue
            logged = rllib_logged(run_dir) if (run_dir / "tensorboard").is_dir() else frozen_logged(run_dir)
            stats = {key: summary(v, LATE // SNAPSHOT) for key, v in alive(run_dir).items()}
            stats.update({key: summary(v, LATE) for key, v in logged.items()})
            runs[(arm, seed)] = stats
    keys = ALIVE + LOGGED
    output = {"late_iterations": LATE, "ema_coeff": EMA_COEFF, "arms": {}}
    for arm in ARMS:
        seeds = [seed for seed in SEEDS if (arm, seed) in runs]
        output["arms"][arm] = {
            phase: {
                key: {
                    "mean": float(np.mean([runs[(arm, s)][key][phase] for s in seeds])),
                    "seeds": {str(s): runs[(arm, s)][key][phase] for s in seeds},
                }
                for key in keys
            }
            for phase in ("whole_run", "late")
        }
    path = MODULE / "analysis" / "results" / "baldwin_ecology.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written {path}")
    if missing:
        print("not finished:", ", ".join(missing))
    for phase in ("whole_run", "late"):
        print(f"\n{phase}: arm mean over seeds [per seed]")
        for key in keys:
            print(key)
            for arm in ARMS:
                cell = output["arms"][arm][phase][key]
                print(f"  {arm:12s} {cell['mean']:10.4f}  [" + ", ".join(f"{v:.4f}" for v in cell["seeds"].values()) + "]")


if __name__ == "__main__":
    main()
