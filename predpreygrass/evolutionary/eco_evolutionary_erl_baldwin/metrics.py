"""Population metrics and functional-constraint tracking.

Functional-constraint analysis (Ackley & Littman 1991, Section 4) is
the paper's method for detecting genetic assimilation (the Baldwin
Effect) directly: track, per genome site, how much that site's value
actually changes across a lineage over generations. Sites that matter
for survival get purged of mutations that would break them (low
observed change = "functionally constrained"); irrelevant sites drift
freely (high observed change). Watching the eval-weight sites and the
action-weight sites separately over time is the direct signature of
assimilation: if action-weight sites become more constrained over
generations while eval-weight sites (for the same behavior) do not,
learned behavior is being fixed into the genome.

This is intentionally a different detection method than the population-
mean-drift / individual-reproduction-correlation approach used
elsewhere in this project (see eco_evolutionary_metabolic_rate_positive_control),
chosen because it doesn't rely on a fitness correlation being visible
above population-size noise -- it directly measures whether mutations
at a site survive to be inherited, at any population size.
"""

import csv
from pathlib import Path

import numpy as np


class FunctionalConstraintTracker:
    """Tracks per-site absolute change in genome value at each reproduction event."""

    def __init__(self, obs_dim: int, n_actions: int):
        self.eval_dim = obs_dim + 1  # weights + bias
        self.action_dim = obs_dim * n_actions + n_actions  # weights + bias
        self.eval_abs_change = np.zeros(self.eval_dim)
        self.action_abs_change = np.zeros(self.action_dim)
        self.n_reproductions = 0

    def record(self, parent_flat: np.ndarray, child_flat: np.ndarray):
        diff = np.abs(child_flat - parent_flat)
        self.eval_abs_change += diff[: self.eval_dim]
        self.action_abs_change += diff[self.eval_dim :]
        self.n_reproductions += 1

    def rates(self) -> dict[str, float]:
        if self.n_reproductions == 0:
            return {"eval_site_change_rate": float("nan"), "action_site_change_rate": float("nan")}
        return {
            "eval_site_change_rate": float(self.eval_abs_change.mean() / self.n_reproductions),
            "action_site_change_rate": float(self.action_abs_change.mean() / self.n_reproductions),
        }

    def reset_window(self):
        """Call periodically so rates() reflects a recent window, not the whole run."""
        self.eval_abs_change[:] = 0.0
        self.action_abs_change[:] = 0.0
        self.n_reproductions = 0


def lineage_fieldnames(obs_dim: int) -> list[str]:
    """Column order for `lineage_record` rows -- pass the same `obs_dim` used
    to build the run's genomes (World.obs_dim; 8 instead of 7 under S/ERLS)."""
    return (
        ["agent_id", "generation", "born_step", "death_step", "lifespan", "offspring_count", "censored"]
        + [f"eval_weight_{i}" for i in range(obs_dim)]
        + ["eval_bias"]
    )


def lineage_record(agent, death_step: int, censored: bool) -> dict:
    """One agent's completed (or run-truncated) lifetime, for the Singh/Lewis/Barto
    "proximate vs. ultimate reward" analysis: pairs the agent's evolved, lifetime-fixed
    evaluation-network weights (its genetically inherited proximate reward) with its
    realized fitness (`offspring_count`), the ultimate criterion evolution selects on.

    `censored=True` means the agent was still alive when the run ended (right-censored:
    its final `offspring_count` and `lifespan` are lower bounds, not its true lifetime
    total) -- pass `death_step=world.current_step` for those, not an actual death event.
    """
    row = {
        "agent_id": agent.agent_id,
        "generation": agent.generation,
        "born_step": agent.born_step,
        "death_step": death_step,
        "lifespan": death_step - agent.born_step,
        "offspring_count": agent.offspring_count,
        "censored": censored,
        "eval_bias": float(agent.genome.eval_bias),
    }
    for i, w in enumerate(agent.genome.eval_weights):
        row[f"eval_weight_{i}"] = float(w)
    return row


def truncate_csv_after_step(path: Path, step_field: str, max_step: int):
    """Drop any row with `step_field` > `max_step`, rewriting the file in place.

    Used when resuming from a checkpoint at `max_step`: progress.csv/
    lineage_fitness.csv may have been flushed past that point before a crash
    (checkpoints are saved less often than log rows), so rows beyond `max_step`
    describe steps the checkpoint never captured and that resuming will
    re-simulate -- appending without truncating first would duplicate them.
    No-op if the file doesn't exist yet (fresh run, not a resume).
    """
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if int(row[step_field]) <= max_step]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str], append: bool = False):
        """`append=True` (for resuming a run from a checkpoint, see checkpoint.py)
        appends to an existing file instead of overwriting it. A zero-byte file
        (or one that doesn't exist yet) still gets a header written; a non-empty
        file's existing header must match `fieldnames` exactly, or this raises --
        silently appending rows with the wrong schema would corrupt the file
        rather than just failing loudly."""
        self.path = path
        self.fieldnames = fieldnames
        has_content = append and path.exists() and path.stat().st_size > 0
        if has_content:
            with open(path, newline="") as f:
                existing_header = next(csv.reader(f), None)
            if existing_header != fieldnames:
                raise ValueError(
                    f"CsvLogger(append=True): {path} has a header that doesn't match "
                    f"fieldnames -- refusing to append.\n  existing: {existing_header}\n"
                    f"  expected: {fieldnames}"
                )
        self._file = open(path, "a" if append else "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not has_content:
            self._writer.writeheader()

    def log(self, row: dict):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()
