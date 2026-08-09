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


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: dict):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()
