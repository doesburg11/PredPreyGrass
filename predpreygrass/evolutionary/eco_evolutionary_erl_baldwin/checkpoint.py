"""Checkpoint save/load for ErlWorld.

Pure-Python/NumPy world (no RLlib/Ray), so there's no framework-provided
checkpointing -- this pickles the whole `ErlWorld` object directly rather than
hand-serializing its fields. That's deliberate: `world.occupant` holds the same
`Agent`/`Carnivore` objects as `world.agents`/`world.carnivores`, and pickling
the object graph in one call preserves that shared identity automatically
(load a checkpoint, mutate an agent found via `world.occupant`, and the same
change shows up through `world.agents` -- a hand-rolled per-field serializer
would have to reconstruct that linkage itself and could easily get it wrong).

Two uses:
  - run_erl_simulation.py: periodic checkpoints so a long run (the paper's own
    scale is up to 1,000,000+ steps, see README.md) can resume after a crash
    or manual interruption instead of losing all progress.
  - eval_checkpoint.py: load any checkpoint standalone, no training loop
    needed, to visualize/inspect an evolved population without retraining.
"""

import pickle
from pathlib import Path

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld

CHECKPOINT_GLOB = "checkpoint_step_*.pkl"


def save_checkpoint(world: ErlWorld, path: Path):
    """Atomic write (temp file + rename) so a crash mid-write never corrupts
    the most recent good checkpoint -- rename is atomic on POSIX filesystems."""
    # world.on_agent_death is typically a closure over an open CsvLogger file
    # handle (see run_erl_simulation.py) -- not picklable, and callers must
    # reattach their own callback after load_checkpoint anyway (see below).
    callback = world.on_agent_death
    world.on_agent_death = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(world, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
    finally:
        world.on_agent_death = callback


def load_checkpoint(path: Path) -> ErlWorld:
    """Returns the world with `on_agent_death` unset (see save_checkpoint) --
    reattach a callback via `world.on_agent_death = ...` if the caller needs
    lineage logging to continue past the resume point."""
    with open(path, "rb") as f:
        world = pickle.load(f)
    world.on_agent_death = None
    return world


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Highest step-numbered checkpoint_step_*.pkl in `checkpoint_dir`, or None
    if the directory doesn't exist or has none. Convenience for `--resume-from`
    callers that want "just continue the last run" without typing a step number."""
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(
        checkpoint_dir.glob(CHECKPOINT_GLOB),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None
