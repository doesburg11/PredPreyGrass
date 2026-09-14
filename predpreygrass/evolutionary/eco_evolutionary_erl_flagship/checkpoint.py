"""Checkpoint save/load for Trial 13.

Unlike eco_evolutionary_erl_baldwin/checkpoint.py (which pickles the whole
pure-Python `ErlWorld` object directly), this pickles a plain dict of the pieces
needed to reconstruct a running `Trial13Driver`: flagship's own
`env.get_state_snapshot()`/`restore_state_snapshot()` pair
(predpreygrass_rllib_env.py:808-845) for the grid/energy/position state, the
prey genome registry, the shared predator policy's LEARNED weights and its
per-predator temporal state, RNG state, current step, and the run's cfg -- not
the `PredPreyGrass` object itself, since it's an RLlib `MultiAgentEnv` and
reconstructing state via its own documented snapshot API is more robust than
pickling framework internals wholesale. The predator policy's weights are real
state that must be checkpointed too, not just re-initialized on resume --
unlike a frozen or rule-based predator, CentralizedPredatorPolicy has actually
learned something over the run.

To resume: construct a fresh `PredPreyGrass(cfg["config_env"])`, call `env.reset()`
(so `__init__`-only state like `observation_spaces`/`possible_agents` is properly
built), THEN `env.restore_state_snapshot(payload["env_snapshot"])` to overwrite it
with the checkpointed grid/energy/position state -- see run_trial13_simulation.py.
"""

import pickle
from pathlib import Path

CHECKPOINT_GLOB = "checkpoint_step_*.pkl"


def save_checkpoint(driver, cfg: dict, config_env: dict, path: Path):
    """Atomic write (temp file + rename) so a crash mid-write never corrupts the
    most recent good checkpoint -- rename is atomic on POSIX filesystems."""
    payload = {
        "env_snapshot": driver.env.get_state_snapshot(),
        "registry": driver.registry,
        "predator_policy": driver.predator_policy,
        "predator_registry": driver.predator_registry,
        "rng_state": driver.rng.bit_generator.state,
        "current_step": driver.current_step,
        "cfg": cfg,
        "config_env": config_env,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict:
    """Returns the raw payload dict; the caller reconstructs env/driver from it
    (see this module's docstring)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Highest step-numbered checkpoint_step_*.pkl in `checkpoint_dir`, or None if
    the directory doesn't exist or has none."""
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(
        checkpoint_dir.glob(CHECKPOINT_GLOB),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None
