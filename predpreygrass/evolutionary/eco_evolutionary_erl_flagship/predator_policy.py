"""Frozen, pretrained PPO predator for Trial 13 -- inference only, no training.

Predators keep flagship's existing behavior (per this project's decision: only
prey get an evolved reward genome). Rather than hand-coding a rule-based hunter,
this loads an already-converged `predator_policy` RLModule checkpoint from a prior
base_environment PPO run and runs it in pure inference mode -- the same proven,
no-Ray-runtime-needed pattern already used by master_tournament_matrix.py and
evaluate_ppo_from_checkpoint_debug.py in that module (`RLModule.from_checkpoint` +
`_forward_inference`).

Deliberately NOT importing `policy_pi`/`policy_mapping_fn` from
evaluate_ppo_from_checkpoint_debug.py despite the logic being identical: that
module also imports pygame/cv2/ray at module level for its own interactive-viewer
purpose, which this lightweight, headless, no-Ray-runtime driver (see this
module's README.md, "RLlib or not") has no reason to depend on. The ~15 lines of
actual inference logic are duplicated here instead.
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

# --- NumPy checkpoint compatibility shim ------------------------------------
# Copied from evaluate_ppo_from_checkpoint_debug.py: older checkpoints may pickle
# references to the private path 'numpy._core.numeric', which may not exist in
# newer NumPy -- pre-emptively alias it if missing.
try:
    import importlib

    if "numpy._core.numeric" not in sys.modules:
        core_numeric = importlib.import_module("numpy.core.numeric")
        shim_pkg = types.ModuleType("numpy._core")
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = shim_pkg
        sys.modules["numpy._core.numeric"] = core_numeric
except Exception:  # noqa: BLE001 -- best-effort; only affects legacy checkpoints
    pass


class FrozenPredatorPolicy:
    """Wraps one loaded `predator_policy` RLModule for standalone action inference.

    Sampling uses its OWN `torch.Generator`, seeded explicitly (`seed`), rather
    than `torch.distributions.Categorical(...).sample()`'s implicit draw from
    PyTorch's global RNG -- found the hard way: `--seed N` run twice produced
    wildly different population trajectories (one run's predators nearly wiped
    out prey by step 80; the same seed's other run sustained calm coexistence to
    step 200), because predator action sampling was silently running on
    whatever random state PyTorch's global generator happened to be in that
    process -- never touched by `--seed`, which only seeds `driver.rng` (a NumPy
    `Generator`, for genome/mutation/prey action sampling). `torch.multinomial`
    accepts an explicit `generator`, so that's used instead of `Categorical`,
    which doesn't expose one."""

    def __init__(self, checkpoint_dir: str | Path, deterministic: bool = False, seed: int | None = None):
        module_path = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "predator_policy"
        if not module_path.is_dir():
            raise FileNotFoundError(
                f"Expected RLModule directory not found: {module_path}\n"
                "(checkpoint_dir should be a base_environment PPO checkpoint_* directory, "
                "e.g. config.py's DEFAULT_PREDATOR_CHECKPOINT_DIR)."
            )
        self._module = RLModule.from_checkpoint(module_path)
        self.deterministic = deterministic
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def act(self, observation: np.ndarray) -> int:
        """Single-agent action for one (4, obs_range, obs_range) observation."""
        obs_tensor = torch.tensor(observation).float().unsqueeze(0)
        with torch.no_grad():
            action_output = self._module._forward_inference({"obs": obs_tensor})
        logits = action_output.get("action_dist_inputs")
        if logits is None:
            raise KeyError("FrozenPredatorPolicy.act: action_dist_inputs not found in action_output.")
        if self.deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=self._generator).item())
