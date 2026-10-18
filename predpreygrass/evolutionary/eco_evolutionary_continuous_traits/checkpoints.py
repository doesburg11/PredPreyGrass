"""Reading trained checkpoints: their environment settings and policies.

Checkpoints of add_walls runs in the milestone world (from the
PredPreyGrassContinuous development repository) work too: this
environment reproduces add_walls exactly (SPEC.md, section 10), and their
settings are translated.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.environment import (
    DEFAULT_CONFIG,
    PredPreyGrassContinuous,
    pooled_gene_pool,
)

SPECIES = ("predator", "prey")

# add_walls settings without a counterpart here, with the only values the
# milestone world (and so this environment) allows.
ADD_WALLS_ONLY = {
    "reward_scale": 1.0,
    "prey_reward": 0,
    "prey_punishment": 0,
    "predator_reward": 0,
    "predator_shaping": 0.0,
    "obs_stack": 1,
    "target_speed_actions": True,
    "grass_food_reward": 0.0,
    "grass_predators_observe": True,
    "walls_layout": None,
    "energy_predator_acceleration_cost": 0.0,
    "energy_prey_acceleration_cost": 0.0,
}
IGNORED = {
    "render_mode",
    "render_window_size",
    "shaping_gamma",
    "walls_thickness",
    "walls_rectangles",
    "walls_occlusion",
    "walls_rays",
    "walls_visibility_cell",
}


def run_settings(checkpoint):
    """The settings train.py saved for the checkpoint's run
    (source_code/settings.json: config_env and config_ppo), or None.

    train.py saves source_code/ next to its checkpoint directory, so it is
    found one level up from a final checkpoint (<run>/checkpoint) and two
    levels up from an intermediate one (<run>/checkpoint/iter_<n>)."""
    checkpoint = Path(checkpoint)
    for run in (checkpoint.parent, checkpoint.parent.parent):
        settings = run / "source_code" / "settings.json"
        if settings.is_file():
            return json.loads(settings.read_text())
    return None


def checkpoint_env_config(checkpoint):
    """The checkpoint's env_config, translated from add_walls if needed.

    Read from the run's source_code/settings.json (written by train.py) when
    it exists: RLlib's own copy in class_and_ctor_args.pkl is a pickle that
    refers to the modules the run was trained with, which need not be
    importable here."""
    checkpoint = Path(checkpoint)
    settings = run_settings(checkpoint)
    if settings is not None:
        config = dict(settings["config_env"])
    else:
        with open(checkpoint / "class_and_ctor_args.pkl", "rb") as f:
            saved = pickle.load(f)
        config = dict(saved["ctor_args_and_kwargs"][0][0]["env_config"])
    for key, value in ADD_WALLS_ONLY.items():
        if key in config and config.pop(key) != value:
            raise ValueError(f"{key} must be {value!r} for this environment")
    # Aquarium's predator age limit: only one that can never be reached
    # (energy is the only way to starve here).
    max_age = config.pop("predator_max_age", None)
    longest = config.get("max_time_steps", DEFAULT_CONFIG["max_time_steps"]) + 2
    if max_age is not None and max_age <= longest:
        raise ValueError(f"predator_max_age {max_age} would end predators' lives")
    for key in IGNORED:
        config.pop(key, None)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"settings without a counterpart: {sorted(unknown)}")
    return config


class _NumPy2Unpickler(pickle.Unpickler):
    """Reads pickles written under NumPy 2 also under NumPy 1.x: NumPy 2
    renamed numpy.core to numpy._core, and its pickles refer to the new
    name. Only this unpickler translates the name; nothing else changes."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_modules(checkpoint):
    """Each species' RLModule, on the CPU, in evaluation mode."""
    return {species: load_module(checkpoint, species) for species in SPECIES}


def load_module(checkpoint, species):
    """The species' RLModule from a checkpoint, on the CPU, in evaluation
    mode."""
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

    path = Path(checkpoint) / "learner_group" / "learner" / "rl_module" / f"{species}_policy"
    try:
        module = RLModule.from_checkpoint(str(path.resolve()))
    except ModuleNotFoundError as error:
        if "numpy._core" not in str(error):
            raise
        module = None  # written under NumPy 2, read under NumPy 1.x
    if not isinstance(module, TorchRLModule):
        # RLlib could not restore the network (its pickles were written
        # under another NumPy version) or fell back to a bare RLModule:
        # rebuild the default PPO network that train.py uses and load the
        # trained weights into it.
        module = rebuild_default_module(checkpoint, species, path)
    module.to("cpu")
    module.eval()
    return module


def rebuild_default_module(checkpoint, species, path):
    """RLlib's default PPO network for the species, built from the
    environment's spaces and the run's model settings, with the weights in
    path/module_state.pkl.

    Only for runs started by train.py, which always uses RLlib's default PPO
    network with default model settings except vf_share_layers (both read
    from the run's saved settings). The saved weights must match the
    rebuilt network exactly, name by name and shape by shape."""
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    settings = run_settings(checkpoint)
    ppo = (settings or {}).get("config_ppo", {})
    if "vf_share_layers" not in ppo:
        raise ValueError(
            f"cannot load {path}: its network could not be restored, and without "
            "the run's source_code/settings.json from train.py its architecture is unknown"
        )
    env = PredPreyGrassContinuous(checkpoint_env_config(checkpoint))
    module = DefaultPPOTorchRLModule(
        observation_space=env.get_observation_space(f"{species}_0"),
        action_space=env.get_action_space(f"{species}_0"),
        inference_only=False,
        model_config=DefaultModelConfig(vf_share_layers=ppo["vf_share_layers"]),
    )
    with open(Path(path) / "module_state.pkl", "rb") as f:
        state = _NumPy2Unpickler(f).load()
    expected = {name: tuple(value.shape) for name, value in module.state_dict().items()}
    saved = {name: tuple(getattr(value, "shape", ())) for name, value in state.items()}
    if saved != expected:
        raise ValueError(
            f"the weights in {path} do not match RLlib's default PPO network: "
            f"missing {sorted(set(expected) - set(saved))}, "
            f"unexpected {sorted(set(saved) - set(expected))}, "
            f"other shapes {sorted(n for n in set(saved) & set(expected) if saved[n] != expected[n])}"
        )
    module.set_state(state)
    return module


GENE_POOLS = "gene_pools.json"


def load_gene_pool(checkpoint):
    """{species: (n, genes) array} of the genomes alive at a checkpoint, over
    every simulation worker (a worker's founder pool for a species with none
    alive), or None if the checkpoint saved no gene pools."""
    path = Path(checkpoint) / GENE_POOLS
    if not path.is_file():
        return None
    return pooled_gene_pool(path)


def use_gene_pool(env, pool):
    """Make every episode of env start from founders drawn from pool, which
    stays fixed (None: founders from the environment's founder
    distribution)."""
    env.gene_pool = {species: None if pool is None else pool[species] for species in ("predator", "prey")}
    env.freeze_gene_pool = True
