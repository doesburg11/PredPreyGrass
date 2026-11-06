"""Reading trained checkpoints: their environment settings and policies,
and untrained policies for comparison."""

import json
import pickle
from pathlib import Path

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import DEFAULT_CONFIG, StagHuntRulePrey


# Settings added after earlier runs, which trained as these values say.
LATER_SETTINGS = {
    "alive_reward": 0.0,
    "alive_reward_shared": False,
    "rabbit_lean_energy": 3.0,
    "rabbit_fat_energy": 0.0,
    "rabbit_party_radius": 64.0,
    "rabbit_catch_cooldown": 8,
    "rabbit_genes": False,
    "rabbit_genes_expressed": True,
    "rabbit_gene_mutation_std": 0.05,
    "rabbit_gene_founders": None,
    "rabbit_gene_log": "",
}


def run_settings(checkpoint):
    """The settings train.py saved for the checkpoint's run
    (source_code/settings.json: config_env and config_ppo), or None. Found
    one level up from a final checkpoint (<run>/checkpoint) and two levels
    up from an intermediate one (<run>/checkpoint/iter_<n>)."""
    checkpoint = Path(checkpoint)
    for run in (checkpoint.parent, checkpoint.parent.parent):
        settings = run / "source_code" / "settings.json"
        if settings.is_file():
            return json.loads(settings.read_text())
    return None


def checkpoint_env_config(checkpoint):
    """The checkpoint's env_config, from the run's saved settings. Every
    setting must be there (none silently taken from today's defaults), and
    the env_config RLlib saved inside the checkpoint itself must agree, so
    settings found next to a checkpoint are those it was trained with. Only
    LATER_SETTINGS may be absent."""
    settings = run_settings(checkpoint)
    if settings is None:
        raise ValueError(f"no source_code/settings.json for {checkpoint}")
    config = dict(settings["config_env"])
    unknown = set(config) - set(DEFAULT_CONFIG)
    missing = set(DEFAULT_CONFIG) - set(config) - set(LATER_SETTINGS)
    if unknown or missing:
        raise ValueError(f"settings without a counterpart: {sorted(unknown)}; missing settings: {sorted(missing)}")
    saved = Path(checkpoint) / "class_and_ctor_args.pkl"
    if saved.is_file():
        with open(saved, "rb") as f:
            inside = dict(pickle.load(f)["ctor_args_and_kwargs"][0][0]["env_config"])
        if inside != config:
            differ = sorted(k for k in set(inside) | set(config) if inside.get(k) != config.get(k))
            raise ValueError(f"{checkpoint}: its own env_config differs from the run's settings.json in {differ}")
    return {**LATER_SETTINGS, **config}


def default_module(env, species, vf_share_layers=False):
    """RLlib's default PPO network for the species, as train.py builds it,
    with fresh (untrained) weights."""
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    return DefaultPPOTorchRLModule(
        observation_space=env.get_observation_space(f"{species}_0"),
        action_space=env.get_action_space(f"{species}_0"),
        inference_only=False,
        model_config=DefaultModelConfig(vf_share_layers=vf_share_layers),
    )


def load_modules(checkpoint):
    """Each species' RLModule from a checkpoint, on the CPU, in evaluation
    mode: the default PPO network with the trained weights. The weights are
    checked by name and shape only: that they belong to this run rests on
    the checkpoint's layout (checkpoint_env_config's check)."""
    settings = run_settings(checkpoint) or {}
    vf_share_layers = settings.get("config_ppo", {}).get("vf_share_layers", False)
    env = StagHuntRulePrey(checkpoint_env_config(checkpoint))
    modules = {}
    for species in ("predator",):
        path = Path(checkpoint) / "learner_group" / "learner" / "rl_module" / f"{species}_policy" / "module_state.pkl"
        module = default_module(env, species, vf_share_layers)
        with open(path, "rb") as f:
            state = pickle.load(f)
        expected = {name: tuple(value.shape) for name, value in module.state_dict().items()}
        saved = {name: tuple(getattr(value, "shape", ())) for name, value in state.items()}
        if saved != expected:
            raise ValueError(f"the weights in {path} do not match the default PPO network")
        module.set_state(state)
        module.to("cpu")
        module.eval()
        modules[species] = module
    return modules


def untrained_modules(config, seed):
    """Freshly initialised policies (the network train.py starts from),
    seeded, on the CPU, in evaluation mode."""
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.config.config_ppo import config_ppo

    torch.manual_seed(seed)
    env = StagHuntRulePrey(config)
    modules = {}
    for species in ("predator",):
        module = default_module(env, species, config_ppo["vf_share_layers"])
        module.to("cpu")
        module.eval()
        modules[species] = module
    return modules
