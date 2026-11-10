"""Reading checkpoints of all three arms: their settings, gene pools and the
policies that act in them.

Layout of a run (train.py for L and B, evolve.py for E):

    runs/<run>/source_code/settings.json   {"config_env", "config_ppo"} (arm in config_ppo)
    runs/<run>/deer_policy/                 section 1: the frozen deer policy
    runs/<run>/deer_schedule/               section 2: the replayed deer (deer_schedule.py)
                                            (section 3: the deer learn; each RLlib
                                            checkpoint holds its own deer module)
    runs/<run>/checkpoint/[iter_<n>/]       gene_pools.json, budget.json (and,
                                            L and B, RLlib's checkpoint)
"""

import json
import pickle
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import genes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import DEFAULT_CONFIG, EvolvingDeerStagHunt

GENE_POOLS = "gene_pools.json"
BUDGET = "budget.json"
DEER_POLICY = "deer_policy"


def run_dir(checkpoint):
    """The run folder of a final (<run>/checkpoint) or intermediate
    (<run>/checkpoint/iter_<n>) checkpoint, or None."""
    checkpoint = Path(checkpoint)
    for run in (checkpoint.parent, checkpoint.parent.parent):
        if (run / "source_code" / "settings.json").is_file():
            return run
    return None


def run_settings(checkpoint):
    run = run_dir(checkpoint)
    return None if run is None else json.loads((run / "source_code" / "settings.json").read_text())


def arm(checkpoint):
    return run_settings(checkpoint)["config_ppo"]["arm"]


def checkpoint_env_config(checkpoint):
    """The checkpoint's env_config from the run's saved settings, complete
    (no setting silently taken from today's defaults) and, for an RLlib
    checkpoint, equal to the env_config saved inside it."""
    settings = run_settings(checkpoint)
    if settings is None:
        raise ValueError(f"no source_code/settings.json for {checkpoint}")
    config = dict(settings["config_env"])
    unknown, missing = set(config) - set(DEFAULT_CONFIG), set(DEFAULT_CONFIG) - set(config)
    if unknown or missing:
        raise ValueError(f"settings without a counterpart: {sorted(unknown)}; missing settings: {sorted(missing)}")
    saved = Path(checkpoint) / "class_and_ctor_args.pkl"
    if saved.is_file():
        with open(saved, "rb") as f:
            inside = dict(pickle.load(f)["ctor_args_and_kwargs"][0][0]["env_config"])
        if inside != config:
            differ = sorted(k for k in set(inside) | set(config) if inside.get(k) != config.get(k))
            raise ValueError(f"{checkpoint}: its own env_config differs from the run's settings.json in {differ}")
    return config


def evaluation_env_config(checkpoint, overrides=None):
    """The world of the checkpoint, with its pooled gene pool as the
    founders of every evaluated episode."""
    config = checkpoint_env_config(checkpoint)
    config.update(overrides or {})
    pool = Path(checkpoint) / GENE_POOLS
    if not pool.is_file():
        raise ValueError(f"{checkpoint} has no {GENE_POOLS}")
    config["gene_initial_pool"] = str(pool)
    return config


def _default_module(env, species, model_config, module_class=None):
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

    cls = module_class or DefaultPPOTorchRLModule
    return cls(
        observation_space=env.get_observation_space(f"{species}_0"),
        action_space=env.get_action_space(f"{species}_0"),
        inference_only=False,
        model_config=model_config,
    )


def _load(module, path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    expected = {name: tuple(value.shape) for name, value in module.state_dict().items()}
    saved = {name: tuple(getattr(value, "shape", ())) for name, value in state.items()}
    if saved != expected:
        raise ValueError(f"the weights in {path} do not match the network")
    module.set_state(state)
    module.to("cpu")
    module.eval()
    return module


def deer_module_at(path, env):
    """A deer policy (RLlib's default PPO network, vf_share_layers False, as
    every deer_stag_hunt run trained it) from a module directory."""
    import dataclasses

    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    config = dataclasses.asdict(DefaultModelConfig(vf_share_layers=False))
    return _load(_default_module(env, "deer", config), Path(path) / "module_state.pkl")


def deer_module(run, env):
    """Section 1: the run's one frozen deer policy."""
    return deer_module_at(Path(run) / DEER_POLICY, env)


def checkpoint_deer(checkpoint):
    """The deer module directory a checkpoint acts with: section 3, the
    deer module saved inside the checkpoint itself; section 2, the
    replayed deer it was saved with (deer.json, hash checked); section 1,
    the run's frozen deer."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import deer_schedule

    run = run_dir(checkpoint)
    if (run_settings(checkpoint) or {}).get("config_ppo", {}).get("deer_learn"):
        # Section 3: the deer trained in this run, inside this checkpoint.
        path = Path(checkpoint) / "learner_group" / "learner" / "rl_module" / "deer_policy"
        if not (path / "module_state.pkl").is_file():
            raise ValueError(f"{checkpoint}: a learning-deer checkpoint without its deer module")
        return path
    active = deer_schedule.active(checkpoint)
    if active is None:
        if deer_schedule.manifest(run) is not None:
            raise ValueError(f"{checkpoint}: its run replays deer but the checkpoint records none")
        return Path(run) / DEER_POLICY
    path = deer_schedule.deer_path(run, active["source_iteration"])
    if deer_schedule.sha256(path / "module_state.pkl") != active["sha256"]:
        raise RuntimeError(f"{checkpoint}: its deer differ from the ones it records")
    return path


class ModuleActor:
    """Move logits from an RLModule (forward_inference)."""

    def __init__(self, module):
        self.module = module

    def logits(self, obs):
        import torch
        from ray.rllib.core.columns import Columns

        with torch.inference_mode():
            return self.module.forward_inference({Columns.OBS: torch.from_numpy(obs)})[Columns.ACTION_DIST_INPUTS]


class EArmActor:
    """The E arm's move logits (genes.e_arm_logits)."""

    def __init__(self, layout, beta=genes.BETA0):
        self.layout, self.beta = layout, beta

    def logits(self, obs):
        import torch

        return torch.from_numpy(genes.e_arm_logits(obs, self.layout, self.beta))


def load_actors(checkpoint, env=None, deer=None):
    """{species: actor} acting in a checkpoint's world: the trained
    predator module (arms L and B) or the gene rule (arm E), and the deer
    (checkpoint_deer, or the deer module directory `deer`)."""
    import dataclasses

    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.rl_module import GeneBiasPPOTorchRLModule

    settings = run_settings(checkpoint)
    run = run_dir(checkpoint)
    env = env or EvolvingDeerStagHunt(checkpoint_env_config(checkpoint))
    ppo = settings["config_ppo"]
    actors = {"deer": ModuleActor(deer_module_at(deer or checkpoint_deer(checkpoint), env))}
    if ppo["arm"] == "E":
        actors["predator"] = EArmActor(genes.layout(env), ppo["beta"])
    else:
        config = dict(dataclasses.asdict(DefaultModelConfig(vf_share_layers=ppo.get("vf_share_layers", False))), gene_bias={"beta": ppo["beta"], "layout": genes.layout(env)})
        path = Path(checkpoint) / "learner_group" / "learner" / "rl_module" / "predator_policy" / "module_state.pkl"
        actors["predator"] = ModuleActor(_load(_default_module(env, "predator", config, GeneBiasPPOTorchRLModule), path))
    return actors


def budget(checkpoint):
    """Lifetime counters summed over the checkpoint's environments."""
    entries = json.loads((Path(checkpoint) / BUDGET).read_text())
    total = {key: sum(e[key] for e in entries) for key in ("physics_steps", "episodes", "predator_births", "selection_episodes")}
    total["max_generation"] = max(e["max_generation"] for e in entries)
    total["mean_max_generation"] = float(np.mean([e["max_generation"] for e in entries]))
    total["selection_sum"] = {name: sum(e["selection_sum"][name] for e in entries) for name in genes.GENES}
    total["environments"] = len(entries)
    return total


def selection_between(earlier, later):
    """Per gene, the mean selection differential of the episodes that
    finished between two checkpoints of a run (budget dicts), or None."""
    episodes = later["selection_episodes"] - (earlier["selection_episodes"] if earlier else 0)
    if episodes <= 0:
        return None
    return {name: (later["selection_sum"][name] - (earlier["selection_sum"][name] if earlier else 0.0)) / episodes for name in genes.GENES}


def checkpoint_at(run, iteration):
    """runs/<run>/checkpoint/iter_<n>, or the final checkpoint for the last
    iteration."""
    run = Path(run)
    settings = json.loads((run / "source_code" / "settings.json").read_text())
    if iteration == settings["config_ppo"]["iterations"]:
        return run / "checkpoint"
    return run / "checkpoint" / f"iter_{iteration:06d}"


def gene_means(checkpoint):
    """Mean of each gene over the checkpoint's pooled gene pool."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import pooled_gene_pool

    genomes, _ = pooled_gene_pool(Path(checkpoint) / GENE_POOLS)
    return {name: float(np.mean(genomes[:, k])) for k, name in enumerate(genes.GENES)}
