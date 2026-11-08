"""The predators' module (rl_module.py): the bias is added in every forward
pass, so exploration (which RLlib's sampler stores log-probabilities from)
and training see one distribution; beta 0 is RLlib's default module."""

import dataclasses

import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import genes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import MOVES, EvolvingDeerStagHunt
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.rl_module import GeneBiasPPOTorchRLModule


def build(env, beta, cls=GeneBiasPPOTorchRLModule):
    config = dataclasses.asdict(DefaultModelConfig(vf_share_layers=False))
    if cls is GeneBiasPPOTorchRLModule:
        config["gene_bias"] = {"beta": beta, "layout": genes.layout(env)}
    return cls(observation_space=env.get_observation_space("predator_0"), action_space=env.get_action_space("predator_0"), inference_only=False, model_config=config)


@pytest.fixture(scope="module")
def batch():
    env = EvolvingDeerStagHunt(dict(width=300, height=300, predator_view_distance=150, deer_view_distance=150, predator_count=10, deer_count=14))
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(1)
    for p in env.predators:
        p.genome = rng.uniform(-1, 1, 4)
    rows = []
    for _ in range(30):
        obs, *_ = env.step({a: int(rng.integers(MOVES)) for a in env.active})
        rows += [obs[p.id] for p in env.predators if p.id in obs]
    return env, torch.from_numpy(np.stack(rows))


def test_beta_zero_is_the_default_module(batch):
    env, obs = batch
    torch.manual_seed(0)
    ours = build(env, 0.0)
    default = build(env, 0.0, DefaultPPOTorchRLModule)
    assert set(ours.state_dict()) == set(default.state_dict())
    default.set_state(ours.get_state())
    for name in ("forward_exploration", "forward_inference", "forward_train"):
        a = getattr(ours, name)({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
        b = getattr(default, name)({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
        torch.testing.assert_close(a, b)


def test_bias_is_added_in_every_pass(batch):
    env, obs = batch
    torch.manual_seed(0)
    ours = build(env, genes.BETA0)
    default = build(env, 0.0, DefaultPPOTorchRLModule)
    default.set_state(ours.get_state())
    expected_bias = torch.from_numpy(genes.bias(obs.numpy(), genes.layout(env), genes.BETA0)).float()
    assert expected_bias.abs().max() > 1
    outputs = {}
    for name in ("forward_exploration", "forward_inference", "forward_train"):
        out = getattr(ours, name)({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
        net = getattr(default, name)({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
        torch.testing.assert_close(out - net, expected_bias, atol=1e-5, rtol=0)
        outputs[name] = out
    # The behaviour distribution equals the one PPO trains on (ratio 1
    # before an update).
    torch.testing.assert_close(outputs["forward_exploration"], outputs["forward_train"])
    dist = ours.get_train_action_dist_cls().from_logits(outputs["forward_train"])
    actions = dist.sample()
    explore = ours.get_exploration_action_dist_cls().from_logits(outputs["forward_exploration"])
    torch.testing.assert_close(dist.logp(actions), explore.logp(actions))


def test_value_function_sees_the_genes(batch):
    env, obs = batch
    torch.manual_seed(0)
    ours = build(env, genes.BETA0)
    changed = obs.clone()
    lay = genes.layout(env)
    changed[:, lay["gene_start"] : lay["gene_start"] + 4] = -changed[:, lay["gene_start"] : lay["gene_start"] + 4]
    v1 = ours.compute_values({Columns.OBS: obs})
    v2 = ours.compute_values({Columns.OBS: changed})
    assert not torch.allclose(v1, v2)
