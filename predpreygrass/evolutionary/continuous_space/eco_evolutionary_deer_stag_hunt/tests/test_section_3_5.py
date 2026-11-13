"""Section 3.5's bias-off evaluation (launchers/bias_off.py) and its tests
(launchers/verdict_3_5.py)."""

import pytest
import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import genes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import ModuleActor
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.bias_off import switch_off
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict_3_5 import sign_flip_p
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.tests.test_rl_module import batch, build  # noqa: F401


def test_switch_off_leaves_the_network_without_the_bias(batch):  # noqa: F811
    env, obs = batch
    torch.manual_seed(0)
    module = build(env, genes.BETA0)
    default = build(env, 0.0, DefaultPPOTorchRLModule)
    default.set_state(module.get_state())
    actor = ModuleActor(module)
    before = actor.logits(obs.numpy())
    switch_off({"predator": actor}, genes.BETA0)
    after = actor.logits(obs.numpy())
    with torch.inference_mode():
        plain = default.forward_inference({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
    torch.testing.assert_close(after, plain)
    assert (before - after).abs().max() > 1


def test_switch_off_refuses_a_module_without_the_trained_beta(batch):  # noqa: F811
    env, _ = batch
    with pytest.raises(ValueError):
        switch_off({"predator": ModuleActor(build(env, 0.0))}, 0.0)
    with pytest.raises(ValueError):
        switch_off({"predator": ModuleActor(build(env, genes.BETA0))}, 1.0)
    with pytest.raises(TypeError):
        switch_off({"predator": ModuleActor(build(env, 0.0, DefaultPPOTorchRLModule))}, genes.BETA0)


def test_sign_flip_p():
    # All six positive: only the all-plus arrangement is as large -> 1/64.
    mean, p = sign_flip_p([1, 2, 3, 4, 5, 6])
    assert mean == pytest.approx(3.5) and p == pytest.approx(1 / 64)
    # All zero: every arrangement ties -> 1.
    assert sign_flip_p([0] * 6)[1] == 1.0
    # One negative of the smallest size: arrangements >= observed are the
    # observed one and the all-plus one -> 2/64.
    assert sign_flip_p([-1, 2, 3, 4, 5, 6])[1] == pytest.approx(2 / 64)
