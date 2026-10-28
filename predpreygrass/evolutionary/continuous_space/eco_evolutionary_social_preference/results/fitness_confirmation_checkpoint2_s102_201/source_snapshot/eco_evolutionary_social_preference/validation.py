"""Fail-closed checkpoint checks for this default-MLP proof of concept."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.core.columns import Columns

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import (
    _NumPy2Unpickler, checkpoint_env_config, load_gene_pool, use_gene_pool,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous, SPECIES
from .social_bias import biased_logits


def validate_checkpoint(checkpoint, modules, seed=42):
    """Compare every saved tensor and independently calculate actor logits.

    This intentionally supports only the saved default two-layer tanh MLP.
    Shape/key or activation changes must fail rather than silently validate.
    """
    env = PredPreyGrassContinuous(checkpoint_env_config(checkpoint))
    use_gene_pool(env, load_gene_pool(checkpoint))
    obs, _ = env.reset(seed=seed)
    report = {}
    for species in SPECIES:
        path = Path(checkpoint) / 'learner_group/learner/rl_module' / f'{species}_policy/module_state.pkl'
        with path.open('rb') as stream:
            saved = _NumPy2Unpickler(stream).load()
        module = modules[species]
        with (path.parent / 'class_and_ctor_args.pkl').open('rb') as stream:
            constructor = _NumPy2Unpickler(stream).load()
        saved_config = constructor['ctor_args_and_kwargs'][1]['model_config']
        assert type(module) is constructor['class'], 'saved module class differs'
        for key, value in saved_config.items():
            assert module.model_config.get(key) == value, f'saved model setting differs: {key}'
        actual = module.state_dict()
        assert set(saved) == set(actual), f'{species}: state keys differ'
        for name, value in saved.items():
            np.testing.assert_array_equal(actual[name].detach().cpu().numpy(), value, err_msg=name)
        assert module.model_config['fcnet_hiddens'] == [256, 256]
        assert module.model_config['fcnet_activation'] == 'tanh'
        assert module.model_config['head_fcnet_hiddens'] == []
        batch = torch.from_numpy(np.stack([obs[a] for a in sorted(obs) if a.startswith(species)]))
        # Exercise the network beyond the initial observations too.
        random = torch.rand((64, batch.shape[1]), generator=torch.Generator().manual_seed(seed)) * 2 - 1
        batch = torch.cat([batch, random])
        def linear(x, prefix):
            return F.linear(x, torch.as_tensor(saved[prefix + '.weight']), torch.as_tensor(saved[prefix + '.bias']))
        with torch.inference_mode():
            x = torch.tanh(linear(batch, 'encoder.actor_encoder.net.mlp.0'))
            x = torch.tanh(linear(x, 'encoder.actor_encoder.net.mlp.2'))
            reference = linear(x, 'pi.net.mlp.0')
            logits = module.forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)
        for row, observation in zip(logits.numpy(), batch.numpy()):
            np.testing.assert_array_equal(row, biased_logits(row, observation, 0, 1, env.observe_count['predator']))
        report[species] = {
            'class': type(module).__name__, 'saved_constructor_config_equal': True, 'saved_tensors_equal': len(saved),
            'checkpoint_state_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'independent_actor_max_abs_error': float((logits-reference).abs().max()),
            'probe_observations': len(batch),
        }
    env.close()
    return report
