"""Small MLP categorical policy, plus per-sample score-function utilities.

Foerster et al. (2018)'s own LOLA-PG (`foerster2018/policy_gradient/
rollout.py` in the sibling reproduction repo) computes the score function
`d/dtheta log pi(a|s)` in closed form, because a memory-one matrix-game
policy is just 5 independent Bernoulli parameters -- there's an exact,
one-line formula for it (`y - p`). A neural-network policy has no such
closed form: `d/dtheta log pi(a|s)` is the gradient of a whole network's
log-probability output with respect to (typically) thousands of weights.

This module gets that gradient, *per sample in the batch*, using
`torch.func.grad` + `torch.func.vmap` rather than looping over the batch in
Python (which would be correct but far too slow to be usable). This is the
direct neural-network analog of Foerster2018's `_bernoulli_score` -- same
role in the pipeline, different (unavoidably more expensive) computation.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim). Returns action logits, (B, n_actions)."""
        return self.net(obs)


def flat_param_count(params: Dict[str, torch.Tensor]) -> int:
    return sum(p.numel() for p in params.values())


def flatten_per_sample_grads(grads: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    """`grads`: pytree matching a params dict, each leaf shaped (B, *param.shape)
    (the output of `vmap(grad(...))` over that params dict). Returns (B, P),
    P = total parameter count, with a fixed, deterministic ordering across
    calls (Python dicts preserve insertion order, and `named_parameters()`
    always yields the same order for a given module)."""
    return torch.cat([g.reshape(batch_size, -1) for g in grads.values()], dim=1)


def make_per_sample_score_fn(model: nn.Module):
    """Returns a function `(params, obs_batch, action_batch) -> (B, P)`: the
    per-sample score d/dtheta log pi(action_b | obs_b), flattened and
    stacked over the batch. `params` is a name->tensor dict, e.g. from
    `dict(model.named_parameters())`.
    """

    def single_sample_logprob(params, obs_single, action_single):
        logits = functional_call(model, params, (obs_single.unsqueeze(0),))
        logp = torch.log_softmax(logits, dim=-1)  # (1, n_actions)
        # Plain fancy-indexing with a vmapped action tensor isn't supported
        # under vmap (data-dependent indexing); gather is.
        selected = logp[0].gather(0, action_single.view(1))
        return selected.squeeze(0)

    grad_fn = grad(single_sample_logprob)
    per_sample_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

    def score_fn(params, obs_batch, action_batch):
        grads = per_sample_grad_fn(params, obs_batch, action_batch)
        flat = flatten_per_sample_grads(grads, obs_batch.shape[0])
        # These are meant to be plain numbers fed into explicit score-function
        # algebra downstream (`opponent_shaping.pairwise_lola_pg`), never
        # differentiated further -- matching Foerster2018's own rollout.py
        # convention for its closed-form scores. `params` still requires
        # grad (needed for `torch.func.grad` above), so without detaching,
        # every stored score would needlessly drag its computation graph
        # along for the whole rollout.
        return flat.detach()

    return score_fn


def flat_grad_from_params_grad(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flattens a params-shaped pytree of *non-batched* tensors (e.g. a
    module's own `.grad` after `.backward()`) into a single (P,) vector,
    using the same ordering as `flatten_per_sample_grads`."""
    return torch.cat([p.reshape(-1) for p in params.values()])


def unflatten_to_params(flat: torch.Tensor, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Inverse of concatenation: splits a (P,) vector back into a dict with
    the same shapes as `params`, in the same order."""
    out = {}
    offset = 0
    for name, p in params.items():
        n = p.numel()
        out[name] = flat[offset : offset + n].view_as(p)
        offset += n
    return out
