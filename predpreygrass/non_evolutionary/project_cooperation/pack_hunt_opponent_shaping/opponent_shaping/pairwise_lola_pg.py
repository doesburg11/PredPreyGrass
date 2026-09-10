"""N-player pairwise opponent-shaping policy-gradient updates.

Direct generalization of Foerster et al. (2018)'s Eq. 4.5-4.7
(`foerster2018/policy_gradient/lola_pg.py` in the sibling reproduction
repo) from 2 players to N: agent i's update is its own naive
policy-gradient term plus a *sum* of pairwise LOLA corrections, one per
other agent j, each built exactly the way Eq. 4.7 builds the 2-player
correction:

    correction_{i from j} = (grad_i grad_j E[R_j]) @ (grad_j E[R_i])

i.e. "how would nudging my own parameters change agent j's own gradient
direction, and how would agent j actually moving in that direction affect
my return" -- summed over every other agent j, not just a single opponent.
This is the pairwise-decomposed N-player opponent-shaping construction used
by this module's README Section 8 (and by "Leading the Pack: N-player
Opponent Shaping", arXiv:2312.12564) rather than the exact 2-player method's
full joint-state enumeration, which is exponential in the number of agents
and not usable here.

Departure from Foerster2018's own implementation, forced by using neural-
network policies instead of a 5-parameter table: `grad_i grad_j E[R_j]` is
formally a (P_i, P_j) matrix, and Foerster2018's `_cross_term_matrix`
materializes it explicitly -- fine at P=5, but P is in the hundreds to
thousands here (a whole network's worth of parameters), so materializing a
full P_i x P_j matrix per ordered pair, every training iteration, would be
needlessly expensive. `_correction_from_pair` below computes the exact same
quantity, `(grad_i grad_j E[R_j]) @ (grad_j E[R_i])`, as a
Hessian-*vector* product -- reassociating the sums so the (P_i, P_j) matrix
is never formed -- the same kind of trick Foerster2018's *exact*-gradient
implementation already uses (`torch.autograd.grad` instead of a materialized
Hessian) applied here to the policy-gradient estimator instead. Verified
against the naive materialize-the-matrix computation for exact numerical
equivalence at small parameter counts before being used at network scale --
see `tests/test_pairwise_lola_pg.py`.
"""

from typing import Dict, List

import torch


def reward_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """`rewards`: (T, B). Returns (T, B) with `out[t] = sum_{l=t}^{T-1}
    gamma^{l-t} * rewards[l]` -- identical to Foerster2018's own
    `_reward_to_go`."""
    horizon = rewards.shape[0]
    out = torch.zeros_like(rewards)
    running = rewards.new_zeros(rewards.shape[1])
    for t in range(horizon - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def reinforce_grad(scores: torch.Tensor, rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Foerster2018's Eq. 4.5 estimator, with the same leave-one-out
    baseline and *absolute*-time `gamma**t` factor as the reproduction
    repo's own `_reinforce_grad` (see that function's docstring for why the
    `gamma**t` factor is easy to drop by mistake -- caught there via a
    4x-magnitude bug against the closed-form exact gradient).

    `scores`: (T, B, P). `rewards`: (T, B). Returns (P,).
    """
    horizon, batch_size = rewards.shape
    r2g = reward_to_go(rewards, gamma)  # (T, B)
    if batch_size > 1:
        total = r2g.sum(dim=1, keepdim=True)
        baseline = (total - r2g) / (batch_size - 1)
    else:
        baseline = torch.zeros_like(r2g)
    advantage = r2g - baseline
    gamma_powers = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    weighted = scores * (gamma_powers.view(horizon, 1) * advantage).unsqueeze(-1)
    return weighted.sum(dim=0).mean(dim=0)


def cross_term_matrix(score_i: torch.Tensor, score_j: torch.Tensor, rewards_j: torch.Tensor, gamma: float) -> torch.Tensor:
    """Eq. 4.6, generalized: `grad_i grad_j E[R_j]`, as an explicit
    (P_i, P_j) matrix. Only used by the correctness test against
    `_correction_from_pair`'s Hessian-vector-product version below -- not
    called from the actual training loop, since materializing this matrix
    is the thing `_correction_from_pair` exists to avoid."""
    horizon = score_i.shape[0]
    cum_i = torch.cumsum(score_i, dim=0)
    cum_j = torch.cumsum(score_j, dim=0)
    gamma_powers = gamma ** torch.arange(horizon, device=rewards_j.device, dtype=rewards_j.dtype)
    weight = gamma_powers.view(horizon, 1) * rewards_j
    weighted_cum_i = cum_i * weight.unsqueeze(-1)
    matrix = torch.einsum("tbi,tbj->ij", weighted_cum_i, cum_j)
    batch_size = score_i.shape[1]
    return matrix / batch_size


def correction_from_pair(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    rewards_i: torch.Tensor,
    rewards_j: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """`(grad_i grad_j E[R_j]) @ (grad_j E[R_i])`, computed without ever
    forming the (P_i, P_j) matrix -- see module docstring. Returns (P_i,).
    """
    grad_j_Ri = reinforce_grad(score_j, rewards_i, gamma)  # (P_j,)

    horizon = score_i.shape[0]
    batch_size = score_i.shape[1]
    cum_i = torch.cumsum(score_i, dim=0)  # (T, B, P_i)
    cum_j = torch.cumsum(score_j, dim=0)  # (T, B, P_j)
    gamma_powers = gamma ** torch.arange(horizon, device=rewards_j.device, dtype=rewards_j.dtype)
    weight = gamma_powers.view(horizon, 1) * rewards_j  # (T, B)
    weighted_cum_i = cum_i * weight.unsqueeze(-1)  # (T, B, P_i)

    dot_tb = torch.einsum("tbj,j->tb", cum_j, grad_j_Ri)  # (T, B)
    correction = torch.einsum("tbi,tb->i", weighted_cum_i, dot_tb) / batch_size
    return correction


def naive_pg_update(scores_i: torch.Tensor, rewards_i: torch.Tensor, gamma: float, delta: float) -> torch.Tensor:
    """Agent i's plain policy-gradient update, no opponent-awareness."""
    return delta * reinforce_grad(scores_i, rewards_i, gamma)


def opponent_shaping_pg_update(
    agent_id: str,
    scores: Dict[str, torch.Tensor],
    rewards: Dict[str, torch.Tensor],
    gamma: float,
    delta: float,
    eta: float,
) -> torch.Tensor:
    """Agent `agent_id`'s full update: its own naive term plus the summed
    pairwise correction against every other agent in `scores`/`rewards`.
    """
    grad_i_Ri = reinforce_grad(scores[agent_id], rewards[agent_id], gamma)
    total_correction = torch.zeros_like(grad_i_Ri)
    for other_id in scores:
        if other_id == agent_id:
            continue
        total_correction = total_correction + correction_from_pair(
            scores[agent_id], scores[other_id], rewards[agent_id], rewards[other_id], gamma
        )
    return delta * grad_i_Ri + delta * eta * total_correction
