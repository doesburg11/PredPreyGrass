"""Batched multi-agent rollout collection for pack_hunt_opponent_shaping.

Foerster2018's own `rollout.py` samples a batch of matrix-game episodes in
one vectorized pass, because the whole "environment" there is a tiny lookup
table. PackHuntEnv has real (if simple) spatial dynamics with no such
closed form, so this collects a batch by stepping `batch_size` independent
environment instances together, one Python-level step per timestep -- slower
than a truly vectorized env, but simple, obviously correct, and fast enough
at this environment's scale (a handful of numpy ops per instance per step).

Returns, per agent: a (T, B, P) tensor of per-step score-function vectors
(via `utils.policy_network`) and a (T, B) tensor of rewards -- exactly the
shapes `opponent_shaping.pairwise_lola_pg` expects, matching Foerster2018's
own `Rollout` dataclass fields in spirit.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.predpreygrass_rllib_env import (
    PackHuntEnv,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.utils.policy_network import (
    make_per_sample_score_fn,
)


@dataclass
class MultiAgentRollout:
    scores: Dict[str, torch.Tensor]  # agent_id -> (T, B, P_i)
    rewards: Dict[str, torch.Tensor]  # agent_id -> (T, B)
    mean_reward: Dict[str, float]
    mean_engagement_rate: float
    mean_catches_per_episode: float
    mean_escapes_per_episode: float


def collect_rollout(
    models: Dict[str, torch.nn.Module],
    env_config: dict,
    horizon: int,
    batch_size: int,
    seed: int = None,
) -> MultiAgentRollout:
    max_episode_steps = env_config.get("max_episode_steps", 500)
    if horizon > max_episode_steps:
        # PackHuntEnv truncates __all__ at max_episode_steps; stepping past
        # that silently continues an already-truncated env (undefined from
        # the env's own perspective) rather than resetting, which would
        # splice two logical episodes into one trajectory. Caught by review
        # before it could bite anyone raising horizon past the default.
        raise ValueError(
            f"rollout horizon ({horizon}) must not exceed env_config['max_episode_steps'] "
            f"({max_episode_steps}); collect_rollout does not reset mid-rollout."
        )

    envs: List[PackHuntEnv] = [PackHuntEnv(env_config) for _ in range(batch_size)]
    agent_ids = list(models.keys())
    score_fns = {a: make_per_sample_score_fn(models[a]) for a in agent_ids}
    params = {a: dict(models[a].named_parameters()) for a in agent_ids}
    ref_param = next(iter(params[agent_ids[0]].values()))
    device, dtype = ref_param.device, ref_param.dtype

    rng = np.random.default_rng(seed)
    obs_batch = {a: [] for a in agent_ids}
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for a in agent_ids:
            obs_batch[a].append(obs[a])

    scores = {
        a: torch.zeros(horizon, batch_size, sum(p.numel() for p in params[a].values()), device=device, dtype=dtype)
        for a in agent_ids
    }
    rewards = {a: torch.zeros(horizon, batch_size, device=device, dtype=dtype) for a in agent_ids}

    n_engaged_total = 0
    n_catches = 0
    n_escapes = 0

    for t in range(horizon):
        obs_tensors = {a: torch.as_tensor(np.stack(obs_batch[a]), device=device, dtype=dtype) for a in agent_ids}

        actions_tensors = {}
        with torch.no_grad():
            for a in agent_ids:
                logits = models[a](obs_tensors[a])  # (B, n_actions)
                dist = torch.distributions.Categorical(logits=logits)
                actions_tensors[a] = dist.sample()  # (B,)

        for a in agent_ids:
            scores[a][t] = score_fns[a](params[a], obs_tensors[a], actions_tensors[a])

        next_obs_batch = {a: [] for a in agent_ids}
        for i, env in enumerate(envs):
            action_dict = {a: int(actions_tensors[a][i].item()) for a in agent_ids}
            obs, step_rewards, terminations, truncations, _ = env.step(action_dict)
            for a in agent_ids:
                rewards[a][t, i] = step_rewards[a]
                next_obs_batch[a].append(obs[a])
            n_engaged_total += sum(env.engaged_this_step.values())
            if env.last_event_step == env.current_step:
                if env.last_event == "catch":
                    n_catches += 1
                elif env.last_event == "escape":
                    n_escapes += 1
        obs_batch = next_obs_batch

    mean_reward = {a: rewards[a].mean().item() for a in agent_ids}
    n_predators = len(agent_ids)
    mean_engagement_rate = n_engaged_total / (horizon * batch_size * n_predators)
    mean_catches_per_episode = n_catches / batch_size
    mean_escapes_per_episode = n_escapes / batch_size

    return MultiAgentRollout(
        scores=scores,
        rewards=rewards,
        mean_reward=mean_reward,
        mean_engagement_rate=mean_engagement_rate,
        mean_catches_per_episode=mean_catches_per_episode,
        mean_escapes_per_episode=mean_escapes_per_episode,
    )
