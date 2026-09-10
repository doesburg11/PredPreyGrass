"""Pairwise N-player opponent-shaping training loop for pack_hunt_opponent_shaping.

Condition 2 of the module README's Section 8: each predator's update is its
own naive policy-gradient term plus a summed pairwise LOLA-style correction
against every other predator (`opponent_shaping.pairwise_lola_pg`), applied
as a direct parameter update -- `theta += delta*grad + delta*eta*correction`
-- matching Foerster et al. (2018)'s own update convention (plain gradient
ascent with a fixed step size, not an adaptive optimizer like Adam), rather
than RLlib's PPO used for the naive baseline in `tune_ppo.py`. See that
script's and this module's README's docstrings for why: PPO's clipped,
multi-epoch surrogate objective is exactly the kind of "differentiate a
surrogate loss twice" shortcut that silently drops the terms this
correction needs, so this condition needs its own training loop rather than
reusing the PPO one.

`opponent_shaping.pairwise_lola_pg` is verified (see the module's own
correctness checks) to reduce exactly to Foerster2018's own `lola_pg_update`
at N=2 -- this is a generalization of that reproduction, not a loose
reinterpretation of it.
"""

import json
from datetime import datetime
from pathlib import Path

import torch

from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.config.config_env_pack_hunt_opponent_shaping import (
    config_env,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.config.config_opponent_shaping import (
    config_opponent_shaping,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.opponent_shaping.pairwise_lola_pg import (
    opponent_shaping_pg_update,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.opponent_shaping.rollout import (
    collect_rollout,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.predpreygrass_rllib_env import (
    PackHuntEnv,
)
from predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.utils.policy_network import (
    MLPPolicy,
    unflatten_to_params,
)


def apply_update(model: torch.nn.Module, update: torch.Tensor) -> None:
    params = dict(model.named_parameters())
    update_by_name = unflatten_to_params(update, params)
    with torch.no_grad():
        for name, p in params.items():
            p.add_(update_by_name[name])


def main():
    cfg = dict(config_opponent_shaping)
    env_cfg = dict(config_env)
    if cfg.get("seed") is not None:
        torch.manual_seed(cfg["seed"])

    sample_env = PackHuntEnv(env_cfg)
    sample_env.reset(seed=cfg.get("seed"))
    agent_ids = list(sample_env.agents)
    obs_dim = sample_env.observation_spaces[agent_ids[0]].shape[0]
    n_actions = sample_env.action_spaces[agent_ids[0]].n
    del sample_env

    models = {a: MLPPolicy(obs_dim, n_actions, hidden=cfg["hidden_size"]) for a in agent_ids}

    out_dir = Path(__file__).resolve().parent / "opponent_shaping_runs"
    run_dir = out_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({"config_env": env_cfg, "config_opponent_shaping": cfg}, f, indent=2, default=str)

    log_path = run_dir / "training_log.csv"
    with open(log_path, "w") as f:
        header = ["iteration", "mean_engagement_rate", "mean_catches_per_episode", "mean_escapes_per_episode"]
        header += [f"mean_reward_{a}" for a in agent_ids]
        f.write(",".join(header) + "\n")

    for iteration in range(cfg["iterations"]):
        rollout = collect_rollout(
            models,
            env_cfg,
            horizon=cfg["horizon"],
            batch_size=cfg["batch_size"],
            seed=None if cfg.get("seed") is None else cfg["seed"] + iteration,
        )

        updates = {
            a: opponent_shaping_pg_update(a, rollout.scores, rollout.rewards, cfg["gamma"], cfg["delta"], cfg["eta"])
            for a in agent_ids
        }
        for a in agent_ids:
            apply_update(models[a], updates[a])

        if iteration % cfg["log_every"] == 0 or iteration == cfg["iterations"] - 1:
            row = [str(iteration), f"{rollout.mean_engagement_rate:.4f}", f"{rollout.mean_catches_per_episode:.4f}", f"{rollout.mean_escapes_per_episode:.4f}"]
            row += [f"{rollout.mean_reward[a]:.4f}" for a in agent_ids]
            with open(log_path, "a") as f:
                f.write(",".join(row) + "\n")
            print(
                f"iter {iteration:4d}  engagement={rollout.mean_engagement_rate:.3f}  "
                f"catches/ep={rollout.mean_catches_per_episode:.3f}  "
                f"escapes/ep={rollout.mean_escapes_per_episode:.3f}  "
                f"rewards={ {a: round(rollout.mean_reward[a], 3) for a in agent_ids} }"
            )

    for a in agent_ids:
        torch.save(models[a].state_dict(), run_dir / f"policy_{a}.pt")
    print(f"Wrote logs and final policies to {run_dir}")


if __name__ == "__main__":
    main()
