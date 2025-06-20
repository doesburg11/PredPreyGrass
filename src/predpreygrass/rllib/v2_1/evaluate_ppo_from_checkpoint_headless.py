import os
import json
from datetime import datetime

import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env

from predpreygrass.rllib.v2_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_0.config.config_env_eval import config_env

SAVE_EVAL_RESULTS = True
MAX_STEPS = 1000
SEED = 1

def policy_mapping_fn(agent_id):
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    raise ValueError(f"Invalid agent_id format: {agent_id}")

def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("Missing 'action_dist_inputs' in output.")
    return torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(logits=logits).sample().item()

def setup_environment_and_modules():
    checkpoint_root = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/v2_0/trained_policies/excl_speed_2/checkpoint_iter_1000"
    rl_module_dir = os.path.join(checkpoint_root, "learner_group", "learner", "rl_module")
    module_paths = {
        pid: os.path.join(rl_module_dir, pid)
        for pid in os.listdir(rl_module_dir)
        if os.path.isdir(os.path.join(rl_module_dir, pid))
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}
    env = PredPreyGrass(config=config_env)
    return env, rl_modules, checkpoint_root

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    register_env("PredPreyGrass", lambda config: PredPreyGrass(config))

    env, rl_modules, checkpoint_root = setup_environment_and_modules()
    eval_output_dir = os.path.join(os.path.dirname(checkpoint_root), f"eval_checkpoint_iter_1000_{now}")
    os.makedirs(eval_output_dir, exist_ok=True)

    observations, _ = env.reset(seed=SEED)
    total_reward = 0

    for _ in range(MAX_STEPS):
        action_dict = {
            aid: policy_pi(observations[aid], rl_modules[policy_mapping_fn(aid)])
            for aid in env.agents
            if policy_mapping_fn(aid) in rl_modules
        }
        observations, rewards, terminations, truncations, _ = env.step(action_dict)
        total_reward += sum(rewards.values())

        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"\nEvaluation complete! Total Reward: {total_reward:.2f}")
    for aid, r in env.cumulative_rewards.items():
        print(f"{aid:20}: {r:.2f}")

    death_stats = {"eaten": 0, "starved": 0}
    for cause in env.death_cause_prey.values():
        if cause in death_stats:
            death_stats[cause] += 1
    print(f"\nPrey Deaths: Eaten={death_stats['eaten']} Starved={death_stats['starved']}")

    if SAVE_EVAL_RESULTS:
        with open(os.path.join(eval_output_dir, "config_env.json"), "w") as f:
            json.dump(config_env, f, indent=4)
        with open(os.path.join(eval_output_dir, "reward_summary.txt"), "w") as f:
            f.write(f"Total Reward: {total_reward:.2f}\n")
            for aid, r in env.cumulative_rewards.items():
                f.write(f"{aid:20}: {r:.2f}\n")
        with open(os.path.join(eval_output_dir, "prey_death_causes.txt"), "w") as f:
            for iid, cause in env.death_cause_prey.items():
                f.write(f"Prey internal_id {iid:4d}: {cause}\n")

    ray.shutdown()
