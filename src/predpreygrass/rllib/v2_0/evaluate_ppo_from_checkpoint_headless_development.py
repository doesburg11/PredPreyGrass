import os
import json
from datetime import datetime

import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env

from predpreygrass.rllib.v2_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_0.config.config_env_eval import config_env
from predpreygrass.utils.renderer import CombinedEvolutionVisualizer, PreyDeathCauseVisualizer

SAVE_EVAL_RESULTS = True
MAX_STEPS = 1000
NUM_EVAL_RUNS = 5
SEED = 1


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    speed = parts[1]
    role = parts[2]
    return f"speed_{speed}_{role}"


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("Missing 'action_dist_inputs' in policy output.")
    return torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(logits=logits).sample().item()


def setup_environment_and_modules():
    checkpoint_root = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/v2_0/trained_policies/incl_speed_2/checkpoint_iter_1000"
    module_paths = {
        pid: os.path.join(checkpoint_root, "learner_group", "learner", "rl_module", pid)
        for pid in ["speed_1_predator", "speed_2_predator", "speed_1_prey", "speed_2_prey"]
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}
    env = PredPreyGrass(config=config_env)
    return env, rl_modules, checkpoint_root


def print_summary(env, total_reward):
    print(f"Total reward: {total_reward:.2f}")
    print("--- Agent Rewards ---")
    for aid, reward in env.cumulative_rewards.items():
        print(f"{aid:20} : {reward:.2f}")
    print("--- Prey Death Causes ---")
    death_stats = {"eaten": 0, "starved": 0}
    for cause in env.death_cause_prey.values():
        if cause in death_stats:
            death_stats[cause] += 1
    for k, v in death_stats.items():
        print(f"{k.capitalize():15}: {v}")


def save_results(env, total_reward, eval_output_dir, now, ceviz, pdviz):
    ceviz.plot()
    pdviz.plot()

    with open(os.path.join(eval_output_dir, "config_env.json"), "w") as f:
        json.dump(config_env, f, indent=4)

    with open(os.path.join(eval_output_dir, "reward_summary.txt"), "w") as f:
        f.write(f"Total Reward: {total_reward:.2f}\n")
        for aid, reward in env.cumulative_rewards.items():
            f.write(f"{aid:20}: {reward:.2f}\n")

    with open(os.path.join(eval_output_dir, "prey_death_causes.txt"), "w") as f:
        for iid, cause in env.death_cause_prey.items():
            f.write(f"Prey internal_id {iid:4d}: {cause}\n")


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrass", lambda config: PredPreyGrass(config))

    checkpoint_root = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/v2_0/trained_policies/incl_speed_2/checkpoint_iter_1000"
    eval_base_dir = os.path.join(os.path.dirname(checkpoint_root), f"eval_checkpoint_iter_1000_{now}")
    os.makedirs(eval_base_dir, exist_ok=True)

    env, rl_modules, _ = setup_environment_and_modules()

    for run_idx in range(NUM_EVAL_RUNS):
        run_dir = os.path.join(eval_base_dir, f"run_{run_idx:02d}")
        os.makedirs(run_dir, exist_ok=True)

        observations, _ = env.reset(seed=SEED + run_idx)

        ceviz = CombinedEvolutionVisualizer(run_dir, timestamp=now)
        pdviz = PreyDeathCauseVisualizer(run_dir, timestamp=now)

        total_reward = 0
        for _ in range(MAX_STEPS):
            action_dict = {
                agent_id: policy_pi(observations[agent_id], rl_modules[policy_mapping_fn(agent_id)])
                for agent_id in env.agents
            }
            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            total_reward += sum(rewards.values())

            ceviz.record(agent_ids=env.agents, internal_ids=env.agent_internal_ids, agent_ages=env.agent_ages)
            pdviz.record(env.death_cause_prey)

            if all(terminations.values()) or all(truncations.values()):
                break

        print(f"\n--- Evaluation Run {run_idx + 1}/{NUM_EVAL_RUNS} ---")
        print_summary(env, total_reward)

        if SAVE_EVAL_RESULTS:
            save_results(env, total_reward, run_dir, now, ceviz, pdviz)

    ray.shutdown()
