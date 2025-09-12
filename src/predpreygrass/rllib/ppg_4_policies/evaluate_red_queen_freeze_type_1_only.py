import os
import json
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env
import ray
import torch
from predpreygrass.rllib.ppg_4_policies.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.ppg_4_policies.config.config_env_eval import config_env


def load_frozen_rl_modules(pred_ckpt_path, prey_ckpt_path):
    """
    Load frozen policies for type_1_predator and type_1_prey from different checkpoints.
    """
    rl_modules = {}

    rl_modules["type_1_predator"] = RLModule.from_checkpoint(
        os.path.join(pred_ckpt_path, "learner_group", "learner", "rl_module", "type_1_predator")
    )
    rl_modules["type_1_prey"] = RLModule.from_checkpoint(
        os.path.join(prey_ckpt_path, "learner_group", "learner", "rl_module", "type_1_prey")
    )

    return rl_modules


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "_".join(agent_id.split("_")[:3])  # 'type_1_predator' or 'type_1_prey'


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("policy_pi: action_dist_inputs not found in action_output.")
    return torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(logits=logits).sample().item()


def run_freeze_test(pred_ckpt_path, prey_ckpt_path, label, max_steps=1000, seed=42):
    """
    Run one freeze–unfreeze evaluation episode with fixed predator and prey policies.
    """
    print(f"\n=== Running: {label} ===")
    ray.init(ignore_reinit_error=True)
    env = PredPreyGrass(config=config_env)
    register_env("PredPreyGrass", lambda cfg: env)
    rl_modules = load_frozen_rl_modules(pred_ckpt_path, prey_ckpt_path)

    obs, _ = env.reset(seed=seed)
    total_reward = 0

    for _ in range(max_steps):
        action_dict = {}
        for agent_id in env.agents:
            policy_id = policy_mapping_fn(agent_id)
            policy_module = rl_modules[policy_id]
            action_dict[agent_id] = policy_pi(obs[agent_id], policy_module)

        obs, rewards, terminations, truncations, _ = env.step(action_dict)
        total_reward += sum(rewards.values())

        if terminations.get("__all__") or truncations.get("__all__"):
            break

    ray.shutdown()

    print(f"{label} - Steps: {env.current_step}")
    print(f"{label} - Total reward: {total_reward:.2f}")
    offspring = env.get_total_offspring_by_type()
    print(f"{label} - Offspring counts: {json.dumps(offspring, indent=2)}")

    # Summarize prey fitness
    prey_stats = [s for s in env.unique_agent_stats.values() if "prey" in s["policy_group"]]
    if prey_stats:
        avg_offspring = sum(a["offspring_count"] for a in prey_stats) / len(prey_stats)
        avg_lifetime = sum(a["death_step"] - a["birth_step"] for a in prey_stats if a["death_step"]) / len(prey_stats)
        print(f"{label} - Avg prey offspring: {avg_offspring:.2f}")
        print(f"{label} - Avg prey lifespan : {avg_lifetime:.2f}")
    else:
        print(f"{label} - No surviving prey agents.")


if __name__ == "__main__":
    # === Customize these paths to your actual checkpoint folders ===
    base_path = "/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/PPO_2025-07-27_23-54-21"
    ckpt_500 = os.path.join(base_path, "checkpoint_iter_500")
    ckpt_1000 = os.path.join(base_path, "checkpoint_iter_1000")

    # === Run the four combinations ===
    run_freeze_test(pred_ckpt_path=ckpt_1000, prey_ckpt_path=ckpt_500, label="Frozen Prey")
    run_freeze_test(pred_ckpt_path=ckpt_500, prey_ckpt_path=ckpt_1000, label="Frozen Predator")
    run_freeze_test(pred_ckpt_path=ckpt_500, prey_ckpt_path=ckpt_500, label="Static Baseline")
    run_freeze_test(pred_ckpt_path=ckpt_1000, prey_ckpt_path=ckpt_1000, label="Fully Co-Evolved")
