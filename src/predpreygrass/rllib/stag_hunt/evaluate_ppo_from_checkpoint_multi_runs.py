from predpreygrass.rllib.stag_hunt.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt.config.config_env_stag_hunt import config_env
from predpreygrass.rllib.stag_hunt.utils.matplot_renderer import CombinedEvolutionVisualizer

# external libraries
import os
import json
from datetime import datetime
import ray
import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env


SAVE_EVAL_RESULTS = True
N_RUNS = 10  # Number of evaluation runs
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


def setup_modules():
    # GRID_30_PRED_OBS_RANGE_9_MAMMOTHS_V_2_2025-12-25_00-36-33/PPO_PredPreyGrass_618a3_00000_0_2025-12-25_00-36-33/checkpoint_000049
    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt/ray_results/pred_decay_0_20/"
    checkpoint_root = "GRID_30_PRED_OBS_RANGE_9_MAMMOTHS_V_2_2025-12-25_00-36-33/PPO_PredPreyGrass_618a3_00000_0_2025-12-25_00-36-33/"
    checkpoint_nr = "checkpoint_000049"
    checkpoint_path = os.path.abspath(ray_results_dir + checkpoint_root + checkpoint_nr)
    rl_module_dir = os.path.join(checkpoint_path, "learner_group", "learner", "rl_module")
    module_paths = {
        pid: os.path.join(rl_module_dir, pid)
        for pid in os.listdir(rl_module_dir)
        if os.path.isdir(os.path.join(rl_module_dir, pid))
    }
    rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}
    return rl_modules, checkpoint_path


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrass", lambda config: PredPreyGrass(config))
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for run in range(N_RUNS):
        seed = SEED + run
        print(f"\n=== Evaluation Run {run + 1} / {N_RUNS} ===")
        print(f"Using seed: {seed}")
        rl_modules, checkpoint_path = setup_modules()
        env = PredPreyGrass(config=config_env)
        observations, _ = env.reset(seed=SEED + run)  # Use different seed per run
        if SAVE_EVAL_RESULTS:
            eval_output_dir = os.path.join(checkpoint_path, f"eval_multiple_runs_PRED_DECAY_0_20_PRED_OBS_RANGE_9_GRID_30_INITS_15_{now}")
            os.makedirs(eval_output_dir, exist_ok=True)
            visualizer = CombinedEvolutionVisualizer(destination_path=eval_output_dir, timestamp=now, run_nr=run + 1)
        else:
            visualizer = None

        total_reward = 0
        terminated = False
        truncated = False 

        while not terminated and not truncated:
            action_dict = {aid: policy_pi(observations[aid], rl_modules[policy_mapping_fn(aid)]) for aid in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            if visualizer:
                visualizer.record(
                    agent_ids=env.agents,
                )

            total_reward += sum(rewards.values())
            # print(f"Step {i} Total Reward so far: {total_reward:.2f}")
            terminated = terminations.get("__all__", False)
            truncated = truncations.get("__all__", False)

        print(f"Evaluation complete! Total Reward: {total_reward:.2f}")
        print(f"Total Steps: {env.current_step}")
        agent_stats = env.get_all_agent_stats()

        if SAVE_EVAL_RESULTS:
            visualizer.plot()
            config_env_dir = os.path.join(eval_output_dir, "config_env")
            os.makedirs(config_env_dir, exist_ok=True)
            summary_data_dir = os.path.join(eval_output_dir, "summary_data")
            os.makedirs(summary_data_dir, exist_ok=True)
            with open(os.path.join(config_env_dir, "config_env_" + str(run + 1) + ".json"), "w") as f:
                json.dump(config_env, f, indent=4)
            with open(os.path.join(summary_data_dir, "reward_summary_" + str(run + 1) + ".txt"), "w") as f:
                f.write(f"Total Reward: {total_reward:.2f}\n")
                for aid, rec in agent_stats.items():
                    r = rec.get("cumulative_reward", 0.0)
                    f.write(f"{aid:20}: {r:.2f}\n")

    ray.shutdown()
