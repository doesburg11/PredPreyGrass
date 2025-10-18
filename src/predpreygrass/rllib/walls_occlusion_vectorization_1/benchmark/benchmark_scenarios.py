import time
import numpy as np
import importlib

# Config import (assumes same config for both envs)
from predpreygrass.rllib.walls_occlusion_proper_termination.config.config_env_walls_occlusion_proper_termination import config_env

ENV_MODULES = [
    "predpreygrass.rllib.walls_occlusion_proper_termination.predpreygrass_rllib_env_old",
    "predpreygrass.rllib.walls_occlusion_proper_termination.predpreygrass_rllib_env",
]
ENV_NAMES = ["Original", "Vectorized"]

SCENARIOS = [
    {"grid_size": 25, "n_predators": 10, "n_prey": 10, "n_grass": 100},
    {"grid_size": 50, "n_predators": 40, "n_prey": 40, "n_grass": 400},
    {"grid_size": 100, "n_predators": 160, "n_prey": 160, "n_grass": 1600},
    {"grid_size": 200, "n_predators": 640, "n_prey": 640, "n_grass": 6400},
    {"grid_size": 400, "n_predators": 5000, "n_prey": 5000, "n_grass": 50000},
    ]
N_STEPS = 200
SEED = 42

def run_env(env_module_name, env_name, scenario, n_steps=N_STEPS, seed=SEED):
    print(f"\n--- Running {env_name} | Scenario: {scenario} ---")
    env_mod = importlib.import_module(env_module_name)
    env_cls = getattr(env_mod, "PredPreyGrass")
    cfg = dict(config_env)
    cfg.update(scenario)
    cfg["seed"] = seed
    cfg["debug_mode"] = False  # Disable debug prints for timing
    env = env_cls(cfg)
    obs, _ = env.reset(seed=seed)
    step_times = []
    for step in range(n_steps):
        action_dict = {agent_id: env.action_spaces[agent_id].sample() for agent_id in env.agents}
        t0 = time.perf_counter()
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        if any(terminations.values()) or any(truncations.values()):
            break
    step_times = np.array(step_times)
    print(f"{env_name} - Steps run: {len(step_times)}")
    print(f"{env_name} - Mean step time: {step_times.mean():.6f}s")
    print(f"{env_name} - Min step time: {step_times.min():.6f}s")
    print(f"{env_name} - Max step time: {step_times.max():.6f}s")
    print(f"{env_name} - Std step time: {step_times.std():.6f}s")
    return step_times

def main():
    all_stats = []
    for i, scenario in enumerate(SCENARIOS, 1):
        scenario_name = f"Scenario {i}"
        results = {}
        stats = {}
        for env_mod, env_name in zip(ENV_MODULES, ENV_NAMES):
            times = run_env(env_mod, env_name, scenario)
            results[env_name] = times
            stats[env_name] = {
                "Steps": len(times),
                "Mean": times.mean(),
                "Min": times.min(),
                "Max": times.max(),
                "Std": times.std(),
            }
        all_stats.append((scenario_name, stats))

    # Print one big table
    print("\n| Scenario   | Env         | Steps | Mean Step Time (s) | Min (s) | Max (s) | Std (s) |")
    print("|------------|-------------|-------|-------------------|---------|---------|---------|")
    for scenario_name, stats in all_stats:
        for env_name in ENV_NAMES:
            s = stats[env_name]
            print(f"| {scenario_name:10} | {env_name:11} | {s['Steps']:5} | {s['Mean']:.6f}           | {s['Min']:.6f} | {s['Max']:.6f} | {s['Std']:.6f} |")

    # Print speedup table after main table
    print("\n| Scenario   | Original/Vectorized Mean Step Time |")
    print("|------------|------------------------------------|")
    for scenario_name, stats in all_stats:
        orig = stats[ENV_NAMES[0]]['Mean']
        vect = stats[ENV_NAMES[1]]['Mean']
        ratio = orig / vect if vect != 0 else float('inf')
        print(f"| {scenario_name:10} | {ratio:34.2f} |")

if __name__ == "__main__":
    main()
