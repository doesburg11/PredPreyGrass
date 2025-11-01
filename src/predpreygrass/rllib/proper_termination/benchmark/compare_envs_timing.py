import time
import numpy as np
import importlib

# Config import (assumes same config for both envs)
from predpreygrass.rllib.walls_occlusion_factored_reset.config.config_env_walls_occlusion_proper_termination import config_env

ENV_MODULES = [
    "predpreygrass.rllib.walls_occlusion_proper_termination.predpreygrass_rllib_env_old",
    "predpreygrass.rllib.walls_occlusion_proper_termination.predpreygrass_rllib_env",
]
ENV_NAMES = ["Original", "Vectorized"]

N_STEPS = 200
SEED = 42


def run_env(env_module_name, env_name, n_steps=N_STEPS, seed=SEED):
    print(f"\n--- Running {env_name} ---")
    env_mod = importlib.import_module(env_module_name)
    env_cls = getattr(env_mod, "PredPreyGrass")
    cfg = dict(config_env)
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
    results = {}
    for env_mod, env_name in zip(ENV_MODULES, ENV_NAMES):
        results[env_name] = run_env(env_mod, env_name)
    print("\n--- Comparison Summary ---")
    for env_name in ENV_NAMES:
        print(f"{env_name}: Mean step time = {results[env_name].mean():.6f}s")

if __name__ == "__main__":
    main()
