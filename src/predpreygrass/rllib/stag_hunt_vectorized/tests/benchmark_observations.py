import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/home/doesburg/Projects/PredPreyGrass/src")))

from predpreygrass.rllib.stag_hunt_vectorized.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt_vectorized.config.config_env_stag_hunt_vectorized import (
    config_env as base_config_env,
)


def build_env(
    *,
    active_multiplier: float = 1.0,
    grid_size: int | None = None,
    active_type_1_predator: int | None = None,
    active_type_2_predator: int | None = None,
    active_type_1_prey: int | None = None,
    active_type_2_prey: int | None = None,
    obs_range: int | None = None,
    predator_obs_range: int | None = None,
    prey_obs_range: int | None = None,
):
    config = dict(base_config_env)
    if grid_size is not None:
        config["grid_size"] = int(grid_size)
    if obs_range is not None:
        predator_obs_range = obs_range
        prey_obs_range = obs_range
    if predator_obs_range is not None:
        config["predator_obs_range"] = int(predator_obs_range)
    if prey_obs_range is not None:
        config["prey_obs_range"] = int(prey_obs_range)

    if active_multiplier != 1.0:
        active_keys = (
            "n_initial_active_type_1_predator",
            "n_initial_active_type_2_predator",
            "n_initial_active_type_1_prey",
            "n_initial_active_type_2_prey",
        )
        for key in active_keys:
            base_val = int(config.get(key, 0))
            new_val = int(round(base_val * active_multiplier))
            if base_val > 0 and new_val == 0:
                new_val = 1
            config[key] = new_val

        possible_pairs = (
            ("n_possible_type_1_predators", "n_initial_active_type_1_predator"),
            ("n_possible_type_2_predators", "n_initial_active_type_2_predator"),
            ("n_possible_type_1_prey", "n_initial_active_type_1_prey"),
            ("n_possible_type_2_prey", "n_initial_active_type_2_prey"),
        )
        for possible_key, active_key in possible_pairs:
            if int(config.get(possible_key, 0)) < int(config.get(active_key, 0)):
                config[possible_key] = int(config.get(active_key, 0))

    explicit_active = {
        "n_initial_active_type_1_predator": active_type_1_predator,
        "n_initial_active_type_2_predator": active_type_2_predator,
        "n_initial_active_type_1_prey": active_type_1_prey,
        "n_initial_active_type_2_prey": active_type_2_prey,
    }
    if any(value is not None for value in explicit_active.values()):
        for key, value in explicit_active.items():
            if value is None:
                continue
            config[key] = int(value)
        possible_pairs = (
            ("n_possible_type_1_predators", "n_initial_active_type_1_predator"),
            ("n_possible_type_2_predators", "n_initial_active_type_2_predator"),
            ("n_possible_type_1_prey", "n_initial_active_type_1_prey"),
            ("n_possible_type_2_prey", "n_initial_active_type_2_prey"),
        )
        for possible_key, active_key in possible_pairs:
            if int(config.get(possible_key, 0)) < int(config.get(active_key, 0)):
                config[possible_key] = int(config.get(active_key, 0))
    return PredPreyGrass(config)


def bench(fn, iters):
    times = []
    for _ in range(5):
        fn()
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times)


def summary(label, arr):
    ms = arr * 1e3
    print(
        f"{label}: mean={ms.mean():.3f}ms p50={np.percentile(ms,50):.3f}ms "
        f"p90={np.percentile(ms,90):.3f}ms p99={np.percentile(ms,99):.3f}ms"
    )


def bench_steps(env, step_iters, *, seed=123, disable_obs=False):
    original_batch = None
    original_single = None
    if disable_obs:
        original_batch = env._get_observations_batch
        original_single = env._get_observation
        channels = env.num_obs_channels + (1 if env.include_visibility_channel else 0)
        pred_zero = np.zeros((channels, env.predator_obs_range, env.predator_obs_range), dtype=np.float32)
        prey_zero = np.zeros((channels, env.prey_obs_range, env.prey_obs_range), dtype=np.float32)

        def fast_batch(agent_list):
            obs = {}
            for agent in agent_list:
                if "predator" in agent:
                    obs[agent] = pred_zero
                elif "prey" in agent:
                    obs[agent] = prey_zero
            return obs

        def fast_single(agent):
            if "predator" in agent:
                return pred_zero
            if "prey" in agent:
                return prey_zero
            return pred_zero

        env._get_observations_batch = fast_batch
        env._get_observation = fast_single

    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(5):
            actions = {agent: 0 for agent in obs}
            obs, _, terms, truncs, _ = env.step(actions)
            if terms.get("__all__") or truncs.get("__all__"):
                obs, _ = env.reset(seed=seed)

        times = []
        for _ in range(step_iters):
            actions = {agent: 0 for agent in obs}
            t0 = time.perf_counter()
            obs, _, terms, truncs, _ = env.step(actions)
            times.append(time.perf_counter() - t0)
            if terms.get("__all__") or truncs.get("__all__"):
                obs, _ = env.reset(seed=seed)
        return np.array(times)
    finally:
        if disable_obs:
            env._get_observations_batch = original_batch
            env._get_observation = original_single


def profile_steps(
    env,
    step_iters,
    *,
    seed=123,
    disable_obs=False,
    sort_by="tottime",
    top=30,
):
    original_batch = None
    original_single = None
    if disable_obs:
        original_batch = env._get_observations_batch
        original_single = env._get_observation
        channels = env.num_obs_channels + (1 if env.include_visibility_channel else 0)
        pred_zero = np.zeros((channels, env.predator_obs_range, env.predator_obs_range), dtype=np.float32)
        prey_zero = np.zeros((channels, env.prey_obs_range, env.prey_obs_range), dtype=np.float32)

        def fast_batch(agent_list):
            obs = {}
            for agent in agent_list:
                if "predator" in agent:
                    obs[agent] = pred_zero
                elif "prey" in agent:
                    obs[agent] = prey_zero
            return obs

        def fast_single(agent):
            if "predator" in agent:
                return pred_zero
            if "prey" in agent:
                return prey_zero
            return pred_zero

        env._get_observations_batch = fast_batch
        env._get_observation = fast_single

    def run_steps():
        obs, _ = env.reset(seed=seed)
        for _ in range(step_iters):
            actions = {agent: 0 for agent in obs}
            obs, _, terms, truncs, _ = env.step(actions)
            if terms.get("__all__") or truncs.get("__all__"):
                obs, _ = env.reset(seed=seed)

    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(5):
            actions = {agent: 0 for agent in obs}
            obs, _, terms, truncs, _ = env.step(actions)
            if terms.get("__all__") or truncs.get("__all__"):
                obs, _ = env.reset(seed=seed)

        prof = cProfile.Profile()
        prof.enable()
        run_steps()
        prof.disable()
        stream = io.StringIO()
        stats = pstats.Stats(prof, stream=stream).strip_dirs().sort_stats(sort_by)
        stats.print_stats(top)
        print(stream.getvalue())
    finally:
        if disable_obs:
            env._get_observations_batch = original_batch
            env._get_observation = original_single


def main():
    parser = argparse.ArgumentParser(description="Benchmark batched vs per-agent observations.")
    parser.add_argument("--iters", type=int, default=200, help="Timing iterations per method.")
    parser.add_argument(
        "--active-multiplier",
        type=float,
        default=1.0,
        help="Multiply initial active agent counts by this factor.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Override grid size for the benchmark.",
    )
    parser.add_argument("--active-type-1-predator", type=int, default=None)
    parser.add_argument("--active-type-2-predator", type=int, default=None)
    parser.add_argument("--active-type-1-prey", type=int, default=None)
    parser.add_argument("--active-type-2-prey", type=int, default=None)
    parser.add_argument(
        "--obs-range",
        type=int,
        default=None,
        help="Override both predator and prey observation ranges.",
    )
    parser.add_argument(
        "--predator-obs-range",
        type=int,
        default=None,
        help="Override predator observation range.",
    )
    parser.add_argument(
        "--prey-obs-range",
        type=int,
        default=None,
        help="Override prey observation range.",
    )
    parser.add_argument(
        "--bench-steps",
        type=int,
        default=0,
        help="If set >0, benchmark env.step for this many iterations.",
    )
    parser.add_argument(
        "--bench-steps-noobs",
        action="store_true",
        help="Use zero observations in the step benchmark to estimate non-obs cost.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="If set >0, run cProfile on env.step for this many iterations.",
    )
    parser.add_argument(
        "--profile-steps-noobs",
        action="store_true",
        help="Use zero observations during step profiling to estimate non-obs cost.",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        default="tottime",
        help="Sort key for cProfile stats (tottime, cumtime, calls, etc.).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=30,
        help="Top N cProfile rows to print.",
    )
    args = parser.parse_args()

    env = build_env(
        active_multiplier=args.active_multiplier,
        grid_size=args.grid_size,
        active_type_1_predator=args.active_type_1_predator,
        active_type_2_predator=args.active_type_2_predator,
        active_type_1_prey=args.active_type_1_prey,
        active_type_2_prey=args.active_type_2_prey,
        obs_range=args.obs_range,
        predator_obs_range=args.predator_obs_range,
        prey_obs_range=args.prey_obs_range,
    )
    obs, _ = env.reset(seed=123)
    agents = list(obs.keys())
    print("agents:", len(agents))

    batch_times = bench(lambda: env._get_observations_batch(agents), iters=args.iters)
    loop_times = bench(lambda: {a: env._get_observation(a) for a in agents}, iters=args.iters)

    summary("batch", batch_times)
    summary("loop", loop_times)
    print("speedup x", round(loop_times.mean() / batch_times.mean(), 2))

    if args.bench_steps > 0:
        step_times = bench_steps(env, args.bench_steps, seed=123, disable_obs=args.bench_steps_noobs)
        label = "step(noobs)" if args.bench_steps_noobs else "step"
        summary(label, step_times)

    if args.profile_steps > 0:
        label = "step(noobs)" if args.profile_steps_noobs else "step"
        print(f"profile ({label}) top {args.profile_top} by {args.profile_sort}")
        profile_steps(
            env,
            args.profile_steps,
            seed=123,
            disable_obs=args.profile_steps_noobs,
            sort_by=args.profile_sort,
            top=args.profile_top,
        )


if __name__ == "__main__":
    main()
