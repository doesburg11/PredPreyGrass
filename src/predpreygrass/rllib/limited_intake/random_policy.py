"""
Random policy for the PredPreyGrass environment.
No backward stepping is implemented in this version,
because that is pointless for debugging and testing
with a random policy.
"""
"""Random policy viewer with wall visualization (limited_intake variant).

This script intentionally imports the local limited_intake environment & renderer
so that the `walls` parameter on `PyGameRenderer.update` is available. If you
accidentally import the older limited_intake renderer, the call with `walls=`
will raise a TypeError (unexpected keyword). Ensure the imports below stay
pointing at `limited_intake` when using walls.
"""

from predpreygrass.rllib.limited_intake.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.limited_intake.config.config_env_limited_intake import config_env
from predpreygrass.rllib.limited_intake.utils.pygame_grid_renderer_rllib import PyGameRenderer

# external libraries
import pygame
import numpy as np
import random


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    # Use config as-is (no fallbacks)
    cfg = dict(config_env)
    # Reproducibility: fix random seeds for env, numpy, python, and action spaces
    base_seed = int(cfg["seed"])
    random.seed(base_seed)
    np.random.seed(base_seed)

    env = env_creator(cfg)
    observations, _ = env.reset(seed=base_seed)

    # Seed all action spaces deterministically (stable order)
    for i, agent_id in enumerate(sorted(env.action_spaces.keys())):
        # Gymnasium spaces accept 32-bit seeds; keep it bounded
        env.action_spaces[agent_id].seed((base_seed + i) % (2**32))

    # Debug: print one observation shape to confirm visibility channel present
    if observations:
        first_agent = next(iter(observations))
        print(f"Sample observation shape for {first_agent}: {observations[first_agent].shape}")

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(
        grid_size,
        enable_speed_slider=False,
        enable_tooltips=False,
        predator_obs_range=cfg["predator_obs_range"],
        prey_obs_range=cfg["prey_obs_range"],
        show_fov=True,
        fov_alpha=40,
        fov_agents=["type_1_predator_0", "type_1_prey_0"],
        fov_respect_walls=True,
    )
    clock = pygame.time.Clock()

    # Run loop until termination
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # --- Step forward using random actions ---
        action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
        print("-----------------------------")
        print(f"step {env.current_step}")
        print("-----------------------------")
        observations, rewards, terminations, truncations, _ = env.step(action_dict)
        print(f"counts predators={env.active_num_predators} prey={env.active_num_prey} agents={len(env.agents)}")
        predators_list = sorted([aid for aid in env.agents if "predator" in aid])
        prey_list = sorted([aid for aid in env.agents if "prey" in aid])
        print(f"predators \n {predators_list}")
        print(f"prey \n {prey_list}")
        print(f"rewards \n {rewards}")
        print(f"cumulative rewards \n {env.cumulative_rewards}")
        print(f"terminations \n { {k: v for k, v in terminations.items() if k != '__all__'} }")

        # Extra diagnostics: show death causes (eaten vs starved) for agents terminated this step
        terminated_agents = [aid for aid, done in terminations.items() if aid != "__all__" and done]
        death_causes = {}
        for aid in terminated_agents:
            # Reconstruct the unique id used for stats: f"{agent_id}_{reuse_index}"
            # reuse_index for the most recent life is activation_count - 1
            try:
                reuse_index = env.agent_activation_counts.get(aid, 0) - 1
                if reuse_index >= 0:
                    uid = f"{aid}_{reuse_index}"
                    cause = env.death_agents_stats.get(uid, {}).get("death_cause", None)
                    step = env.death_agents_stats.get(uid, {}).get("death_step", None)
                    death_causes[aid] = {"cause": cause, "death_step": step}
            except Exception:
                # best-effort only
                pass
        if death_causes:
            print(f"death causes \n {death_causes}")

        # Also list predators that ate this step (energy_eating > 0)
        if env.per_step_agent_data:
            last_step = env.per_step_agent_data[-1]
            ate_predators = sorted([
                aid for aid, data in last_step.items()
                if "predator" in aid and float(data.get("energy_eating", 0.0)) > 0.0
            ])
            if ate_predators:
                print(f"predators ate this step \n {ate_predators}")
        print("-----------------------------")

        # --- Update visualizer ---
        visualizer.update(
            grass_positions=env.grass_positions,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate,
            per_step_agent_data=env.per_step_agent_data,
            walls=env.wall_positions,
        )
        # Only end the episode when the environment signals global termination/truncation via '__all__'
        terminated = bool(terminations.get("__all__", False))
        truncated = bool(truncations.get("__all__", False))

        # Frame rate control
        clock.tick(visualizer.target_fps)

    visualizer.close()
    env.close()
