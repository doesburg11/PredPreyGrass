"""
Random policy for the PredPreyGrass environment.
No backward stepping is implemented in this version,
because that is pointless for debugging and testing
with a random policy.
"""
"""Random policy viewer with wall visualization (ppg_visibility variant).

This script intentionally imports the local ppg_visibility environment & renderer
so that the `walls` parameter on `PyGameRenderer.update` is available. If you
accidentally import the older ppg_visibility renderer, the call with `walls=`
will raise a TypeError (unexpected keyword). Ensure the imports below stay
pointing at `ppg_visibility` when using walls.
"""

from predpreygrass.rllib.ppg_visibility.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.ppg_visibility.config.config_env_train_2_policies import config_env as base_config_env
from predpreygrass.rllib.ppg_visibility.utils.pygame_grid_renderer_rllib import PyGameRenderer

# external libraries
import pygame


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    # Inject walls into config (if not already present)
    cfg = dict(base_config_env)
    cfg.setdefault("num_walls", 20)  # default number of walls for visualization
    # To use manual wall placement instead of random, uncomment and edit below:
    cfg["wall_placement_mode"] = "manual"
    cfg["manual_wall_positions"] = [
        # Top (y=4)
        (4,4),(5,4),(6,4),(7,4),(10,4),(11,4),(12,4),(13,4),(14,4),(15,4),
        # Bottom (y=15)
        (4,15),(5,15),(6,15),(7,15),(10,15),(11,15),(12,15),(13,15),(14,15),(15,15),
        # Left (x=4), excluding corners and opening at y=9,10
        (4,5),(4,6),(4,7),(4,8),(4,11),(4,12),(4,13),(4,14),
        # Right (x=15), excluding corners and opening at y=9,10
        (15,5),(15,6),(15,7),(15,8),(15,11),(15,12),(15,13),(15,14),
        ]  # list of (x,y) coordinates within grid bounds
    # Enable visibility (occlusion) channel so observations include LOS mask as final channel
    cfg.setdefault("include_visibility_channel", True)
    env = env_creator(cfg)
    observations, _ = env.reset(seed=cfg.get("seed", 42))

    # Debug: print one observation shape to confirm visibility channel present
    if observations:
        first_agent = next(iter(observations))
        print(f"Sample observation shape for {first_agent}: {observations[first_agent].shape}")

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(
        grid_size,
        enable_speed_slider=False,
        enable_tooltips=False,
        predator_obs_range=cfg.get("predator_obs_range"),
        prey_obs_range=cfg.get("prey_obs_range"),
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
        observations, rewards, terminations, truncations, _ = env.step(action_dict)

        # --- Update visualizer ---
        try:
            visualizer.update(
                grass_positions=env.grass_positions,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate,
                per_step_agent_data=env.per_step_agent_data,
                walls=getattr(env, "wall_positions", None),
            )
        except TypeError:
            # Fallback for legacy renderer without `walls` kwarg
            visualizer.update(
                grass_positions=env.grass_positions,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate,
                per_step_agent_data=env.per_step_agent_data,
            )

        terminated = any(terminations.values())
        truncated = any(truncations.values())

        # Frame rate control
        clock.tick(visualizer.target_fps)

    visualizer.close()
    env.close()
