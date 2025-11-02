"""
Random policy for the PredPreyGrass environment.
No backward stepping is implemented in this version,
because that is pointless for debugging and testing
with a random policy.
"""
"""Random policy viewer with wall visualization (walls_occlusion_proper_termination variant).

This script intentionally imports the local walls_occlusion_proper_termination environment & renderer
so that the `walls` parameter on `PyGameRenderer.update` is available. If you
accidentally import the older walls_occlusion_proper_termination renderer, the call with `walls=`
will raise a TypeError (unexpected keyword). Ensure the imports below stay
pointing at `walls_occlusion_proper_termination` when using walls.
"""

from predpreygrass.rllib.walls_occlusion_termination.predpreygrass_rllib_env_termination import PredPreyGrass
from predpreygrass.rllib.walls_occlusion_termination.config.config_env_walls_occlusion_termination import config_env
from predpreygrass.rllib.walls_occlusion_termination.utils.pygame_grid_renderer_rllib import PyGameRenderer

# external libraries
import pygame


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    # Inject walls into config (if not already present)
    cfg = dict(config_env)
    cfg.setdefault("num_walls", 20)  # default number of walls for visualization
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
