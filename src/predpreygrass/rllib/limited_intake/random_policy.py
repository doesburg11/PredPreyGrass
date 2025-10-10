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
from predpreygrass.rllib.limited_intake.utils.pygame_grid_renderer_rllib import PyGameRenderer, ViewerControlHelper

# external libraries
import pygame



def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


if __name__ == "__main__":
    # Use config as-is, only override carcass/observation settings for viewer friendliness
    cfg = dict(config_env)
    cfg["include_visibility_channel"] = True
    cfg["max_eating_predator"] = 1.0
    cfg["max_eating_prey"] = 1.0
    cfg["carcass_decay_per_step"] = 0.0
    cfg["carcass_max_lifetime"] = None
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
        enable_tooltips=True,  # Enable tooltips
        predator_obs_range=cfg.get("predator_obs_range"),
        prey_obs_range=cfg.get("prey_obs_range"),
        show_fov=True,
        fov_alpha=40,
        fov_agents=["type_1_predator_0", "type_1_prey_0"],
        fov_respect_walls=True,
    )
    clock = pygame.time.Clock()

    # Add interactive pause/play/step controls
    control = ViewerControlHelper(initial_paused=False)


    # Run loop with pause/play/step controls
    terminated = False
    truncated = False
    while not terminated and not truncated:
        control.handle_events()

        if not control.paused or control.step_once:
            # --- Step forward using random actions ---
            action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(action_dict)
            control.step_once = False

        # --- Update visualizer ---
        try:
            tooltip_data = None
            if hasattr(env, 'get_tooltip_data'):
                try:
                    tooltip_data = env.get_tooltip_data()
                except Exception:
                    tooltip_data = None
            visualizer.update(
                grass_positions=env.grass_positions,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate,
                per_step_agent_data=env.per_step_agent_data,
                walls=getattr(env, "wall_positions", None),
                carcass_positions=getattr(env, "carcass_positions", None),
                carcass_energies=getattr(env, "carcass_energies", None),
                tooltip_data=tooltip_data,
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
