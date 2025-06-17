"""
Random policy for the PredPreyGrass environment.
No backward stepping is implemented in this version,
because that is pointless for debugging and testing
with a random policy.
"""
from predpreygrass.rllib.v2_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_0.config.config_env_random import config_env
from predpreygrass.utils.pygame_renderer import PyGameRenderer, ViewerControlHelper, LoopControlHelper
import pygame


def env_creator(config):
    return PredPreyGrass(config)


def random_policy_pi(agent_id, env):
    return env.action_spaces[agent_id].sample()


def step_forward(env, observations, control, visualizer, clock):
    action_dict = {agent_id: random_policy_pi(agent_id, env) for agent_id in env.agents}
    observations, rewards, terminations, truncations, _ = env.step(action_dict)

    visualizer.update(
        agent_positions=env.agent_positions,
        grass_positions=env.grass_positions,
        agent_energies=env.agent_energies,
        grass_energies=env.grass_energies,
        agents_just_ate=env.agents_just_ate,
        step=env.current_step,
    )

    control.fps_slider_rect = visualizer.slider_rect
    control.step_once = False
    clock.tick(visualizer.target_fps)

    return observations, terminations, truncations


def render_static_if_paused(env, visualizer):
    visualizer.update(
        agent_positions=env.agent_positions,
        grass_positions=env.grass_positions,
        agent_energies=env.agent_energies,
        grass_energies=env.grass_energies,
        agents_just_ate=env.agents_just_ate,
        step=env.current_step,
    )


if __name__ == "__main__":
    env = env_creator(config_env)
    observations, _ = env.reset(seed=config_env.get("seed", 42))

    grid_size = (env.grid_size, env.grid_size)
    visualizer = PyGameRenderer(grid_size, ennable_speed_slider=True)
    control = ViewerControlHelper(initial_paused=False)
    loop_helper = LoopControlHelper()
    control.visualizer = visualizer
    control.fps_slider_update_fn = lambda new_fps: setattr(visualizer, "target_fps", new_fps)
    control.fps_slider_rect = visualizer.slider_rect
    clock = pygame.time.Clock()

    while not loop_helper.simulation_terminated:
        control.handle_events()

        if loop_helper.should_step(control):
            observations, terminations, truncations = step_forward(
                env, observations, control, visualizer, clock
            )
            loop_helper.update_simulation_terminated(terminations, truncations)
        else:
            render_static_if_paused(env, visualizer)
            pygame.time.wait(50)

    visualizer.close()
    env.close()
