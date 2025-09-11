"""Run and visualize a random policy in the continuous PredPreyGrass environment.

Uses the environment's built-in matplotlib rendering (human or rgb_array).
Configure render_mode in env_config below. For an interactive window set to 'human'.
"""
from __future__ import annotations
import time
import numpy as np
from predpreygrass.rllib.continues_model.predpreygrass_continuous_env import PredPreyGrassContinuous

# Basic environment config (adjust as desired)
env_config = dict(
    seed=0,
    n_initial_predators=16,
    n_initial_prey=8,
    n_grass_patches=50,
    max_steps=1000,
    world_size=25.0,
    # Per-type observation radii
    vision_radius_predator=3.0,
    vision_radius_prey=4.0,
    render_mode='human',  # 'human' for live window; switch to 'rgb_array' to collect frames
)

def random_action(env, agent_id):
    return env.action_space(agent_id).sample()


def main():
    env = PredPreyGrassContinuous(env_config)
    obs, _ = env.reset()

    # Simple timing control
    target_fps = 15
    dt = 1.0 / target_fps
    last = time.time()

    done_all = False
    step = 0
    try:
        while not done_all:
            actions = {aid: random_action(env, aid) for aid in list(env.agents)}
            obs, rew, terms, truncs, infos = env.step(actions)
            frame = env.render()  # returns None in human mode
            done_all = terms.get('__all__', False) or truncs.get('__all__', False)
            step += 1
            # Basic console feedback every 25 steps
            if step % 25 == 0:
                n_pred = sum(1 for a in env.agents if a.startswith('predator'))
                n_prey = sum(1 for a in env.agents if a.startswith('prey'))
                print(f"Step {step}: predators={n_pred} prey={n_prey}")
            # Frame pacing
            now = time.time()
            sleep_time = dt - (now - last)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last = time.time()
            if step >= env.max_steps:
                break
    finally:
        env.close()

if __name__ == '__main__':
    main()
