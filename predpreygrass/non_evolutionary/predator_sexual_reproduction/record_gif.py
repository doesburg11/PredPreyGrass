"""
Record a GIF of trained predator_sexual_reproduction policies (headless: no window needed). Same idea as predator_bands/record_gif.py:
deterministic actions from the checkpoint's three policies, the env the run was trained in (from run_config.json), every --every-th step
rendered with the viewer's renderer, written as an optimised GIF.

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.record_gif \
      ~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_PROP_K05_MB1024_SEED42/PPO_*/checkpoint_000029 \
      --steps 400 --every 5 --out predpreygrass/non_evolutionary/predator_sexual_reproduction/results_figures/trained_policies.gif
"""
import argparse
import glob
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from PIL import Image
from ray.rllib.core.rl_module.rl_module import RLModule

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_energy_sources import make_instrumented_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.evaluate_ppo_from_checkpoint_debug import (
    _find_run_config,
    policy_mapping_fn,
    policy_pi,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.utils.pygame_grid_renderer_rllib import PyGameRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="checkpoint directory (glob patterns allowed)")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--colors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = sorted(glob.glob(os.path.expanduser(args.checkpoint)))[0]
    root = os.path.join(checkpoint, "learner_group", "learner", "rl_module")
    modules = {pid: RLModule.from_checkpoint(os.path.join(root, pid)) for pid in ("predator_male_policy", "predator_female_policy", "prey_policy")}
    env = make_instrumented_env(_find_run_config(checkpoint) or {})
    observations, _ = env.reset(seed=args.seed)
    active = list(observations.keys())
    renderer = PyGameRenderer((env.grid_size, env.grid_size), ennable_speed_slider=False)

    def food_scores():
        out = {}
        for key in ("male", "female"):
            tag = f"predator_{key}"
            out[key] = {
                "prey": sum(v for a, v in env.energy_from_prey.items() if tag in a),
                "fruit": sum(v for a, v in env.energy_from_fruit.items() if tag in a),
                "n_prey": sum(v for a, v in env.n_prey_caught.items() if tag in a),
                "n_fruit": sum(v for a, v in env.n_fruit_eaten.items() if tag in a),
            }
        return out

    frames = []
    for step in range(args.steps):
        actions = {a: policy_pi(observations[a], modules[policy_mapping_fn(a)], True) for a in active}
        observations, _, terminations, truncations, _ = env.step(actions)
        active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
        # update() EVERY step (the population chart adds one point per call), but keep only every k-th frame
        renderer.update(
                agent_positions=env.agent_positions, grass_positions=env.grass_positions, agent_energies=env.agent_energies,
                grass_energies=env.grass_energies, fruit_positions=env.fruit_positions, fruit_energies=env.fruit_energies,
                agents_just_ate=env.agents_just_ate, step=env.current_step, food_scores=food_scores(),
            )
        if step % args.every == 0:
            img = Image.frombytes("RGB", renderer.screen.get_size(), pygame.image.tostring(renderer.screen, "RGB"))
            img = img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.LANCZOS)
            frames.append(img.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        if terminations.get("__all__") or truncations.get("__all__"):
            break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB, ended at step {env.current_step}")


if __name__ == "__main__":
    main()
