"""
Record a GIF of trained predator_bands policies (headless: no window needed).

Rolls out the checkpoint's policies (deterministic actions, the env the run was trained in, from its run_config.json), renders every
--every-th step with the PyGame viewer's renderer (band-coloured symbols and rings, food scores, energy pies, threats and mammoths if
the run had them) and writes an optimised GIF.

Example:
  python -m predpreygrass.non_evolutionary.predator_bands.record_gif \
      ~/simulation_results/ray_results/PPO_PREDATOR_BANDS_CALIB_F008_MEAT060_SEED42/PPO_*/checkpoint_000019 \
      --steps 400 --every 4 --out predpreygrass/non_evolutionary/predator_bands/results_figures/trained_policies.gif
"""
import argparse
import glob
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import torch
from PIL import Image
from ray.rllib.core.rl_module.rl_module import RLModule

from predpreygrass.non_evolutionary.predator_bands.analyze_band_behavior import InstrumentedBandsEnv
from predpreygrass.non_evolutionary.predator_bands.evaluate_ppo_from_checkpoint_debug import _find_run_config, policy_pi
from predpreygrass.non_evolutionary.predator_bands.utils.pygame_grid_renderer_rllib import PyGameRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="checkpoint directory (glob patterns allowed)")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=4, help="render every k-th env step")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.6, help="resize factor for the frames")
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = sorted(glob.glob(os.path.expanduser(args.checkpoint)))[0]
    root = os.path.join(checkpoint, "learner_group", "learner", "rl_module")
    modules = {
        pid: RLModule.from_checkpoint(os.path.join(root, pid)) for pid in ("predator_male_policy", "predator_female_policy")
    }
    config = _find_run_config(checkpoint) or {}
    env = InstrumentedBandsEnv(config)
    observations, _ = env.reset(seed=args.seed)
    active = list(observations.keys())
    renderer = PyGameRenderer((env.grid_size, env.grid_size), ennable_speed_slider=False)

    def food_scores():
        out = {}
        for key in ("male", "female"):
            tag = f"predator_{key}"
            out[key] = {
                "prey": sum(v for a, v in env.own["meat"].items() if tag in a),
                "fruit": sum(v for a, v in env.own["fruit"].items() if tag in a),
                "n_prey": sum(v for a, v in env.n_items["meat"].items() if tag in a),
                "n_fruit": sum(v for a, v in env.n_items["fruit"].items() if tag in a),
            }
        return out

    frames = []
    for step in range(args.steps):
        actions = {
            a: policy_pi(observations[a], modules["predator_male_policy" if "predator_male" in a else "predator_female_policy"], True)
            for a in active
        }
        observations, _, terminations, truncations, _ = env.step(actions)
        active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
        if step % args.every == 0:
            renderer.update(
                agent_positions=env.agent_positions, grass_positions=env.grass_positions, agent_energies=env.agent_energies,
                grass_energies=env.grass_energies, fruit_positions=env.fruit_positions, fruit_energies=env.fruit_energies,
                agents_just_ate=env.agents_just_ate, step=env.current_step, food_scores=food_scores(),
                agent_bands=env.agent_band, agent_fruit_stores=env.agent_fruit_store, threat_positions=env.threat_positions,
            )
            raw = pygame.image.tostring(renderer.screen, "RGB")
            img = Image.frombytes("RGB", renderer.screen.get_size(), raw)
            img = img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.LANCZOS)
            frames.append(img.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        if terminations.get("__all__") or truncations.get("__all__"):
            break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB, ended at step {env.current_step}")


if __name__ == "__main__":
    main()
