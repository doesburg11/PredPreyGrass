"""Play a trained checkpoint (or a uniform-random baseline) and watch it.

Settings come from config/config_eval.py; edit it, then run from the
repository root (no command-line arguments):

    python predpreygrass/non_evolutionary/base_environment_continuous/eval.py

source "checkpoint" loads each species' policy from a checkpoint and samples
actions as in training (stochastic=True) or takes the argmax; "random"
samples every agent's action uniformly. Each episode uses its own seeded
generator, and agents in sorted order, so an episode can be replayed
exactly. render "window" shows a live pygame window, "video" also saves an
mp4 per episode, "none" only prints the summary.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ray.rllib.core.columns import Columns

from predpreygrass.non_evolutionary.base_environment_continuous.checkpoints import SPECIES, checkpoint_env_config, load_modules
from predpreygrass.non_evolutionary.base_environment_continuous.config.config_env import config_env
from predpreygrass.non_evolutionary.base_environment_continuous.config.config_eval import config_eval
from predpreygrass.non_evolutionary.base_environment_continuous.environment import PredPreyGrassContinuous

SOURCES = ("checkpoint", "random")
RENDERS = ("window", "video", "none")


def load_settings():
    """Copy and validate config_eval without modifying the source dictionary."""
    settings = dict(config_eval)
    if settings["source"] not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if settings["render"] not in RENDERS:
        raise ValueError(f"render must be one of {RENDERS}")
    if settings["source"] == "checkpoint" and not settings["checkpoint"]:
        raise ValueError('source "checkpoint" needs a checkpoint path')
    for name in ("episodes", "fps"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings["window_fps"] is not None and (
        isinstance(settings["window_fps"], bool) or not isinstance(settings["window_fps"], int) or settings["window_fps"] <= 0
    ):
        raise ValueError("window_fps must be a positive integer or None")
    if isinstance(settings["seed"], bool) or not isinstance(settings["seed"], int):
        raise ValueError("seed must be an integer")
    for name in ("stochastic", "draw_view_cones"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be True or False")
    if not isinstance(settings["env_overrides"], dict):
        raise ValueError("env_overrides must be a dict of config_env settings")
    return SimpleNamespace(**settings)


@torch.inference_mode()
def select_actions(modules, obs, generator, stochastic):
    """One batched forward pass per species, agents in sorted order."""
    actions = {}
    for species in SPECIES:
        agents = sorted(agent for agent in obs if agent.startswith(species))
        if not agents:
            continue
        batch = torch.from_numpy(np.stack([obs[agent] for agent in agents]))
        logits = modules[species].forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        if stochastic:
            chosen = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            chosen = chosen.squeeze(1)
        else:
            chosen = torch.argmax(logits, dim=-1)
        actions.update(zip(agents, chosen.tolist()))
    return actions


class VideoWriter:
    """An mp4 writer; frames are encoded as they come, not kept in memory."""

    def __init__(self, path, fps):
        self.path, self.fps, self.writer = path, fps, None

    def append_data(self, frame):
        import cv2

        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        self.writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()


def open_video(out_dir, episode, fps):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"episode_{episode}.mp4"
    return path, VideoWriter(path, fps)


def main():
    args = load_settings()
    torch.set_num_threads(1)
    if args.source == "checkpoint":
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        env_config = checkpoint_env_config(checkpoint)
        modules = load_modules(checkpoint)
    else:
        env_config = dict(config_env)
        modules = None
    env_config.update(args.env_overrides)
    if args.env_overrides:
        print(f"environment overrides: {args.env_overrides}")
    env = PredPreyGrassContinuous(env_config)
    viewer = None
    if args.render != "none":
        from predpreygrass.non_evolutionary.base_environment_continuous.render import Viewer

        viewer = Viewer(
            env,
            window_size=args.window_size,
            fps=args.window_fps if args.render == "window" else None,
            draw_view_cones=args.draw_view_cones,
        )
    csv_file = None
    video = None
    if args.population_csv:
        csv_file = open(args.population_csv, "w", encoding="utf-8")  # noqa: SIM115
        csv_file.write("episode,decision,physics_step,predators,prey,predator_births,prey_births\n")
    try:
        for episode in range(1, args.episodes + 1):
            seed = args.seed + episode - 1
            generator = torch.Generator().manual_seed(seed)
            action_rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=seed)
            if args.render == "video":
                video_path, video = open_video(args.out_dir, episode, args.fps)
            if viewer is not None:
                viewer.reset()
                frame = viewer.draw()
                if video is not None:
                    video.append_data(frame)
            counts: dict[str, int] = dict.fromkeys(
                ("caught", "predator_births", "prey_births", "predators_starved", "prey_starved", "grass_eaten"),
                0,
            )
            peak = {"predator": len(env.predators), "prey": len(env.prey)}
            decisions = 0
            while True:
                acting = {agent: obs[agent] for agent in env.active}
                if modules is None:
                    actions = {agent: int(action_rng.integers(48)) for agent in sorted(acting)}
                else:
                    actions = select_actions(modules, acting, generator, args.stochastic)
                obs, _, terms, truncs, infos = env.step(actions)
                decisions += 1
                for agent, info in infos.items():
                    species = agent.split("_")[0]
                    counts["grass_eaten"] += info.get("grass_eaten", 0)
                    if info.get("born"):
                        counts[f"{species}_births"] += 1
                    if info.get("starved"):
                        counts["predators_starved" if species == "predator" else "prey_starved"] += 1
                    elif species == "prey" and terms.get(agent):
                        counts["caught"] += 1
                peak["predator"] = max(peak["predator"], len(env.predators))
                peak["prey"] = max(peak["prey"], len(env.prey))
                if csv_file is not None:
                    csv_file.write(
                        f"{episode},{decisions},{env.time_step},{len(env.predators)},"
                        f"{len(env.prey)},{counts['predator_births']},{counts['prey_births']}\n"
                    )
                if viewer is not None:
                    frame = viewer.draw()
                    if viewer.closed:
                        return
                    if video is not None:
                        video.append_data(frame)
                if terms["__all__"] or truncs["__all__"]:
                    break
            print(
                f"episode {episode}/{args.episodes} (seed {seed}): "
                f"{decisions} decisions, {env.time_step} physics steps, "
                f"final {len(env.predators)} predators / {len(env.prey)} prey "
                f"(peak {peak['predator']} / {peak['prey']}), "
                f"births {counts['predator_births']} / {counts['prey_births']}, "
                f"prey caught {counts['caught']}, starved {counts['predators_starved']} / "
                f"{counts['prey_starved']}, grass eaten {counts['grass_eaten']}"
            )
            if video is not None:
                video.close()
                video = None
                print(f"saved {video_path}")
    finally:
        if video is not None:
            video.close()
        if csv_file is not None:
            csv_file.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("eval.py takes no command-line arguments. Edit config/config_eval.py.")
    main()
