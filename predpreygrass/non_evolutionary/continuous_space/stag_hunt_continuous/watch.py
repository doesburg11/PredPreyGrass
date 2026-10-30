"""Watch one episode of trained (or untrained) policies in a window, or save
it as an mp4.

Actions are sampled as in training and as evaluate.py samples them (one
torch generator seeded with the episode's seed), so an episode watched
here is the same as evaluate.py's episode with that seed.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.watch \\
        <checkpoint dir | untrained> [--seed N] [--video out.mp4] [--fps N]

On a machine without a display, set SDL_VIDEODRIVER=dummy with --video.
"""

import argparse

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import checkpoint_env_config, load_modules, untrained_modules
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import StagHuntContinuous
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.evaluate import sample_actions
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.render import Viewer


class VideoWriter:
    """An mp4 writer; frames are encoded as they come, not kept in memory."""

    def __init__(self, path, fps):
        self.path, self.fps, self.writer = path, fps, None

    def append_data(self, frame):
        import cv2

        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot write {self.path} (mp4v codec or path)")
        self.writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", help="save an mp4 here instead of only showing a window")
    parser.add_argument("--fps", type=int, default=8, help="decisions per second, in the window and the video (8: about real time)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint == "untrained":
        from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.config.config_env import config_env

        config = dict(config_env)
        modules = untrained_modules(config, seed=0)
    else:
        config = checkpoint_env_config(args.checkpoint)
        modules = load_modules(args.checkpoint)
    env = StagHuntContinuous(config)
    viewer = Viewer(env, window_size=800 if args.video else "fit", fps=None if args.video else args.fps)
    video = VideoWriter(args.video, args.fps) if args.video else None
    generator = torch.Generator().manual_seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    viewer.reset()
    try:
        while not viewer.closed:
            frame = viewer.draw()
            if video is not None:
                video.append_data(frame)
            obs, _, terms, truncs, _ = env.step(sample_actions(modules, {a: obs[a] for a in sorted(obs) if a in env.active}, generator))
            if terms["__all__"] or truncs["__all__"]:
                frame = viewer.draw()
                if video is not None:
                    video.append_data(frame)
                break
    finally:
        if video is not None:
            video.close()
        viewer.close()
    stats = env.episode_hunt_stats or env.hunt_stats
    print(
        f"seed {args.seed}: {env.time_step} physics steps; alive: "
        + ", ".join(f"{s} {len(env.animals[s])}" for s in env.animals)
        + f"; hunts by party 1/2/3/4+: {[stats[f'hunts_party_{n}'] for n in range(1, 5)]}"
        + f", kills: {[stats[f'kills_party_{n}'] for n in range(1, 5)]}"
    )


if __name__ == "__main__":
    main()
