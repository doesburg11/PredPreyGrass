from __future__ import annotations

"""
Create an animated GIF from the PNG frames in output/dft_debug/prey_u_steps/.

Run:
  PYTHONPATH=./src /home/doesburg/Projects/PredPreyGrass/.conda/bin/python -m predpreygrass.rllib.dynamic_field_theory.debug.make_gif_preych_u
"""

import glob
from pathlib import Path


def save_gif(frames, out_path: Path, per_step_sec: float = 1.0, final_pause_sec: float = 2.0, base_ms: int = 100):
    """Save a GIF with robust timing.

    We duplicate frames so even viewers that ignore per-frame duration lists
    display ~1s per step and a 2s pause on the last frame. base_ms controls
    the duration of each duplicated frame.
    """
    # Compute duplication counts per frame
    n = max(len(frames), 1)
    per_step_ms = int(round(1000 * float(per_step_sec)))
    last_ms = int(round(1000 * (float(per_step_sec) + float(final_pause_sec))))
    repeats = [max(per_step_ms // base_ms, 1)] * n
    repeats[-1] = max(last_ms // base_ms, 1)

    expanded = []
    for f, r in zip(frames, repeats):
        expanded.extend([f] * r)

    try:
        from PIL import Image
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in expanded]
        imgs[0].save(
            out_path,
            save_all=True,
            append_images=imgs[1:],
            duration=base_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        print(f"GIF saved (PIL): {out_path}")
    except Exception as e_pil:
        try:
            import imageio.v2 as imageio
            imgs = [imageio.imread(str(f)) for f in expanded]
            imageio.mimsave(out_path, imgs, duration=base_ms / 1000.0, loop=0)
            print(f"GIF saved (imageio): {out_path}")
        except Exception as e_io:
            raise RuntimeError(f"Failed to save GIF with PIL and imageio. PIL error={e_pil}; imageio error={e_io}")


def main():
    frames = sorted(glob.glob("output/dft_debug/prey_u_steps/step_*.png"))
    if not frames:
        print("No frames found. Run visualize_preych_u_five_steps.py first.")
        return
    out_path = Path("output/dft_debug/prey_u_steps/anim.gif")
    save_gif(frames, out_path, per_step_sec=1.0, final_pause_sec=2.0)


if __name__ == "__main__":
    main()
