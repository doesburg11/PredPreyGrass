from __future__ import annotations

"""
Visualize five consecutive steps for type_1_predator_0:
- Left: Observation prey channel (ch2) 7x7 as a 3D surface ("Reality").
- Right: DFT prey-trace u as a 3D surface ("DFT memory bump").

Outputs PNGs to output/dft_debug/prey_u_steps/step_XXX.png

Run:
  PYTHONPATH=./src /home/doesburg/Projects/PredPreyGrass/.conda/bin/python -m predpreygrass.rllib.dynamic_field_theory.debug.visualize_preych_u_five_steps
"""

import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

from predpreygrass.rllib.dynamic_field_theory.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.dynamic_field_theory.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.dynamic_field_theory.models.dft_memory_model import DFTMemoryConvModel
from predpreygrass.rllib.dynamic_field_theory.debug.make_gif_preych_u import save_gif as _save_gif


def make_mesh(h: int, w: int):
    X = np.arange(w)
    Y = np.arange(h)
    XX, YY = np.meshgrid(X, Y)
    return XX, YY


def draw_cylinder(ax, x_center: float, y_center: float, radius: float, height: float,
                  color: str = "#1f77b4", alpha: float = 0.9, resolution: int = 32):
    """Draw a rounded cylinder centered in a grid cell.

    ax: 3D axis
    (x_center, y_center): center in grid coordinates
    radius: cylinder radius (<= 0.5 fits within a 1x1 cell)
    height: cylinder height (prey energy)
    """
    if height <= 0:
        return

    theta = np.linspace(0.0, 2 * np.pi, resolution)
    z = np.linspace(0.0, height, 2)
    Theta, Z = np.meshgrid(theta, z)
    X = x_center + radius * np.cos(Theta)
    Y = y_center + radius * np.sin(Theta)

    # Lateral surface
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0.2, edgecolor="k", antialiased=True)

    # Caps (top and bottom)
    circle_top = [(x_center + radius * np.cos(t), y_center + radius * np.sin(t), height) for t in theta]
    circle_bottom = [(x_center + radius * np.cos(t), y_center + radius * np.sin(t), 0.0) for t in theta]
    ax.add_collection3d(Poly3DCollection([circle_top], facecolors=color, edgecolors="k", alpha=alpha))
    ax.add_collection3d(Poly3DCollection([circle_bottom], facecolors=color, edgecolors="k", alpha=alpha))


def draw_disc(ax, x_center: float, y_center: float, radius: float, z: float = 0.02,
              color: str = "#2ca02c", alpha: float = 0.6, resolution: int = 40):
    """Draw a flat filled disc at a given z (for grass overlay)."""
    theta = np.linspace(0.0, 2 * np.pi, resolution)
    circle = [(x_center + radius * np.cos(t), y_center + radius * np.sin(t), z) for t in theta]
    ax.add_collection3d(Poly3DCollection([circle], facecolors=color, edgecolors=None, alpha=alpha))


 


def plot_surfaces(prey_ch: np.ndarray, grass_ch: np.ndarray, u: np.ndarray, step: int, out_path: Path,
                  *, ate_last_step: bool = False, max_prey_energy: float | None = None):
    h, w = prey_ch.shape
    XX, YY = make_mesh(h, w)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("observation type_1_predator_0", y=0.98)
    # Reality (prey channel) as rounded cylinders with height = energy
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # Disable built-in gridlines; we draw our own thin floor cell borders
    ax1.grid(False)
    try:
        ax1.xaxis._axinfo["grid"]["linewidth"] = 0
        ax1.yaxis._axinfo["grid"]["linewidth"] = 0
        ax1.zaxis._axinfo["grid"]["linewidth"] = 0
    except Exception:
        pass
    max_pre = float(np.max(prey_ch)) if prey_ch.size else 0.0
    max_gr = float(np.max(grass_ch)) if grass_ch.size else 0.0
    for iy in range(h):
        for ix in range(w):
            val = float(prey_ch[iy, ix])
            if val > 1e-9:
                # draw a cylinder centered on the grid cell
                draw_cylinder(ax1, ix, iy, radius=0.35, height=val, color="#1f77b4", alpha=0.95, resolution=36)
            # draw grass as square bars (centered in cell) with height = grass energy
            gval = float(grass_ch[iy, ix])
            if gval > 1e-9:
                dx = dy = 0.8  # bar footprint size (square)
                x0 = ix - dx / 2.0
                y0 = iy - dy / 2.0
                ax1.bar3d(x0, y0, 0.0, dx, dy, gval, color="#2ca02c", alpha=0.65, shade=True, edgecolor="k", linewidth=0.3)
    # If predator ate in previous step, draw a green ring at the center cell
    if ate_last_step:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        theta = np.linspace(0, 2 * np.pi, 100)
        ring_r = 0.55
        xring = cx + ring_r * np.cos(theta)
        yring = cy + ring_r * np.sin(theta)
        zring = np.full_like(theta, 0.02)
        ax1.plot(xring, yring, zring, color="lime", linewidth=3.0)

    title = f"Reality: prey (t={step})"
    if max_prey_energy is not None:
        title += f"  E_max={max_prey_energy:.2f}"
    ax1.set_title(title)
    ax1.set_xlabel("X (grid)")
    ax1.set_ylabel("Y (grid)")
    ax1.set_zlabel("Energy level")
    ax1.set_xlim(-0.5, w - 0.5)
    ax1.set_ylim(-0.5, h - 0.5)
    ax1.set_zlim(0, max(6.0, max(max_pre, max_gr) + 0.5))
    # Show every integer tick 0..6 instead of 0,2,4,6
    ax1.set_xticks(np.arange(w))
    ax1.set_yticks(np.arange(h))
    ax1.set_xticklabels([str(i) for i in range(w)])
    ax1.set_yticklabels([str(i) for i in range(h)])
    # Per-unit floor gridlines (cell borders) helper; used on both panels (thinner lines)
    def draw_unit_grid(ax, hh: int, ww: int, *, z: float = 0.01, color: str = "#222222", lw: float = 0.6, alpha: float = 0.7):
        x_min, x_max = -0.5, ww - 0.5
        y_min, y_max = -0.5, hh - 0.5
        for k in range(ww + 1):
            x = -0.5 + k
            ax.plot([x, x], [y_min, y_max], [z, z], color=color, linewidth=lw, alpha=alpha)
        for k in range(hh + 1):
            y = -0.5 + k
            ax.plot([x_min, x_max], [y, y], [z, z], color=color, linewidth=lw, alpha=alpha)
    # Plane grid at x=0: Y divisions at integer cells and Z divisions (7 levels from 0 to z_top)
    def draw_plane_x0_grid(ax, hh: int, *, z_top: float, color: str = "#222222", lw: float = 0.6, alpha: float = 0.7):
        x0 = -0.5
        # vertical (along Z) lines at each integer Y boundary
        for k in range(hh + 1):
            y = -0.5 + k
            ax.plot([x0, x0], [y, y], [0.0, z_top], color=color, linewidth=lw, alpha=alpha)
        # horizontal (along Y) lines at evenly spaced Z levels (0..z_top in 6 steps -> 7 lines)
        for z in np.linspace(0.0, z_top, 7):
            ax.plot([x0, x0], [-0.5, hh - 0.5], [z, z], color=color, linewidth=lw, alpha=alpha)
    # Plane grid at y=hh-1: X divisions at integer cells and Z divisions (7 levels)
    def draw_plane_ymax_grid(ax, hh: int, ww: int, *, z_top: float, color: str = "#222222", lw: float = 0.6, alpha: float = 0.7):
        y0 = float(hh - 0.5)
        # vertical (along Z) lines at each integer X boundary
        for k in range(ww + 1):
            x = -0.5 + k
            ax.plot([x, x], [y0, y0], [0.0, z_top], color=color, linewidth=lw, alpha=alpha)
        # horizontal (along X) lines at evenly spaced Z levels (0..z_top in 6 steps -> 7 lines)
        for z in np.linspace(0.0, z_top, 7):
            ax.plot([-0.5, ww - 0.5], [y0, y0], [z, z], color=color, linewidth=lw, alpha=alpha)
    # Wall gridlines helper (vertical ZX and ZY planes)
    def draw_wall_grid(ax, hh: int, ww: int, *, z_top: float, color: str = "#222222", lw: float = 0.6, alpha: float = 0.7):
        x_min, x_max = -0.5, ww - 0.5
        y_min, y_max = -0.5, hh - 0.5
        # ZX plane at back wall only (y = y_min)
        for k in range(ww + 1):
            x = -0.5 + k
            ax.plot([x, x], [y_min, y_min], [0.0, z_top], color=color, linewidth=lw, alpha=alpha)
        # ZY plane at back-left wall only (x = x_min)
        for k in range(hh + 1):
            y = -0.5 + k
            ax.plot([x_min, x_min], [y, y], [0.0, z_top], color=color, linewidth=lw, alpha=alpha)
        # Top edges for neatness
        ax.plot([x_min, x_max], [y_min, y_min], [z_top, z_top], color=color, linewidth=lw, alpha=alpha)
        ax.plot([x_min, x_min], [y_min, y_max], [z_top, z_top], color=color, linewidth=lw, alpha=alpha)
    draw_unit_grid(ax1, h, w, z=0.01)
    draw_plane_x0_grid(ax1, h, z_top=ax1.get_zlim()[1])
    draw_plane_ymax_grid(ax1, h, w, z_top=ax1.get_zlim()[1])
    # No floating borders drawn over cylinders

    # Legend for Reality panel (prey cylinder, grass bar)
    prey_patch = mpatches.Patch(color="#1f77b4", label="prey (energy)")
    grass_patch = mpatches.Patch(color="#2ca02c", label="grass (energy)")
    ax1.legend(handles=[prey_patch, grass_patch], loc="upper left", bbox_to_anchor=(0.0, 1.05), frameon=False, fontsize=8)

    # DFT memory u
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Remove default axis gridlines so only our floor grid is visible
    ax2.grid(False)
    try:
        ax2.xaxis._axinfo["grid"]["linewidth"] = 0
        ax2.yaxis._axinfo["grid"]["linewidth"] = 0
        # Ensure panes don't occlude our custom lines
        for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
            try:
                axis.pane.set_alpha(0.0)
                axis.pane.fill = False
            except Exception:
                pass
    except Exception:
        pass
    surf2 = ax2.plot_surface(XX, YY, u, cmap="viridis", edgecolor="k", linewidth=0.3, antialiased=False)
    ax2.set_title(f"DFT memory bump u (t={step})")
    ax2.set_xlabel("X (grid)")
    ax2.set_ylabel("Y (grid)")
    ax2.set_zlabel("Activation")
    # Match left panel: show every integer tick and consistent bounds
    ax2.set_xlim(-0.5, w - 0.5)
    ax2.set_ylim(-0.5, h - 0.5)
    ax2.set_xticks(np.arange(w))
    ax2.set_yticks(np.arange(h))
    ax2.set_xticklabels([str(i) for i in range(w)])
    ax2.set_yticklabels([str(i) for i in range(h)])
    ax2.set_zlim(0, 1.0)

    # Floor gridlines plus plane at x=0 for reference
    draw_unit_grid(ax2, h, w, z=0.01)
    draw_plane_x0_grid(ax2, h, z_top=ax2.get_zlim()[1])
    draw_plane_ymax_grid(ax2, h, w, z_top=ax2.get_zlim()[1])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    out_dir = Path("output/dft_debug/prey_u_steps")
    print(f"[viz] Start at {datetime.now().isoformat(timespec='seconds')} -> {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_run_marker.txt").write_text(datetime.now().isoformat())

    cfg = dict(config_env)
    # Make grass less dense for visualization and reduce initial grass height/growth
    cfg.update({
        "initial_num_grass": 40,          # fewer grass tiles overall
        "initial_energy_grass": 1.0,      # shorter initial bars
        "energy_gain_per_step_grass": 0.02,  # slower regrowth
    })
    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=cfg.get("seed", 42))

    target = "type_1_predator_0"
    if target not in obs:
        # Fallback to any available agent so we still produce frames
        if len(obs) > 0:
            target = next(iter(obs.keys()))
            print(f"[viz] Default target missing; falling back to {target}", flush=True)
        else:
            print("[viz] No agents active after reset; aborting.", flush=True)
            return

    # DFT model (u only)
    obs_space = env.observation_spaces[target]
    act_space = env.action_spaces[target]
    model = DFTMemoryConvModel(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=act_space.n,
        model_config={
            "custom_model_config": {
                "prey_channel_index": 2,
                "dual_traces": False,
                "dft_dt": 1.0,
                "dft_tau": 10.0,
                "dft_gamma": 0.55,
                "dft_exc_sigma": 1.2,
                "dft_exc_gain": 0.60,
                "dft_input_gain": 1.60,
                "dft_zmin": 0.0,
                "dft_zmax": 1.0,
                "dft_kernel_size": 5,
            }
        },
        name="viz_dft_u_only",
    )
    state = model.get_initial_state()

    # Wait until prey visible (best effort); will proceed even if not visible
    def prey_visible(o):
        try:
            return np.any(o[2] > 0)
        except Exception:
            return False

    wait = 0
    # Wait until the target exists and prey is visible
    while ((target not in obs) or (not prey_visible(obs[target]))) and wait < 200:
        actions = {}
        for aid in env.agents:
            if "predator" in aid:
                actions[aid] = 4
            else:
                if "type_1" in aid:
                    actions[aid] = np.random.randint(0, env.type_1_act_range ** 2)
                else:
                    actions[aid] = np.random.randint(0, env.type_2_act_range ** 2)
        obs, _, _, _, _ = env.step(actions)
        wait += 1

    if (target not in obs):
        # Try to acquire any agent again
        if len(obs) > 0:
            target = next(iter(obs.keys()))
            print(f"[viz] Target missing; switching to {target}", flush=True)
            # Rebuild model for new spaces
            obs_space = env.observation_spaces[target]
            act_space = env.action_spaces[target]
            model = DFTMemoryConvModel(
                obs_space=obs_space,
                action_space=act_space,
                num_outputs=act_space.n,
                model_config=model.model_config,
                name="viz_dft_u_only",
            )
            state = model.get_initial_state()
        else:
            print("[viz] No agents present; cannot proceed.", flush=True)
            return

    # Capture more steps to see longer prey movement
    NUM_STEPS = 20
    for i in range(NUM_STEPS):
        ob = obs[target]
        prey_ch = ob[2]
        grass_ch = ob[3]
        ob_t = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
        _, state_out = model({"obs": ob_t}, state, None)
        u = state_out[0][0].cpu().numpy()

        out_path = out_dir / f"step_{i:03d}.png"
        ate_last_step = target in getattr(env, "agents_just_ate", set())
        prey_cells = int(np.count_nonzero(prey_ch > 0))
        # 1) Log tracked predator observation with prey energy summary and whether it ate
        tracked_prey_energy = float(prey_ch.max()) if prey_cells > 0 else 0.0
        print(
            f"Frame {i}: ate_last_step={ate_last_step}, prey_cells_in_FOV={prey_cells}, "
            f"max_prey_energy={tracked_prey_energy:.2f}, env_step={env.current_step}",
            flush=True,
        )

        # 2) Overlay grass patches on the Reality panel
        plot_surfaces(
            prey_ch,
            grass_ch,
            u,
            env.current_step,
            out_path,
            ate_last_step=ate_last_step,
            max_prey_energy=tracked_prey_energy,
        )
        print(f"Saved: {out_path}", flush=True)

        state = state_out

        # advance env
        actions = {}
        for aid in env.agents:
            if "predator" in aid:
                actions[aid] = 4
            else:
                if "type_1" in aid:
                    actions[aid] = np.random.randint(0, env.type_1_act_range ** 2)
                else:
                    actions[aid] = np.random.randint(0, env.type_2_act_range ** 2)
        obs, _, _, _, _ = env.step(actions)
        # Try to reacquire the target if it disappears
        reacq = 0
        while target not in obs and reacq < 50:
            # recompute actions for current agents set
            actions = {}
            for aid in env.agents:
                if "predator" in aid:
                    actions[aid] = 4
                else:
                    if "type_1" in aid:
                        actions[aid] = np.random.randint(0, env.type_1_act_range ** 2)
                    else:
                        actions[aid] = np.random.randint(0, env.type_2_act_range ** 2)
            obs, _, _, _, _ = env.step(actions)
            reacq += 1
        if target not in obs:
            print(f"{target} did not reappear within {reacq} steps; stopping.")
            break

    # Auto-build GIF after frames are saved
    try:
        import glob as _glob
        frames = sorted(_glob.glob(str(out_dir / "step_*.png")))
        if frames:
            gif_path = out_dir / "anim.gif"
            print(f"[viz] Building GIF -> {gif_path}", flush=True)
            _save_gif(frames, gif_path, per_step_sec=1.0, final_pause_sec=2.0)
        else:
            print("[viz] No frames to build GIF.", flush=True)
    except Exception as e:
        print(f"[viz] GIF build failed: {e}", flush=True)


if __name__ == "__main__":
    main()
