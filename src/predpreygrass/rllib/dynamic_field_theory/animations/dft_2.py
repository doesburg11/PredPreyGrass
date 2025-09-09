"""
DFT vs Reality (side-by-side, fixed camera)

Left  = Reality: BLUE prey pillar, height = energy (~6). z-axis: "Energy level" (0..6).
Right = DFT memory field: activation bump peaks ~0.8 by default (auto-calibrated). z-axis: "Activation" (0..1).

Outputs:
    - dft_vs_reality_pillar.gif  (needs Pillow)

Notes:
    - You can let the prey make a half-way 90° turn (left or right). See TURN_HALF_WAY and TURN_DIR.
    - You can adjust the bump peak via BUMP_TARGET (default 0.8). Set AUTO_CALIBRATE=False to disable.
    - You can switch the prey pillar to a round cylinder via PREY_PILLAR_ROUND=True.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter

# ----------------------------
# CONFIG
# ----------------------------
GRID          = 15
TIMESTEPS     = 35
MOVE_STEPS    = 18          # prey visible for this many steps, then disappears

# Reality scale (energy)
PREY_ENERGY   = 6.0          # pillar height
ZLIM_REAL     = (0.0, 6.0)   # Reality z-range

# DFT dynamics & scale (activation in 0..1)
DT            = 1.0          # integration step
TAU           = 10.0         # memory length
GAMMA         = 0.55         # global inhibition strength
EXC_SIGMA     = 1.2          # local excitation kernel width (cells)
EXC_GAIN      = 0.60         # local excitation gain  (↑ makes bump taller)
INPUT_GAIN    = 1.60         # stimulus gain           (↑ makes bump taller)
ZLIM_DFT      = (0.0, 1.0)   # DFT z-range (activation)

# Target peak and auto-calibration
BUMP_TARGET   = 0.80         # desired peak activation when prey present
AUTO_CALIBRATE = True        # if True, scales INPUT_GAIN to reach ~BUMP_TARGET
CALIB_STEPS    = 32          # short run to estimate peak for scaling

# Rendering: hide near-zero activation to remove colored ground surface
SURFACE_MIN    = 1e-3        # values below this are masked from the 3D surface

# Reality rendering (left): pillar shape
PREY_PILLAR_ROUND   = True   # draw prey as a round cylinder instead of a box
PREY_PILLAR_RADIUS  = 0.45   # radius in cell units (cell center at +0.5)
PREY_PILLAR_SIDES   = 48     # angular resolution for the circular shape
PREY_PILLAR_RADIAL  = 16     # radial resolution for the top disk fill
PREY_PILLAR_EDGE_COLOR = "black"  # outline color for the top disk
PREY_PILLAR_EDGE_WIDTH = 1.5       # linewidth for the outline

SAVE_GIF      = True
SAVE_MP4      = True        # set True if you have ffmpeg
GIF_NAME      = "dft_vs_reality_pillar.gif"

# Camera (fixed; no rotation)
ELEV, AZIM = 35, 45

# Stepwise prey path (integer cells), optional half-way turn, then vanish
start_xy = np.array([2, 4])      # (x, y)
step_vec = np.array([1, 0])      # initial heading: move right 1 cell per frame

# Turn configuration
TURN_HALF_WAY = True             # if True, turn after MOVE_STEPS // 2 steps
TURN_DIR      = "left"           # "left" (CCW) or "right" (CW) relative to heading

def turn_vector(v: np.ndarray, direction: str) -> np.ndarray:
    """Return v turned 90° left (CCW) or right (CW). v must be one of unit lattice vectors.

    Mapping (x,y):
      - left (CCW):  (1,0)->(0,1)->(-1,0)->(0,-1)->(1,0)
      - right (CW):  (1,0)->(0,-1)->(-1,0)->(0,1)->(1,0)
    """
    vx, vy = int(v[0]), int(v[1])
    if direction.lower() == "left":      # CCW
        return np.array([-vy, vx])
    elif direction.lower() == "right":   # CW
        return np.array([vy, -vx])
    else:
        raise ValueError(f"TURN_DIR must be 'left' or 'right', got: {direction}")

path_xy = []
for t in range(TIMESTEPS):
    if t < MOVE_STEPS:
        if TURN_HALF_WAY and t >= (MOVE_STEPS // 2):
            # First half along initial heading, second half after 90° turn
            first_half_steps = MOVE_STEPS // 2
            turned_vec = turn_vector(step_vec, TURN_DIR)
            pos = start_xy + first_half_steps * step_vec + (t - first_half_steps) * turned_vec
        else:
            pos = start_xy + t * step_vec
        pos = np.clip(pos, [0, 0], [GRID - 1, GRID - 1])
        path_xy.append((int(pos[0]), int(pos[1])))
    else:
        path_xy.append(None)

# ----------------------------
# Helpers
# ----------------------------
x = np.arange(GRID)
y = np.arange(GRID)
X, Y = np.meshgrid(x, y)

def draw_base_grid(ax):
    """Light grey grid on z=0 (PPG-like)."""
    for i in range(GRID + 1):
        ax.plot([i, i], [0, GRID], [0, 0], color="#d0d0d0", linewidth=0.5, alpha=0.9)
        ax.plot([0, GRID], [i, i], [0, 0], color="#d0d0d0", linewidth=0.5, alpha=0.9)

def draw_prey_pillar(ax, cell, height):
    """Draw the prey as a pillar in the cell.

    If PREY_PILLAR_ROUND is True, draw a cylinder centered in the cell; otherwise use a box.
    """
    if cell is None:
        return
    cx, cy = cell
    color = "#015089"
    alpha = 0.95
    if not PREY_PILLAR_ROUND:
        dx = dy = 0.95
        ax.bar3d(cx + (1 - dx) / 2.0, cy + (1 - dy) / 2.0, 0.0, dx, dy, height,
                 color=color, alpha=alpha, shade=False)
        return

    # Cylinder: lateral surface
    x0 = cx + 0.5
    y0 = cy + 0.5
    r  = PREY_PILLAR_RADIUS
    n_theta = PREY_PILLAR_SIDES
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z_vals = np.array([0.0, height])
    TH, ZZ = np.meshgrid(thetas, z_vals)
    XX = x0 + r * np.cos(TH)
    YY = y0 + r * np.sin(TH)
    ax.plot_surface(XX, YY, ZZ, color=color, alpha=alpha, linewidth=0, shade=False)

    # Top disk to give a solid cap
    n_rad = PREY_PILLAR_RADIAL
    rr = np.linspace(0.0, r, n_rad)
    THt, RRt = np.meshgrid(thetas, rr)
    Xtop = x0 + RRt * np.cos(THt)
    Ytop = y0 + RRt * np.sin(THt)
    Ztop = np.full_like(Xtop, height)
    ax.plot_surface(Xtop, Ytop, Ztop, color=color, alpha=alpha, linewidth=0, shade=False)

    # Outline the top disk with a black circular edge
    theta_edge = np.linspace(0.0, 2.0 * np.pi, max(64, n_theta))
    Xedge = x0 + r * np.cos(theta_edge)
    Yedge = y0 + r * np.sin(theta_edge)
    Zedge = np.full_like(theta_edge, height)
    ax.plot(Xedge, Yedge, Zedge, color=PREY_PILLAR_EDGE_COLOR, linewidth=PREY_PILLAR_EDGE_WIDTH)

def stimulus_from_prey(cell, energy):
    """Narrow Gaussian stimulus centered at the prey cell, scaled to 'energy'."""
    if cell is None:
        return np.zeros((GRID, GRID), dtype=float)
    cx, cy = cell
    sigma = 0.8  # ~1 cell wide
    stim = np.exp(-0.5 * (((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2)))
    stim = (stim / (stim.max() + 1e-8)) * energy
    return stim

# ----------------------------
# Simulate DFT field (right plot)
# - Normalize stimulus to 0..1 so DFT activation is on 0..1 scale
# - Tune INPUT_GAIN/EXC_GAIN/GAMMA so peak reaches ~0.9 when visible
# ----------------------------
history_dft = []

def simulate_history(input_gain: float):
    """Run the DFT dynamics over the configured path and return (history, peak)."""
    u_sim = np.zeros((GRID, GRID), dtype=float)
    hist = []
    peak = 0.0
    for t in range(TIMESTEPS):
        stim_energy = stimulus_from_prey(path_xy[t], PREY_ENERGY)      # 0..6
        stim_norm   = stim_energy / (PREY_ENERGY + 1e-8)               # 0..1

        exc  = EXC_GAIN * gaussian_filter(u_sim, sigma=EXC_SIGMA, mode="nearest")
        global_inh = GAMMA * u_sim.mean()

        du = (-u_sim + input_gain * stim_norm + exc - global_inh) / TAU
        u_sim  = np.clip(u_sim + DT * du, ZLIM_DFT[0], ZLIM_DFT[1])
        hist.append(u_sim.copy())
        if u_sim.max() > peak:
            peak = float(u_sim.max())
    return hist, peak

# Optional: auto-calibrate INPUT_GAIN so that the peak activation ≈ BUMP_TARGET (using moving path)
if AUTO_CALIBRATE:
    _orig_input_gain = INPUT_GAIN
    _, obs_peak = simulate_history(INPUT_GAIN)
    if obs_peak > 1e-6:
        scale = BUMP_TARGET / obs_peak
        scale = float(np.clip(scale, 0.25, 6.0))
        INPUT_GAIN *= scale
        print(f"[DFT] Auto-calibrate (path): peak≈{obs_peak:.3f} → target {BUMP_TARGET:.2f}. INPUT_GAIN {_orig_input_gain:.3f} → {INPUT_GAIN:.3f} (×{scale:.2f})")
    else:
        print("[DFT] Auto-calibrate (path): peak was ~0; skipping scaling.")

# Final simulation with (possibly) adjusted INPUT_GAIN
history_dft, final_peak = simulate_history(INPUT_GAIN)
print(f"[DFT] Final observed peak along path: {final_peak:.3f}")

# ----------------------------
# Animate side-by-side
# ----------------------------
fig = plt.figure(figsize=(12, 5))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axR = fig.add_subplot(1, 2, 2, projection="3d")

def style_axes_left(ax, title):
    ax.set_zlim(*ZLIM_REAL)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("Energy level")   # requested label
    ax.set_title(title)

def style_axes_right(ax, title):
    ax.set_zlim(*ZLIM_DFT)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("Activation")     # requested label
    ax.set_title(title)

def init():
    axL.clear(); axR.clear()
    style_axes_left(axL,  "Reality: single prey (pillar = energy)")
    style_axes_right(axR, "DFT memory bump")
    draw_base_grid(axL); draw_base_grid(axR)
    draw_prey_pillar(axL, path_xy[0], PREY_ENERGY)
    Z0 = history_dft[0]
    Zp = np.where(Z0 < SURFACE_MIN, np.nan, Z0)
    axR.plot_surface(X, Y, Zp, cmap="viridis", rstride=1, cstride=1,
                     linewidth=0, antialiased=True)
    plt.tight_layout()
    return []

def animate(i):
    axL.clear(); axR.clear()
    style_axes_left(axL,  f"Reality: prey @ {path_xy[i]} (t={i})")
    style_axes_right(axR, f"DFT memory bump (t={i})")
    draw_base_grid(axL); draw_base_grid(axR)
    draw_prey_pillar(axL, path_xy[i], PREY_ENERGY if path_xy[i] is not None else 0.0)
    Z0 = history_dft[i]
    Zp = np.where(Z0 < SURFACE_MIN, np.nan, Z0)
    axR.plot_surface(X, Y, Zp, cmap="viridis", rstride=1, cstride=1,
                     linewidth=0, antialiased=True)
    return []

ani = animation.FuncAnimation(fig, animate, frames=TIMESTEPS, init_func=init,
                              blit=False, interval=220)
plt.tight_layout()

if SAVE_GIF:
    ani.save(GIF_NAME, writer="pillow", fps=6)
    print("Saved GIF →", os.path.abspath(GIF_NAME))

if SAVE_MP4:
    try:
        ani.save(GIF_NAME.replace(".gif", ".mp4"), writer="ffmpeg", fps=18, dpi=150)
        print("Saved MP4 →", os.path.abspath(GIF_NAME.replace('.gif', '.mp4')))
    except Exception as e:
        print("Could not save MP4 (need ffmpeg):", e)

plt.show()
