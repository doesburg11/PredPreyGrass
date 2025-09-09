"""
DFT vs Reality (side-by-side, no rotation)

Left  = Reality: a prey moving on the grid (narrow Gaussian "spike")
Right = DFT memory field: bump forms from the stimulus, persists after prey disappears

What you'll see:
- Prey moves for MOVE_STEPS frames, then disappears.
- The memory bump on the right keeps a peak for a while and then fades.

Deps:
  pip install numpy matplotlib scipy pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
import os

# ----------------------------
# CONFIG
# ----------------------------
GRID          = 40           # grid size (GRID x GRID)
TIMESTEPS     = 45           # total frames
MOVE_STEPS    = 18           # prey is visible/moving for this many steps, then disappears
DT            = 1.0          # integration step
TAU           = 10.0         # time constant (memory length)
GAMMA         = 0.60         # global inhibition strength (competition)
INPUT_GAIN    = 1.0          # stimulus gain
EXC_SIGMA     = 1.2          # local excitation kernel width (cells) - boundary-safe
EXC_GAIN      = 0.45         # excitation strength
REAL_SIGMA    = 0.8          # how "pointy" the prey spike looks in Reality
ZLIM          = (0.0, 1.0)   # z-axis limits for both plots
SAVE_GIF      = True         # set True to save GIF
GIF_NAME      = "dft_vs_reality_side_by_side.gif"

# Camera (fixed; no rotation)
ELEV, AZIM = 35, 45

# Prey path: simple horizontal move, then vanish
# (Customize this path if you want more complex motion.)
start_xy = np.array([10, 14])  # (x, y)
step_vec = np.array([1,  0])   # move right 1 per frame
path_xy = []
for t in range(TIMESTEPS):
    if t < MOVE_STEPS:
        pos = start_xy + t * step_vec
        # keep inside the grid
        pos = np.clip(pos, [0,0], [GRID-1, GRID-1])
        path_xy.append(tuple(pos.tolist()))
    else:
        path_xy.append(None)  # prey absent

# ----------------------------
# Helpers
# ----------------------------
x = np.arange(GRID)
y = np.arange(GRID)
X, Y = np.meshgrid(x, y)  # shape (GRID, GRID)

def gaussian_2d(center_xy, sigma):
    """Narrow Gaussian 'spike' at center_xy (x,y)."""
    cx, cy = center_xy
    return np.exp(-0.5 * (((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2)))

def reality_frame(t):
    """Return the 'Reality' surface at time t: prey spike if present, else zeros."""
    if path_xy[t] is None:
        return np.zeros((GRID, GRID), dtype=float)
    return gaussian_2d(path_xy[t], REAL_SIGMA)

# ----------------------------
# Simulate DFT field (right plot)
# ----------------------------
u = np.zeros((GRID, GRID), dtype=float)
history_real = []
history_dft  = []

for t in range(TIMESTEPS):
    stim = reality_frame(t)                       # SAME signal as the left plot, but used as stimulus
    exc  = EXC_GAIN * gaussian_filter(u, sigma=EXC_SIGMA, mode="nearest")  # boundary safe
    global_inh = GAMMA * u.mean()

    du = (-u + INPUT_GAIN * stim + exc - global_inh) / TAU
    u  = u + DT * du
    u  = np.clip(u, 0.0, 1.0)

    history_real.append(stim.copy())
    history_dft.append(u.copy())

# ----------------------------
# Animate side-by-side 3D surfaces
# ----------------------------
fig = plt.figure(figsize=(12, 5))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axR = fig.add_subplot(1, 2, 2, projection="3d")

def draw_surface(ax, Z, title):
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_zlim(*ZLIM)
    ax.view_init(elev=ELEV, azim=AZIM)  # FIXED VIEW (no rotation)
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("Activation")
    ax.set_title(title)

# Optional: base plane to make the 0-level obvious
def draw_base_plane(ax):
    ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", linewidth=0.2, alpha=0.35)

def init():
    draw_surface(axL, history_real[0], "Reality: Prey (spike)")
    draw_base_plane(axL)
    draw_surface(axR, history_dft[0] , "DFT memory bump")
    draw_base_plane(axR)
    plt.tight_layout()
    return []

def animate(i):
    draw_surface(axL, history_real[i], f"Reality: Prey (t={i})")
    draw_base_plane(axL)
    draw_surface(axR, history_dft[i],  f"DFT memory bump (t={i})")
    draw_base_plane(axR)
    return []

ani = animation.FuncAnimation(fig, animate, frames=TIMESTEPS, init_func=init,
                              blit=False, interval=180)

plt.tight_layout()

if SAVE_GIF:
    try:
        ani.save(GIF_NAME, writer="pillow", fps=7)
        print(f"Saved GIF → {os.path.abspath(GIF_NAME)}")
    except Exception as e:
        print("Could not save GIF (need Pillow):", e)

plt.show()
