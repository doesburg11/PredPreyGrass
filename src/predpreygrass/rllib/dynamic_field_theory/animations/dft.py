"""
DFT 2D bump movie (3D surface animation)

- A Gaussian "prey" stimulus appears on a 2D grid for some steps, then disappears.
- A simple DFT-like field u(x,y,t) evolves with:
      u <- u + dt/τ * ( -u + input_gain*stim + exc(u) - γ*mean(u) )
  where exc(u) is local excitation via Gaussian convolution (FFT-based).

Outputs:
  - dft_2d_bump.gif  (always, needs Pillow)
  - dft_2d_bump.mp4  (optional, needs ffmpeg installed)

Deps:
  pip install numpy matplotlib pillow
  # optional for mp4:
  sudo apt-get install ffmpeg   (Linux)  or brew install ffmpeg (macOS)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from scipy.ndimage import gaussian_filter

# ----------------------------
# Config (tweak these freely)
# ----------------------------
GRID = 40                 # grid size (GRID x GRID)
TIMESTEPS = 50            # number of frames
DT = 1.0                  # integration step
TAU = 10.0                # time constant (memory length)
GAMMA = 0.55              # global inhibition strength (0.4–0.8 is nice)
INPUT_GAIN = 1.0          # stimulus gain
EXC_SIGMA = 1.5           # width of local excitation kernel (in cells)
EXC_GAIN = 0.7            # how strongly neighbors excite each other
ZLIM = (0.0, 1.0)         # z-axis limits for 3D plot
STIM_DURATION = 20        # stimulus is present for first N steps
STIM_CENTER = (25, 15)    # (x, y) center of the stimulus
STIM_SIGMA = 3.0          # width of the stimulus (in cells)
ROTATE_VIEW = True        # slowly rotate camera
SAVE_GIF = True
SAVE_MP4 = True           # needs ffmpeg; set False if you don't have it

# ----------------------------
# Helpers
# ----------------------------
def gaussian_2d(X, Y, cx, cy, sigma):
    return np.exp(-0.5 * (((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2)))

def make_gaussian_kernel(shape, sigma):
    """FFT-friendly Gaussian kernel with same shape as field."""
    h, w = shape
    y = np.fft.fftfreq(h) * h
    x = np.fft.fftfreq(w) * w
    X, Y = np.meshgrid(x, y)
    K = np.exp(-0.5 * (X**2 + Y**2) / (sigma**2))
    # normalize so sum(kernel) ~ 1 in spatial domain
    K /= K.max()
    return K

def conv2d_fft(u, K_fft):
    """Circular convolution via FFT (same shape, fast)."""
    return np.fft.ifft2(np.fft.fft2(u) * K_fft).real

# ----------------------------
# Build grid, kernel, stimulus
# ----------------------------
x = np.arange(GRID)
y = np.arange(GRID)
X, Y = np.meshgrid(x, y)  # note: Y rows, X cols; both shape (GRID, GRID)

# Pre-compute FFT kernel for local excitation
K_fft = np.fft.fft2(make_gaussian_kernel((GRID, GRID), EXC_SIGMA))

def stimulus(t):
    if t < STIM_DURATION:
        return gaussian_2d(X, Y, STIM_CENTER[0], STIM_CENTER[1], STIM_SIGMA)
    else:
        return np.zeros_like(X, dtype=float)

# ----------------------------
# Simulate field history
# ----------------------------
u = np.zeros((GRID, GRID), dtype=float)
history = []

for t in range(TIMESTEPS):
    stim = stimulus(t)
    exc = EXC_GAIN * gaussian_filter(u, sigma=EXC_SIGMA, mode="nearest")
    global_inh = GAMMA * u.mean()                    # global inhibition
    du = (-u + INPUT_GAIN * stim + exc - global_inh) / TAU
    u = u + DT * du
    u = np.clip(u, 0.0, 1.0)                         # keep it in a nice range (optional)
    history.append(u.copy())

# ----------------------------
# Animate (3D surface)
# ----------------------------
fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(111, projection="3d")

# Pre-allocate once for speed
surf = [None]

def init():
    ax.clear()
    ax.set_zlim(*ZLIM)
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("Activation")
    return []

def animate(i):
    ax.clear()
    Z = history[i]
    surf[0] = ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_zlim(*ZLIM)
    ax.set_title(f"DFT 2D bump (t={i})")
    ax.set_xlabel("X (grid)")
    ax.set_ylabel("Y (grid)")
    ax.set_zlabel("Activation")
    if ROTATE_VIEW:
        # slowly rotate around z to make it clearer
        ax.view_init(elev=35, azim=30 + i * 2.0)
    return [surf[0]]

ani = animation.FuncAnimation(fig, animate, frames=TIMESTEPS, init_func=init, blit=False, interval=150)
plt.tight_layout()

# ----------------------------
# Save files
# ----------------------------
out_gif = "dft_2d_bump.gif"
out_mp4 = "dft_2d_bump.mp4"

if SAVE_GIF:
    try:
        ani.save(out_gif, writer="pillow", fps=7)
        print(f"Saved GIF → {os.path.abspath(out_gif)}")
    except Exception as e:
        print("Could not save GIF (need Pillow?):", e)

if SAVE_MP4:
    try:
        ani.save(out_mp4, writer="ffmpeg", fps=20, dpi=150)
        print(f"Saved MP4 → {os.path.abspath(out_mp4)}")
    except Exception as e:
        print("Could not save MP4 (need ffmpeg?):", e)

plt.close(fig)
