#!/usr/bin/env python3
"""
emerging_cooperation.py

Minimal ecology (no learning) with:
1) Heatmap of local clustering (local neighborhood mean cooperation level)
2) Spatial animation (live grid)
3) Lotka–Volterra-style oscillation plot (Predators vs Prey + phase plot)
4) Trait evolution: continuous cooperation level in [0,1]

Animation (maximum clarity):
- Base layer: clustering heatmap (local mean predator cooperation)
- Overlay: prey density heatmap with:
    * interpolation="nearest"
    * log scaling via LogNorm (so dense patches don’t wash out everything)
    * zeros masked (LogNorm can’t represent 0)
- Predators: open circles, edge color encodes cooperation trait

Fixes:
- Robust scatter.set_offsets() with true empty (0,2) numpy arrays
- Keep animation alive via fig.ani

Run:
  python emerging_cooperation.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm


# ============================================================
# CONFIG
# ============================================================

W, H = 60, 60

PRED_INIT = 250
PREY_INIT = 600

STEPS = 2500

# --- Predator energetics ---
METAB_PRED = 0.06
MOVE_COST = 0.008
COOP_COST = 0.18          # cost per tick * coop_level
BIRTH_THRESH_PRED = 3.0
MUT_RATE = 0.03
MUT_SIGMA = 0.08
LOCAL_BIRTH_R = 1

# --- Hunt mechanics ---
HUNT_R = 1
P0 = 0.08
KILL_ENERGY = 4.0
SHARE_MIN_COOP = 0.02

# --- Prey dynamics ---
PREY_MOVE_PROB = 0.25
PREY_REPRO_PROB = 0.04
PREY_MAX = 1600

# --- Visualization ---
ANIMATE = True
ANIM_STEPS = 500
ANIM_INTERVAL_MS = 40

CLUST_R = 2

# Predator marker look
PRED_SIZE = 70
PRED_EDGE_LINEWIDTH = 1.2

# Layering (alpha)
PREY_DENSITY_ALPHA = 0.35   # overlay strength
CLUSTER_ALPHA = 1.0         # base clustering heatmap alpha

SEED = None                 # set to int for reproducibility (e.g. 42)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Predator:
    x: int
    y: int
    energy: float
    coop: float  # continuous trait in [0,1]


@dataclass
class Prey:
    x: int
    y: int


def wrap(v: int, L: int) -> int:
    return v % L


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# ============================================================
# CORE ECOLOGY
# ============================================================

def step_world(preds: List[Predator], preys: List[Prey]) -> Tuple[List[Predator], List[Prey]]:
    """One tick update: prey update, predator hunting, predator costs/move/repro/death."""

    # ---- Prey move + reproduce
    new_preys: List[Prey] = []
    prey_count = len(preys)

    crowd = prey_count / max(1, PREY_MAX)
    repro_scale = max(0.0, 1.0 - crowd)

    for pr in preys:
        if random.random() < PREY_MOVE_PROB:
            pr.x = wrap(pr.x + random.choice([-1, 0, 1]), W)
            pr.y = wrap(pr.y + random.choice([-1, 0, 1]), H)

        new_preys.append(pr)

        if random.random() < PREY_REPRO_PROB * repro_scale:
            cx = wrap(pr.x + random.choice([-1, 0, 1]), W)
            cy = wrap(pr.y + random.choice([-1, 0, 1]), H)
            new_preys.append(Prey(cx, cy))

    preys = new_preys

    # ---- Index prey by cell
    prey_by_cell: Dict[Tuple[int, int], List[int]] = {}
    for i, pr in enumerate(preys):
        prey_by_cell.setdefault((pr.x, pr.y), []).append(i)

    # ---- Index predators by cell
    pred_by_cell: Dict[Tuple[int, int], List[int]] = {}
    for i, pd in enumerate(preds):
        pred_by_cell.setdefault((pd.x, pd.y), []).append(i)

    # ---- Hunting
    prey_killed_indices = set()

    for (cx, cy), pred_idxs in pred_by_cell.items():
        candidates: List[int] = []
        for dy in range(-HUNT_R, HUNT_R + 1):
            yy = (cy + dy) % H
            for dx in range(-HUNT_R, HUNT_R + 1):
                xx = (cx + dx) % W
                candidates.extend(prey_by_cell.get((xx, yy), []))

        if not candidates:
            continue

        sum_contrib = sum(preds[i].coop for i in pred_idxs)
        pkill = 1.0 - (1.0 - P0) ** (sum_contrib + 1e-6)

        if random.random() < pkill:
            victim = None
            for idx in candidates:
                if idx not in prey_killed_indices:
                    victim = idx
                    break
            if victim is None:
                continue

            prey_killed_indices.add(victim)

            eligible = [(i, preds[i].coop) for i in pred_idxs if preds[i].coop > SHARE_MIN_COOP]
            if eligible:
                tot = sum(w for _, w in eligible)
                for i, w in eligible:
                    preds[i].energy += KILL_ENERGY * (w / tot)

    if prey_killed_indices:
        preys = [pr for i, pr in enumerate(preys) if i not in prey_killed_indices]

    # ---- Predator costs, movement, reproduction, death
    new_preds: List[Predator] = []

    random.shuffle(preds)
    for pd in preds:
        pd.energy -= METAB_PRED
        pd.energy -= MOVE_COST
        pd.energy -= COOP_COST * pd.coop

        pd.x = wrap(pd.x + random.choice([-1, 0, 1]), W)
        pd.y = wrap(pd.y + random.choice([-1, 0, 1]), H)

        if pd.energy >= BIRTH_THRESH_PRED:
            pd.energy *= 0.5
            child = Predator(pd.x, pd.y, pd.energy, pd.coop)

            child.x = wrap(child.x + random.randint(-LOCAL_BIRTH_R, LOCAL_BIRTH_R), W)
            child.y = wrap(child.y + random.randint(-LOCAL_BIRTH_R, LOCAL_BIRTH_R), H)

            if random.random() < MUT_RATE:
                child.coop = clamp01(child.coop + random.gauss(0.0, MUT_SIGMA))

            new_preds.append(child)

        if pd.energy > 0.0:
            new_preds.append(pd)

    return new_preds, preys


# ============================================================
# CLUSTERING HEATMAP
# ============================================================

def compute_local_clustering_field(preds: List[Predator], r: int) -> np.ndarray:
    """HxW field: mean predator coop in neighborhood radius r around each cell."""
    cell_sum = np.zeros((H, W), dtype=float)
    cell_cnt = np.zeros((H, W), dtype=int)

    for pd in preds:
        cell_sum[pd.y, pd.x] += pd.coop
        cell_cnt[pd.y, pd.x] += 1

    field = np.zeros((H, W), dtype=float)

    for y in range(H):
        for x in range(W):
            s = 0.0
            c = 0
            for dy in range(-r, r + 1):
                yy = (y + dy) % H
                for dx in range(-r, r + 1):
                    xx = (x + dx) % W
                    s += cell_sum[yy, xx]
                    c += cell_cnt[yy, xx]
            field[y, x] = (s / c) if c > 0 else 0.0

    return field


def compute_prey_density(preys: List[Prey]) -> np.ndarray:
    """HxW array: prey count per cell."""
    dens = np.zeros((H, W), dtype=float)
    for pr in preys:
        dens[pr.y, pr.x] += 1.0
    return dens


def mask_zeros_for_lognorm(arr: np.ndarray) -> np.ma.MaskedArray:
    """Mask zeros (and negatives) so LogNorm can be used safely."""
    return np.ma.masked_less_equal(arr, 0.0)


# ============================================================
# RUN SIMULATION
# ============================================================

def run_sim() -> Tuple[
    List[int], List[int], List[float], List[float], List[List[Predator]], List[List[Prey]], List[Predator]
]:
    if SEED is not None:
        random.seed(SEED)

    preds: List[Predator] = [
        Predator(random.randrange(W), random.randrange(H), 1.5, random.random())
        for _ in range(PRED_INIT)
    ]
    preys: List[Prey] = [
        Prey(random.randrange(W), random.randrange(H))
        for _ in range(PREY_INIT)
    ]

    pred_hist: List[int] = []
    prey_hist: List[int] = []
    mean_coop_hist: List[float] = []
    var_coop_hist: List[float] = []

    preds_snaps: List[List[Predator]] = []
    preys_snaps: List[List[Prey]] = []

    for t in range(STEPS):
        preds, preys = step_world(preds, preys)

        pred_n = len(preds)
        prey_n = len(preys)

        pred_hist.append(pred_n)
        prey_hist.append(prey_n)

        if pred_n > 0:
            mu = sum(p.coop for p in preds) / pred_n
            var = sum((p.coop - mu) ** 2 for p in preds) / pred_n
        else:
            mu = 0.0
            var = 0.0

        mean_coop_hist.append(mu)
        var_coop_hist.append(var)

        if ANIMATE and t < ANIM_STEPS:
            preds_snaps.append([Predator(p.x, p.y, p.energy, p.coop) for p in preds])
            preys_snaps.append([Prey(p.x, p.y) for p in preys])

        if (t + 1) % 200 == 0:
            print(f"t={t+1:4d} preds={pred_n:4d} preys={prey_n:4d} mean_coop={mu:.3f} var={var:.3f}")

        if pred_n == 0 and prey_n == 0:
            print(f"Total extinction at step {t}.")
            break

    return pred_hist, prey_hist, mean_coop_hist, var_coop_hist, preds_snaps, preys_snaps, preds


# ============================================================
# PLOTS
# ============================================================

def plot_lv_style(pred_hist: List[int], prey_hist: List[int]) -> None:
    plt.figure()
    plt.plot(prey_hist, label="Prey")
    plt.plot(pred_hist, label="Predators")
    plt.xlabel("Time step")
    plt.ylabel("Count")
    plt.title("Population oscillations (Lotka–Volterra style)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(prey_hist, pred_hist)
    plt.xlabel("Prey count")
    plt.ylabel("Predator count")
    plt.title("Phase plot (Predators vs Prey)")
    plt.show()


def plot_trait_evolution(mean_coop_hist: List[float], var_coop_hist: List[float]) -> None:
    plt.figure()
    plt.plot(mean_coop_hist)
    plt.xlabel("Time step")
    plt.ylabel("Mean cooperation level")
    plt.title("Trait evolution: mean cooperation level over time")
    plt.ylim(0, 1)
    plt.show()

    plt.figure()
    plt.plot(var_coop_hist)
    plt.xlabel("Time step")
    plt.ylabel("Variance of cooperation level")
    plt.title("Trait evolution: variance over time")
    plt.show()


def plot_clustering_heatmap(preds_final: List[Predator]) -> None:
    field = compute_local_clustering_field(preds_final, CLUST_R)
    plt.figure()
    plt.imshow(field, origin="lower", interpolation="nearest")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Local clustering heatmap (mean coop in radius {CLUST_R})")
    plt.colorbar(label="Local mean cooperation level")
    plt.show()


# ============================================================
# ANIMATION (cluster heatmap + prey density overlay (nearest + log) + predators edge-colored by coop)
# ============================================================

def animate_world(preds_snaps: List[List[Predator]], preys_snaps: List[List[Prey]]) -> None:
    if not preds_snaps:
        print("No snapshots recorded for animation.")
        return

    fig, ax = plt.subplots()

    # Base: clustering heatmap
    clust0 = compute_local_clustering_field(preds_snaps[0], CLUST_R)
    clust_im = ax.imshow(clust0, origin="lower", alpha=CLUSTER_ALPHA, interpolation="nearest")
    clust_cb = plt.colorbar(clust_im, ax=ax)
    clust_cb.set_label("Local mean cooperation level")

    # Overlay: prey density with LogNorm (mask zeros)
    prey0 = compute_prey_density(preys_snaps[0])
    prey0m = mask_zeros_for_lognorm(prey0)
    prey_vmax0 = max(1.0, float(prey0.max()))
    prey_norm = LogNorm(vmin=1.0, vmax=prey_vmax0)

    prey_im = ax.imshow(
        prey0m,
        origin="lower",
        alpha=PREY_DENSITY_ALPHA,
        interpolation="nearest",
        norm=prey_norm,
    )
    prey_cb = plt.colorbar(prey_im, ax=ax)
    prey_cb.set_label("Prey density (log scale; zeros masked)")

    # Predators: open circles, edge color by coop
    empty_xy = np.empty((0, 2), dtype=float)

    cmap = cm.get_cmap()  # default colormap
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    pred_scatter = ax.scatter(
        [], [],
        marker="o",
        s=PRED_SIZE,
        facecolors="none",
        edgecolors=[],
        linewidths=PRED_EDGE_LINEWIDTH,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    pred_cb = plt.colorbar(sm, ax=ax)
    pred_cb.set_label("Predator coop trait (edge color)")

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)

    def init():
        pred_scatter.set_offsets(empty_xy)
        pred_scatter.set_edgecolors(np.empty((0, 4), dtype=float))
        ax.set_title("Live grid: clustering + prey density (log) + predators (edge color = coop)")
        return clust_im, prey_im, pred_scatter

    def update(frame_idx: int):
        preds = preds_snaps[frame_idx]
        preys = preys_snaps[frame_idx]

        # update clustering layer
        clust = compute_local_clustering_field(preds, CLUST_R)
        clust_im.set_data(clust)

        # update prey density layer (masked zeros + dynamic vmax)
        prey_d = compute_prey_density(preys)
        prey_dm = mask_zeros_for_lognorm(prey_d)

        vmax = max(1.0, float(prey_d.max()))
        # update norm vmax so contrast stays useful
        prey_im.norm = LogNorm(vmin=1.0, vmax=vmax)
        prey_im.set_data(prey_dm)

        # update predators
        if preds:
            pred_xy = np.array([(p.x, p.y) for p in preds], dtype=float)
            coop = np.array([p.coop for p in preds], dtype=float)
            colors = cmap(norm(coop))
            pred_scatter.set_offsets(pred_xy)
            pred_scatter.set_edgecolors(colors)
        else:
            pred_scatter.set_offsets(empty_xy)
            pred_scatter.set_edgecolors(np.empty((0, 4), dtype=float))

        ax.set_title(f"Live grid (step {frame_idx+1}/{len(preds_snaps)})")
        return clust_im, prey_im, pred_scatter

    fig.ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(preds_snaps),
        init_func=init,
        interval=ANIM_INTERVAL_MS,
        blit=False,
        repeat=False,
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    pred_hist, prey_hist, mean_coop_hist, var_coop_hist, preds_snaps, preys_snaps, preds_final = run_sim()

    plot_lv_style(pred_hist, prey_hist)
    plot_trait_evolution(mean_coop_hist, var_coop_hist)

    if preds_final:
        plot_clustering_heatmap(preds_final)
    else:
        print("No predators at end -> clustering heatmap skipped.")

    if ANIMATE:
        animate_world(preds_snaps, preys_snaps)


if __name__ == "__main__":
    main()
