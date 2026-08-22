"""Lightweight matplotlib renderer for ErlWorld's grid state -- walls, trees,
plants, corpses, agents, carnivores. Purely for inspection/debugging; not
part of the ERL mechanism, and not meant to be left on for a full-scale
1,000,000-step comparative run (see README.md's steps/sec numbers) -- use
sparingly via --render-every.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import (
    ErlWorld,
    TERRAIN_WALL,
    TERRAIN_TREE,
)

EMPTY, WALL, TREE, PLANT, CORPSE_AGENT, CORPSE_CARNIVORE, AGENT, CARNIVORE, AGENT_IN_TREE = range(9)

_LABELS = [
    "empty", "wall", "tree", "plant", "agent corpse", "carnivore corpse",
    "agent", "carnivore", "agent (sheltered)",
]
_COLORS = [
    "#f5f5f0",  # EMPTY
    "#3a3a3a",  # WALL
    "#8b5a2b",  # TREE
    "#7fc97f",  # PLANT
    "#8a7ca8",  # CORPSE_AGENT
    "#4a1f1f",  # CORPSE_CARNIVORE
    "#2266cc",  # AGENT
    "#cc2222",  # CARNIVORE
    "#66d9e8",  # AGENT_IN_TREE -- distinct from AGENT so shelter state is visible
]
_CMAP = ListedColormap(_COLORS)
_NORM = BoundaryNorm(list(range(len(_COLORS) + 1)), _CMAP.N)


def build_grid(world: ErlWorld) -> np.ndarray:
    """Render current world state to an (n, n) int8 grid of category codes.
    Draw order (later overwrites earlier) matches physical layering: terrain,
    then plants, then corpses, then live occupants on top.
    """
    n = world.grid_size
    grid = np.full((n, n), EMPTY, dtype=np.int8)
    grid[world.terrain == TERRAIN_WALL] = WALL
    grid[world.terrain == TERRAIN_TREE] = TREE
    grid[world.plant] = PLANT
    for (r, c), corpse in world.corpses.items():
        grid[r, c] = CORPSE_AGENT if corpse.kind == "agent" else CORPSE_CARNIVORE
    for (r, c), occ in world.occupant.items():
        if hasattr(occ, "carnivore_id"):
            grid[r, c] = CARNIVORE
        else:
            grid[r, c] = AGENT_IN_TREE if occ.in_tree else AGENT
    return grid


class WorldRenderer:
    """Reusable figure -- `show` for a live-updating window, `save` for PNG
    snapshots to disk. Both draw from the same categorical grid via
    `imshow`, cheap regardless of population size.
    """

    def __init__(self, grid_size: int, figsize: float = 7.0):
        self.fig, self.ax = plt.subplots(figsize=(figsize, figsize))
        self.im = self.ax.imshow(
            np.zeros((grid_size, grid_size), dtype=np.int8),
            cmap=_CMAP,
            norm=_NORM,
            interpolation="nearest",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.title = self.ax.set_title("")
        legend_handles = [Patch(color=c, label=lbl) for c, lbl in zip(_COLORS, _LABELS)]
        self.ax.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
            ncol=4, fontsize=8, frameon=False,
        )
        self.fig.tight_layout()

    def draw(self, world: ErlWorld, step: int):
        self.im.set_data(build_grid(world))
        counts = world.population_counts()
        self.title.set_text(f"step {step}   agents={counts['agent']}   carnivores={counts['carnivore']}")

    def show(self, world: ErlWorld, step: int, pause: float = 0.001):
        self.draw(world, step)
        plt.pause(pause)

    def save(self, world: ErlWorld, step: int, out_dir: Path):
        self.draw(world, step)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_dir / f"frame_{step:07d}.png", dpi=110)

    def close(self):
        plt.close(self.fig)
