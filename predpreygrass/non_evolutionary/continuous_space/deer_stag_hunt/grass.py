"""Grass: renewable food patches in clusters that move by seed dispersal and
collapse when overgrazed (SPEC.md, section 4).

Two ways of regrowing (regrowth):
- "instant" (the world of RESULTS.md sections 1 and 2): an eaten patch is
  gone for respawn_delay physics steps, then back whole.
- "logistic": each patch has a biomass b in [0, 1] (1 = a whole patch,
  energy_grass_gain to a deer) that grows every physics step by
  growth_rate * (b + root_reserve) * (1 - b): slowly when grazed short
  (little leaf left; the small root reserve lets bare grass start again),
  fastest at half grown, slowly again near full. A deer eats all of a
  patch's biomass and gains that share of energy_grass_gain. With the
  defaults (growth_rate 0.0187, root_reserve 0.01) bare grass reaches 0.95
  in about 400 physics steps, as instant grass returns; a deer keeping a
  patch bare gains about 0.0004 energy per physics step, below a resting
  deer's cost (0.0016), so nibbling does not pay. An eaten patch re-seeds
  at once next to a mature patch (b >= MATURE) of its cluster, or from the
  seed bank, and grows there from 0; a cluster collapses when it has no
  more mature patches than overgrazing_threshold of its patches. Patches
  start whole.
- "grazed" (RESULTS.md section 4): biomass and growth as "logistic", but a
  deer crops instead of uprooting. Each physics step, a patch in reach is
  bitten by its closest deer, which takes bite of biomass (at most down to
  stubble, below which it cannot graze) and gains that share of
  energy_grass_gain. A grazed patch keeps its place and regrows from its
  stubble. A cluster with no patch at MATURE or above collapses as under
  "logistic" (killed back to bare, regrowing elsewhere).

The procedures and the order of their random draws follow add_walls's
GrassLayer exactly, without its wall handling; and stag_hunt_continuous's
grass, without its seasons and burrows.
"""

import math
import random
from dataclasses import dataclass

import numpy as np

REGROWTH = ("instant", "logistic", "grazed")
MATURE = 0.5  # logistic regrowth: biomass of a patch that seeds and counts as living

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.geometry import Torus, Vector


@dataclass
class GrassPatch:
    position: Vector
    ready_at: float = 0
    biomass: float = 1.0  # used by logistic regrowth only
    cluster: int | None = None
    # Where the patch regrows when it does: None keeps its spot, "disperse"
    # next to a living patch of its cluster, "centre" around the centre.
    regrow: str | None = None


class Grass:
    """Grass patches with their own random stream and physics-step clock.

    clustered: patches split evenly over cluster_count clusters, each centred
    in its own cell of a grid over the arena, scattered around the centre
    (normal offsets with sd cluster_spread). An eaten patch regrows after
    respawn_delay physics steps: in place, at a random spot of its cluster
    (random_respawn), or next to a living patch of its cluster (dispersal).
    overgrazing: a cluster grazed down to overgrazing_threshold of its
    patches collapses and regrows after overgrazing_delay steps around a new
    centre in a grid cell no other cluster uses.
    """

    def __init__(
        self,
        width,
        height,
        count=100,
        consume_radius=12.0,
        respawn_delay=400,
        clustered=True,
        cluster_count=20,
        cluster_spread=24.0,
        random_respawn=False,
        dispersal=True,
        dispersal_distance=24.0,
        overgrazing=True,
        overgrazing_threshold=0.0,
        overgrazing_delay=400,
        regrowth="instant",
        growth_rate=0.0187,
        root_reserve=0.01,
        bite=0.1,
        stubble=0.2,
    ):
        if regrowth not in REGROWTH:
            raise ValueError(f"grass_regrowth must be one of {REGROWTH}")
        for name, value in (("growth_rate", growth_rate), ("root_reserve", root_reserve), ("bite", bite)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"grass_{name} must be in (0, 1]")
        if isinstance(stubble, bool) or not isinstance(stubble, (int, float)) or not math.isfinite(stubble) or not 0 <= stubble < 1:
            raise ValueError("grass_stubble must be in [0, 1)")
        for name, value, minimum in (
            ("count", count, 0),
            ("respawn_delay", respawn_delay, 1),
            ("cluster_count", cluster_count, 1),
            ("overgrazing_delay", overgrazing_delay, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"grass_{name} must be an integer >= {minimum}")
        for name, value in (
            ("clustered", clustered),
            ("random_respawn", random_respawn),
            ("dispersal", dispersal),
            ("overgrazing", overgrazing),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"grass_{name} must be True or False")
        for name, value, positive in (
            ("consume_radius", consume_radius, True),
            ("cluster_spread", cluster_spread, True),
            ("dispersal_distance", dispersal_distance, True),
            ("overgrazing_threshold", overgrazing_threshold, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (positive and value == 0)
            ):
                raise ValueError(f"grass_{name} must be finite and " f"{'positive' if positive else 'nonnegative'}")
        if not 0 <= overgrazing_threshold < 1:
            raise ValueError("grass_overgrazing_threshold must be in [0, 1)")
        if (dispersal or overgrazing) and not clustered:
            raise ValueError("grass_dispersal and grass_overgrazing require grass_clustered")
        if dispersal and random_respawn:
            raise ValueError(
                "grass_dispersal and grass_random_respawn both choose where an " "eaten patch regrows; enable only one"
            )
        self.width = width
        self.height = height
        self.torus = Torus(width, height)
        self.count = count
        self.consume_radius = consume_radius
        self.respawn_delay = respawn_delay
        self.clustered = clustered
        self.cluster_count = cluster_count
        self.cluster_spread = cluster_spread
        self.random_respawn = random_respawn
        self.dispersal = dispersal
        self.dispersal_distance = dispersal_distance
        self.overgrazing = overgrazing
        self.overgrazing_threshold = overgrazing_threshold
        self.overgrazing_delay = overgrazing_delay
        self.regrowth = regrowth
        self.logistic = regrowth in ("logistic", "grazed")  # biomass that grows
        self.grazed = regrowth == "grazed"
        self.growth_rate = growth_rate
        self.root_reserve = root_reserve
        self.bite = bite
        self.stubble = stubble
        self.rng = random.Random()
        self.centres = []
        self.collapsed = []
        self.patches = []
        self.tick = 0
        self.total_consumed = 0
        self.total_collapses = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        if self.clustered:
            self.centres = self.cluster_centres()
            self.patches = []
            for i in range(self.count):
                cluster = i % self.cluster_count
                self.patches.append(GrassPatch(self.spot(cluster), cluster=cluster))
        else:
            self.centres = []
            self.patches = [GrassPatch(self.spot()) for _ in range(self.count)]
        self.collapsed = [False] * len(self.centres)
        self.tick = 0
        self.total_consumed = 0
        self.total_collapses = 0

    def grid(self):
        """Columns and rows of the grid that spreads clusters over the arena
        (5 x 4 for 20 clusters on a square arena)."""
        columns = math.ceil(math.sqrt(self.cluster_count * self.width / self.height))
        return columns, math.ceil(self.cluster_count / columns)

    def cell_of(self, position):
        columns, rows = self.grid()
        column = min(int(position.x / (self.width / columns)), columns - 1)
        row = min(int(position.y / (self.height / rows)), rows - 1)
        return row * columns + column

    def centre_in(self, cell):
        """A random point in the middle half of a grid cell."""
        columns, rows = self.grid()
        cell_w, cell_h = self.width / columns, self.height / rows
        return Vector(
            (cell % columns + self.rng.uniform(0.25, 0.75)) * cell_w,
            (cell // columns + self.rng.uniform(0.25, 0.75)) * cell_h,
        )

    def cluster_centres(self):
        """One centre per cluster, each in its own grid cell."""
        columns, rows = self.grid()
        cells = self.rng.sample(range(columns * rows), self.cluster_count)
        return [self.centre_in(cell) for cell in cells]

    def near(self, position, spread):
        """A point scattered around position (wrapped across the edges)."""
        return Vector(
            (position.x + self.rng.gauss(0, spread)) % self.width,
            (position.y + self.rng.gauss(0, spread)) % self.height,
        )

    def spot(self, cluster=None):
        """A random patch position: uniform over the arena, or scattered
        around a cluster's centre."""
        if cluster is None:
            return Vector(self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))
        return self.near(self.centres[cluster], self.cluster_spread)

    def available(self, patch):
        return patch.ready_at <= self.tick

    def edible(self, patch):
        if not self.available(patch):
            return False
        if self.grazed:
            return patch.biomass > self.stubble
        return not self.logistic or patch.biomass > 0

    def mature(self, patch):
        """Logistic regrowth: a patch that seeds new ones and keeps its
        cluster alive."""
        return self.available(patch) and patch.regrow is None and patch.biomass >= MATURE

    def grow(self):
        """Logistic regrowth: one physics step of growth for every patch
        that is in place."""
        r, b0 = self.growth_rate, self.root_reserve
        for patch in self.patches:
            if patch.regrow is None and self.available(patch):
                b = patch.biomass
                patch.biomass = min(1.0, b + r * (b + b0) * (1.0 - b))

    def advance(self, prey):
        """One physics step: regrowth placement (and, logistic, growth),
        grazing by the living prey (each patch to the closest prey within
        consume_radius, ties by agent ID) and overgrazing. Returns {prey id:
        patches eaten}, in whole patches (logistic, grazed: the biomass eaten)."""
        self.tick += 1
        for patch in self.patches:
            if patch.regrow is not None and self.available(patch):
                self.place(patch)
        if self.logistic:
            self.grow()
        eaten = {}
        living = [p for p in prey if p.alive]
        near = self.prey_in_reach(living)
        for index, patch in enumerate(self.patches):
            if not self.edible(patch):
                continue
            # The pre-filter only narrows the search; the exact torus distance
            # and tie-break decide.
            candidates = []
            for j in near.get(index, ()):
                distance = self.torus.distance(living[j].position, patch.position)
                if distance <= self.consume_radius:
                    candidates.append((distance, living[j].id))
            if candidates and self.grazed:
                _, agent = min(candidates)
                take = min(self.bite, patch.biomass - self.stubble)
                eaten[agent] = eaten.get(agent, 0) + take
                patch.biomass -= take
                self.total_consumed += 1  # bites
            elif candidates and self.logistic:
                _, agent = min(candidates)
                eaten[agent] = eaten.get(agent, 0) + patch.biomass
                patch.biomass = 0.0
                if self.random_respawn:
                    patch.position = self.spot(patch.cluster)
                elif self.dispersal:
                    patch.regrow = "disperse"
                    self.place(patch)
                self.total_consumed += 1
            elif candidates:
                _, agent = min(candidates)
                eaten[agent] = eaten.get(agent, 0) + 1
                patch.ready_at = self.tick + self.respawn_delay
                if self.random_respawn:
                    patch.position = self.spot(patch.cluster)
                elif self.dispersal:
                    patch.regrow = "disperse"
                self.total_consumed += 1
        if self.overgrazing:
            self.check_overgrazing()
        return eaten

    def prey_in_reach(self, prey):
        """{patch index: indices into prey} of the prey that may be within
        consume_radius of each edible patch (vectorized, float64, with a
        margin; advance() confirms each candidate exactly)."""
        if not prey:
            return {}
        indices = [i for i, patch in enumerate(self.patches) if self.edible(patch)]
        if not indices:
            return {}
        patch_xy = np.array(
            [(self.patches[i].position.x, self.patches[i].position.y) for i in indices],
            dtype=np.float64,
        )
        prey_xy = np.array([(p.position.x, p.position.y) for p in prey], dtype=np.float64)
        delta = np.abs(patch_xy[:, None, :] - prey_xy[None, :, :])
        delta = np.minimum(delta, np.array([self.width, self.height], dtype=np.float64) - delta)
        margin = 1e-5 * max(self.consume_radius, self.width, self.height) + 1e-6
        reach = np.hypot(delta[..., 0], delta[..., 1]) <= self.consume_radius + margin
        rows, columns = np.nonzero(reach)
        near = {}
        for row, column in zip(rows.tolist(), columns.tolist()):
            near.setdefault(indices[row], []).append(column)
        return near

    def place(self, patch):
        """Position a patch that regrows now."""
        if patch.regrow == "disperse":
            parents = [
                other
                for other in self.patches
                if other is not patch
                and other.cluster == patch.cluster
                and (self.mature(other) if self.logistic else other.regrow is None and self.available(other))
            ]
            if parents:
                parent = self.rng.choice(parents)
                patch.position = self.near(parent.position, self.dispersal_distance)
            else:  # nothing left to seed from: regrow from the seed bank
                patch.position = self.spot(patch.cluster)
        elif patch.regrow == "centre":
            patch.position = self.spot(patch.cluster)
        patch.regrow = None

    def check_overgrazing(self):
        """Collapse clusters grazed down to the threshold; re-arm recovered
        ones."""
        for cluster in range(len(self.centres)):
            members = [p for p in self.patches if p.cluster == cluster]
            living = sum((self.mature(p) if self.logistic else self.available(p)) for p in members)
            limit = self.overgrazing_threshold * len(members)
            if self.collapsed[cluster]:
                if living > limit:
                    self.collapsed[cluster] = False
            elif living <= limit:
                self.collapse(cluster, members)

    def collapse(self, cluster, members):
        """The cluster dies back and regrows elsewhere after a delay."""
        self.collapsed[cluster] = True
        self.total_collapses += 1
        used = {self.cell_of(c) for i, c in enumerate(self.centres) if i != cluster}
        used.add(self.cell_of(self.centres[cluster]))
        columns, rows = self.grid()
        free = [cell for cell in range(columns * rows) if cell not in used]
        if not free:  # every cell taken: any cell but its own
            own = self.cell_of(self.centres[cluster])
            free = [cell for cell in range(columns * rows) if cell != own] or [own]
        self.centres[cluster] = self.centre_in(self.rng.choice(free))
        regrows = self.tick + self.overgrazing_delay
        for patch in members:
            patch.ready_at = regrows
            patch.regrow = "centre"
            if self.logistic:
                patch.biomass = 0.0

    def observation_size(self):
        return 5 if self.logistic else 4

    def observation(self, observer, view_distance, fov):
        """The nearest edible patch in the observer's view cone, as
        [seen, dx / view_distance, dy / view_distance, distance /
        view_distance] (logistic, grazed: and its biomass), or zeros. Grazed:
        patches at stubble are not edible, so not seen."""
        visible = []
        for patch in self.patches:
            if not self.edible(patch):
                continue
            if not self.torus.in_view(observer, patch.position, view_distance, fov):
                continue
            dx, dy = self.torus.offset(observer.position, patch.position)
            visible.append((math.hypot(dx, dy), dx, dy, patch.biomass))
        if not visible:
            return np.zeros(self.observation_size(), dtype=np.float32)
        distance, dx, dy, biomass = min(visible)
        values = [1.0, dx / view_distance, dy / view_distance, distance / view_distance]
        if self.logistic:
            values.append(biomass)
        return np.asarray(values, dtype=np.float32)
