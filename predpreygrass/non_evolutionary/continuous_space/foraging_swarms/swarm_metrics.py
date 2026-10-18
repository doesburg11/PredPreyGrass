"""Swarm measurements on one species' positions and velocities (README,
"Measurements"). Pure functions of NumPy arrays on a torus; the environment
and the policies are not involved, so each can be tested on hand-made
configurations.

Conventions: positions and velocities are (n, 2) arrays in arena units;
the arena is a width x height torus. Every function measures one species
(conspecifics only).
"""

import math

import numpy as np


def torus_offsets(positions, width, height):
    """(n, n, 2) wrapped offsets from each animal (row) to each other
    (column)."""
    d = positions[None, :, :] - positions[:, None, :]
    d[..., 0] -= width * np.round(d[..., 0] / width)
    d[..., 1] -= height * np.round(d[..., 1] / height)
    return d


def torus_distances(positions, width, height):
    """(n, n) wrapped distances; the diagonal is infinite."""
    distances = np.hypot(*np.moveaxis(torus_offsets(positions, width, height), -1, 0))
    np.fill_diagonal(distances, np.inf)
    return distances


def nearest_neighbour_distances(positions, width, height):
    """Each animal's distance to its nearest conspecific (empty with fewer
    than two animals)."""
    if len(positions) < 2:
        return np.empty(0)
    return torus_distances(positions, width, height).min(axis=1)


_NULL_CACHE: dict = {}


def random_nearest_neighbour_mean(n, width, height, samples=200, seed=0):
    """Mean nearest-neighbour distance of n animals placed uniformly at
    random on the same torus (Monte Carlo, cached per n). The null for
    clustering: the Clark-Evans ratio R = observed / this; R < 1 means
    closer together than chance at the same density. Real animals cannot
    overlap (the bump rule), which pushes R up, so R < 1 is conservative."""
    key = (n, width, height, samples, seed)
    if key not in _NULL_CACHE:
        rng = np.random.default_rng(seed)
        means = [
            nearest_neighbour_distances(rng.uniform((0, 0), (width, height), size=(n, 2)), width, height).mean()
            for _ in range(samples)
        ]
        _NULL_CACHE[key] = float(np.mean(means))
    return _NULL_CACHE[key]


def unit_vectors(velocities):
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        units = velocities / speeds[:, None]
    return units, speeds


def local_alignment(positions, velocities, width, height, radius, eligible=None, focal=None):
    """Per-animal alignment with its neighbours within radius: the cosine
    between its own heading and the mean heading of its eligible
    neighbours. Returns an array with one value per focal animal that has
    at least one eligible neighbour (others are left out).

    eligible: boolean mask of animals that count (moving, old enough);
    default all. focal: the eligible animals to report on; default all of
    them. 1 means moving exactly as the neighbours do, 0 unrelated, -1
    opposite."""
    n = len(positions)
    if eligible is None:
        eligible = np.ones(n, dtype=bool)
    if focal is None:
        focal = eligible
    index = np.flatnonzero(eligible)
    if len(index) < 2:
        return np.empty(0)
    units, _ = unit_vectors(velocities[index])
    distances = torus_distances(positions[index], width, height)
    near = distances <= radius
    values = []
    for i in range(len(index)):
        if not focal[index[i]]:
            continue
        neighbours = near[i]
        if not neighbours.any():
            continue
        mean = units[neighbours].sum(axis=0)
        norm = math.hypot(*mean)
        if norm == 0:
            continue
        values.append(float(units[i] @ (mean / norm)))
    return np.asarray(values)


def shuffled_alignment(positions, velocities, width, height, radius, eligible, rng, shuffles=20):
    """The null for local_alignment: the same positions, with headings
    shuffled among the eligible animals. Its mean is what alignment would
    be if headings were unrelated to who is near whom, at the same density
    and with the same headings in the population (e.g. all prey drifting
    the same way would count here too)."""
    index = np.flatnonzero(eligible)
    if len(index) < 2:
        return np.empty(0)
    values = []
    for _ in range(shuffles):
        shuffled = velocities.copy()
        shuffled[index] = velocities[rng.permutation(index)]
        values.append(local_alignment(positions, shuffled, width, height, radius, eligible))
    return np.concatenate(values)


def polarization(velocities, eligible=None):
    """Length of the mean heading of the eligible animals: 1 when all move
    the same way, near 0 for random headings (nan for none)."""
    if eligible is not None:
        velocities = velocities[eligible]
    if len(velocities) == 0:
        return math.nan
    units, _ = unit_vectors(velocities)
    return float(math.hypot(*units.mean(axis=0)))


def clusters(positions, width, height, link):
    """Groups as connected components: two animals are linked when they are
    at most link apart (single linkage). Returns a label per animal."""
    n = len(positions)
    labels = np.arange(n)
    if n < 2:
        return labels

    def find(i):
        while labels[i] != i:
            labels[i] = labels[labels[i]]
            i = labels[i]
        return i

    linked = torus_distances(positions, width, height) <= link
    for i, j in zip(*np.nonzero(np.triu(linked, 1))):
        a, b = find(i), find(j)
        if a != b:
            labels[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def group_sizes(labels):
    """Size of the group of each animal, in the order of labels."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts[inverse]


class PairTracker:
    """How long pairs of animals stay within link of each other, in
    decisions. A spell ends when the two part; when one of them dies, or
    the episode ends, the spell is censored (its true length is unknown:
    at least what was seen). excluded(a, b) -> True leaves a pair out
    (e.g. parent and offspring)."""

    def __init__(self, excluded=None):
        self.excluded = excluded or (lambda a, b: False)
        self.running: dict = {}
        self.finished: list = []
        self.censored: list = []

    def update(self, ids, positions, width, height, link):
        together = set()
        if len(ids) >= 2:
            close = torus_distances(positions, width, height) <= link
            for i, j in zip(*np.nonzero(np.triu(close, 1))):
                pair = tuple(sorted((ids[i], ids[j])))
                if not self.excluded(*pair):
                    together.add(pair)
        present = set(ids)
        for pair in list(self.running):
            if pair not in together:
                parted = pair[0] in present and pair[1] in present
                (self.finished if parted else self.censored).append(self.running.pop(pair))
        for pair in together:
            self.running[pair] = self.running.get(pair, 0) + 1

    def close(self):
        """(finished spell lengths, censored spell lengths)."""
        censored = self.censored + list(self.running.values())
        self.running = {}
        return self.finished, censored


class GroupTracker:
    """Lifetimes of groups of at least min_size, in decisions. A group
    continues when a group in the next decision shares at least
    min_overlap of its members (Jaccard index), the best match first;
    otherwise it ends. Groups still alive at the episode's end are
    censored."""

    def __init__(self, min_size=3, min_overlap=0.5):
        self.min_size, self.min_overlap = min_size, min_overlap
        self.running: list = []  # (members, age)
        self.finished: list = []

    def update(self, ids, labels):
        groups = {}
        for agent, label in zip(ids, labels):
            groups.setdefault(label, set()).add(agent)
        current = [members for members in groups.values() if len(members) >= self.min_size]
        matches = sorted(
            (
                (len(old & new) / len(old | new), i, j)
                for i, (old, _) in enumerate(self.running)
                for j, new in enumerate(current)
            ),
            reverse=True,
        )
        used_old, used_new, continued = set(), set(), []
        for overlap, i, j in matches:
            if overlap < self.min_overlap or i in used_old or j in used_new:
                continue
            used_old.add(i)
            used_new.add(j)
            continued.append((current[j], self.running[i][1] + 1))
        for i, (_, age) in enumerate(self.running):
            if i not in used_old:
                self.finished.append(age)
        continued += [(members, 1) for j, members in enumerate(current) if j not in used_new]
        self.running = continued

    def close(self):
        censored = [age for _, age in self.running]
        self.running = []
        return self.finished, censored


def kaplan_meier(finished, censored):
    """Kaplan-Meier survival of spell lengths with right-censoring: a list
    of (length t, share of spells lasting longer than t) at each length
    where a spell ended. A censored spell of length c counts as lasting at
    least c (it is at risk up to and including c)."""
    finished = np.asarray(finished, dtype=float)
    censored = np.asarray(censored, dtype=float)
    survival, curve = 1.0, []
    for t in np.unique(finished):
        at_risk = (finished >= t).sum() + (censored >= t).sum()
        survival *= 1.0 - (finished == t).sum() / at_risk
        curve.append((float(t), float(survival)))
    return curve


def km_median(finished, censored):
    """Median spell length from kaplan_meier (nan if more than half the
    spells outlast every finished one)."""
    for t, survival in kaplan_meier(finished, censored):
        if survival <= 0.5:
            return t
    return math.nan


def km_share_lasting(finished, censored, length):
    """Share of spells lasting at least length decisions, from
    kaplan_meier."""
    survival = 1.0
    for t, s in kaplan_meier(finished, censored):
        if t >= length:
            break
        survival = s
    return survival
