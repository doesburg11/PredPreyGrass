"""Read-only physical measurements; hidden state never enters action selection."""
from collections import defaultdict

import numpy as np


def wrapped_offset(start, end, size):
    """Shortest displacement on a rectangular torus, in world units."""
    return (np.asarray(end) - np.asarray(start) + np.asarray(size) / 2) % size - np.asarray(size) / 2


def nearest_neighbors(positions, size):
    """ID -> (nearest ID, distance); no neighbor is defined for a singleton."""
    if len(positions) < 2:
        return {}
    ids = list(positions)
    points = np.array([positions[agent] for agent in ids])
    offsets = wrapped_offset(points[:, None, :], points[None, :, :], size)
    distances = np.linalg.norm(offsets, axis=-1)
    np.fill_diagonal(distances, np.inf)
    indices = distances.argmin(axis=1)
    return {agent: (ids[j], float(distances[i, j])) for i, (agent, j) in enumerate(zip(ids, indices))}


def capture(env):
    """Copy positions and identify the same nearest visible target as the slots."""
    positions = {a.id: np.array([a.position.x, a.position.y]) for a in env.predators}
    size = np.array([env.width, env.height], dtype=float)
    targets = {}
    if env.observe_count['predator'] > 0:
        for animal in env.predators:
            seen = [other for other in env.predators if other is not animal and
                    env.torus.in_view(animal, other.position, animal.view_distance, animal.fov)]
            if seen:
                target = min(seen, key=lambda other: np.linalg.norm(wrapped_offset(positions[animal.id], positions[other.id], size)))
                if np.linalg.norm(wrapped_offset(positions[animal.id], positions[target.id], size)) > 1e-12:
                    targets[animal.id] = target.id
    return positions, targets


class MovementMetrics:
    """Agent-decision means, with explicit denominators and survivor exclusions.

    Projection and pair closing are rates per physics tick over each decision
    interval. Cosine excludes zero-displacement survivors; it describes net
    displacement, not the final velocity. NN distance includes unseen animals.
    """
    def __init__(self, size, distance_scale):
        self.size = np.array(size, dtype=float)
        self.scale = distance_scale
        self.values = defaultdict(list)
        self.counts = defaultdict(int)

    def record(self, before, after_positions, elapsed):
        positions, targets = before
        if elapsed <= 0:
            raise ValueError('elapsed physics ticks must be positive')
        self.values['predator_population'].append(len(positions))
        neighbors = nearest_neighbors(positions, self.size)
        for _, distance in neighbors.values():
            self.values['all_predator_nearest_neighbor_distance_normalized'].append(distance / self.scale)
        self.counts['decision_intervals'] += 1
        self.counts['physics_ticks'] += elapsed
        for agent, target in targets.items():
            self.counts['visible_target_intervals'] += 1
            if agent not in after_positions:
                self.counts['focal_death_intervals'] += 1
                continue
            direction = wrapped_offset(positions[agent], positions[target], self.size)
            distance = np.linalg.norm(direction)
            delta = wrapped_offset(positions[agent], after_positions[agent], self.size)
            displacement = np.linalg.norm(delta)
            projection = float(delta @ (direction / distance))
            self.values['visible_target_projection_per_tick'].append(projection / elapsed)
            self.values['visible_target_displacement_per_tick'].append(displacement / elapsed)
            self.values['visible_target_stationary_fraction'].append(float(displacement <= 1e-12))
            if displacement > 1e-12:
                self.values['visible_target_displacement_alignment_when_moving'].append(projection / displacement)
            if target not in after_positions:
                self.counts['target_death_intervals_with_focal_surviving'] += 1
                continue
            end_distance = np.linalg.norm(wrapped_offset(after_positions[agent], after_positions[target], self.size))
            self.values['visible_pair_closing_per_tick'].append(float(distance - end_distance) / elapsed)

    def result(self):
        names = ('predator_population', 'all_predator_nearest_neighbor_distance_normalized',
                 'visible_target_projection_per_tick', 'visible_target_displacement_per_tick',
                 'visible_target_stationary_fraction', 'visible_target_displacement_alignment_when_moving',
                 'visible_pair_closing_per_tick')
        return {
            'means': {name: float(np.mean(self.values[name])) if self.values[name] else None for name in names},
            'samples': {name: len(self.values[name]) for name in names},
            'counts': dict(self.counts),
        }
