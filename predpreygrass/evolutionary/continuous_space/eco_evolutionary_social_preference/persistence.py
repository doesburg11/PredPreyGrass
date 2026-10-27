"""Identity-preserving proximity measurements at every physics tick.

Read-only world access is for measurement. No hidden state enters inference.
"""
from collections import defaultdict
import json
import gzip

import numpy as np

from .movement_metrics import nearest_neighbors, wrapped_offset


class Spells:
    """Track continuous key membership; deaths and observation ends censor it."""
    def __init__(self, horizon=100):
        self.horizon = horizon
        self.active = {}
        self.finished = []

    def update(self, keys, tick, alive, newborn=()):
        keys = set(keys)
        for key in set(self.active) - keys:
            start, left, birth = self.active.pop(key)
            reason = 'separation' if all(member in alive for member in key) else 'death'
            self.finished.append((tick - start, reason, left, birth))
        for key in keys - set(self.active):
            self.active[key] = (tick, tick == 0, any(member in newborn for member in key))

    def result(self, tick):
        rows = self.finished + [(tick - start, 'episode_end', left, birth)
                                for start, left, birth in self.active.values()]
        rows = [row for row in rows if row[0] > 0]
        def subset(exclude_birth):
            eligible = [r for r in rows if not r[2] and (not exclude_birth or not r[3])]
            success = sum(duration >= self.horizon for duration, *_ in eligible)
            failure = sum(duration < self.horizon and reason == 'separation' for duration, reason, *_ in eligible)
            censored = len(eligible) - success - failure
            completed = [duration for duration, reason, *_ in eligible if reason == 'separation']
            return {
                'observed_onsets': len(eligible), 'survive_horizon': success,
                'separate_before_horizon': failure, 'censored_before_horizon': censored,
                'retention_probability_among_known': success / (success + failure) if success + failure else None,
                'retention_lower_bound': success / len(eligible) if eligible else None,
                'retention_upper_bound': (success + censored) / len(eligible) if eligible else None,
                'completed_spells': len(completed),
                'mean_completed_duration_ticks': float(np.mean(completed)) if completed else None,
                'median_completed_duration_ticks': float(np.median(completed)) if completed else None,
            }
        return {
            'horizon_ticks': self.horizon,
            'nonbirth_onsets': subset(True), 'all_observed_onsets': subset(False),
            'left_censored_spells': sum(r[2] for r in rows),
            'birth_onset_spells': sum(r[3] for r in rows),
            'record_fields': ['duration_ticks', 'end_reason', 'left_censored', 'birth_onset'],
            'spell_records': rows,
        }


class PersistenceObserver:
    """Assign each integer-tick state to [tick,tick+1); final state has no weight.

    Initial spells are left-censored. A spell starting on the tick either
    member first appears after reset is birth-associated. Its removal from
    nonbirth analysis does not remove all possible parent/offspring encounters.
    Nearest-neighbor spells are directed and need not be mutually nearest.
    """
    def __init__(self, thresholds=(75, 100, 125), horizon=100, trace_dir=None, trace_seeds=(62, 63), condition='neutral'):
        self.thresholds = tuple(thresholds)
        self.horizon = horizon
        self.trace_dir = trace_dir
        self.trace_seeds = set(trace_seeds)
        self.condition = condition
        self.stream = None

    def reset(self, env, seed):
        if self.stream is not None:
            raise RuntimeError('previous episode observer was not finished')
        self.size = np.array([env.width, env.height], dtype=float)
        self.pairs = {r: Spells(self.horizon) for r in self.thresholds}
        self.neighbors = Spells(self.horizon)
        self.exposure = defaultdict(float)
        self.previous = None
        self.seen_ids = set()
        self.seed = seed
        if self.trace_dir is not None and seed in self.trace_seeds:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.stream = gzip.open(self.trace_dir / f'{self.condition}_seed{seed}.jsonl.gz', 'xt', compresslevel=3)
        self.sample(env)

    def sample(self, env):
        positions = {a.id: np.array([a.position.x, a.position.y]) for a in env.predators}
        tick = env.time_step
        if self.previous is not None:
            old_tick, old_n, old_neighbors, old_pairs, old_close_focals = self.previous
            elapsed = tick - old_tick
            if elapsed != 1:
                raise RuntimeError('persistence observer requires every physics tick')
            self.exposure['ticks'] += elapsed
            self.exposure['predator_ticks'] += old_n * elapsed
            self.exposure['potential_pair_ticks'] += old_n * (old_n - 1) / 2 * elapsed
            self.exposure['nn_distance_sum'] += sum(distance for _, distance in old_neighbors.values()) * elapsed
            self.exposure['nn_predator_ticks'] += len(old_neighbors) * elapsed
            for radius in self.thresholds:
                self.exposure[f'close_pair_ticks_{radius}'] += len(old_pairs[radius]) * elapsed
                self.exposure[f'close_predator_ticks_{radius}'] += len(old_close_focals[radius]) * elapsed
        ids = list(positions)
        n = len(ids)
        if n:
            points = np.array([positions[a] for a in ids])
            distances = np.linalg.norm(wrapped_offset(points[:, None], points[None, :], self.size), axis=-1)
        else:
            distances = np.empty((0, 0))
        pairs, close_focals = {}, {}
        newborn = set(positions) - self.seen_ids if tick > 0 else set()
        for radius in self.thresholds:
            i, j = np.where(np.triu(distances <= radius, k=1))
            pairs[radius] = {tuple(sorted((ids[a], ids[b]))) for a, b in zip(i, j)}
            close_focals[radius] = {member for pair in pairs[radius] for member in pair}
            self.pairs[radius].update(pairs[radius], tick, positions, newborn)
        neighbors = nearest_neighbors(positions, self.size)
        self.neighbors.update({(agent, other) for agent, (other, _) in neighbors.items()}, tick, positions, newborn)
        self.seen_ids.update(positions)
        self.previous = (tick, n, neighbors, pairs, close_focals)
        if self.stream is not None and (tick % 8 == 0 or tick == 0):
            self._trace(env, positions)

    def _trace(self, env, positions):
        self.stream.write(json.dumps({'tick': env.time_step, 'predators': {a: p.tolist() for a, p in positions.items()},
                                      'prey': {a.id: [a.position.x, a.position.y] for a in env.prey}}) + '\n')

    def finish(self, env):
        if self.stream is not None:
            if env.time_step % 8:
                self._trace(env, {a.id: np.array([a.position.x, a.position.y]) for a in env.predators})
            self.stream.close()
            self.stream = None
        e = self.exposure
        def ratio(a, b):
            return e[a] / e[b] if e[b] else None
        return {
            'sample_interval_ticks': 1, 'physics_ticks': int(e['ticks']),
            'mean_population_time_weighted': ratio('predator_ticks', 'ticks'),
            'mean_nn_distance_world_agent_time': ratio('nn_distance_sum', 'nn_predator_ticks'),
            'thresholds': {str(radius): {
                'close_pair_fraction_pair_time': ratio(f'close_pair_ticks_{radius}', 'potential_pair_ticks'),
                'close_neighbor_fraction_predator_time': ratio(f'close_predator_ticks_{radius}', 'predator_ticks'),
                'spells': self.pairs[radius].result(env.time_step),
            } for radius in self.thresholds},
            'nearest_neighbor_spells': self.neighbors.result(env.time_step),
            'exposure': dict(e),
        }
