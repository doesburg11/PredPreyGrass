"""Descriptive encounter measurements by fixed preference type."""
from collections import defaultdict
from itertools import combinations_with_replacement

from .individual_preferences import VALUES
from .persistence import PersistenceObserver, Spells

PAIR_TYPES = tuple(combinations_with_replacement(VALUES, 2))


def pair_label(pair):
    return ','.join(str(v) for v in pair)


class PreferencePersistenceObserver(PersistenceObserver):
    def __init__(self, preferences, **kwargs):
        self.preferences = preferences
        super().__init__(**kwargs)

    def reset(self, env, seed):
        self.pair_types = {pair: Spells(self.horizon) for pair in PAIR_TYPES}
        self.neighbor_types = {value: Spells(self.horizon) for value in VALUES}
        self.type_exposure = defaultdict(float)
        self.previous_types = None
        super().reset(env, seed)

    def sample(self, env):
        alive = {a.id for a in env.predators}
        newborn = alive - self.seen_ids if env.time_step > 0 else set()
        super().sample(env)
        counts = {value: sum(self.preferences.get(a) == value for a in alive) for value in VALUES}
        close_pairs = self.previous[3][100]
        grouped = {pair: set() for pair in PAIR_TYPES}
        for pair in close_pairs:
            types = tuple(sorted(self.preferences.get(a) for a in pair))
            grouped[types].add(pair)
        for types, keys in grouped.items():
            self.pair_types[types].update(keys, env.time_step, alive, newborn)
        neighbors = self.previous[2]
        for value in VALUES:
            self.neighbor_types[value].update({(agent, other) for agent, (other, _) in neighbors.items()
                                               if self.preferences.get(agent) == value}, env.time_step, alive, newborn)
        close_agents = {a for pair in close_pairs for a in pair}
        close_counts = {value: sum(self.preferences.get(a) == value for a in close_agents) for value in VALUES}
        if self.previous_types is not None:
            old_counts, old_grouped, old_close = self.previous_types
            # The base observer enforces elapsed==1; measure the preceding tick.
            for value in VALUES:
                self.type_exposure[f'predator_ticks_{value}'] += old_counts[value]
                self.type_exposure[f'close_predator_ticks_{value}'] += old_close[value]
            for a,b in PAIR_TYPES:
                label = pair_label((a,b))
                potential = old_counts[a]*(old_counts[a]-1)/2 if a == b else old_counts[a]*old_counts[b]
                self.type_exposure[f'potential_pair_ticks_{label}'] += potential
                self.type_exposure[f'close_pair_ticks_{label}'] += len(old_grouped[(a,b)])
        self.previous_types = (counts, grouped, close_counts)

    def finish(self, env):
        result = super().finish(env)
        e = self.type_exposure
        def ratio(a,b):
            return e[a]/e[b] if e[b] else None
        result['preference_groups'] = {
            'threshold_world_units': 100,
            'pairs': {pair_label(pair): {
                'spells': tracker.result(env.time_step),
                'close_pair_fraction_pair_time': ratio(f'close_pair_ticks_{pair_label(pair)}',f'potential_pair_ticks_{pair_label(pair)}'),
            } for pair,tracker in self.pair_types.items()},
            'focal_types': {str(value): {
                'close_neighbor_fraction_predator_time': ratio(f'close_predator_ticks_{value}',f'predator_ticks_{value}'),
                'nearest_neighbor_spells': self.neighbor_types[value].result(env.time_step),
            } for value in VALUES},
            'exposure': dict(e),
        }
        return result
