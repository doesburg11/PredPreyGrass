"""Externally assigned lifetime preferences; no genome, ancestry or simulation RNG."""
import hashlib
from itertools import permutations
import re

VALUES = (-1, 0, 1)
PERMUTATIONS = tuple(permutations(VALUES))


class IndividualPreferences:
    """Balanced randomized ID blocks, invariant to observation order and survival.

    Each consecutive ID block of three receives a permutation of -1,0,+1.
    Randomization is a pure hash of assignment seed, episode seed and block.
    This balances consecutive births, not the currently surviving population.
    Uniform modes exercise the same per-agent lookup for regression checks.
    """
    def __init__(self, mode='mixed', assignment_seed=20261027):
        if mode not in ('mixed', 'neutral', 'avoidance', 'attraction2'):
            raise ValueError('unknown preference assignment mode')
        self.mode = mode
        self.assignment_seed = assignment_seed
        self.uniform_value = {'neutral': 0, 'avoidance': -1, 'attraction2': 1}.get(mode)
        self.assignments = {}

    @property
    def is_uniform_neutral(self):
        return self.uniform_value == 0

    def start_episode(self, env, seed):
        self.episode_seed = seed
        self.assignments = {}
        self.register(env)

    def value_for_id(self, agent):
        match = re.fullmatch(r'predator_(\d+)', agent)
        if match is None:
            raise ValueError('expected a predator ID')
        if self.uniform_value is not None:
            return self.uniform_value
        block, index = divmod(int(match[1]), 3)
        key = f'{self.assignment_seed}:{self.episode_seed}:{block}'.encode()
        permutation = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(PERMUTATIONS)
        return PERMUTATIONS[permutation][index]

    def register(self, env):
        for animal in env.predators:
            if animal.id not in self.assignments:
                self.assignments[animal.id] = {'value': self.value_for_id(animal.id),
                                              'first_seen_tick': env.time_step,
                                              'founder': env.time_step == 0}

    def get(self, agent):
        return self.assignments[agent]['value']

    def result(self, env):
        def counts(records):
            return {str(value): sum(row['value'] == value for row in records) for value in VALUES}
        return {'mode': self.mode, 'assignment_seed': self.assignment_seed, 'episode_seed': self.episode_seed,
                'rule': 'SHA-256 shuffled permutations in consecutive predator ID blocks of three',
                'inherited': False, 'assignments': dict(self.assignments),
                'founder_counts': counts([r for r in self.assignments.values() if r['founder']]),
                'birth_counts': counts([r for r in self.assignments.values() if not r['founder']]),
                'final_living_counts': counts([self.assignments[a.id] for a in env.predators])}
