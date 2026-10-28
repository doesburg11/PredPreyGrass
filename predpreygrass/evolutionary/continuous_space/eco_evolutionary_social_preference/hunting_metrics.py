"""Read-only hooks into existing captures, capped feeding and metabolic costs."""
import math
from .fitness_metrics import FitnessObserver
from .individual_preferences import VALUES

TOTALS = ('predator_ticks', 'captures', 'offered_hunting_energy', 'retained_hunting_energy',
          'movement_energy', 'resting_energy', 'reproduction_energy')


class HuntingObserver(FitnessObserver):
    def reset(self, env, seed):
        self.budgets = {}
        self.in_energy = False
        if getattr(self, '_hooked_env', None) is not env:
            self._install(env)
        super().reset(env, seed)

    def _ensure(self, animal):
        if animal.id not in self.budgets:
            self.budgets[animal.id] = {key: 0 for key in TOTALS}
            self.budgets[animal.id].update(initial_energy=float(animal.energy), final_energy=float(animal.energy))
        return self.budgets[animal.id]

    def _install(self, env):
        original_gain, original_energy = env._gain, env._energy

        def measured_gain(animal, amount):
            before = animal.energy
            result = original_gain(animal, amount)
            if self.in_energy and animal.species == 'predator':
                self.budgets[animal.id]['retained_hunting_energy'] += animal.energy - before
            return result

        def measured_energy(catches, eaten, terminateds, truncateds, infos):
            predators = tuple(env.predators)  # Keep references to animals dying this tick.
            for animal in predators:
                budget = self.budgets[animal.id]
                budget['predator_ticks'] += 1
                budget['movement_energy'] += env.speed_cost['predator'] * animal.velocity.mag()
                budget['resting_energy'] += env.resting_cost['predator'] * animal.cost_multiplier
            for predator, prey_energy in catches:
                budget = self.budgets[predator.id]
                budget['captures'] += 1
                if predator.alive:
                    budget['offered_hunting_energy'] += env.catch_efficiency * prey_energy
            self.in_energy = True
            try:
                result = original_energy(catches, eaten, terminateds, truncateds, infos)
            finally:
                self.in_energy = False
            for animal in predators:
                self.budgets[animal.id]['final_energy'] = float(animal.energy)
            return result

        env._gain, env._energy = measured_gain, measured_energy
        self._hooked_env = env

    def sample(self, env):
        super().sample(env)
        for animal in env.predators:
            self._ensure(animal)['final_energy'] = float(animal.energy)

    def finish(self, env):
        result = super().finish(env)
        records = []
        for life in result['fitness']['individuals']:
            budget = dict(self.budgets[life['id']])
            if budget['predator_ticks'] != life['observed_ticks']:
                raise RuntimeError('hunting and lifetime exposure differ')
            budget['reproduction_energy'] = life['offspring'] * env.initial_energy['predator']
            expected_change = (budget['retained_hunting_energy'] - budget['movement_energy']
                               - budget['resting_energy'] - budget['reproduction_energy'])
            error = budget['final_energy'] - budget['initial_energy'] - expected_change
            if not math.isclose(error, 0., abs_tol=1e-8):
                raise RuntimeError(f'energy ledger does not balance for {life["id"]}: {error}')
            records.append({'id': life['id'], 'preference': life['preference'], 'founder': life['founder'],
                            **budget, 'energy_balance_error': error})
        groups = {}
        for value in VALUES:
            groups[str(value)] = {}
            for cohort in ('all', 'founders', 'born'):
                rows = [r for r in records if r['preference'] == value and
                        (cohort == 'all' or r['founder'] == (cohort == 'founders'))]
                sums = {key: sum(r[key] for r in rows) for key in TOTALS}
                ticks = sums['predator_ticks']
                derived = {key+'_per_1000_predator_ticks': 1000*sums[key]/ticks if ticks else None
                           for key in TOTALS if key != 'predator_ticks'}
                sums['storage_cap_loss'] = sums['offered_hunting_energy'] - sums['retained_hunting_energy']
                sums['net_hunting_minus_metabolism'] = sums['retained_hunting_energy'] - sums['movement_energy'] - sums['resting_energy']
                derived['net_hunting_minus_metabolism_per_1000_predator_ticks'] = 1000*sums['net_hunting_minus_metabolism']/ticks if ticks else None
                cost = sums['movement_energy'] + sums['resting_energy']
                derived['retained_hunting_energy_per_metabolic_energy'] = sums['retained_hunting_energy']/cost if cost else None
                groups[str(value)][cohort] = {'count': len(rows), **sums, **derived}
        result['hunting'] = {'groups': groups, 'individuals': records,
                             'max_abs_energy_balance_error': max((abs(r['energy_balance_error']) for r in records), default=0.)}
        return result
