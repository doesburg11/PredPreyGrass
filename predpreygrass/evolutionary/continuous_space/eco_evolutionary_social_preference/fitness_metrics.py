"""Read-only lifetime, reproduction and energy observations at physics ticks."""
from .individual_preferences import VALUES
from .preference_metrics import PreferencePersistenceObserver


class FitnessObserver(PreferencePersistenceObserver):
    """Retain existing encounter output and add individual life histories.

    Energy is integrated from the alive sample at the start of each interval.
    Remaining animals are censored at episode end, never counted as deaths.
    Offspring counts come from the environment's retained family register.
    """
    def reset(self, env, seed):
        self.energy_bins = {}
        self.lives = {}
        self.energy_previous = {}
        self.fitness_tick = None
        super().reset(env, seed)  # Dispatches sample once at tick zero.

    def sample(self, env):
        super().sample(env)
        tick = env.time_step
        alive = {a.id: a for a in env.predators}
        if self.fitness_tick is not None:
            if tick != self.fitness_tick + 1:
                raise ValueError('fitness sampling requires every physics tick')
            interval = self.energy_bins.setdefault(self.fitness_tick // 1000,
                {str(v): {'energy_integral': 0., 'predator_ticks': 0} for v in VALUES})
            for agent, energy in self.energy_previous.items():
                group = interval[str(self.preferences.get(agent))]
                group['energy_integral'] += energy
                group['predator_ticks'] += 1
                self.lives[agent]['energy_integral'] += energy
                self.lives[agent]['observed_ticks'] += 1
                if agent not in alive:
                    self.lives[agent]['death_tick'] = tick
        for agent, animal in alive.items():
            if agent not in self.lives:
                if agent in self.energy_previous:
                    raise RuntimeError('inconsistent lifetime record')
                self.lives[agent] = {
                    'id': agent, 'preference': self.preferences.get(agent),
                    'birth_tick': tick, 'founder': tick == 0,
                    'death_tick': None, 'observed_ticks': 0, 'energy_integral': 0.,
                }
            elif self.lives[agent]['death_tick'] is not None:
                raise RuntimeError('predator ID reused after death')
        self.energy_previous = {agent: float(a.energy) for agent, a in alive.items()}
        self.fitness_tick = tick

    def finish(self, env):
        if env.time_step != self.fitness_tick:
            raise ValueError('finish requires a sample at the final tick')
        result = super().finish(env)
        records = []
        for agent, life in sorted(self.lives.items()):
            row = dict(life)
            row['right_censored'] = row['death_tick'] is None
            row['end_tick'] = env.time_step if row['right_censored'] else row['death_tick']
            row['offspring'] = int(env.family[agent][2])
            row['mean_energy'] = row['energy_integral']/row['observed_ticks'] if row['observed_ticks'] else None
            if row['observed_ticks'] != row['end_tick'] - row['birth_tick']:
                raise RuntimeError('lifetime exposure mismatch')
            records.append(row)
        total_births = sum(not r['founder'] for r in records)
        if sum(r['offspring'] for r in records) != total_births:
            raise RuntimeError('offspring register does not match observed births')
        groups = {}
        for value in VALUES:
            groups[str(value)] = {}
            for cohort in ('founders', 'born', 'all'):
                rows = [r for r in records if r['preference'] == value and
                        (cohort == 'all' or r['founder'] == (cohort == 'founders'))]
                ticks = sum(r['observed_ticks'] for r in rows)
                offspring = sum(r['offspring'] for r in rows)
                groups[str(value)][cohort] = {
                    'count': len(rows), 'deaths': sum(not r['right_censored'] for r in rows),
                    'right_censored': sum(r['right_censored'] for r in rows),
                    'predator_ticks': ticks, 'offspring': offspring,
                    'mean_observed_lifespan': ticks/len(rows) if rows else None,
                    'mean_observed_offspring': offspring/len(rows) if rows else None,
                    'offspring_per_1000_predator_ticks': 1000*offspring/ticks if ticks else None,
                    'mean_energy_predator_time': sum(r['energy_integral'] for r in rows)/ticks if ticks else None,
                }
        bins = []
        for index, values in sorted(self.energy_bins.items()):
            bins.append({'start_tick': index*1000, 'end_tick': min((index+1)*1000, env.time_step),
                         'groups': {v: {**g, 'mean_energy': g['energy_integral']/g['predator_ticks']
                                      if g['predator_ticks'] else None} for v,g in values.items()}})
        result['fitness'] = {'episode_end_tick': env.time_step, 'groups': groups,
                             'energy_bins_1000_ticks': bins, 'individuals': records}
        return result
