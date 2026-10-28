from .fitness_metrics import FitnessObserver
from .individual_preferences import IndividualPreferences
from .test_persistence import world
import pytest


def test_birth_death_censoring_and_energy_intervals():
    p = IndividualPreferences('neutral')
    family = {'predator_0': ['predator', None, 0]}
    def sample(tick, energies):
        env = world(tick, {agent: (i*10, 0) for i,agent in enumerate(energies)})
        for a in env.predators: a.energy = energies[a.id]
        env.family = family
        return env
    env = sample(0, {'predator_0': 5.})
    p.start_episode(env, 82)
    observer = FitnessObserver(p, trace_seeds=())
    observer.reset(env, 82)
    family['predator_0'][2] = 1
    family['predator_1'] = ['predator', None, 0]
    env = sample(1, {'predator_0': 2., 'predator_1': 5.})
    p.register(env); observer.sample(env)
    env = sample(2, {'predator_1': 4.})
    p.register(env); observer.sample(env)
    result = observer.finish(env)['fitness']
    parent, child = result['individuals']
    assert parent['death_tick'] == 2
    assert parent['observed_ticks'] == 2 and parent['mean_energy'] == 3.5
    assert parent['offspring'] == 1 and not parent['right_censored']
    assert child['birth_tick'] == 1 and child['observed_ticks'] == 1
    assert child['mean_energy'] == 5. and child['right_censored']
    assert child['death_tick'] is None
    group = result['groups']['0']
    assert group['founders']['mean_observed_lifespan'] == 2
    assert group['all']['predator_ticks'] == 3
    assert group['all']['mean_energy_predator_time'] == 4.
    energy = result['energy_bins_1000_ticks'][0]
    assert energy['start_tick'] == 0 and energy['end_tick'] == 2
    assert energy['groups']['0']['predator_ticks'] == 3
    assert energy['groups']['0']['mean_energy'] == 4.
    assert result['groups']['1']['all']['mean_observed_lifespan'] is None
    family['predator_1'][2] = 1
    with pytest.raises(RuntimeError, match='offspring register'):
        observer.finish(env)


def test_end_tick_birth_has_zero_exposure_and_is_censored():
    env = world(0, {'predator_0': (0, 0)})
    env.predators[0].energy = 5.
    env.family = {'predator_0': ['predator', None, 0]}
    p = IndividualPreferences('neutral'); p.start_episode(env, 1)
    observer = FitnessObserver(p, trace_seeds=()); observer.reset(env, 1)
    final = world(1, {'predator_0': (0, 0), 'predator_1': (1, 0)})
    for a in final.predators: a.energy = 5.
    final.family = {'predator_0': ['predator', None, 1], 'predator_1': ['predator', None, 0]}
    p.register(final); observer.sample(final)
    result = observer.finish(final)['fitness']['individuals'][1]
    assert result['right_censored'] and result['observed_ticks'] == 0
    assert result['mean_energy'] is None
    assert not result['founder'] and result['birth_tick'] == result['end_tick'] == 1
    final.time_step = 2
    with pytest.raises(ValueError, match='final tick'): observer.finish(final)


def test_energy_bin_boundary_and_preference_partition():
    p = IndividualPreferences()
    env = world(0, {f'predator_{i}': (i*10,0) for i in range(3)})
    env.family = {a.id: ['predator',None,0] for a in env.predators}
    p.start_episode(env, 82)
    for a in env.predators: a.energy = 10.+p.get(a.id)
    observer = FitnessObserver(p, trace_seeds=()); observer.reset(env,82)
    for tick in range(1,1002):
        env.time_step = tick
        observer.sample(env)
    result = observer.finish(env)['fitness']
    assert [b['end_tick']-b['start_tick'] for b in result['energy_bins_1000_ticks']] == [1000,1]
    for v in (-1,0,1):
        rows = [b['groups'][str(v)] for b in result['energy_bins_1000_ticks']]
        assert sum(r['predator_ticks'] for r in rows) == 1001
        assert all(r['mean_energy'] == 10.+v for r in rows)
        assert result['groups'][str(v)]['founders']['right_censored'] == 1
    env.time_step += 2
    with pytest.raises(RuntimeError, match="every physics tick"): observer.sample(env)
