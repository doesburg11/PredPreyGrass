import numpy as np

from .persistence import Spells, PersistenceObserver


def test_left_censoring_births_and_death_are_distinguished():
    tracker = Spells(horizon=3)
    tracker.update({('a', 'b')}, 0, {'a', 'b'})
    tracker.update(set(), 2, {'a', 'b'})  # left-censored encounter
    tracker.update({('a', 'b')}, 3, {'a', 'b'})
    tracker.update(set(), 7, {'a', 'b'})  # observed separation after horizon
    tracker.update({('a', 'c')}, 8, {'a', 'b', 'c'}, {'c'})
    tracker.update(set(), 9, {'a', 'b'})  # birth-associated death
    tracker.update({('a', 'b')}, 10, {'a', 'b'})
    result = tracker.result(11)  # short episode censoring
    primary = result['nonbirth_onsets']
    assert result['left_censored_spells'] == 1
    assert result['birth_onset_spells'] == 1
    assert primary['survive_horizon'] == 1
    assert primary['censored_before_horizon'] == 1
    assert primary['retention_lower_bound'] == 0.5
    assert primary['retention_upper_bound'] == 1
    assert primary['mean_completed_duration_ticks'] == 4
    assert result['all_observed_onsets']['censored_before_horizon'] == 2


def test_reentry_starts_a_new_spell_and_failure_before_horizon():
    tracker = Spells(horizon=3)
    tracker.update(set(), 0, {'a', 'b'})
    tracker.update({('a', 'b')}, 1, {'a', 'b'})
    tracker.update(set(), 2, {'a', 'b'})
    tracker.update({('a', 'b')}, 3, {'a', 'b'})
    result = tracker.result(6)['nonbirth_onsets']
    assert result['survive_horizon'] == 1
    assert result['separate_before_horizon'] == 1
    assert result['retention_probability_among_known'] == 0.5


def world(tick, points):
    from types import SimpleNamespace
    return SimpleNamespace(width=100, height=100, time_step=tick, prey=[],
                           predators=[SimpleNamespace(id=key, position=SimpleNamespace(x=x, y=y)) for key, (x,y) in points.items()])


def test_tick_weighted_exposure_wrap_and_zero_population():
    observer = PersistenceObserver(thresholds=(5,), horizon=2)
    observer.reset(world(0, {'a': (99, 0), 'b': (1, 0)}), 1)
    observer.sample(world(1, {'a': (20, 0), 'b': (40, 0)}))
    observer.sample(world(2, {}))
    result = observer.finish(world(2, {}))
    assert result['physics_ticks'] == 2
    assert result['mean_population_time_weighted'] == 2
    assert result['mean_nn_distance_world_agent_time'] == 11
    assert result['thresholds']['5']['close_pair_fraction_pair_time'] == 0.5
    assert result['thresholds']['5']['close_neighbor_fraction_predator_time'] == 0.5


def test_birth_parent_pair_not_in_primary_nonbirth_analysis():
    observer = PersistenceObserver(thresholds=(10,), horizon=1)
    observer.reset(world(0, {'a': (0, 0)}), 1)
    observer.sample(world(1, {'a': (0, 0), 'child': (1, 0)}))
    observer.sample(world(2, {'a': (0, 0), 'child': (1, 0)}))
    spells = observer.finish(world(2, {'a': (0, 0), 'child': (1, 0)}))['thresholds']['10']['spells']
    assert spells['nonbirth_onsets']['observed_onsets'] == 0
    assert spells['all_observed_onsets']['survive_horizon'] == 1


def test_missing_tick_is_rejected():
    import pytest
    observer = PersistenceObserver()
    observer.reset(world(0, {}), 1)
    with pytest.raises(RuntimeError, match='every physics tick'):
        observer.sample(world(2, {}))


def test_nearest_neighbor_switch_and_return_are_separate_directed_spells():
    observer = PersistenceObserver(thresholds=(5,), horizon=2)
    observer.reset(world(0, {'a': (0,0), 'b': (4,0), 'c': (20,0)}), 1)
    observer.sample(world(1, {'a': (0,0), 'b': (4,0), 'c': (1,0)}))
    assert ('a','c') in observer.neighbors.active
    assert ('a','b') not in observer.neighbors.active
    observer.sample(world(2, {'a': (0,0), 'b': (4,0), 'c': (20,0)}))
    assert observer.neighbors.active[('a','b')][0] == 2
    assert ('a','c') not in observer.neighbors.active
    observer.sample(world(3, {'a': (0,0), 'b': (4,0), 'c': (20,0)}))
    result = observer.finish(world(3, {}))
    assert result['physics_ticks'] == 3
    assert result['nearest_neighbor_spells']['nonbirth_onsets']['censored_before_horizon'] > 0
