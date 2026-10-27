"""Physical metric checks independent of PPO or Ray."""
import numpy as np
import pytest

from .movement_metrics import MovementMetrics, nearest_neighbors, wrapped_offset


def test_toroidal_nearest_neighbor_and_singleton():
    positions = {'a': [99, 5], 'b': [1, 5], 'c': [50, 5]}
    assert nearest_neighbors(positions, (100, 100))['a'] == ('b', 2)
    assert nearest_neighbors({'a': [0, 0]}, (100, 100)) == {}
    np.testing.assert_allclose(wrapped_offset([99, 5], [1, 5], (100, 100)), [2, 0])


def test_realized_wrap_crossing_approach_and_pair_closing():
    metric = MovementMetrics((100, 100), 50)
    metric.record(({'a': [99, 5], 'b': [5, 5]}, {'a': 'b'}),
                  {'a': [1, 5], 'b': [4, 5]}, 2)
    report = metric.result()
    assert report['means']['visible_target_projection_per_tick'] == 1
    assert report['means']['visible_target_displacement_alignment_when_moving'] == 1
    assert report['means']['visible_pair_closing_per_tick'] == 1.5
    assert report['means']['all_predator_nearest_neighbor_distance_normalized'] == 6/50


def test_stationary_focal_can_have_pair_approach_from_target_motion():
    metric = MovementMetrics((100, 100), 50)
    metric.record(({'a': [0, 0], 'b': [10, 0]}, {'a': 'b'}), {'a': [0, 0], 'b': [8, 0]}, 1)
    means = metric.result()['means']
    assert means['visible_target_projection_per_tick'] == 0
    assert means['visible_target_displacement_alignment_when_moving'] is None
    assert means['visible_target_stationary_fraction'] == 1
    assert means['visible_pair_closing_per_tick'] == 2


def test_deaths_excluded_without_substituting_newborn_target():
    metric = MovementMetrics((100, 100), 50)
    metric.record(({'a': [0, 0], 'b': [10, 0]}, {'a': 'b', 'b': 'a'}),
                  {'a': [1, 0], 'newborn': [3, 0]}, 1)
    report = metric.result()
    assert report['samples']['visible_target_projection_per_tick'] == 1
    assert report['samples']['visible_pair_closing_per_tick'] == 0
    assert report['counts']['focal_death_intervals'] == 1
    assert report['counts']['target_death_intervals_with_focal_surviving'] == 1


def test_avoidance_negative_projection_and_closing():
    metric = MovementMetrics((100, 100), 50)
    metric.record(({'a': [20, 0], 'b': [30, 0]}, {'a': 'b'}), {'a': [18, 0], 'b': [30, 0]}, 2)
    assert metric.result()['means']['visible_target_projection_per_tick'] == -1
    assert metric.result()['means']['visible_pair_closing_per_tick'] == -1
    with pytest.raises(ValueError):
        metric.record(({}, {}), {}, 0)


def test_capture_matches_observation_target_and_copies_positions():
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous, SELF_SIZE
    from .movement_metrics import capture
    env = PredPreyGrassContinuous({'predator_count': 6})
    observations, _ = env.reset(seed=42)
    positions, targets = capture(env)
    for animal in env.predators:
        slot = observations[animal.id][SELF_SIZE:SELF_SIZE + 6]
        assert (animal.id in targets) == bool(slot[0])
        if animal.id in targets:
            reach = env.distance_scale if env.fixed_scale else animal.view_distance
            offset = wrapped_offset(positions[animal.id], positions[targets[animal.id]], (env.width, env.height))
            np.testing.assert_allclose(offset / reach, slot[1:3], atol=1e-7)
    old_x = positions[env.predators[0].id][0]
    env.predators[0].position.x += 1
    assert positions[env.predators[0].id][0] == old_x
    env.close()
