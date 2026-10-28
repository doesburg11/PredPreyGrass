import pytest

from .summarize import METRICS, summarize


def row(seed, value):
    return {'seed': seed, 'movement': {'means': {name: value for name in METRICS}}}


def test_pairs_by_seed_rather_than_input_order():
    results = {'-1.0': [row(2, 19), row(1, 9)], '0.0': [row(1, 10), row(2, 20)],
               '1.0': [row(2, 22), row(1, 12)]}
    report = summarize(results)
    for metric in report['metrics'].values():
        assert metric['differences_from_neutral']['-1.0']['mean'] == -1
        assert metric['differences_from_neutral']['1.0']['bootstrap_95_percent_interval'] == [2, 2]


def test_missing_and_duplicate_seed_rejected():
    with pytest.raises(ValueError):
        summarize({'0.0': [row(1, 1)], '-1.0': [row(2, 1)], '1.0': [row(1, 1)]})
    with pytest.raises(ValueError):
        summarize({'0.0': [row(1, 1), row(1, 1)]})


def test_missing_metric_uses_same_subset_in_all_conditions():
    results = {key: [row(1, None), row(2, 2)] for key in ('-1.0', '0.0', '1.0')}
    metric = summarize(results)['metrics'][METRICS[0]]
    assert metric['paired_seeds'] == [2]
    assert metric['differences_from_neutral']['1.0']['bootstrap_95_percent_interval'] is None


def test_attraction_only_and_named_strengths_do_not_require_avoidance():
    results = {'0.0': [row(1, 10), row(2, 20)],
               'attraction_2': [row(1, 12), row(2, 22)],
               'attraction_4': [row(1, 14), row(2, 24)]}
    report = summarize(results)
    for metric in report['metrics'].values():
        assert set(metric['differences_from_neutral']) == {'attraction_2', 'attraction_4'}
        assert metric['differences_from_neutral']['attraction_2']['bootstrap_95_percent_interval'] == [2, 2]


def test_custom_persistence_metric_names_are_supported():
    results = {'0.0': [{'seed': 1, 'movement': {'means': {'retention': 0.2}}}],
               'attraction2': [{'seed': 1, 'movement': {'means': {'retention': 0.4}}}]}
    report = summarize(results, metrics=('retention',))
    assert set(report['metrics']) == {'retention'}
    assert report['metrics']['retention']['differences_from_neutral']['attraction2']['mean'] == 0.2
