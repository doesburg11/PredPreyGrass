import copy
import pytest
from .fitness_confirmation import confirmation_summary


def episodes():
    result = []
    for seed in (102, 103):
        groups = {}
        for value in (-1,0,1):
            row = {'count': 2, 'deaths': 1, 'right_censored': 1, 'offspring': 4+2*value,
                   'predator_ticks': 1000+200*value, 'mean_observed_lifespan': 500.+100*value,
                   'mean_observed_offspring': 2.+value, 'mean_energy_predator_time': 5.,
                   'offspring_per_1000_predator_ticks': 4.}
            groups[str(value)] = {cohort: dict(row) for cohort in ('all','founders','born')}
        result.append({'seed': seed, 'persistence': {'fitness': {
            'groups': groups, 'individuals': [{'founder': True, 'right_censored': True}],
            'episode_end_tick': 8000}}})
    return result


def test_predeclared_primary_rule_and_episode_order():
    rows = episodes()
    result = confirmation_summary(list(reversed(rows)))
    assert result['seeds'] == [102,103]
    assert result['joint_positive_confirmation']
    assert result['primary']['founder_observed_offspring']['bootstrap_97_5_percent_interval'] == [1.,1.]
    # Early administrative censoring must prevent an unsupported full-horizon claim.
    rows[0]['persistence']['fitness']['episode_end_tick'] = 7000
    early = confirmation_summary(rows)
    assert early['early_episode_seeds'] == [102]
    assert early['early_censored_founders_all_types'] == 1
    assert not early['joint_positive_confirmation']


def test_both_primary_endpoints_needed_and_no_duplicate_seeds():
    rows = episodes()
    for row in rows:
        row['persistence']['fitness']['groups']['1']['founders']['mean_observed_offspring'] = 2.
    assert not confirmation_summary(rows)['joint_positive_confirmation']
    with pytest.raises(ValueError, match='unique'): confirmation_summary(rows+[copy.deepcopy(rows[0])])
    with pytest.raises(ValueError, match='nonempty'): confirmation_summary([])


def test_missing_primary_outcome_is_rejected():
    rows = episodes()
    rows[0]['persistence']['fitness']['groups']['1']['founders']['mean_observed_offspring'] = None
    with pytest.raises(ValueError, match='missing primary'): confirmation_summary(rows)


def test_assignment_seed_is_pinned_and_chunks_cover_only_fresh_seeds():
    from .fitness_confirmation import validate_pilot, seed_chunks
    pilot = {'seeds': list(range(82,102)), 'strength': 2, 'inherited': False,
             'all_prior_fields_match': True, 'assignment_seed': 20261027,
             'episodes': [{'seed': s} for s in range(82,102)]}
    validate_pilot(pilot)
    pilot['assignment_seed'] = 0
    with pytest.raises(ValueError, match='assignment seed'): validate_pilot(pilot)
    chunks = seed_chunks()
    assert [s for chunk in chunks for s in chunk] == list(range(102,202))
    assert all(len(chunk) == 25 for chunk in chunks)


def test_full_checkpoint_hash_covers_constructor_metadata(tmp_path):
    from .fitness_confirmation import checkpoint_files
    meta = tmp_path/'learner_group/learner/rl_module/predator_policy/class_and_ctor_args.pkl'
    meta.parent.mkdir(parents=True)
    meta.write_bytes(b'constructor version 1')
    first = checkpoint_files(tmp_path)
    meta.write_bytes(b'constructor version 2')
    assert checkpoint_files(tmp_path) != first
    assert str(meta.resolve()) in first


def test_variable_effects_produce_non_degenerate_primary_intervals():
    rows = []
    for seed in range(102,112):
        row = copy.deepcopy(episodes()[0]); row['seed'] = seed
        row['persistence']['fitness']['groups']['1']['founders']['mean_observed_offspring'] = 0. if seed%2 else 4.
        rows.append(row)
    summary = confirmation_summary(rows)
    low, high = summary['primary']['founder_observed_offspring']['bootstrap_97_5_percent_interval']
    assert low < 0 < high
    assert not summary['joint_positive_confirmation']


def test_worker_rejects_checkpoint_file_drift_before_loading(tmp_path, monkeypatch):
    from . import fitness_confirmation as runner
    (tmp_path/'class_and_ctor_args.pkl').write_bytes(b'changed metadata')
    def unexpected_load(_):
        raise AssertionError('must reject before loading modules')
    monkeypatch.setattr(runner, 'load_modules', unexpected_load)
    with pytest.raises(RuntimeError, match='files changed before load'):
        runner._worker(tmp_path, [102], 20261027, tmp_path, 0, {}, {})


def test_confirmation_secondary_output_keeps_counts_without_episode_details():
    result = confirmation_summary(episodes())
    for metric in result['descriptive_secondary']['summary']['metrics'].values():
        assert metric['paired_episode_count'] == 2
        assert 'paired_seeds' not in metric
        assert all('seed_differences' not in contrast for contrast in metric['differences_from_neutral'].values())
