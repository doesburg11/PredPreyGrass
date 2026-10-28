from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.fitness_summary import build_summary


def test_summary_pairs_missing_values_and_totals():
    episodes = []
    for seed in (82,83):
        groups = {}
        for value in (-1,0,1):
            row = {'count': 2, 'deaths': 1, 'right_censored': 1, 'offspring': 2,
                   'predator_ticks': 10, 'mean_observed_lifespan': 5.+value,
                   'mean_observed_offspring': 1.+value,
                   'mean_energy_predator_time': None if seed == 83 and value == -1 else 4.+value,
                   'offspring_per_1000_predator_ticks': 200.}
            groups[str(value)] = {cohort: dict(row) for cohort in ('founders','born','all')}
        episodes.append({'seed':seed, 'persistence':{'fitness': {'episode_end_tick':8000,'groups':groups}}})
    result = build_summary({'episodes':episodes})
    life = result['summary']['metrics']['founder_observed_lifespan_ticks']
    assert life['paired_seeds'] == [82,83]
    assert life['differences_from_neutral']['attraction']['mean'] == 1.
    assert life['differences_from_neutral']['avoidance']['mean'] == -1.
    energy = result['summary']['metrics']['founder_mean_energy']
    assert energy['paired_seeds'] == [82]
    assert energy['differences_from_neutral']['avoidance']['bootstrap_95_percent_interval'] is None
    assert result['totals']['attraction']['founders']['count'] == 4
    assert result['totals']['attraction']['founders']['right_censored'] == 2
    assert result['episode_end_ticks'] == [8000]
    episodes[1]['persistence']['fitness']['episode_end_tick'] = 7000
    assert 'unequal episode horizons' in build_summary({'episodes':episodes})['lifespan_interpretation']
