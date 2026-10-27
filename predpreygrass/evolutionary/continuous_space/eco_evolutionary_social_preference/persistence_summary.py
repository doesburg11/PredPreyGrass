"""Paired episode-level summaries of the fresh persistence experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from .summarize import summarize


def values(episode):
    p = episode['persistence']
    result = {'mean_nn_distance_world': p['mean_nn_distance_world_agent_time'],
              'mean_population_time_weighted': p['mean_population_time_weighted']}
    for radius, row in p['thresholds'].items():
        spells = row['spells']['nonbirth_onsets']
        for name in ('retention_probability_among_known', 'retention_lower_bound', 'retention_upper_bound',
                     'mean_completed_duration_ticks', 'median_completed_duration_ticks'):
            result[f'pair_{radius}_{name}'] = spells[name]
        result[f'close_neighbor_fraction_{radius}'] = row['close_neighbor_fraction_predator_time']
        result[f'close_pair_fraction_{radius}'] = row['close_pair_fraction_pair_time']
    for name in ('retention_probability_among_known', 'mean_completed_duration_ticks'):
        result[f'nearest_neighbor_{name}'] = p['nearest_neighbor_spells']['nonbirth_onsets'][name]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    paths = {key: next(args.directory.glob(f'{prefix}_s*.json')) for key, prefix in [('0.0','neutral'),('attraction2','attraction2')]}
    raw = {key: json.loads(path.read_text()) for key,path in paths.items()}
    reference = raw['0.0']
    for run in raw.values():
        for field in ('seeds','checkpoint','input_sha256','source_sha256','runtime','validation',
                      'primary_threshold_world_units','sensitivity_thresholds_world_units','persistence_horizon_ticks'):
            if run[field] != reference[field]:
                raise ValueError(f'unmatched experiment inputs: {field}')
        if [r['seed'] for r in run['episodes']] != run['seeds']:
            raise ValueError('episode seeds differ from declared metadata')
        if run['validation_after'] != run['validation']:
            raise ValueError('post-run checkpoint validation differs')
    if reference['condition'] != 'neutral' or raw['attraction2']['condition'] != 'attraction2':
        raise ValueError('unexpected condition labels')
    if not all(r['neutral_original_actions_verified'] for r in reference['episodes']):
        raise ValueError('neutral inference validation missing')
    keys = tuple(values(reference['episodes'][0]))
    def rows(episodes):
        return [{'seed': r['seed'], 'movement': {'means': values(r)}} for r in episodes]
    paired = {key: rows(run['episodes']) for key,run in raw.items()}
    result = {'primary_threshold_world_units': 100, 'persistence_horizon_ticks': 100,
              'definition': 'same unordered pair continuously close; primary spells exclude tick-0 and birth onsets; death/episode censoring reported',
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
              'summary': summarize(paired, metrics=keys)}
    excluded = {r['seed'] for run in raw.values() for r in run['episodes'] if r['persistence']['physics_ticks'] < 8000}
    remaining = {key: [r for r in rows if r['seed'] not in excluded] for key,rows in paired.items()}
    result['early_episode_seeds'] = sorted(excluded)
    if all(remaining.values()):
        result['full_horizon_sensitivity'] = summarize(remaining, metrics=keys)
    with (args.directory/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for metric,row in result['summary']['metrics'].items():
        print(metric, row['condition_means'], row['differences_from_neutral']['attraction2'])


if __name__ == '__main__':
    main()
