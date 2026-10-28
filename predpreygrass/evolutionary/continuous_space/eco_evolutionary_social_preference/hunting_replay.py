"""Replay all 100 confirmation seeds and append read-only hunting ledgers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform

import torch
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from .evaluate import evaluate
from .fitness_confirmation import checkpoint_files, seed_chunks
from .hunting_metrics import HuntingObserver, TOTALS
from .individual_preferences import IndividualPreferences
from .mixed_evaluate import canonical
from .strength_sweep import fingerprint
from .summarize import summarize
from .validation import validate_checkpoint

RATE_METRICS = ('captures_per_1000_predator_ticks', 'retained_hunting_energy_per_1000_predator_ticks',
                'movement_energy_per_1000_predator_ticks', 'resting_energy_per_1000_predator_ticks',
                'offered_hunting_energy_per_1000_predator_ticks',
                'net_hunting_minus_metabolism_per_1000_predator_ticks',
                'retained_hunting_energy_per_metabolic_energy')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_replay(row, reference):
    stripped = dict(row)
    stripped['persistence'] = {k:v for k,v in row['persistence'].items() if k != 'hunting'}
    if canonical(stripped) != canonical(reference):
        raise RuntimeError(f'previous saved fields differ at seed {row["seed"]}')


def hunting_summary(episodes):
    metrics = tuple(f'{cohort}_{metric}' for cohort in ('all','founders') for metric in RATE_METRICS)
    paired, totals = {}, {}
    for value, condition in [('0','0.0'),('-1','avoidance'),('1','attraction')]:
        paired[condition] = []
        totals[condition] = {cohort: {key: 0 for key in TOTALS} for cohort in ('all','founders','born')}
        for row in episodes:
            groups = row['persistence']['hunting']['groups'][value]
            means = {f'{cohort}_{metric}': groups[cohort][metric]
                     for cohort in ('all','founders') for metric in RATE_METRICS}
            paired[condition].append({'seed': row['seed'], 'movement': {'means': means}})
            for cohort, fields in totals[condition].items():
                for key in fields: fields[key] += groups[cohort][key]
    summary = summarize(paired, metrics=metrics)
    for metric in summary['metrics'].values():
        metric['paired_episode_count'] = len(metric.pop('paired_seeds'))
        for contrast in metric['differences_from_neutral'].values(): contrast.pop('seed_differences')
    return {'summary': summary, 'totals': totals,
            'max_abs_individual_energy_balance_error': max(r['persistence']['hunting']['max_abs_energy_balance_error'] for r in episodes),
            'early_episode_seeds': [r['seed'] for r in episodes if r['persistence']['physics_ticks'] < 8000],
            'interpretation': 'Exploratory paired episode-mean rates, 95% intervals without multiplicity adjustment. Same seeds replayed, not independent replication. Capture credit is the environment original catcher, not evidence of cooperative hunting. Exposure and age/body composition differ across preferences.'}


def _worker(checkpoint, seeds, baseline_path, output, index, validation, expected_files):
    torch.set_num_threads(1)
    if checkpoint_files(checkpoint) != expected_files: raise RuntimeError('checkpoint changed before worker load')
    references = {r['seed']:r for r in (json.loads(line) for line in Path(baseline_path).read_text().splitlines())}
    if sorted(references) != seeds: raise RuntimeError('baseline shard seed mismatch')
    modules = load_modules(checkpoint)
    if validate_checkpoint(checkpoint, modules, 82) != validation: raise RuntimeError('worker checkpoint validation mismatch')
    preferences = IndividualPreferences('mixed', assignment_seed=20261027)
    observer = HuntingObserver(preferences, trace_seeds=())
    with (Path(output)/f'worker_{index}.episodes.jsonl').open('x') as journal:
        def completed(_, row):
            check_replay(row, references[row['seed']])
            journal.write(json.dumps(row)+'\n'); journal.flush()
        episodes = evaluate(checkpoint, 0., len(seeds), 2., seeds[0], modules, completed, observer, preferences)
    if validate_checkpoint(checkpoint, modules, 82) != validation or checkpoint_files(checkpoint) != expected_files:
        raise RuntimeError('worker checkpoint changed')
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    baseline_file = args.baseline_dir/'fitness.json'
    baseline = json.loads(baseline_file.read_text())
    seeds = list(range(102,202))
    if (baseline['seeds'] != seeds or baseline['strength'] != 2 or baseline['inherited']
            or baseline['assignment_seed'] != 20261027 or baseline['preference_mode'] != 'mixed'
            or baseline['episode_count'] != 100 or not baseline['parallel_fidelity_verified']
            or baseline['validation'] != baseline['validation_after']):
        parser.error('expected validated 100-seed fixed mixed confirmation baseline')
    expected_names = {f'worker_{i}.episodes.jsonl' for i in range(4)}
    if set(baseline['episode_artifacts_sha256']) != expected_names: parser.error('unexpected baseline shards')
    for name, value in baseline['episode_artifacts_sha256'].items():
        if digest(args.baseline_dir/name) != value: raise RuntimeError('baseline journal hash mismatch')
    for name, value in baseline['source_and_baseline_sha256'].items():
        if digest(name) != value: raise RuntimeError('previous source or pilot changed')
    runtime = {'python': platform.python_version(), **{k:version(k) for k in ('numpy','torch','ray','gymnasium')}}
    if runtime != baseline['runtime']: raise RuntimeError('runtime differs from confirmation')
    if args.output_dir.exists(): parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    checkpoint = Path(baseline['checkpoint'])
    inputs, full_files = fingerprint(checkpoint), checkpoint_files(checkpoint)
    modules = load_modules(checkpoint)
    validation = validate_checkpoint(checkpoint, modules, 82)
    if inputs != baseline['input_sha256'] or full_files != baseline['all_checkpoint_files_sha256'] or validation != baseline['validation']:
        raise RuntimeError('checkpoint differs from confirmation')
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((Path(__file__).parent.parent/'eco_evolutionary_continuous_traits').glob('*.py'))
    hashes = {str(p):digest(p) for p in sources + [baseline_file] + [args.baseline_dir/name for name in sorted(expected_names)]}
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'seeds': seeds, 'checkpoint': str(checkpoint),
        'strength': 2., 'assignment_seed': 20261027, 'preference_mode': 'mixed', 'inherited': False, 'workers': 4,
        'purpose': 'Append hunting measurements to exact saved 100-seed trajectories; no independent replication or tuning',
        'main_descriptive_metrics': list(RATE_METRICS[:3]), 'unit': 'episode seed; paired differences against neutral within a mixed world',
        'intervals': '10000 paired episode percentile bootstrap draws; seed 20261026; exploratory 95%, no multiplicity adjustment',
        'measurement': 'Original catches list and actual before/after _gain energy; original post-movement velocity metabolic costs; full per-life energy conservation including offspring cost',
        'early_end_rule': 'Retain all seeds and observed exposures, including seed 133; no imputed future hunting',
        'stopping_rule': 'All 100 original seeds; no interim outcome analysis or tuning',
        'input_sha256': inputs, 'all_checkpoint_files_sha256': full_files, 'validation': validation,
        'source_and_baseline_sha256': hashes, 'runtime': runtime,
    }
    with (args.output_dir/'experiment_plan.json').open('x') as stream: json.dump(plan, stream, indent=2)
    episodes = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker, checkpoint, chunk, args.baseline_dir/f'worker_{i}.episodes.jsonl',
                               args.output_dir, i, validation, full_files) for i,chunk in enumerate(seed_chunks())]
        for future in as_completed(futures): episodes.extend(future.result())
    episodes.sort(key=lambda row: row['seed'])
    if [r['seed'] for r in episodes] != seeds: raise RuntimeError('unexpected final seeds')
    # Verify the new ledger as well as old fields across parallel versus serial inference.
    preferences = IndividualPreferences('mixed', assignment_seed=20261027)
    serial = evaluate(checkpoint, 0., 1, 2., 102, modules, observer=HuntingObserver(preferences, trace_seeds=()), individual_preferences=preferences)[0]
    if canonical(serial) != canonical(episodes[0]): raise RuntimeError('serial/parallel hunting replay mismatch')
    after = validate_checkpoint(checkpoint, modules, 82)
    if after != validation or fingerprint(checkpoint) != inputs or checkpoint_files(checkpoint) != full_files:
        raise RuntimeError('checkpoint changed')
    if any(digest(p) != value for p,value in hashes.items()): raise RuntimeError('source or baseline changed during run')
    artifacts = {p.name:digest(p) for p in sorted(args.output_dir.glob('worker_*.episodes.jsonl'))}
    with (args.output_dir/'hunting.json').open('x') as stream:
        json.dump({**plan, 'episode_count':100, 'all_100_prior_episodes_match':True,
                   'serial_parallel_seed102_all_fields_match':True, 'validation_after':after,
                   'episode_artifacts_sha256':artifacts}, stream, indent=2)
    with (args.output_dir/'summary.json').open('x') as stream: json.dump(hunting_summary(episodes), stream, indent=2)
    print(args.output_dir/'summary.json', flush=True)


if __name__ == '__main__': main()
