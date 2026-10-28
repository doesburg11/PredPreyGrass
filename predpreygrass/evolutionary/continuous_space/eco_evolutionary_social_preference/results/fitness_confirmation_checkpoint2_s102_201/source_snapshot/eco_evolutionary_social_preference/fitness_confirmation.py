"""Predeclared fresh-seed confirmation of fixed mixed-preference outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform

import numpy as np
import torch

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from .evaluate import evaluate
from .fitness_metrics import FitnessObserver
from .fitness_summary import build_summary
from .individual_preferences import IndividualPreferences
from .mixed_evaluate import canonical
from .strength_sweep import fingerprint
from .validation import validate_checkpoint

PRIMARY = ('founder_observed_lifespan_ticks', 'founder_observed_offspring')


def confirmation_summary(episodes):
    """Two primary attraction-minus-neutral contrasts; episode is the unit."""
    if not episodes or len({r['seed'] for r in episodes}) != len(episodes):
        raise ValueError('expected nonempty unique episode seeds')
    episodes = sorted(episodes, key=lambda r: r['seed'])
    descriptive = build_summary({'episodes': episodes})
    rng = np.random.default_rng(20261028)
    indices = rng.integers(0, len(episodes), size=(10000, len(episodes)))
    primary = {}
    for metric in PRIMARY:
        contrast = descriptive['summary']['metrics'][metric]['differences_from_neutral']['attraction']
        delta = np.asarray(contrast['seed_differences'], dtype=float)
        if len(delta) != len(episodes) or not np.isfinite(delta).all():
            raise ValueError('missing primary outcome')
        interval = np.quantile(delta[indices].mean(axis=1), [.0125, .9875]).tolist() if len(delta) > 1 else None
        primary[metric] = {'mean': float(delta.mean()), 'bootstrap_97_5_percent_interval': interval,
                           'positive_adjusted_interval': interval is not None and interval[0] > 0}
    # Retain detailed seed outcomes only in the hashed journals.
    for metric in descriptive['summary']['metrics'].values():
        metric['paired_episode_count'] = len(metric.pop('paired_seeds'))
        for contrast in metric['differences_from_neutral'].values():
            contrast.pop('seed_differences')
    early = [r['seed'] for r in episodes if r['persistence']['fitness']['episode_end_tick'] < 8000]
    early_censored = sum(
        life['founder'] and life['right_censored']
        for row in episodes if row['seed'] in early
        for life in row['persistence']['fitness']['individuals'])
    return {
        'fresh_seeds_only': True, 'seeds': [r['seed'] for r in episodes],
        'primary_comparison': 'attraction minus neutral within each mixed episode',
        'primary_interval_method': '10000 paired episode percentile resamples, RNG seed 20261028; 97.5% marginal intervals for two endpoints (Bonferroni-style approximate 5% familywise coverage)',
        'primary': primary, 'early_episode_seeds': early,
        'early_censored_founders_all_types': early_censored,
        'joint_positive_confirmation': not early_censored and all(r['positive_adjusted_interval'] for r in primary.values()),
        'descriptive_secondary': descriptive,
        'interpretation': 'Observed founder time is restricted by episode end; early-censored founders prevent a full-8000-tick survival claim. Intervals are bootstrap approximations, and this is one checkpoint with interacting fixed, non-heritable preferences.',
    }


def _worker(checkpoint, seeds, assignment_seed, output, index, expected_validation, expected_files):
    torch.set_num_threads(1)
    if checkpoint_files(checkpoint) != expected_files:
        raise RuntimeError('worker checkpoint files changed before load')
    modules = load_modules(checkpoint)
    before = validate_checkpoint(checkpoint, modules, 82)
    if before != expected_validation:
        raise RuntimeError('worker checkpoint validation differs')
    preferences = IndividualPreferences('mixed', assignment_seed=assignment_seed)
    observer = FitnessObserver(preferences, trace_seeds=())
    with (Path(output)/f'worker_{index}.episodes.jsonl').open('x') as journal:
        def completed(_, row):
            for value in ('-1', '0', '1'):
                if row['persistence']['fitness']['groups'][value]['founders']['count'] != 2:
                    raise RuntimeError('expected two founders per preference')
            journal.write(json.dumps(row)+'\n'); journal.flush()
        episodes = evaluate(checkpoint, 0., len(seeds), 2., seeds[0], modules, completed, observer, preferences)
    if validate_checkpoint(checkpoint, modules, 82) != before or checkpoint_files(checkpoint) != expected_files:
        raise RuntimeError('worker weights changed')
    if [r['seed'] for r in episodes] != seeds:
        raise RuntimeError('worker seed mismatch')
    return episodes


def checkpoint_files(checkpoint):
    """Hash every checkpoint file, including per-policy constructor metadata."""
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(checkpoint).rglob('*')) if p.is_file()}


def validate_pilot(baseline):
    if (baseline['seeds'] != list(range(82,102)) or baseline['strength'] != 2
            or baseline['inherited'] or not baseline['all_prior_fields_match']
            or baseline['assignment_seed'] != 20261027
            or [r['seed'] for r in baseline['episodes']] != baseline['seeds']):
        raise ValueError('expected validated pilot with seeds 82–101 and assignment seed 20261027')


def seed_chunks():
    return [list(range(first, first+25)) for first in (102,127,152,177)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path, help='previous 20-episode fitness.json')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    seeds = list(range(102, 202))
    try:
        validate_pilot(baseline)
    except ValueError as error:
        parser.error(str(error))
    if set(seeds) & set(baseline['seeds']):
        raise RuntimeError('confirmation seeds overlap pilot')
    if args.output_dir.exists():
        parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    checkpoint = Path(baseline['checkpoint'])
    inputs = fingerprint(checkpoint)
    complete_checkpoint = checkpoint_files(checkpoint)
    modules = load_modules(checkpoint)
    validation = validate_checkpoint(checkpoint, modules, 82)
    if inputs != baseline['input_sha256'] or validation != baseline['validation']:
        raise RuntimeError('pilot checkpoint mismatch')
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((Path(__file__).parent.parent/'eco_evolutionary_continuous_traits').glob('*.py'))
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p): digest(p) for p in sources + [args.baseline]}
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': seeds, 'checkpoint': str(checkpoint), 'strength': 2.,
        'assignment_seed': baseline['assignment_seed'], 'inherited': False,
        'preference_mode': 'mixed', 'workers': 4, 'planned_episode_horizon': 8000,
        'primary_metrics': list(PRIMARY), 'primary_comparison': 'attraction minus neutral within mixed episodes',
        'primary_intervals': '10000 paired episode bootstrap resamples, seed 20261028, 97.5% marginal intervals for two primary endpoints',
        'confirmation_rule': 'Both primary lower interval bounds > 0, with no founder censored by an early episode end; otherwise benefit not jointly confirmed',
        'early_end_rule': 'Keep every seed. Report observed time, offspring, early endings and right censoring. Do not treat episode-end survivors as dead or make a full-horizon survival claim with early-censored founders.',
        'parallel_fidelity': 'Replay fresh seed 102 serially after all 100 finish; compare every saved field',
        'secondary': 'avoidance comparisons, all-animal reproduction rates, energy averages and 1000-tick bins; descriptive 95% intervals',
        'stopping_rule': 'Run all 100 seeds with no interim outcome analysis, tuning, seed removal or pilot pooling',
        'validation': validation, 'input_sha256': inputs, 'all_checkpoint_files_sha256': complete_checkpoint, 'source_and_baseline_sha256': hashes,
        'runtime': {'python': platform.python_version(), **{k: version(k) for k in ('numpy','torch','ray','gymnasium')}},
    }
    with (args.output_dir/'experiment_plan.json').open('x') as stream: json.dump(plan, stream, indent=2)
    # One saved pilot replay checks the unchanged observer/inference path before fresh outcomes.
    preferences = IndividualPreferences('mixed', assignment_seed=baseline['assignment_seed'])
    replay = evaluate(checkpoint, 0., 1, 2., 82, modules, observer=FitnessObserver(preferences, trace_seeds=()), individual_preferences=preferences)[0]
    if canonical(replay) != canonical(baseline['episodes'][0]):
        raise RuntimeError('pilot replay differs in an existing field, including fitness')
    with (args.output_dir/'pilot_replay_verification.json').open('x') as stream:
        json.dump({'seed': 82, 'all_saved_fields_including_fitness_match': True}, stream, indent=2)
    chunks = seed_chunks()
    episodes = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker, checkpoint, chunk, baseline['assignment_seed'], args.output_dir, i, validation, complete_checkpoint)
                   for i,chunk in enumerate(chunks)]
        for future in as_completed(futures): episodes.extend(future.result())
    episodes.sort(key=lambda row: row['seed'])
    if [r['seed'] for r in episodes] != seeds:
        raise RuntimeError('final seeds differ from predeclared seeds')
    preferences = IndividualPreferences('mixed', assignment_seed=baseline['assignment_seed'])
    serial = evaluate(checkpoint, 0., 1, 2., seeds[0], modules,
                      observer=FitnessObserver(preferences, trace_seeds=()), individual_preferences=preferences)[0]
    if canonical(serial) != canonical(episodes[0]):
        raise RuntimeError('parallel versus serial fresh replay differs')
    with (args.output_dir/'parallel_fidelity.json').open('x') as stream:
        json.dump({'seed': seeds[0], 'all_fields_match': True}, stream, indent=2)
    after = validate_checkpoint(checkpoint, modules, 82)
    if after != validation or fingerprint(checkpoint) != inputs or checkpoint_files(checkpoint) != complete_checkpoint:
        raise RuntimeError('checkpoint changed')
    if any(digest(Path(p)) != h for p,h in hashes.items()):
        raise RuntimeError('source or pilot baseline changed during experiment')
    with (args.output_dir/'fitness.json').open('x') as stream:
        artifacts = {p.name: digest(p) for p in sorted(args.output_dir.glob('worker_*.episodes.jsonl'))}
        json.dump({**plan, 'validation_after': after, 'pilot_replay_verified': True,
                   'parallel_fidelity_verified': True, 'episode_count': len(episodes),
                   'episode_artifacts_sha256': artifacts}, stream, indent=2)
    with (args.output_dir/'summary.json').open('x') as stream:
        json.dump(confirmation_summary(episodes), stream, indent=2)
    print(args.output_dir/'summary.json', flush=True)


if __name__ == '__main__': main()
