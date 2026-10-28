"""Replay a saved mixed experiment, verifying prior fields and adding fitness."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from .evaluate import evaluate
from .fitness_metrics import FitnessObserver
from .individual_preferences import IndividualPreferences
from .mixed_evaluate import canonical
from .strength_sweep import fingerprint
from .validation import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    seeds = baseline['seeds']
    if (baseline['condition'] != 'mixed' or baseline['strength'] != 2 or baseline['inherited']
            or not seeds or seeds != list(range(seeds[0], seeds[0]+len(seeds)))
            or [r['seed'] for r in baseline['episodes']] != seeds):
        parser.error('expected a consecutive-seed, non-inherited mixed strength-2 baseline')
    if args.output_dir.exists():
        parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    checkpoint = Path(baseline['checkpoint'])
    inputs = fingerprint(checkpoint)
    modules = load_modules(checkpoint)
    validation = validate_checkpoint(checkpoint, modules, seeds[0])
    if inputs != baseline['input_sha256'] or validation != baseline['validation']:
        raise RuntimeError('checkpoint differs from saved experiment')
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((Path(__file__).parent.parent/'eco_evolutionary_continuous_traits').glob('*.py'))
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p): digest(p) for p in sources + [args.baseline]}
    plan = {
        'seeds': seeds, 'strength': 2, 'checkpoint': str(checkpoint),
        'assignment_seed': baseline['assignment_seed'], 'inherited': False,
        'primary': 'founder mean observed lifespan and offspring; survivors right censored',
        'interpretation': 'founder lifespan is restricted survival only when all episodes share the full horizon',
        'secondary': 'all-animal offspring per 1000 predator-ticks and time-weighted energy',
        'energy': 'left-endpoint alive energy integrated every physics tick; also grouped in 1000-tick bins',
        'runtime': {'python': platform.python_version(), **{k: version(k) for k in ('numpy','torch','ray','gymnasium')}},
        'baseline_runtime': baseline['runtime'],
        'provenance_scope': 'Empirical agreement of all prior fields on these seeds; historical source/runtime equivalence is not asserted',
        'comparisons': 'paired episode-level differences vs neutral, exploratory bootstrap intervals',
        'source_and_baseline_sha256': hashes, 'input_sha256': inputs,
    }
    with (args.output_dir/'experiment_plan.json').open('x') as f: json.dump(plan, f, indent=2)
    preferences = IndividualPreferences('mixed', assignment_seed=baseline['assignment_seed'])
    observer = FitnessObserver(preferences, trace_seeds=())
    references = {r['seed']: r for r in baseline['episodes']}
    with (args.output_dir/'episodes.jsonl').open('x') as journal:
        def completed(_, row):
            stripped = dict(row)
            stripped['persistence'] = {k:v for k,v in row['persistence'].items() if k != 'fitness'}
            if canonical(stripped) != canonical(references[row['seed']]):
                raise RuntimeError(f'replay differs at seed {row["seed"]}')
            journal.write(json.dumps({'all_prior_fields_match': True, **row})+'\n')
            journal.flush()
        episodes = evaluate(checkpoint, 0., len(seeds), 2., seeds[0], modules, completed, observer, preferences)
    after = validate_checkpoint(checkpoint, modules, seeds[0])
    if after != validation or fingerprint(checkpoint) != inputs:
        raise RuntimeError('checkpoint changed during evaluation')
    if any(digest(Path(p)) != h for p,h in hashes.items()):
        raise RuntimeError('source or baseline changed during evaluation')
    payload = {**plan, 'all_prior_fields_match': True, 'validation': validation,
               'validation_after': after, 'episodes': episodes}
    with (args.output_dir/'fitness.json').open('x') as f: json.dump(payload, f, indent=2)
    print(args.output_dir/'fitness.json', flush=True)


if __name__ == '__main__':
    main()
