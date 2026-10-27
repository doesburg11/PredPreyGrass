"""Fresh-seed neutral/attraction-2 persistence evaluation and position traces."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from .evaluate import evaluate
from .persistence import PersistenceObserver
from .strength_sweep import fingerprint
from .validation import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=62)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'results/persistence_s62_81')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    last = args.seed + args.episodes - 1
    conditions = {'neutral': 0.0, 'attraction2': 1.0}
    output_paths = {c: args.output_dir / f'{c}_s{args.seed}_{last}.json' for c in conditions}
    trace_seeds = tuple(range(args.seed, min(args.seed + 2, last + 1)))
    paths = list(output_paths.values()) + [p.with_suffix('.episodes.jsonl') for p in output_paths.values()]
    paths += [args.output_dir / 'traces' / f'{c}_seed{seed}.jsonl.gz' for c in conditions for seed in trace_seeds]
    if any(p.exists() for p in paths):
        parser.error('an output or trace already exists; choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest = fingerprint(args.checkpoint)
    source_paths = list(Path(__file__).parent.glob('*.py'))
    source_paths += list((Path(__file__).parent.parent / 'eco_evolutionary_continuous_traits').glob('*.py'))
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    modules = load_modules(args.checkpoint)
    validation = validate_checkpoint(args.checkpoint, modules, args.seed)
    for condition, preference in conditions.items():
        observer = PersistenceObserver(trace_dir=args.output_dir / 'traces', trace_seeds=trace_seeds, condition=condition)
        output = output_paths[condition]
        with output.with_suffix('.episodes.jsonl').open('x') as stream:
            def completed(p, episode):
                stream.write(json.dumps({'condition': condition, **episode}) + '\n')
                stream.flush()
            episodes = evaluate(args.checkpoint, preference, args.episodes, 2.0, args.seed, modules, completed, observer)
        after = validate_checkpoint(args.checkpoint, modules, args.seed)
        if after != validation or fingerprint(args.checkpoint) != manifest:
            raise RuntimeError('checkpoint inputs or loaded weights changed')
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != sources[str(p)] for p in source_paths):
            raise RuntimeError('experiment source changed during evaluation')
        payload = {
            'condition': condition, 'preference': preference, 'strength': 2.0,
            'seeds': list(range(args.seed, last + 1)), 'checkpoint': str(args.checkpoint),
            'primary_threshold_world_units': 100, 'sensitivity_thresholds_world_units': [75,125],
            'persistence_horizon_ticks': 100, 'trace_seeds': trace_seeds, 'trace_interval_ticks': 8,
            'runtime': {'python': platform.python_version(), **{k: version(k) for k in ('numpy','torch','ray','gymnasium')}},
            'input_sha256': manifest, 'source_sha256': sources,
            'validation': validation, 'validation_after': after, 'episodes': episodes,
        }
        with output.open('x') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
        print(output, flush=True)


if __name__ == '__main__':
    main()
