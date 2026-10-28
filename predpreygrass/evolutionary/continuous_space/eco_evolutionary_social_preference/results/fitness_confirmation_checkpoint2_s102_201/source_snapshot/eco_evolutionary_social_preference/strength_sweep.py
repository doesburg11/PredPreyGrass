"""Evaluate stronger attraction, reusing a validated matched neutral baseline.

Existing strength-1 results stay unchanged. Run from the repository root:
python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.strength_sweep CHECKPOINT
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from .evaluate import evaluate
from .summarize import summarize
from .validation import validate_checkpoint


def fingerprint(checkpoint):
    """Pin saved weights, environment settings and founder pool during the sweep."""
    root = Path(checkpoint)
    paths = [root / 'gene_pools.json', root.parent / 'source_code/settings.json',
             root.parent.parent / 'source_code/settings.json', root / 'class_and_ctor_args.pkl']
    paths += [root / f'learner_group/learner/rl_module/{species}_policy/module_state.pkl' for species in ('predator', 'prey')]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).parent / 'results/movement_s42_61_strength1.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--strengths', nargs='+', type=float, default=[2.0, 4.0])
    args = parser.parse_args()
    if any(not 1 < strength < float('inf') for strength in args.strengths) or len(set(args.strengths)) != len(args.strengths):
        parser.error('strengths must be unique, finite and greater than 1')
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    if Path(baseline['checkpoint']).resolve() != args.checkpoint.resolve() or baseline['strength'] != 1:
        parser.error('baseline must use strength 1 and the same checkpoint')
    neutral = baseline['results']['0.0']
    seeds = [row['seed'] for row in neutral]
    if not seeds or seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        parser.error('baseline must have consecutive ascending seeds')
    summarize(baseline['results'])  # Validate every historical condition's seed set before running.
    paths = [args.output_dir / f'attraction_s{seeds[0]}_{seeds[-1]}_strength{strength:g}.json' for strength in args.strengths]
    if len(set(paths)) != len(paths):
        parser.error('strengths produce colliding output names; choose more distinct values')
    if any(p.exists() or p.with_suffix('.episodes.jsonl').exists() for p in paths):
        parser.error('an output already exists; choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest = fingerprint(args.checkpoint)
    modules = load_modules(args.checkpoint)
    validation = validate_checkpoint(args.checkpoint, modules, seeds[0])
    if validation != baseline['validation']:
        raise RuntimeError('checkpoint validation differs from baseline')
    # Verify every saved field against a fresh neutral trajectory at the
    # baseline strength, including its counterfactual diagnostic columns.
    probe = evaluate(args.checkpoint, 0, 1, 1, seeds[0], modules)[0]
    if probe != neutral[0]:
        raise RuntimeError('fresh neutral episode differs from the saved baseline')
    for strength, output in zip(args.strengths, paths):
        with output.with_suffix('.episodes.jsonl').open('x') as progress:
            def completed(preference, episode):
                progress.write(json.dumps({'preference': preference, 'strength': strength, **episode}) + '\n')
                progress.flush()
            episodes = evaluate(args.checkpoint, 1.0, len(seeds), strength, seeds[0], modules, completed)
        after = validate_checkpoint(args.checkpoint, modules, seeds[0])
        if fingerprint(args.checkpoint) != manifest or after != validation or args.baseline.read_bytes() != baseline_bytes:
            raise RuntimeError('checkpoint, founder pool, settings or baseline changed during evaluation')
        summary = summarize({'0.0': neutral, '1.0': episodes})
        payload = {
            'checkpoint': str(args.checkpoint), 'preference': 1.0, 'strength': strength,
            'seeds': seeds, 'baseline': str(args.baseline),
            'baseline_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
            'reference_conditions': {'0.0': 'saved neutral', '1.0': f'new attraction at strength {strength:g}'},
            'input_sha256': manifest, 'validation': validation, 'validation_after': after,
            'neutral_probe_matches_baseline': True, 'summary': summary, 'episodes': episodes,
        }
        with output.open('x') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
        print(output, flush=True)


if __name__ == '__main__':
    main()
