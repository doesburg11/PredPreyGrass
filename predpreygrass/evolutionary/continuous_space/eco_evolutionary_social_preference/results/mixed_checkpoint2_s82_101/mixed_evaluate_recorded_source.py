"""Mixed fixed preferences versus saved, matched uniform controls.

A two-condition replay check must pass before saved neutral/attraction
controls are reused. New runs add uniform avoidance and a mixed population.
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from .evaluate import evaluate
from .individual_preferences import IndividualPreferences
from .preference_metrics import PreferencePersistenceObserver
from .strength_sweep import fingerprint
from .validation import validate_checkpoint


def canonical(value):
    """Spell-record iteration order is irrelevant and varies with Python hashing."""
    if isinstance(value, dict):
        return {key: sorted(tuple(row) for row in item) if key == 'spell_records' else canonical(item)
                for key,item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    baselines={}
    paths={}
    for condition in ('neutral','attraction2'):
        matches=list(args.baseline_dir.glob(f'{condition}_s*.json'))
        if len(matches)!=1:parser.error('expected one saved baseline per condition')
        paths[condition]=matches[0]
        baselines[condition]=json.loads(matches[0].read_text())
    neutral=baselines['neutral']
    seeds=neutral['seeds']
    if not seeds or seeds!=list(range(seeds[0],seeds[0]+len(seeds))):parser.error('expected consecutive seed list')
    for condition,pref in [('neutral',0.),('attraction2',1.)]:
        row=baselines[condition]
        if row['condition']!=condition or row['preference']!=pref or row['strength']!=2.:
            parser.error('incorrect baseline treatment')
        for key in ('seeds','checkpoint','input_sha256','validation','runtime','source_sha256'):
            if row[key]!=neutral[key]:parser.error(f'baseline mismatch: {key}')
        if [r['seed'] for r in row['episodes']]!=seeds:parser.error('incorrect baseline episode seeds')
    outputs={c:args.output_dir/f'{c}_s{seeds[0]}_{seeds[-1]}.json' for c in ('avoidance','mixed')}
    if args.output_dir.exists():parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True)
    checkpoint=Path(neutral['checkpoint'])
    torch.set_num_threads(1)
    manifest=fingerprint(checkpoint)
    modules=load_modules(checkpoint)
    validation=validate_checkpoint(checkpoint,modules,seeds[0])
    if validation!=neutral['validation']:raise RuntimeError('baseline weights or architecture differ')
    # Baseline manifests include exactly the same inputs from the same runner.
    if manifest!=neutral['input_sha256']:raise RuntimeError('checkpoint input files differ from saved baseline')
    baseline_hash={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    sources=list(Path(__file__).parent.glob('*.py'))
    sources+=list((Path(__file__).parent.parent/'eco_evolutionary_continuous_traits').glob('*.py'))
    source_hash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    plan={'seeds':seeds,'checkpoint':str(checkpoint),'strength':2,'assignment_seed':20261027,
          'mixed_founder_counts':{'-1':2,'0':2,'1':2},'rule':'ID-block shuffled permutations; lifetime-fixed, non-inherited; births do not read ancestry',
          'new_conditions':['avoidance','mixed'],'reused_conditions':['neutral','attraction2'],
          'primary_metrics':['global pair100 retention','global nearest-neighbor spacing'],
          'secondary':'100-unit encounters by unordered preference pair; descriptive within mixed world',
          'baseline_sha256':baseline_hash,'source_sha256':source_hash}
    with (args.output_dir/'experiment_plan.json').open('x') as stream:json.dump(plan,stream,indent=2)
    smoke={}
    for condition,pref in [('neutral',0.),('attraction2',1.)]:
        preferences=IndividualPreferences(condition)
        observer=PreferencePersistenceObserver(preferences,trace_seeds=())
        row=evaluate(checkpoint,pref,1,2.,seeds[0],modules,observer=observer,individual_preferences=preferences)[0]
        stripped={k:v for k,v in row.items() if k!='individual_preferences'}
        stripped['persistence']={k:v for k,v in row['persistence'].items() if k!='preference_groups'}
        if canonical(stripped)!=canonical(baselines[condition]['episodes'][0]):
            raise RuntimeError(f'{condition} per-agent smoke differs from saved uniform control')
        smoke[condition]=row
    with (args.output_dir/'uniform_smoke.json').open('x') as stream:json.dump(smoke,stream,indent=2)
    for condition,pref in [('avoidance',-1.),('mixed',0.)]:
        preferences=IndividualPreferences(condition)
        observer=PreferencePersistenceObserver(preferences,trace_seeds=())
        output=outputs[condition]
        with output.with_suffix('.episodes.jsonl').open('x') as progress:
            def completed(p,episode):
                progress.write(json.dumps({'condition':condition,**episode})+'\n');progress.flush()
            episodes=evaluate(checkpoint,pref,len(seeds),2.,seeds[0],modules,completed,observer,preferences)
        after=validate_checkpoint(checkpoint,modules,seeds[0])
        if after!=validation or fingerprint(checkpoint)!=manifest:raise RuntimeError('checkpoint changed')
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=source_hash[str(p)] for p in sources):raise RuntimeError('source changed')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in baseline_hash.items()):raise RuntimeError('baseline changed')
        payload={'condition':condition,'strength':2.,'seeds':seeds,'checkpoint':str(checkpoint),
                 'preference_mode':condition,'inherited':False,'assignment_seed':20261027,
                 'uniform_smoke_verified':True,'baseline_sha256':baseline_hash,
                 'validation':validation,'validation_after':after,'input_sha256':manifest,'source_sha256':source_hash,
                 'runtime':{'python':platform.python_version(),**{key:version(key) for key in ('numpy','torch','ray','gymnasium')}},
                 'episodes':episodes}
        with output.open('x') as stream:json.dump(payload,stream,indent=2);stream.write('\n')
        print(output,flush=True)


if __name__=='__main__':main()
