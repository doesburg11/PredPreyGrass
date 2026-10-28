"""Matched short hunts: one neutral, two neutral, or two attracted predators.

Only initial populations/poses and external trial stopping differ from the
checkpoint world. Original physics, feeding, reproduction and rewards run intact.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import pickle
import platform

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import (
    checkpoint_env_config, load_gene_pool, load_modules, use_gene_pool,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.eval import select_actions
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.geometry import Vector, heading_degrees
from .fitness_confirmation import checkpoint_files
from .hunting_metrics import HuntingObserver, TOTALS
from .individual_preferences import IndividualPreferences
from .social_bias import biased_logits, social_scores
from .strength_sweep import fingerprint
from .summarize import summarize
from .validation import validate_checkpoint

CONDITIONS = ('one_neutral', 'two_neutral', 'two_attraction')
SEEDS = tuple(range(202,302))
HORIZON = 256
METRICS = ('capture_success', 'penalized_capture_time', 'team_total_energy_spent',
           'team_movement_energy', 'team_resting_energy', 'team_retained_hunting_energy', 'observed_ticks')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator_for(seed, agent):
    value = int.from_bytes(hashlib.sha256(f'controlled_hunt_v1:{seed}:{agent}'.encode()).digest()[:8], 'big')
    return torch.Generator().manual_seed(value % (2**63))


def setup_trial(config, pool, seed, condition):
    if condition not in CONDITIONS: raise ValueError('unknown condition')
    # Always draw two predators first: changing draw count would change the
    # focal/prey genomes. Remove only the companion for the one-predator arm.
    env = PredPreyGrassContinuous({**config, 'predator_count':2, 'prey_count':1})
    use_gene_pool(env, pool)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed+20261028)
    angle = rng.uniform(-math.pi,math.pi)
    center = rng.uniform([0.,0.],[env.width,env.height])
    distance = (80.,120.,160.)[(seed-SEEDS[0]) % 3]
    def rotate(x,y):
        return np.array([math.cos(angle)*x-math.sin(angle)*y, math.sin(angle)*x+math.cos(angle)*y])
    for i,animal in enumerate(env.predators):
        lateral = -40. if i == 0 else 40.
        xy = center + rotate(-distance,lateral)
        animal.position = Vector(float(xy[0]%env.width),float(xy[1]%env.height))
        # Bisect bearings to prey and companion so both can initially be seen
        # in most sampled phenotypes. Use the same focal pose in the solo arm.
        toward_prey = np.array([distance,-lateral])/math.hypot(distance,lateral)
        direction = toward_prey + np.array([0.,1. if i == 0 else -1.])
        direction /= np.linalg.norm(direction)
        velocity = rotate(*direction)
        animal.velocity = Vector(float(velocity[0]),float(velocity[1]))
        animal.orientation = heading_degrees(animal.velocity)
    prey = env.prey[0]
    prey.position = Vector(float(center[0]),float(center[1]))
    prey_heading = rng.uniform(-math.pi,math.pi)
    prey.velocity = Vector(math.cos(prey_heading),math.sin(prey_heading))
    prey.orientation = heading_degrees(prey.velocity)
    if condition == 'one_neutral':
        companion = env.predators.pop()
        env.entities.remove(companion)
        env.active.remove(companion.id)
        env.agents.remove(companion.id)
        del env.family[companion.id]
    return env, distance


def initial_state(env):
    animals = {}
    for a in env.predators+env.prey:
        animals[a.id] = {'position':[a.position.x,a.position.y], 'velocity':[a.velocity.x,a.velocity.y],
                         'orientation':a.orientation, 'energy':a.energy, 'genome':a.genome.tolist(),
                         'max_speed':a.max_speed, 'view_distance':a.view_distance, 'fov':a.fov,
                         'cost_multiplier':a.cost_multiplier, 'species':a.species, 'alive':a.alive,
                         'acceleration':[a.acceleration.x,a.acceleration.y], 'radius':a.radius, 'mass':a.mass,
                         'max_acceleration':a.max_acceleration, 'max_steer_force':a.max_steer_force}
    return {'animals':animals, 'grass_sha256':hashlib.sha256(pickle.dumps(env.grass)).hexdigest(),
            'birth_rng_sha256':hashlib.sha256(pickle.dumps(env.birth_rng.getstate())).hexdigest(),
            'gene_rng_state':env.gene_rng.bit_generator.state}


@torch.inference_mode()
def actions_for(modules, obs, generators, preference, slots):
    actions = {}; visible = 0; decisions = 0
    for agent in sorted(obs):
        species = agent.split('_')[0]
        generator = generators[agent]
        if species == 'predator':
            decisions += 1
            visible += bool(np.any(social_scores(obs[agent],slots)))
        if species != 'predator' or preference == 0:
            actions.update(select_actions(modules,{agent:obs[agent]},generator,True))
        else:
            batch = torch.from_numpy(np.asarray(obs[agent],dtype=np.float32)[None,:])
            logits = modules[species].forward_inference({Columns.OBS:batch})[Columns.ACTION_DIST_INPUTS]
            adjusted = biased_logits(logits[0].cpu().numpy(),obs[agent],preference,2.,slots)
            distribution = torch.softmax(torch.as_tensor(adjusted,dtype=logits.dtype)[None,:],dim=-1)
            actions[agent] = int(torch.multinomial(distribution,1,generator=generator).item())
    return actions, visible, decisions


def trial(config, pool, modules, seed, condition):
    env, distance = setup_trial(config,pool,seed,condition)
    start = initial_state(env)
    preferences = IndividualPreferences('attraction2' if condition == 'two_attraction' else 'neutral')
    preferences.start_episode(env,seed)
    observer = HuntingObserver(preferences,trace_seeds=()); observer.reset(env,seed)
    generators = {a:generator_for(seed,a) for a in env.active}
    preference = 1 if condition == 'two_attraction' else 0
    visible = decisions = 0
    initial_visible = sum(bool(np.any(social_scores(o,env.observe_count['predator'])))
                          for a,o in env._observations().items() if a.startswith('predator'))
    outcome = None
    for tick in range(HORIZON):
        if tick % env.action_repeat == 0:
            actions, v, n = actions_for(modules,env._observations(),generators,preference,env.observe_count['predator'])
            visible += v; decisions += n
        # Same held action schedule as env.step, but stop on the exact event
        # tick rather than accidentally running the remainder of a decision.
        _,_,terms,truncs,_,_ = env._physics_step(actions)
        preferences.register(env); observer.sample(env)
        captures = sum(b['captures'] for b in observer.budgets.values())
        if captures: outcome = 'captured'
        elif sum(env.births.values()): outcome = 'birth_stop'
        elif not env.prey: outcome = 'target_starved'
        elif not env.predators: outcome = 'predators_starved'
        elif terms['__all__'] or truncs['__all__']: outcome = 'environment_end'
        elif tick == HORIZON-1: outcome = 'timeout'
        if outcome is not None: break
    observed = observer.finish(env)
    records = observed['hunting']['individuals']
    totals = {key:sum(r[key] for r in records) for key in TOTALS}
    if totals['captures'] not in (0,1): raise RuntimeError('trial should contain at most one capture')
    success = int(outcome == 'captured')
    row = {'seed':seed,'condition':condition,'distance_parameter':distance,'initial_state':start,
           'initial_socially_visible_predators':initial_visible,'social_visible_decisions':visible,
           'predator_decisions':decisions,'outcome':outcome,'capture_success':success,
           'capture_tick':env.time_step if success else None,
           'penalized_capture_time':env.time_step if success else HORIZON,
           'observed_ticks':env.time_step,'births':dict(env.births),
           'team_movement_energy':totals['movement_energy'],'team_resting_energy':totals['resting_energy'],
           'team_retained_hunting_energy':totals['retained_hunting_energy'],
           'team_total_energy_spent':totals['movement_energy']+totals['resting_energy']+totals['reproduction_energy'],
           'hunting':observed['hunting']}
    env.close()
    return row


def check_matched(rows):
    reference = rows['two_neutral']['initial_state']
    if rows['two_attraction']['initial_state'] != reference: raise RuntimeError('two-predator starting states differ')
    solo = rows['one_neutral']['initial_state']
    subset = {**reference,'animals':{a:r for a,r in reference['animals'].items() if a != 'predator_1'}}
    if solo != subset: raise RuntimeError('solo focal/prey/grass/RNG state differs')


def wilson_interval(successes, count):
    if count <= 0: raise ValueError('positive trial count required')
    z = 1.959963984540054
    fraction = successes/count
    denominator = 1+z*z/count
    center = (fraction+z*z/(2*count))/denominator
    half = z*math.sqrt(fraction*(1-fraction)/count+z*z/(4*count*count))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def trial_summary(rows):
    by_condition = {condition:[r for r in rows if r['condition']==condition] for condition in CONDITIONS}
    paired = {('0.0' if c=='two_neutral' else c):[{'seed':r['seed'],'movement':{'means':{m:r[m] for m in METRICS}}} for r in entries]
              for c,entries in by_condition.items()}
    result = summarize(paired,metrics=METRICS)
    result['neutral_reference'] = 'two_neutral'
    result['outcomes'] = {c:dict(Counter(r['outcome'] for r in entries)) for c,entries in by_condition.items()}
    result['successes'] = {c:sum(r['capture_success'] for r in entries) for c,entries in by_condition.items()}
    result['capture_probability_wilson_95_percent_intervals'] = {c:wilson_interval(sum(r['capture_success'] for r in entries),len(entries)) for c,entries in by_condition.items()}
    result['initial_social_visibility_counts'] = {c:dict(Counter(r['initial_socially_visible_predators'] for r in entries)) for c,entries in by_condition.items()}
    result['social_opportunity_fraction_predator_decisions'] = {c:sum(r['social_visible_decisions'] for r in entries)/sum(r['predator_decisions'] for r in entries) for c,entries in by_condition.items()}
    references = {r['seed']:r for r in by_condition['two_neutral']}
    result['discordant_capture_pairs'] = {c:sum(r['capture_success']!=references[r['seed']]['capture_success'] for r in entries)
                                          for c,entries in by_condition.items() if c!='two_neutral'}
    result['mean_capture_tick_successes_only'] = {c:float(np.mean([r['capture_tick'] for r in entries if r['capture_success']]))
                                                 if any(r['capture_success'] for r in entries) else None for c,entries in by_condition.items()}
    result['limitations'] = 'Exploratory intervals, no multiplicity adjustment. First birth is a competing stop, not a disabled rule. Failure time score=256 is a penalty, not an imputed capture time. Costs run only until each trial ends; interpret with success/duration. More predators alone do not establish cooperation. Matched per-agent action streams differ from the previous global-stream ecosystem sampler. Bootstrap contrast intervals can collapse with no discordant outcomes and do not prove equivalence; per-condition Wilson intervals are also reported.'
    return result


def verify_core_sources(core_dir, expected_source):
    core_dir = Path(core_dir).resolve()
    actual = {str(p.resolve()):digest(p) for p in core_dir.glob('*.py')}
    expected = {str(Path(p).resolve()):h for p,h in expected_source.items()
                if Path(p).resolve().parent == core_dir and Path(p).suffix == '.py'}
    if not expected or actual != expected:
        raise RuntimeError('existing environment source set/hashes differ from validated hunting baseline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--baseline-manifest',type=Path,default=Path(__file__).parent/'results/hunting_checkpoint2_s102_201/hunting.json')
    args = parser.parse_args()
    if args.output_dir.exists(): parser.error('choose a new output directory')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    config = checkpoint_env_config(args.checkpoint); pool = load_gene_pool(args.checkpoint)
    modules = load_modules(args.checkpoint)
    validation = validate_checkpoint(args.checkpoint,modules,82)
    inputs, files = fingerprint(args.checkpoint), checkpoint_files(args.checkpoint)
    core_sources = list((Path(__file__).parent.parent/'eco_evolutionary_continuous_traits').glob('*.py'))
    baseline = json.loads(args.baseline_manifest.read_text())
    expected_source = {str(Path(p).resolve()):h for p,h in baseline['source_and_baseline_sha256'].items()}
    if (not baseline['all_100_prior_episodes_match'] or baseline['input_sha256']!=inputs
            or baseline['all_checkpoint_files_sha256']!=files or baseline['validation']!=validation):
        raise RuntimeError('validated hunting baseline checkpoint differs')
    verify_core_sources(Path(__file__).parent.parent/'eco_evolutionary_continuous_traits',expected_source)
    sources = list(Path(__file__).parent.glob('*.py')) + core_sources
    hashes = {str(p):digest(p) for p in sources+[args.baseline_manifest]}
    plan = {'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(args.checkpoint),
            'seeds':list(SEEDS),'conditions':list(CONDITIONS),'horizon':HORIZON,'strength':2.,
            'canonical_population':{'predator':2,'prey':1},'distance_parameters':[80,120,160],'lateral_offsets':[-40,40],
            'pose_rule':'random translation/rotation, predator heading bisects prey and companion bearings, random prey heading; identical focal pose in solo arm',
            'main_comparison':'two_attraction versus two_neutral probability of capture before competing stop or deadline; other outcomes descriptive',
            'stopping':'first capture, else first birth, target starvation, all predator starvation, environment end, or 256 ticks; all trials retained',
            'time_metric':'capture tick if successful, otherwise fixed 256 penalty; success-only mean is selected and descriptive',
            'sampling':'per-agent SHA256-seeded Torch streams, same across arms, original neutral categorical sampler, existing strength-2 predator bias',
            'unchanged':'PPO weights, rewards, physics, grass, feeding, reproduction and social non-inheritance',
            'inference':'10000 paired episode percentile resamples; seed 20261026; exploratory 95% intervals, no multiplicity adjustment',
            'source_scope':'Environment sources match validated 100-episode hunting baseline; historical training-code equivalence is not asserted',
            'input_sha256':inputs,'all_checkpoint_files_sha256':files,'source_sha256':hashes,'validation':validation,
            'runtime':{'python':platform.python_version(),**{k:version(k) for k in ('numpy','torch','ray','gymnasium')}}}
    with (args.output_dir/'experiment_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    rows=[]
    with (args.output_dir/'trials.jsonl').open('x') as journal:
        for seed in SEEDS:
            matched = {c:trial(config,pool,modules,seed,c) for c in CONDITIONS}
            check_matched(matched)
            for row in matched.values(): journal.write(json.dumps(row)+'\n'); rows.append(row)
            journal.flush()
            print(f'matched seed {seed} complete',flush=True)
    after=validate_checkpoint(args.checkpoint,modules,82)
    if after!=validation or fingerprint(args.checkpoint)!=inputs or checkpoint_files(args.checkpoint)!=files:raise RuntimeError('checkpoint changed')
    if any(digest(p)!=h for p,h in hashes.items()):raise RuntimeError('source changed')
    with (args.output_dir/'manifest.json').open('x') as f:json.dump({**plan,'validation_after':after,'matched_starting_states_verified':True,'trial_count':len(rows),'trials_sha256':digest(args.output_dir/'trials.jsonl')},f,indent=2)
    with (args.output_dir/'summary.json').open('x') as f:json.dump(trial_summary(rows),f,indent=2)
    print(args.output_dir/'summary.json',flush=True)


if __name__=='__main__':main()
