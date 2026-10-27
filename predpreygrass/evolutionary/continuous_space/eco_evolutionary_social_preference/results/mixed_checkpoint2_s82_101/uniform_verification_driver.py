import hashlib,json
from pathlib import Path
import torch
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.evaluate import evaluate
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.individual_preferences import IndividualPreferences
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.preference_metrics import PreferencePersistenceObserver
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.mixed_evaluate import canonical
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.strength_sweep import fingerprint
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.validation import validate_checkpoint
root=Path('predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference')
base=root/'results/checkpoint2_persistence_s82_101';out=root/'results/mixed_checkpoint2_s82_101'
old={c:json.loads((base/f'{c}_s82_101.json').read_text()) for c in ('neutral','attraction2')}
checkpoint=Path(old['neutral']['checkpoint']);torch.set_num_threads(1)
modules=load_modules(checkpoint);validation=validate_checkpoint(checkpoint,modules,82)
inputs=fingerprint(checkpoint);sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(root.glob('*.py'))+list((root.parent/'eco_evolutionary_continuous_traits').glob('*.py'))}
checks=[]
with (out/'uniform_verification.episodes.jsonl').open('x') as stream:
 for condition,pref in [('neutral',0.),('attraction2',1.)]:
  preferences=IndividualPreferences(condition);observer=PreferencePersistenceObserver(preferences,trace_seeds=())
  references={r['seed']:r for r in old[condition]['episodes']}
  def checked(p,row):
   stripped={k:v for k,v in row.items() if k!='individual_preferences'}
   stripped['persistence']={k:v for k,v in row['persistence'].items() if k!='preference_groups'}
   if canonical(stripped)!=canonical(references[row['seed']]):raise RuntimeError(f'uniform replay mismatch {condition} {row["seed"]}')
   digest=hashlib.sha256(json.dumps(canonical(stripped),sort_keys=True).encode()).hexdigest()
   item={'condition':condition,'seed':row['seed'],'matches_all_saved_fields':True,'canonical_episode_sha256':digest}
   stream.write(json.dumps(item)+'\n');stream.flush();checks.append(item)
  evaluate(checkpoint,pref,20,2.,82,modules,checked,observer,preferences)
if validation!=validate_checkpoint(checkpoint,modules,82) or inputs!=fingerprint(checkpoint):raise RuntimeError('input weights changed')
if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in sources.items()):raise RuntimeError('source changed')
with (out/'uniform_verification.json').open('x') as stream:
 json.dump({'verified_episodes':len(checks),'all_fields_match':True,'source_sha256':sources,'input_sha256':inputs,'baseline_files':{c:str(base/f'{c}_s82_101.json') for c in old},'checks':checks},stream,indent=2)
print('All 40 uniform control episodes exactly reproduce saved fields.',flush=True)
