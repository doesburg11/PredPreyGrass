"""Post-hoc exact paired binary sensitivity; no changes to trial selection."""
from pathlib import Path
import json,math
root=Path('predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/controlled_hunt_checkpoint2_s202_301')
rows=[json.loads(line) for line in (root/'trials.jsonl').read_text().splitlines()]
reference={r['seed']:r for r in rows if r['condition']=='two_neutral'}
result={'status':'post-hoc sensitivity; main paired bootstrap analysis unchanged','comparisons':{}}
for condition in ('one_neutral','two_attraction'):
    pairs=[(r['capture_success'],reference[r['seed']]['capture_success']) for r in rows if r['condition']==condition]
    wins=sum(a>b for a,b in pairs);losses=sum(a<b for a,b in pairs);n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    result['comparisons'][condition]={'condition_only_captures':wins,'two_neutral_only_captures':losses,
        'both_capture':sum(a==b==1 for a,b in pairs),'neither_capture':sum(a==b==0 for a,b in pairs),
        'two_sided_exact_binomial_discordance_p':p}
with (root/'paired_binary_sensitivity.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
