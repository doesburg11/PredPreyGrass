import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/fitness_confirmation_checkpoint2_s102_201')
s=json.loads((root/'summary.json').read_text())
rows=[]
for p in sorted(root.glob('worker_*.episodes.jsonl')):
    rows.extend(json.loads(line) for line in p.read_text().splitlines())
rows.sort(key=lambda r:r['seed'])
assert [r['seed'] for r in rows] == list(range(102,202))
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,metric,title in zip(axes[:2],['founder_observed_lifespan_ticks','founder_observed_offspring'],['Founder observed alive time (ticks)','Founder offspring']):
    v=s['primary'][metric];lo,hi=v['bootstrap_97_5_percent_interval']
    ax.axvline(0,c='#999',lw=1)
    ax.errorbar(v['mean'],0,xerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',capsize=6,color='#c65d3b')
    ax.set_ylim(-.7,.7);ax.set_yticks([0],['Attraction − neutral']);ax.set_title(title)
    ax.set_xlabel('Mean paired difference; 97.5% interval')
for value,label,color in [('-1','Avoidance','#4078a5'),('0','Neutral','#888888'),('1','Attraction','#c65d3b')]:
    samples=np.full((100,8),np.nan)
    for i,row in enumerate(rows):
        for b in row['persistence']['fitness']['energy_bins_1000_ticks']:
            samples[i,b['start_tick']//1000]=b['groups'][value]['mean_energy']
    axes[2].plot(np.arange(8)*1000+500,np.nanmean(samples,axis=0),label=label,c=color)
axes[2].set_title('Energy over time (descriptive)');axes[2].set_xlabel('Physics tick (window midpoint)');axes[2].set_ylabel('Episode-mean alive energy');axes[2].legend(fontsize=8)
fig.suptitle('Fresh confirmation · checkpoint 2 · 100 mixed episodes · seeds 102–201')
fig.tight_layout();fig.savefig(root/'confirmation_comparison.png',dpi=160)
