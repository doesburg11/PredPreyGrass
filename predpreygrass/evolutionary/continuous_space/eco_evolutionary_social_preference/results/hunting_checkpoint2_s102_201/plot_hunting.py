from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/hunting_checkpoint2_s102_201')
s=json.loads((root/'summary.json').read_text())
metrics=['all_captures_per_1000_predator_ticks','all_retained_hunting_energy_per_1000_predator_ticks','all_movement_energy_per_1000_predator_ticks']
labels=['Prey captures','Retained hunting energy','Movement energy charged']
fig,axes=plt.subplots(1,3,figsize=(13,3.3))
for ax,metric,label in zip(axes,metrics,labels):
    contrasts=s['summary']['metrics'][metric]['differences_from_neutral']
    for y,key,color in [(0,'attraction','#c65d3b'),(1,'avoidance','#4078a5')]:
        c=contrasts[key];lo,hi=c['bootstrap_95_percent_interval']
        ax.errorbar(c['mean'],y,xerr=[[c['mean']-lo],[hi-c['mean']]],fmt='o',capsize=5,color=color)
    ax.axvline(0,c='#999',lw=1);ax.set_yticks([0,1],['Attraction − neutral','Avoidance − neutral']);ax.set_ylim(-.65,1.65)
    ax.set_title(label);ax.set_xlabel('Difference per 1,000 predator-ticks')
fig.suptitle('Same 100 mixed episodes · exploratory paired 95% intervals')
fig.tight_layout();fig.savefig(root/'hunting_comparison.png',dpi=160)
