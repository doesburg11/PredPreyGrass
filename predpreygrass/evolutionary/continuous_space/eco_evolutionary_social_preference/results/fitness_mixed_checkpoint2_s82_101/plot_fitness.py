import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/fitness_mixed_checkpoint2_s82_101')
run=json.loads((root/'fitness.json').read_text())
summary=json.loads((root/'summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
keys=['avoidance','0.0','attraction'];labels=['Avoidance','Neutral','Attraction']
colors=['#4078a5','#888888','#c65d3b']
for ax,metric,title in zip(axes[:2],['founder_observed_lifespan_ticks','founder_observed_offspring'],['Founder observed life (ticks)','Founder offspring']):
    means=summary['summary']['metrics'][metric]['condition_means']
    for i,k in enumerate(keys):
        ax.bar(i,means[k],color=colors[i],alpha=.65)
        val={'avoidance':'-1','0.0':'0','attraction':'1'}[k]
        field='mean_observed_lifespan' if 'lifespan' in metric else 'mean_observed_offspring'
        points=[r['persistence']['fitness']['groups'][val]['founders'][field] for r in run['episodes']]
        ax.scatter(np.full(len(points),i)+np.linspace(-.15,.15,len(points)),points,s=10,c=colors[i],edgecolors='white',linewidths=.2)
    ax.set_xticks(range(3),labels,rotation=15);ax.set_title(title)
for value,label,color in zip(['-1','0','1'],labels,colors):
    samples=np.array([[b['groups'][value]['mean_energy'] for b in r['persistence']['fitness']['energy_bins_1000_ticks']] for r in run['episodes']],dtype=float)
    x=np.arange(samples.shape[1])*1000+500
    axes[2].plot(x,np.nanmean(samples,axis=0),label=label,color=color)
axes[2].set_title('Energy over time (all predators)');axes[2].set_xlabel('Physics tick (window midpoint)');axes[2].set_ylabel('Episode-mean alive energy');axes[2].legend(fontsize=8)
fig.suptitle('Fixed mixed preferences · checkpoint 2 · 20 matched episode seeds')
fig.tight_layout();fig.savefig(root/'fitness_comparison.png',dpi=160)
