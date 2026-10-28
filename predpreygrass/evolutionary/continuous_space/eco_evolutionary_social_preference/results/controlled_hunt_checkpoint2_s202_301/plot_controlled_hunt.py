from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/controlled_hunt_checkpoint2_s202_301')
s=json.loads((root/'summary.json').read_text())
conditions=['one_neutral','two_neutral','two_attraction'];labels=['One neutral','Two neutral','Two attraction'];colors=['#6c87a5','#888888','#c65d3b']
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for i,c in enumerate(conditions):
    p=s['successes'][c];lo,hi=s['capture_probability_wilson_95_percent_intervals'][c]
    axes[0].bar(i,p,color=colors[i],alpha=.8)
    axes[0].errorbar(i,p,yerr=[[p-lo*100],[hi*100-p]],fmt='none',c='#333',capsize=5)
    axes[0].text(i,p+1,str(p)+'/100',ha='center',fontsize=9)
    previous=0
    for outcome,col in [('captured','#4a9873'),('birth_stop','#a9a9a9'),('timeout','#d9ad4c')]:
        n=s['outcomes'][c].get(outcome,0)
        axes[1].bar(i,n,bottom=previous,color=col,label=outcome if i==0 else None)
        previous+=n
    key='0.0' if c=='two_neutral' else c
    axes[2].bar(i,s['metrics']['team_total_energy_spent']['condition_means'][key],color=colors[i],alpha=.8)
for ax in axes:ax.set_xticks(range(3),labels,rotation=15)
axes[0].set_ylabel('Capture success (%)');axes[0].set_ylim(0,24);axes[0].set_title('Before birth / stop / deadline')
axes[1].set_ylabel('Trials');axes[1].set_title('Why trials stopped');axes[1].legend(fontsize=8,loc='lower left')
axes[2].set_ylabel('Mean total predator energy spent');axes[2].set_title('Costs only until trial stops')
fig.suptitle('100 matched short-hunt situations · birth stops dominate this protocol')
fig.tight_layout();fig.savefig(root/'controlled_hunt_comparison.png',dpi=160)
