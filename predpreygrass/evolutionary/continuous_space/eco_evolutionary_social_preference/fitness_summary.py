"""Episode-paired descriptive fitness comparisons in fixed mixed populations."""
import argparse
import json
from pathlib import Path
from .summarize import summarize

METRICS = {
    'founder_observed_lifespan_ticks': ('founders', 'mean_observed_lifespan'),
    'founder_observed_offspring': ('founders', 'mean_observed_offspring'),
    'founder_mean_energy': ('founders', 'mean_energy_predator_time'),
    'all_offspring_per_1000_predator_ticks': ('all', 'offspring_per_1000_predator_ticks'),
    'all_mean_energy': ('all', 'mean_energy_predator_time'),
}


def build_summary(run):
    episodes = run['episodes']
    paired = {}
    totals = {}
    for value, key in [('0', '0.0'), ('-1', 'avoidance'), ('1', 'attraction')]:
        paired[key] = []
        totals[key] = {cohort: {field: 0 for field in ('count', 'deaths', 'right_censored', 'offspring', 'predator_ticks')}
                       for cohort in ('founders', 'born', 'all')}
        for row in episodes:
            groups = row['persistence']['fitness']['groups'][value]
            paired[key].append({'seed': row['seed'], 'movement': {'means': {
                metric: groups[cohort][field] for metric,(cohort,field) in METRICS.items()}}})
            for cohort, fields in totals[key].items():
                for field in fields: fields[field] += groups[cohort][field]
    horizons = sorted({r['persistence']['fitness']['episode_end_tick'] for r in episodes})
    return {'summary': summarize(paired, metrics=tuple(METRICS)), 'totals': totals,
            'episode_end_ticks': horizons,
            'lifespan_interpretation': 'founder restricted survival through common episode horizon' if len(horizons) == 1
            else 'observed time only; unequal episode horizons require a separate survival analysis',
            'limitations': 'Preferences fixed, non-inherited. Descriptive mixed-population contrasts; no evolutionary fitness claim. Birth cohorts have unequal follow-up. Energy reflects both feeding and reproductive expenditure.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = build_summary(json.loads(args.run.read_text()))
    with (args.run.parent/'summary.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
