"""Episode-level paired summaries; individual agents are not replicates."""
import numpy as np

METRICS = (
    'visible_target_projection_per_tick',
    'visible_target_displacement_alignment_when_moving',
    'visible_pair_closing_per_tick',
    'all_predator_nearest_neighbor_distance_normalized',
    'predator_population',
)


def summarize(results, bootstrap_samples=10000):
    by_seed = {condition: {row['seed']: row for row in rows} for condition, rows in results.items()}
    if any(len(by_seed[k]) != len(rows) for k, rows in results.items()):
        raise ValueError('duplicate seeds')
    seeds = sorted(by_seed['0.0'])
    if not seeds or any(set(rows) != set(seeds) for rows in by_seed.values()):
        raise ValueError('conditions must contain identical nonempty seed sets')
    rng = np.random.default_rng(20261026)
    indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    summary = {'seeds': seeds, 'replicate_unit': 'episode seed',
               'interval_method': f'paired percentile bootstrap, {bootstrap_samples} resamples; exploratory, no multiplicity adjustment',
               'metrics': {}}
    for metric in METRICS:
        arrays = {key: np.array([rows[seed]['movement']['means'][metric] for seed in seeds], dtype=float)
                  for key, rows in by_seed.items()}
        # Omit a seed for this metric across all conditions if any value is missing.
        valid = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
        arrays = {key: a[valid] for key, a in arrays.items()}
        n = int(valid.sum())
        entry = {'paired_seeds': [seed for seed, keep in zip(seeds, valid) if keep],
                 'condition_means': {key: float(a.mean()) if n else None for key, a in arrays.items()},
                 'differences_from_neutral': {}}
        if n:
            draws = indices if n == len(seeds) else rng.integers(0, n, size=(bootstrap_samples, n))
            for key in ('-1.0', '1.0'):
                delta = arrays[key] - arrays['0.0']
                entry['differences_from_neutral'][key] = {
                    'mean': float(delta.mean()),
                    'seed_differences': delta.tolist(),
                    'bootstrap_95_percent_interval': np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist() if n > 1 else None,
                }
        summary['metrics'][metric] = entry
    return summary
