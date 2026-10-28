No correctness bugs found in the reviewed paths.

- Read-only hooks and repeat-reset guard are sound: [hunting_metrics.py](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_metrics.py:11) avoids double wrapping; the wrappers delegate to the original bound methods without changing simulation values ([lines 25–55](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_metrics.py:25)).
- Capture offer and clipped retained gain correctly match `_energy` / `_gain` semantics: offered energy is credited only for live catchers ([lines 41–45](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_metrics.py:41)); retained gain uses actual before/after energy ([lines 27–32](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_metrics.py:27)).
- Movement/rest costs use the post-movement velocity and include terminal intervals ([lines 35–40](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_metrics.py:35)). Birth cost and final-energy conservation are properly checked ([lines 67–75](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_metrics.py:67)).
- Replay checks retain every pre-existing field while allowing only the new hunting ledger ([hunting_replay.py:34](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_replay.py:34)). The 100 expected seeds are shard-checked and serial-vs-parallel seed 102 is compared over all fields ([lines 64–80](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_replay.py:64), [134–143](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_replay.py:134)).
- Spawn multiprocessing and worker checkpoint checks are appropriate ([lines 130–133](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_replay.py:130)).
- Paired summaries use episode-level rates, align seed sets through `summarize`, and keep raw totals separately ([lines 41–61](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/hunting_replay.py:41)).

Validation run with `.conda/bin/python`:

- `test_hunting_metrics.py`: 3 passed
- Related fitness and replay-summary tests: 6 passed
- Both reviewed modules compile successfully

No files modified; no results JSONs inspected.
