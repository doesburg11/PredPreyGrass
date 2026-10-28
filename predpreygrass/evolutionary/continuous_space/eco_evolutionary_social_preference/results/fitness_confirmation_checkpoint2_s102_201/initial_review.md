Review findings (no files modified):

- **High — violates the “avoid giant results JSONs” requirement:** [fitness_confirmation.py:146](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/fitness_confirmation.py:146) writes all 100 full episode records—including per-animal lifetime records and preference assignments—into one `fitness.json`. The existing 20-episode pilot is already ~17.7 MB, so this will grow substantially. Keep the streaming JSONL as the detailed artifact and write a compact final manifest/summary with hashes and aggregate outcomes instead.

- **Medium — the supposedly fixed protocol leaves the assignment randomization seed selectable through the pilot:** [fitness_confirmation.py:113](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/fitness_confirmation.py:113) copies `baseline['assignment_seed']` without checking it against the predeclared seed (the current pilot uses `20261027`). Since assignment seed determines individual preference allocation, accepting any otherwise-shaped baseline weakens the “fixed fresh-seed protocol.” Pin it explicitly, or include an immutable pilot-file digest/predeclared manifest.

- **Medium — incomplete frozen-checkpoint provenance coverage:** [fitness_confirmation.py:101](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/fitness_confirmation.py:101) relies on `fingerprint()`, which does not hash the per-policy `class_and_ctor_args.pkl` files that [validation.py] reads. Validation makes silent meaningful drift unlikely, but the recorded input hash is not a complete checkpoint freeze. Extend the fingerprint to every file consumed by loading/validation.

- **Coverage gap:** [test_fitness_confirmation.py:22](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/test_fitness_confirmation.py:22) tests only `confirmation_summary` with synthetic inputs. There is no test for:
  - fixed baseline/assignment-seed rejection;
  - worker seed partitioning and sorted deterministic aggregation;
  - worker checkpoint validation before/after;
  - pilot replay verification;
  - compact-output behavior;
  - a non-degenerate bootstrap/multiplicity case.

The primary analysis itself is otherwise sensible: episode-level attraction-minus-neutral differences are paired correctly, shared bootstrap resampling is appropriate, and 97.5% two-sided marginal percentile intervals give the intended conservative Bonferroni-style threshold for the two endpoints. Early administrative censoring is correctly made disqualifying for the stated full-horizon joint claim.

Multiprocessing is structurally well handled: spawn context, single Torch thread per worker, fixed chunks/seeds, and final seed sorting eliminate completion-order effects. It still lacks an integration regression test demonstrating identical compact outcomes across worker ordering/run repetitions.

Tests run successfully:

```text
.conda/bin/python -m pytest -q .../test_fitness_metrics.py .../test_fitness_summary.py .../test_fitness_confirmation.py
7 passed
```
