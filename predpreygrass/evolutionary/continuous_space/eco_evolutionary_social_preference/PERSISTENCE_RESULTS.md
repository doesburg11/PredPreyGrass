# Fresh-seed proximity persistence

Finalized 2026-10-27. Strength-2 attraction makes close encounters last longer
on fresh seeds. Evidence that predators keep a particular nearest neighbor
is inconclusive; stable social groups have not been established.

## Design and validation

Neutral versus attraction +1 at strength 2, with 20 fresh matched seeds
62–81: 40 episodes, all reaching 8000 physics ticks. The same frozen
`blrep2500_both_learn_s1/checkpoint` was used. Seeds 62 and 63 were selected
for position replays before observing the results.

The primary close distance was fixed at 100 world units, with 75 and 125
as sensitivity thresholds. Persistence means that the same unordered pair
remains continuously within the threshold for at least 100 physics ticks.
All distances use toroidal geometry. Measurements run after every physics
tick, including ticks inside action repeats, without changing the transition,
policy inputs, RNG draws, actions, rewards or inheritance.

The observer-enabled seed-42 smoke test reproduced prior action and movement
metrics in both conditions. All 27 local tests pass. Every fresh neutral
action/RNG check passed. Weight tensors, independent actor logits, settings,
founder-pool hashes and source hashes were validated throughout the run.
Both conditions have matching runtime and source/input provenance in the raw
results. No old experiment outputs were overwritten.

## Results

Values are equal-weight averages of 20 episode-level estimates. Intervals
are exploratory paired 95% percentile bootstrap intervals from 10,000
seed-level resamples, without multiple-comparison adjustment.

| Measurement | Neutral | Attraction 2 | Paired difference [95% interval] |
|---|---:|---:|---:|
| Nonbirth close encounters lasting at least 100 ticks, threshold 100 | 15.35% | 19.73% | +4.38 percentage points [2.42, 6.12] |
| Mean completed close-encounter duration, threshold 100 | 55.48 ticks | 66.48 ticks | +11.00 ticks [7.16, 14.70] |
| Predator-time with any neighbor within 100 units | 45.40% | 52.27% | +6.87 percentage points [4.77, 8.89] |
| Same nearest-neighbor identity retained for 100 ticks | 16.43% | 17.35% | +0.92 percentage points [-0.27, 2.28] |
| All-predator nearest-neighbor distance, agent-time weighted | 111.56 units | 105.46 units | -6.09 units [-8.74, -3.51] |
| Mean predator population, time weighted | 13.58 | 12.98 | -0.60 [-1.02, -0.16] |

The fresh-seed spacing result is about 5.5% closer than neutral, consistent
with the preceding 42–61 experiment. Close-encounter retention also increases
at both sensitivity thresholds: 11.28% to 15.55% at 75 units, and 21.12% to
24.58% at 125 units; the paired difference intervals exclude zero.

## Censoring and interpretation

Tick-0 spells are left-censored and excluded from onset-based retention.
A newly close pair on the tick either member first appears is marked
birth-associated and excluded from the primary nonbirth analysis. Other
encounters involving relatives or young animals remain possible.

A physical separation is an observed failure. Death and episode end censor
an encounter; they are not silently counted as separation. Retention among
known outcomes excludes spells censored before the 100-tick horizon. Raw
counts and compact spell records are saved. Treating all unresolved spells
as failures versus successes gives mean retention bounds of 15.00–17.27%
for neutral and 19.27–21.61% for attraction; these are censoring bounds,
not confidence intervals. Completed-duration means exclude censored spells
and should not be interpreted as unbiased lifetime estimates.

The nearest-neighbor identity measure is directed, does not require mutual
nearest neighbors, and does not itself require proximity. Its inconclusive
change limits a stronger interpretation about persistent partner choice.
The positive close-pair result supports longer local encounters, rather
than proof of stable membership, coordinated hunting, cooperation or fitness
benefits. Most observed close encounters still last less than 100 ticks.
Population dynamics, offspring placement and one trained checkpoint remain
important limits. The lower predator population is not a demonstrated fitness
mechanism or a controlled population intervention.

## Matched position replays

Orange points are predators, blue points prey. Green links join pairs within
100 world units, including unseen pairs; they are measurement overlays,
not policy inputs or lines of sight. Predators' trails span up to 64 ticks.
The replays omit grass, energy and field-of-view overlays. Position sampling
is every 8 physics ticks; video playback is 20 frames/second. These are saved
simulation positions, not new inference runs or interpolated trajectories.
Trail segments crossing a torus seam are omitted; positions and wrapped
proximity links remain correct.

- [Matched seed 62 video](results/persistence_s62_81/matched_seed62.mp4)
- [Matched seed 62 snapshots](results/persistence_s62_81/matched_seed62_snapshots.png)
- [Matched seed 63 video](results/persistence_s62_81/matched_seed63.mp4)
- [Matched seed 63 snapshots](results/persistence_s62_81/matched_seed63_snapshots.png)
- [Paired summaries](results/persistence_s62_81/summary.json)
- [Neutral raw results](results/persistence_s62_81/neutral_s62_81.json)
- [Attraction raw results](results/persistence_s62_81/attraction2_s62_81.json)

## Review and reproducibility

The independent Codex review of `persistence.py`, its tests and the observer
hook found no correctness defects. It agreed with per-tick exposure, pair
identity, birth/initial/terminal censoring and unchanged inference/physics.
A directed-neighbor switch/reversion test was added. The seed-42 smoke test
provides recorded integration evidence in addition to unit tests.

A fresh independent runner/replay review completed after the first attempt
was interrupted by CLI usage limits. It confirmed seed-keyed statistics and
flagged open trace-binding, treatment-validation and ambiguous-input-selection
follow-ups. The intended treatment metadata and unique files were directly
verified for this completed run. See REVIEW.md for findings and assessment. Earlier probability
precision and optimized-Python assertion follow-ups remain as previously
documented; recorded evaluations used the standard interpreter and the
physical measurements do not depend on the counterfactual approximation.

Run `persistence_evaluate`, then `persistence_summary`, then
`replay_persistence --seed 62` and `--seed 63` from the repository root.
Each module accepts the checkpoint or results directory as its positional
argument. Choose an unused output directory; existing files are refused.
The saved source hashes identify the code used in the experiment.
