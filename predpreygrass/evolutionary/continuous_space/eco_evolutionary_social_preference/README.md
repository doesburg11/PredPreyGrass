# Social preference: fixed-policy proof of concept

**Status: closed (2026-10-28). Behavioral proof of concept demonstrated;
no survival, reproductive or hunting benefit established in the tested settings.**

The completed work shows that fixed attraction can change predator movement,
spacing and encounter duration. The larger mixed-population evaluation did
not confirm the pilot's apparent performance advantage. The final controlled
hunt was dominated by early prey births and does not settle longer pursuit
performance. This is not proof that attraction can never help.

Read [RESULTS.md](RESULTS.md) for the final assessment, evidence by phase,
limitations and closure decision. Code, checkpoints and existing result
artifacts are retained as a reproducible baseline. No further strength sweeps,
seed extensions, social inheritance or PPO training are planned for this phase.
Reopening it would require a separately specified mechanism and evaluation
protocol, including explicit handling of reproduction in isolated hunts.

This module is an isolated sibling of the continuous-traits environment.
Its training, checkpoints and existing environment rules were preserved.
The social preference remains externally assigned and non-heritable; existing
body-trait inheritance was not changed. The broader learning/evolution and
cooperation objectives remain open, including the separate
[field-of-view experiment](../eco_evolutionary_continuous_fov/README.md).

## Final findings

- **Behavior:** strength-2 uniform attraction produced closer spacing and
  longer local encounters in both evaluated checkpoints. These checkpoints
  share an ancestral training run, so this is not independent replication
  across training histories. Stable group membership was not established.
- **Performance:** in 100 fresh mixed episodes, attraction minus neutral was
  −217 observed founder ticks alive and +0.195 offspring per founder; adjusted
  97.5% intervals include zero for both. One early-ending episode limits
  full-horizon survival interpretation. Hunting rates and energy costs also
  showed no clear attraction advantage.
- **Controlled hunts:** captures were 4/100 with one neutral predator,
  11/100 with two neutral predators, and 9/100 with two attracted predators.
  Births ended 89–95% of trials, usually around tick 37 of a planned maximum
  of 256. This birth-limited diagnostic does not establish cooperation or
  longer-chase performance.
- **Verification:** checkpoint tensors and actor logits were validated,
  neutral inference and matched replays checked, and energy ledgers balanced.
  The latest verification passed 65 current top-level tests. This closure
  changes documentation only; it introduces no new runs or analysis.

The sections below preserve the chronological implementation and experimental
record. Test counts, go/no-go criteria and proposed next steps in that record
refer to their respective stages; the closure decision above and
[final results](RESULTS.md) supersede earlier follow-up suggestions.

## Question

Does a *fixed*, per-agent preference for proximity to visible conspecifics
change the behavior of a frozen predator PPO policy? This is **not** genetic
evolution yet. No social rewards are introduced.

## Mechanism

The original policy produces 48 logits (16 directions × full/half/stop).
For predator actions only, add `strength * preference * cosine(direction,
nearest visible predator)` to moving-action logits; stop-action logits
remain unchanged. Preference -1 = avoidance, 0 = neutral, +1 = attraction.
No visible conspecific => exactly zero bias. Uses the existing predator
observation slots, not hidden environment state.

The direction is global, matching the existing discrete actions. The
preference is applied *after frozen-policy inference and before sampling*.
**Do not use this sampler for PPO training without including the adjusted
log probabilities in PPO's optimization.**

## Run

```bash
python -m pytest predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/test_social_bias.py
python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.evaluate /path/to/checkpoint --episodes 5 --output social_poc.json
```

Uses one frozen PPO checkpoint, matched episode seeds, and the same initial
gene pool for -1, 0, +1 preferences. A stochastic run may still diverge
quickly; replicate across independent seeds before drawing conclusions.

## Original go / no-go criteria (historical)

1. Unit tests: neutrality, no visibility, opposite directional effects.
2. Frozen checkpoint: positive versus negative preference should shift
   mean action alignment in the expected directions.
3. Secondary ecological measurements: normalized distance to visible
   conspecifics and moving-action fraction. These are descriptive, not
   causal proof of improved fitness.
4. If no detectable behavioral effect, stop and debug; **do not add genes**.

No rewards, body genes, selection, policy architecture, or PPO training are
changed in Steps 1–3.

## Fidelity checks and revised measurements

`validation.py` compares all 14 saved tensors for each species, including
actor and critic, exactly with the loaded CPU network. It independently
computes the saved two-layer 256-unit tanh actor MLP and linear 48-logit
head on initial observations plus 64 seeded probes per species. The
checkpoint was written with Ray 2.59.0; the restored modules are
`DefaultPPOTorchRLModule`, with zero actor-logit error. A bare `RLModule`
warning alone therefore does not imply that random weights were used:
the existing loader rebuilds the default network and loads its saved state.
This check establishes compatibility with the saved architecture and state;
it does not replay inference in the original Ray installation.

The evaluator now uses the original evaluator's sorted agents, float32
Torch softmax, batched multinomial sampling and seeded Torch generator.
Every neutral step compares actions and generator state with the original
`eval.select_actions`. The old NumPy sampler had the same intended
categorical distribution but did not reproduce original seeded trajectories;
the first one-episode results are therefore not directly seed-comparable.

Additional metrics report visibility fraction, alignment conditional on both
visibility and a moving action, and movement probability given visibility.
Expected alignment averages the policy's conditional directional expectation
per visible decision; sampled alignment pools selected moving actions, so
these use different weighting. Same-observation counterfactual metrics apply
all three preferences to each visited observation without stepping the world.
These isolate the immediate directional effect from trajectory and visibility
changes. Alignment measures commanded action direction, not realized velocity
or distance change after inertia, collisions and other animals move.

Output uses exclusive creation and refuses to overwrite an existing result.
PPO parameters, rewards, and the environment's existing inheritance mechanisms
are unchanged. Existing body-trait expression and reproduction still operate;
this experiment introduces no social genes or inheritance changes.

The independent Codex CLI review initially failed because the configured model
was rejected. Model access is now repaired and review has completed; see
[REVIEW.md](REVIEW.md) for findings, disagreements and open follow-ups.

## Matched-seed results

Checkpoint: `blrep2500_both_learn_s1/checkpoint`; strength 1.0; seeds 42–44;
one episode per seed and condition. Runtime: project `.conda/bin/python`,
Ray 2.59.0, PyTorch 2.14.1+cu132. Seven unit tests passed.
Full episode data: `results/validated_counterfactual_s42_44.json`.
`results/validated_s42_44.json` is the preceding run with identical actions
and metrics but without the same-observation measurements.

Values below are unweighted means of the three episode means, not pooled
agent decisions or independent-agent statistical estimates.

| Metric | Avoidance -1 | Neutral 0 | Attraction +1 |
|---|---:|---:|---:|
| Alignment given visible and moving | -0.3704 | -0.2502 | -0.1599 |
| Moving-action fraction | 0.3451 | 0.3239 | 0.3359 |
| Visible-decision fraction | 0.3514 | 0.4275 | 0.4052 |
| Visible distance, normalized | 0.1925 | 0.1902 | 0.1899 |

The alignment ordering holds on each seed. On the **same neutral-run
observations**, expected alignment conditional on moving is -0.4345,
-0.3255 and -0.1997 for avoidance, neutral and attraction respectively.
This demonstrates the directional intervention independently of changes in
movement probability and which states each condition visits. Attraction
at this strength still has negative alignment: it weakens the frozen
policy's avoidance tendency rather than yielding net approach.

Distance changes are small and visibility-conditioned; they do not establish
clustering. Only three seeds were tested, observations within episodes are
correlated, and diverging populations consume random draws differently even
with matched starting seeds. Attraction seed 43 also ended sooner (595
steps versus 1237 avoidance and 1229 neutral), so survivor and episode-length
selection may affect aggregate metrics. No fitness or stable social-group
claim follows from these results. All neutral actions and RNG states matched
the original inference implementation throughout all three episodes.

## Actual movement evaluation

`movement_metrics.py` adds read-only measurements without changing actions,
random draws, rewards, physics, PPO, or inheritance. A seed-42 smoke run
(`results/movement_smoke_s42.json`) reproduces every previous metric exactly
for all three conditions. Unit tests cover wrapped movement, nearest neighbors,
stationary focal animals with moving targets, deaths, and paired seed summaries.

For each decision interval, identify the nearest visible predator at its start
using the same visibility geometry as the observation slots. Keep that target
fixed for measurement, even if it later leaves the field of view:

- **Projection per physics tick:** the focal predator's actual net displacement
  projected toward the target's initial position, divided by elapsed ticks.
  Positive means the focal animal moved toward where its neighbor was.
- **Displacement alignment when moving:** cosine between that displacement and
  the initial target direction, excluding zero-displacement intervals.
- **Pair closing per physics tick:** initial minus final toroidal distance to
  that same target, divided by elapsed ticks. Positive means the pair got closer;
  this includes both animals' movements.
- **All-predator nearest-neighbor distance:** shortest toroidal distance for
  every living predator with a neighbor, including unseen neighbors, divided
  by `env.distance_scale` (400 world units here). Lower means closer spacing,
  but population density, births and deaths can also change it.

Focal deaths exclude displacement measurements. Either animal's death excludes
pair-closing measurements; a newborn never substitutes for the original target.
Counts and denominators are saved. Nearest-neighbor distances are sampled at
pre-decision states, including singletons only in population counts. These are
agent-decision means, not physics-time-weighted means; reproduction may shorten
an action-repeat interval. Net displacement uses shortest wrapped offsets; at
this checkpoint's arena, speed and action-repeat settings animals cannot travel
half an arena in one interval. These metrics are intended for this toroidal run.

`evaluate` saves completed episodes to an exclusive `.episodes.jsonl` beside the
final JSON, prints episode progress to stderr, and validates the loaded weights
again at the end. `summarize.py` pairs episodes by seed and estimates exploratory
95% percentile bootstrap intervals for differences from neutral using 10,000
seed-level resamples. Agents and decisions are not treated as independent
replicates. No multiple-comparison adjustment is applied. Final condition means
weight each episode equally. Three of the 20 seeds repeat the earlier pilot.

The 20-seed run uses seeds 42–61, strength 1.0 and the same completed checkpoint:

```bash
nice -n 10 .conda/bin/python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.evaluate \
  predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/runs/blrep2500_both_learn_s1/checkpoint \
  --episodes 20 --seed 42 \
  --output predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/results/movement_s42_61_strength1.json
```

Existing background Python/Ray jobs were left running. This evaluator uses one
CPU thread at low priority. The independent Codex CLI review initially failed
with an unsupported-model error. After changing the CLI default to a tested
model, the review completed. See [REVIEW.md](REVIEW.md).

The 20-seed evaluation is complete. See [MOVEMENT_RESULTS.md](MOVEMENT_RESULTS.md)
for condition means, paired uncertainty intervals, sensitivity checks and the
matched-seed plot. Attraction increases actual movement toward visible neighbors,
but its effects on pair closing and overall spacing remain uncertain. Avoidance
increases separation more consistently. Sixteen tests pass, and all 20 neutral
episodes matched original inference throughout. Checkpoint validation remained
unchanged at the end of the run.

## Stronger attraction sweep

Attraction strengths 2 and 4 were evaluated on seeds 42–61, using the same
frozen checkpoint and physical metrics. Strength 2 gives about 5.7% closer
nearest-neighbor spacing than neutral; strength 4 gives a stronger directional
effect but less consistent spacing and a lower mean predator population.
See [ATTRACTION_STRENGTH_RESULTS.md](ATTRACTION_STRENGTH_RESULTS.md) for
paired intervals, episode-length sensitivity, plots and provenance. Twenty
tests pass, and the independent runner re-review found no remaining material
correctness concerns.

## Fresh-seed persistence and matched replays

The neutral-versus-attraction-2 comparison is complete on fresh seeds 62–81.
Close encounters lasting at least 100 physics ticks increased from 15.4% to
19.7%, and fresh nearest-neighbor spacing was about 5.5% closer. Keeping the
same nearest neighbor remains inconclusive. See
[PERSISTENCE_RESULTS.md](PERSISTENCE_RESULTS.md) for censoring definitions,
paired intervals, raw data and matched videos for preselected seeds 62 and 63.
Twenty-seven local tests pass. No stable-group or fitness claim follows yet.


## Second-checkpoint replication

Neutral versus strength-2 attraction was repeated with
`blrep2500_both_learn_s2/checkpoint` on fresh seeds 82–101. Close-pair
100-tick retention increased from 15.8% to 21.1%, and nearest-neighbor
spacing was about 6.7% closer. Both primary effects replicate the first
checkpoint result; keeping the same nearest neighbor remains inconclusive.
See [SECOND_CHECKPOINT_RESULTS.md](SECOND_CHECKPOINT_RESULTS.md) for paired
intervals, provenance, limitations and an across-checkpoint comparison plot.
No training, reward or inheritance changes were made.

## Fixed individual preferences

Predators can now receive different non-heritable lifetime preferences.
`mixed_evaluate` assigns shuffled ID blocks of -1, 0 and +1, including newborns,
without ancestry or simulation-RNG input. Six founders give two of each type.
The runner verifies every saved neutral and attraction control episode through
the per-agent path before testing uniform avoidance and a mixed population.

On checkpoint 2, seeds 82–101, the balanced mixed population remained close
to neutral overall; attraction/attraction pairs had longer encounters than
avoidance/avoidance pairs within it. See
[MIXED_PREFERENCES_RESULTS.md](MIXED_PREFERENCES_RESULTS.md) for methods,
paired intervals, assignment audits, review assessment and source history.
All 34 final local tests pass. Values are still externally assigned, not genes.

## Survival, offspring and energy

The same 20 mixed-population seeds were replayed with per-animal lifetime,
offspring and 1,000-tick energy measurements. Every prior saved field matched.
Attraction founders averaged longer observed lives and more offspring, but
paired uncertainty intervals include zero: a fitness benefit is not established.
See [FITNESS_RESULTS.md](FITNESS_RESULTS.md) for censoring, tables, energy curves,
raw data and review. All 38 tests pass; policies, rewards and preference
inheritance remain unchanged.

## Fresh 100-episode fitness confirmation

The fixed mixed-preference experiment was repeated on fresh seeds 102–201.
Attraction's pilot advantage was not confirmed: attraction minus neutral was
−217 observed ticks alive (adjusted 97.5% interval −785 to +356) and +0.195
offspring per founder (−0.315 to +0.710). One early episode leaves four
founders censored, so alive time is reported as observed time. All seeds were
retained and the pilot was not pooled. Pilot/serial replay fidelity and
checkpoint/source checks passed; 46 tests pass. See
[FITNESS_CONFIRMATION_RESULTS.md](FITNESS_CONFIRMATION_RESULTS.md) for the
predeclared design, tables, energy curves, artifacts and independent review.

## Hunting efficiency diagnostics

The same 100 mixed episodes now include credited captures, actual retained
hunting energy, and movement/resting energy charges. Every prior saved field
matched. Attraction versus neutral showed no clear difference in captures,
feeding return or movement cost; all exploratory paired 95% intervals include
zero. Per-animal energy conservation passed for 4,297 predators, and all 51
current top-level tests pass. See [HUNTING_RESULTS.md](HUNTING_RESULTS.md) for
definitions, tables, caveats, plots, data and independent review.

## Controlled one-versus-two predator hunts

A short pilot ran 100 matched situations with one neutral, two neutral or two
attracted predators. Captures were 4, 11 and 9 respectively. Attraction added
no captures over the neutral pair. However, 89–95% of trials stopped at a prey
birth, generally around tick 37, so this is a birth-limited endpoint rather
than a full 256-tick chase test. Frozen policies and existing rules were kept.
All 65 current tests pass. See [CONTROLLED_HUNT_RESULTS.md](CONTROLLED_HUNT_RESULTS.md)
for stopping limitations, paired uncertainty, energy costs, source provenance,
review and data.
