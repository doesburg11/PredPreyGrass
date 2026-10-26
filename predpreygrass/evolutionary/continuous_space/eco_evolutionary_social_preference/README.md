# Social preference: fixed-policy proof of concept

**Status: checkpoint fidelity and neutral inference validated; small matched-seed evaluation completed (2026-10-26).**
Isolated sibling module. The existing continuous-traits code, checkpoints,
training and RESULTS.md are untouched.

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

## Go / no-go

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
