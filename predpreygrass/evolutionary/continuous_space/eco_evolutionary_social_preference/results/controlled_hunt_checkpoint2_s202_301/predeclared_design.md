# Controlled short hunts with one or two predators

## Pre-run design

Use the same frozen predator/prey checkpoint (`blrep2500_both_learn_s2`) and
strength-2 attraction, with the original capture, movement, grass, energy,
reproduction and reward rules. Compare 100 matched situations, seeds **202–301**:

1. One predator, neutral preference.
2. Two predators, both neutral.
3. Two predators, both attraction.

The main comparison is two attraction versus two neutral. Adding a predator
provides more hunting effort; an advantage over the solo condition alone is
not evidence that attraction helps, or that predators cooperate.

Each trial lasts at most **256 physics ticks**. Stop at the first capture;
otherwise the first birth, target starvation, loss of all predators, original
environment termination, or the deadline. All trials remain in the denominator.
The endpoint is capture **before a competing stop or the deadline**, not an
estimate of eventual capture probability. Births occur under the original rule;
no reproduction threshold or reward is altered. Stopping at the first birth
keeps subsequent descendants from entering this single-prey test.

### Matched initial state

All arms reset a canonical two-predator/one-prey environment, drawing exactly
the same checkpoint founder genomes. The solo arm then removes only the
companion. This avoids the different gene-RNG draws that separate one/two
population resets would otherwise cause. Focal predator, prey, grass and
birth/gene random states must match exactly; the two-predator arms match their
complete initial states. The reset config still records the canonical two
predators; the actual solo population is explicitly one.

The prey starts at a seeded random translation. Predators are placed at
coordinates `(-d,-40)` and `(-d,+40)` relative to it, rotated by a seeded random
angle; `d` cycles through 80, 120 and 160 across the 100 seeds. These are axial
offsets, not radial distances. Predator headings bisect the bearings to prey
and companion, with original unit initial speed. The focal pose remains
identical in the solo arm. Prey heading is seeded uniform with unit speed.
This deliberately provides nearby partners and potentially visible prey,
rather than waiting for random ecosystem encounters. Initial social visibility
is recorded, not used to select successful situations or retune the design.

These are changed initial conditions, outside the original population setup;
the result is a controlled diagnostic, not an ecological replication.

### Inference and stepping

Each animal uses its own SHA-256-seeded Torch stream, shared across conditions.
Adding a companion therefore does not shift the focal/prey action random
numbers. Neutral/prey actions use the original categorical inference helper;
attraction uses the existing logit bias. This preserves frozen policy
probabilities but intentionally differs from the old shared global sampling
stream, so old ecosystem trajectories are not expected to match.

Actions are held for the saved action-repeat interval. The original
`_physics_step` is called once per tick so the runner can stop precisely at an
event, without simulating the remainder of a decision. A test checks physical state and observations against original `env.step`
over no-event held-action decisions. The direct loop does not reproduce unused
RLlib return aggregation or its `env.agents` bookkeeping; it uses `active` and
current animal observations. Environment source hashes must match the validated 100-episode hunting
baseline. This establishes continuity with those experiments, not historical
training-code equivalence. The unchanged hunting observer records
predator energy costs and verifies each energy ledger.

### Outcomes and analysis

- Capture success before the competing stop/deadline is the main outcome.
- Successful capture tick is recorded. For an all-trial time comparison,
  unsuccessful trials receive a fixed score of 256. This is a penalty score,
  not an imputed capture time. Success-only mean time is descriptive and
  selected on success; it is not a matched causal comparison.
- Total team movement, resting and reproductive energy spent is summed through
  each trial's stopping time. This includes both predators in the two-predator
  arms. Interpret cost jointly with success and duration; early competing
  stops can reduce observed cost without improving hunting efficiency.

Use 10,000 paired situation bootstrap resamples (seed 20261026), with exploratory
95% intervals and no multiplicity adjustment. Also report per-condition Wilson
95% success-rate intervals and discordant capture-pair counts. A bootstrap
contrast interval can collapse when no pairs are discordant; that does not
prove equivalence. Do not analyse interim outcomes or tune the geometry/seeds.

## Reproduce

From the repository root, choose a new output directory:

```bash
nice -n 10 .conda/bin/python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.controlled_hunt \
  predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/runs/blrep2500_both_learn_s2/checkpoint \
  --output-dir /tmp/ppg_controlled_hunt_new
```
