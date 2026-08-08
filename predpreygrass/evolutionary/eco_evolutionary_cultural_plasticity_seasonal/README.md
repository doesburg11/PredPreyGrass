# Cultural Plasticity, Seasonal Variant: Target Dialect Flips Over Time (Trial 9)

## Why this module exists

The original `eco_evolutionary_cultural_plasticity` (Trial 8) evolved a
genetic `plasticity` trait gating adoption of a locally-shared `dialect`. All
3 real replication seeds came back flat — no directional drift, ~zero
individual-fitness correlation (see that module's `RESULTS.md` §3-4). The
diagnosis: the coordination bonus there rewards matching the **local
majority** dialect — a self-referential game with no external "correct
answer" that ever changes. There's nothing for `plasticity` to win by being
fast at, since the crowd is always locally self-consistent with itself. This
is Rogers' Paradox (Rogers, 1988): a gene for social-vs-individual learning
has no fitness advantage in a *static* environment.

A separate exploration (`base_environment_seasonal`, a 6-regime
resource-abundance sweep on the plain, non-evolutionary base env) confirmed a
simple on/off timer keyed on the env's own per-episode step counter works
cleanly with no new plumbing needed. But that mechanism — scaling *how much*
food exists — doesn't test the right thing for Rogers' Paradox either: what's
needed is a change to *which behavior is correct*, not to overall resource
abundance.

**This module reuses that timer pattern, pointed at a different target.**
Instead of scaling grass regrowth, the timer decides which `dialect` is
currently "correct" (bonus-earning), and rotates that pick periodically
(`_current_target_dialect`, cycling through all `n_dialects` every
`dialect_season_length_steps`). After each flip, the population is stuck
matching the *old* answer; agents recover only by noticing they've stopped
earning the bonus and copying a neighbor who's already got the new one —
which is exactly the existing `plasticity`-gated dialect-copying behavior
(`_apply_cultural_learning`), left completely unchanged. This gives
`plasticity` a concrete, repeated, per-episode opportunity to pay off, which
the static local-majority game in Trial 8 never had.

Everything else about the dual-inheritance design (two channels evolving at
different speeds) carries over unchanged from Trial 8:

- **Genetic** (slow, vertical-only, mutation-selection): `plasticity`, a
  continuous trait inherited through the existing `Genome` machinery.
- **Cultural** (fast, vertical *and* horizontal, imitation-driven): `dialect`,
  a live, mutable, per-agent state that can change many times within one
  agent's own lifetime via social learning from neighbors.

Critically, the gene does **not** encode behavior directly — it encodes
*capacity to adopt culture*. This is the literal Baldwin/dual-inheritance
move, and it targets a landscape that is inherently non-smooth (a
coordination game, not a scalar optimum), which is the shape prior modules'
theoretical note said was missing for a Baldwin effect to have anything to
act on.

## The two channels

**`dialect`** (categorical, one of `n_dialects` = 4 arbitrary codes, no code
intrinsically better than another): every agent has a *founder* dialect,
inherited like any other genome field, and a separate *live* dialect
(`agent_live_dialect`), seeded from the founder value at birth but free to
change within that agent's own lifetime. Every `plasticity_check_interval`
steps, each agent looks at the live dialects of same-species neighbors within
`culture_range` (Chebyshev distance) and, with probability equal to its own
`plasticity`, adopts the local majority (`_apply_cultural_learning`). This is
a genuinely per-individual lifetime process, deliberately independent of the
shared PPO policy — the same architectural move `eco_evolutionary_metabolic_code`
used for its per-individual loci-solving, but here decoupled at a landscape
that is frequency-dependent rather than a fixed combination lock.

**`plasticity`** (continuous, `[0, 1]`, Gaussian mutation on reproduction,
same convention as every prior module's scalar trait): the actual Darwinian
gene under test. It sets how readily an agent's culture tracks the local
consensus. High-plasticity lineages track a locally-advantageous dialect
fast; low-plasticity lineages stay put. This is the trait the replication
methodology below tests for selection-driven drift on — not `dialect` itself
(frequency-dependent drift on `dialect` is close to a foregone conclusion and
not informative on its own).

## Fitness effect: a moving target, not a static coordination game

A catch (predator) or graze (prey) event grants `coordination_bonus_multiplier`
(default 1.5x) on the energy gained *if* the agent's current live dialect
matches the environment's **current target dialect** (`_dialect_match_bonus`
→ `_current_target_dialect`) — a square-wave cycle through all `n_dialects`,
`dialect_season_length_steps` steps per phase (default 25; Trial 8 episodes
ran ~180-260 steps, so this gives roughly 2-2.5 full tours per episode, a
first guess not yet tuned). This is **not** what Trial 8 tested: there, the
bonus went to whoever matched the local majority, so being locally popular
was always self-consistently "correct." Here, the target is set externally
and periodically pulled out from under the population, so **being popular
and being correct can diverge** — right after a flip, the whole local
majority is briefly wrong, and only fast adopters (high `plasticity`) recover
the bonus quickly.

`_apply_cultural_learning` (how an agent's live dialect updates by copying
its local majority) and `_local_majority_dialect` are **unchanged from Trial
8** — they're still the mechanism by which dialects spread. The difference is
only in what determines the bonus. The intended emergent loop: right after a
flip, whichever agents already happen to hold the new target (initially by
chance — founder dialect is uniform over `n_dialects` options) earn the
bonus, out-reproduce their neighbors, and spawn near their parent (existing
spatial-viscosity behavior) — so the local majority near them drifts toward
the new target, and `plasticity`-gated neighbors copy it from there. Slow- or
zero-plasticity agents lag and lose the bonus for longer. That lag is the
fitness differential Rogers'-Paradox-resolution models (Boyd & Richerson
1995; Enquist, Eriksson & Ghirlanda 2007) say a social-learning gene needs in
order to show a detectable advantage.

## The reverse leg: policy conditioned on cultural state

`include_culture_in_obs` (default `True`) adds two observation channels: the
agent's own live dialect (normalized) and whether it currently matches the
local majority (unchanged from Trial 8 — note this obs channel still reports
majority-match, not target-match; the policy doesn't directly observe the
season's target dialect). This lets the shared PPO policy learn to condition
movement/hunting on rapidly-changing cultural state (e.g., clustering with
same-dialect neighbors) — a plausible route to the "genome/culture shift
feeds back into learned behavior" leg that every prior module left
unconfirmed.

## What's deliberately unchanged

- **Offspring investment** is a fixed, non-heritable constant
  (`offspring_investment_fraction`, default 0.35) — the heritable channels
  here are plasticity/dialect, not investment fraction.
- **Sustainability mechanism** (predator satiation cooldown + per-catch
  energy cap) is ported unchanged from `eco_evolutionary_metabolic_rate` /
  `eco_evolutionary_investment` / `eco_evolutionary_metabolic_code` — an
  already-validated, orthogonal concern (criteria 1/2 of the project goal),
  not something this trial re-tests.
- **Reproduction/energy dynamics** are otherwise identical to
  `eco_evolutionary_metabolic_code`.

## Predictions and what to watch in training

1. **Primary (dual-inheritance Darwin-signal) test:** live-population mean
   `plasticity` should drift away from the founder mean (0.1) under real
   selection, and — critically — should drift **further than the
   neutral-drift control** (`genome_neutral_drift_control=True`, same
   mechanism ported from every prior module). This is the headline
   comparison in `analyze_replication_seeds.py`
   (`live_culture/{species}_plasticity_mean`), using the |deviation from
   founder| methodology (no a-priori direction, unlike `metabolic_code`'s
   directional mean-WRONG-loci test).
2. **Individual-level cross-check:** `{species}_plasticity_repro_spearman`
   (genome plasticity vs. binary reproduced-or-not, this episode) — a more
   direct test than the population-mean metric, since it doesn't need
   selection to move the aggregate before it's detectable.
3. **Secondary (cultural dynamics) signals, not the headline test:**
   `{species}_dialect_entropy` (0 = one dialect has fixed locally; `ln(4)` =
   maximum diversity) and `{species}_dialect_match_rate` (fraction of
   catch/graze events that earned the coordination bonus — now against the
   season's current target, not the local majority, see above) — these
   should move even under the neutral control (culture is not genetically
   inherited in the sense being tested), so they're diagnostic of the
   cultural-learning mechanism working at all, not of selection on
   `plasticity`. Watch specifically for `dialect_match_rate` **dipping right
   after each phase flip and recovering within the phase** — that dip/
   recovery pattern is the direct, visible signature that the target-flip
   mechanism is actually biting, not just running in the background unused.

Metrics are logged both per-step (`live_culture/*`, currently-alive
population only — `_build_live_culture_metrics`) and per-episode
(`eco_evolution/*`, live + completed agents this episode —
`_build_episode_training_metrics`), mirroring the dual-metric pattern used by
every prior module in this family.

## Methodology this module inherits

Directly ported from `eco_evolutionary_metabolic_code` / `eco_evolutionary_investment`:
- **Satiation-throttle sustainability** — validated, not re-tested here.
- **`genome_neutral_drift_control`** — severs genome inheritance from
  reproductive success (offspring template becomes a uniformly random
  currently-alive same-species agent instead of the true parent); population/
  energy dynamics unchanged. Isolates mutation + finite-population sampling
  noise from genuine selection.
- **Pilot-first discipline** — a short single-seed pilot before committing to
  a full 3-seed-per-group Mann-Whitney replication. Every prior module in
  this project paid for skipping this step at least once.

## What's new here vs. Trial 8 (`eco_evolutionary_cultural_plasticity`)

Only two things changed, both in `predpreygrass_rllib_env.py`:
- New `_current_target_dialect()` (square-wave phase on `current_step`,
  cycling through `n_dialects`, `dialect_season_length_steps` steps per
  phase — same pattern as `base_environment_seasonal`'s
  `_current_season_multiplier`).
- `_dialect_match_bonus` now compares against `_current_target_dialect()`
  instead of `_local_majority_dialect(agent)`.

Everything else — genome structure, `_apply_cultural_learning`,
`_local_majority_dialect` (still used for copying, just not for the bonus),
satiation/sustainability mechanics, reproduction/energy dynamics, the
neutral-drift-control scaffold, `include_culture_in_obs` — is unchanged from
Trial 8.

## Status

Implemented and unit-tested (29 tests: the 27 ported from Trial 8, plus two
new tests for `_current_target_dialect()`'s phase-cycling, plus the two
Trial-8 coordination-bonus tests rewritten to encode the semantic change —
matching the local majority but not the target grants no bonus, and vice
versa — as the regression guard for this module's core behavior change).

**Not yet run.** No PPO training pilot or replication launched yet. Per the
staged discipline every prior module in this family follows: a short
single-seed pilot (a few hundred iterations) should come first, checking
sustainability/coexistence and confirming the target-flip mechanism is
actually visible in the data (`dialect_match_rate` dipping after each flip
and recovering within the phase — see "Predictions" above), before any real
replication. `dialect_season_length_steps` (default 25) is a first guess and
likely needs retuning after that pilot, the same way every prior module's
constants did.
