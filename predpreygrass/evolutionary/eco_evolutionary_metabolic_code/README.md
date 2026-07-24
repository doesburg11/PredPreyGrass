# Metabolic Code: a Combinatorial Needle-in-Haystack Trait

## Why this module exists

Three prior modules in this family (`eco_evolutionary_metabolic_rate`,
`eco_evolutionary_investment`, `eco_evolutionary_cooperation`) each evolve a
single continuous scalar trait with a smooth interior fitness optimum. All
three showed sustainability/coexistence solved but **no detectable
selection-driven genome drift** beyond a neutral-drift control, across
properly-powered multi-seed replications (see
`predpreygrass/evolutionary/RESULTS.md`, Trials 1-6).

Hinton & Nowlan (1987), "How Learning Can Guide Evolution" — the paper that
formalized the Baldwin effect computationally — offers a specific account for
why: *"the main limitation of the Baldwin effect is that it is only effective
in spaces that would be hard to search without an adaptive process to
restructure the space."* A smooth 1-D scalar is exactly the kind of space
ordinary selection can climb on its own, with nothing for learning to add.
Their own demonstration instead uses a **combinatorial genome with a single
narrow fitness spike** — a "needle in a haystack" that blind mutation and
selection struggle to find unassisted, but that individual lifetime learning
can locate by trial and error, manufacturing a detectable gradient for
evolution to climb (genetic assimilation).

This module implements a trait shaped like that one, instead of another
smooth scalar.

## The genome: CORRECT / WRONG / PLASTIC loci

The heritable trait is `loci`, a length-`L=10` tuple. Each locus is defined
**relative to an implicit fixed target** — no target bit-string is stored
anywhere, since (as in the original paper, once its target is fixed WLOG)
only the relative state matters:

- **CORRECT** — permanently matches the target. No action needed.
- **WRONG** — permanently mismatches the target. An individual carrying even
  one `WRONG` locus can never achieve a full match in its own lifetime, no
  matter how it searches. This is the needle: fixed loci are inherited, not
  searchable.
- **PLASTIC** — unresolved at birth. Searched fresh every step (see below).

Founder distribution (`haystack_founder_probs`, per species): each locus i.i.d.
`CORRECT` 0.2 / `WRONG` 0.3 / `PLASTIC` 0.5. With `L=10`, `E[wrong loci]=3`;
`P(zero wrong loci)` per founder ≈ 2.8% — a genuine needle, but with enough
near-miss individuals (one wrong locus ≈ 12% of founders) to give evolution a
partial-credit slope to climb, which is the actual mechanism the source paper
demonstrates.

Mutation (`haystack_mutation.rate`, default 0.05, per locus per reproduction
event): on trigger, a locus is resampled uniformly among the 3 states — the
same "resample on trigger" pattern `utils/genome.py` already used for the
continuous traits in sibling modules, just categorical instead of Gaussian.

## Per-individual lifetime search — a second, independent learning channel

Every step, every living agent that has **zero `WRONG` loci** and hasn't
already solved this life draws one **fresh joint guess** across *all* its
`PLASTIC` loci simultaneously (each 50/50) and checks whether that guess
completes a full match (`utils/genome.py::attempt_resolve`). This mirrors
Hinton & Nowlan's mechanism precisely — a trial tests the whole combination
at once, not locus-by-locus — so an individual with `k` plastic loci needs on
average `2**k` trials. With the founder distribution above, `k` averages ~5
(~32 average trials), comfortably inside a typical lifetime (prey/predator
survive tens to hundreds of steps). Individuals with any `WRONG` locus are
flagged once at birth (`agent_has_wrong`) and skip the check entirely — cheap,
and a directly interpretable "doomed vs. viable" split for metrics.

This is deliberately **not** implemented through the shared PPO policy. The
cross-module theoretical note (`predpreygrass/evolutionary/RESULTS.md`) flags
a structural gap: in this project PPO is population-level policy
optimization, not individual-lifetime search, which is plausibly part of why
the reverse leg (genome shaping learning) has been hard to detect in the
prior traits. Here, PPO keeps doing exactly what it already does — learning
movement/hunting/foraging behavior — while a separate, cheap, per-individual
stochastic process resolves this one trait, matching the source paper's own
mechanism directly instead of approximating it through PPO.

## Fitness effect

Once an agent fully solves (zero `WRONG` loci, successful plastic-locus
guess), a persistent flag multiplies its energy gain (`haystack_bonus_multiplier`,
default 1.5x) at the two existing intake sites for the rest of its life:
predator catch energy and prey grass-eating energy. This reuses the same
"metabolic efficiency" fitness channel `eco_evolutionary_metabolic_rate` used
(so results are directly comparable to that module's null result) but
replaces its smooth `food^alpha` gradient with an all-or-nothing combinatorial
gate. No new energy-cost parameter is introduced — the "cost" of unresolved
plastic loci is the implicit opportunity cost of a delayed bonus, exactly as
in the source paper (which has no explicit per-trial cost either).

## What's deliberately unchanged

- **Offspring investment is a fixed, non-heritable constant** here
  (`offspring_investment_fraction`, default 0.35, applied uniformly) — one
  heritable trait per module, matching every prior module's isolation
  discipline.
- **Sustainability mechanism** (predator satiation cooldown + per-catch energy
  cap) is ported unchanged from `eco_evolutionary_investment` /
  `eco_evolutionary_metabolic_rate` — an already-validated, orthogonal concern
  (criteria 1/2 of the project goal), not something this trial re-tests.
- **The genome stays invisible to the policy's observation space**, same as
  every prior trait. Behavior stays PPO-learned, not genome-conditioned.

## Predictions and what to watch in training

Two predictions, richer than any prior trait's single-scalar drift test:

1. **Primary (Darwin-signal) test:** mean `WRONG`-loci count in the live
   population should fall below the founder expectation (3.0) under real
   selection, and — critically — should fall **further than the neutral-drift
   control** (`genome_neutral_drift_control=True`, same mechanism ported from
   `eco_evolutionary_investment`'s R7). This is the headline comparison in
   `analyze_replication_seeds.py` (`live_haystack/{species}_mean_wrong_loci`).
2. **Secondary (genetic assimilation) signal:** among near-solved lineages,
   mean `PLASTIC`-loci count should also trend down relative to control over
   generations — evolution converting "searched every life" into "known at
   birth." Tracked (`{species}_mean_plastic_loci`,
   `{species}_fraction_solved`, `{species}_mean_steps_to_solve`) but not the
   headline test.

Metrics are logged both per-step (`live_haystack/*`, currently-alive
population only — `predpreygrass_rllib_env.py::_build_live_haystack_metrics`)
and per-episode (`eco_evolution/*`, live + completed agents this episode —
`_build_episode_training_metrics`), mirroring the dual-metric pattern used by
every prior module in this family.

## Methodology this module inherits

Directly ported from `eco_evolutionary_investment`'s R4-R7 (the methodology
established there and validated for `eco_evolutionary_metabolic_rate`):
- **Satiation-throttle sustainability** — validated, not re-tested here.
- **`genome_neutral_drift_control`** — severs genome inheritance from
  reproductive success (offspring template becomes a uniformly random
  currently-alive same-species agent instead of the true parent); population/
  energy dynamics unchanged. Isolates mutation + finite-population sampling
  noise from genuine selection.
- **Pilot-first discipline** — a short single-seed pilot (a few hundred
  iterations) before committing to a full 3-seed-per-group Mann-Whitney
  replication. Do not skip straight to a full replication; every prior trait
  in this project paid for skipping this step at least once.

## Status

Implemented and smoke-tested (see `RESULTS.md`); no pilot or replication run
launched yet. Launching a real pilot is a separate decision, not automatic
once implementation lands — see `predpreygrass/evolutionary/RESULTS.md`'s
Trial 7 entry for the current state of that decision.
