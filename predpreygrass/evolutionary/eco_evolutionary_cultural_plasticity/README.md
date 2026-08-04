# Cultural Plasticity: Gene-Culture Coevolution (Dual Inheritance)

## Why this module exists

Every prior module in this family (`eco_evolutionary_metabolic_rate`,
`eco_evolutionary_investment`, `eco_evolutionary_cooperation`,
`eco_evolutionary_metabolic_code`) evolves a **single heritable channel**: a
genome trait feeding a shared per-species PPO policy. All four came back null
on selection-driven drift after proper neutral-drift-control replication (see
`predpreygrass/evolutionary/RESULTS.md`, Trials 1-7) — either because the trait
sits on a smooth fitness landscape ordinary selection can already climb
unassisted (Hinton & Nowlan's diagnosis for Trials 1-6), or, for the one
genuinely rugged trait tried (`metabolic_code`'s combinatorial loci, Trial 7),
still null.

This module tries a structurally different mechanism instead of another
variation on "one gene, one shared policy": **dual inheritance / gene-culture
coevolution** (Boyd & Richerson, 1985; Cavalli-Sforza & Feldman, 1981). Two
inheritance channels, not one, evolving at different speeds:

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

## Fitness effect: a coordination game, not a gradient

A catch (predator) or graze (prey) event grants `coordination_bonus_multiplier`
(default 1.5x) on the energy gained *if* the agent's current live dialect
matches its local same-species majority at that moment
(`_dialect_match_bonus`). This is deliberately **not** a smooth function of
any single scalar — whether a given dialect pays off depends entirely on what
neighbors currently do, which itself depends on the history of social
learning in that neighborhood. A dialect that is locally dominant in one
region may be a minority elsewhere. This is the "needle in a haystack" shape
Hinton & Nowlan's own limitation note says a Baldwin effect needs, achieved
through a coordination game instead of a combinatorial lock.

## The reverse leg: policy conditioned on cultural state

`include_culture_in_obs` (default `True`) adds two observation channels: the
agent's own live dialect (normalized) and whether it currently matches the
local majority. This lets the shared PPO policy learn to condition
movement/hunting on rapidly-changing cultural state (e.g., clustering with
same-dialect neighbors) — a plausible route to the "genome/culture shift
feeds back into learned behavior" leg that every prior module left
unconfirmed. Every prior module kept the genome fully invisible to the
policy; this module is a deliberate departure from that convention because
the reverse leg specifically requires the policy to be able to *see* the
state genes have leverage over.

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
   catch/graze events that earned the coordination bonus) — these should
   move even under the neutral control (culture is not genetically
   inherited in the sense being tested), so they're diagnostic of the
   cultural-learning mechanism working at all, not of selection on
   `plasticity`.

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

## Status

Implemented and unit-tested (27 tests covering genome sampling/mutation for
both channels, the cultural-learning update rule and its plasticity-gating,
the coordination-bonus energy-gain hook, neutral-drift-control template
selection, and the metrics builders). A 300-iteration single-seed pilot
(seed=1) confirmed sustainability/coexistence under PPO training and a
genuinely active cultural-learning mechanism (`dialect_match_rate` up to
0.82, far above the chance baseline). The full 3-real + 3-neutral-control-seed
replication (1000 iterations each) is now in progress. See
`predpreygrass/evolutionary/RESULTS.md`'s Trial 8 entry for the cross-module
framing, and `RESULTS.md` in this directory for the full pilot results and
replication run log.
