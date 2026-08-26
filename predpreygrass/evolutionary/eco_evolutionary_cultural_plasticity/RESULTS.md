# Training Analysis — eco_evolutionary_cultural_plasticity

**Status: stopped early, after all 3 real seeds but before any control
seed.** See §4 below for why. See `README.md` for the trait design (dual
inheritance: `plasticity` genetic, `dialect` cultural) and
`predpreygrass/evolutionary/RESULTS.md`'s Trial 8 entry for the
cross-module framing.

---

## 1. What's been done so far

- **Implementation.** `predpreygrass_rllib_env.py`, `utils/genome.py`, and
  the config pair (`config_env_eco_evolutionary.py` /
  `config_env_eco_evolutionary_neutral_control.py`) built by adapting
  `eco_evolutionary_metabolic_code`'s scaffold: same reproduction/energy/
  satiation-throttle mechanics, new dual-inheritance genome
  (`plasticity` + `dialect`), new `_apply_cultural_learning` /
  `_local_majority_dialect` / `_dialect_match_bonus` mechanism, two new
  observation channels (`include_culture_in_obs`), and `live_culture/*` /
  `eco_evolution/*` metrics (`_build_live_culture_metrics`,
  `_build_episode_training_metrics`).
- **Unit tests.** 27 tests in `tests/test_eco_evolutionary_validation.py`,
  all passing: genome founder sampling and bounds for both the continuous
  (`plasticity`) and categorical (`dialect`) fields; zero- and full-rate
  mutation behavior for both fields; the cultural-learning update rule
  (check-interval gating, plasticity-gated adoption, zero-plasticity never
  adopts); `_local_majority_dialect` correctness (excluding the focal
  agent's own vote); the coordination-bonus energy-gain hook on both match
  and mismatch; `genome_neutral_drift_control` template selection; the
  RLlib multi-agent contract tests (termination/truncation/observation
  bookkeeping) ported unchanged from `eco_evolutionary_metabolic_code`;
  dialect-entropy and the dependency-free Spearman helper (including the
  tie-averaging fix needed because one of its two inputs, reproduced-or-not,
  is binary and ties heavily).
- **Smoke runs.** 300 steps under a uniform-random policy (real config) and
  100 steps under the neutral-drift-control config, both to completion with
  no errors, plausible-looking `plasticity`/`dialect_entropy`/
  `dialect_match_rate` metrics at episode end. This is not a training run —
  no learning occurred — it only exercises the full step/reproduction/
  cultural-learning/metrics code path at realistic population scale.

## 2. Pilot results (seed=1, 300 iterations, real config)

Single-seed pilot run to check criteria 1/2 (sustainability/coexistence)
and confirm the cultural-learning mechanism is genuinely active before
committing to a full replication. Final-iteration metrics:

- **Sustainability/coexistence: confirmed.** `episode_len_mean` reached
  202 (max 326, min 131) by iteration 300 — predator, prey, and grass all
  coexisted without population collapse under PPO training. The
  satiation-throttle constants ported from `eco_evolutionary_metabolic_code`
  transferred cleanly; the two new observation channels and the
  coordination-bonus energy dynamics did not destabilize the ecology.
- **Cultural-learning mechanism: confirmed genuinely active.**
  `dialect_match_rate` reached 0.688 (predator) / 0.823 (prey), both far
  above the chance baseline for `n_dialects=4` (0.25) — agents are
  actually converging on shared local dialects via the
  `_local_majority_dialect` / plasticity-gated adoption mechanism, not
  just tracking founder noise.
- **`plasticity` drift: flat, as expected at this scale.** `plasticity_mean`
  sat at 0.092 (predator) / 0.084 (prey) against a founder mean of 0.1 —
  no meaningful movement yet. This is not a red flag; 300 iterations is a
  sustainability check, not a selection-detection window (every prior
  module in this family needed the full 1000-iteration, multi-seed
  replication to see any directional signal at all, if one existed).
  `plasticity_repro_spearman` was likewise near zero (0.019 predator,
  -0.011 prey), consistent with "no signal yet" rather than "no signal
  possible."

**Conclusion: cleared to proceed to the full replication** — both
pilot-stage questions (does it survive PPO training, is the cultural
mechanism real) came back positive.

## 3. Full replication — real seeds (completed) before stopping

Launched via `run_replication_seeds.sh`: 3 real seeds (42/43/44) + 3
neutral-control seeds (42/43/44), 1000 iterations each, sequential (GPU
sharing risk — see `predpreygrass/non_evolutionary/reward_shaping/README.md`'s
"Concurrent vs. sequential training"). All 3 **real** seeds completed
1000/1000 iterations before the run was stopped (see §4); no control seed
was run. Final-iteration metrics, real seeds only (founder
`plasticity_mean = 0.1` for both species):

| seed | predator `plasticity_mean` | prey `plasticity_mean` | predator repro-spearman | prey repro-spearman | `dialect_match_rate` (pred / prey) | `episode_len_mean` |
|---|---|---|---|---|---|---|
| 42 | 0.098 | 0.089 | -0.026 | +0.077 | 0.737 / 0.782 | 253.3 |
| 43 | 0.079 | 0.115 | -0.0001 | -0.024 | 0.702 / 0.847 | 182.8 |
| 44 | 0.101 | 0.096 | +0.027 | +0.002 | 0.777 / 0.858 | 261.5 |

Sustainability held in all 3 runs (no population collapse; `episode_len_mean`
stayed in a healthy 183–262 range) and the cultural-learning mechanism stayed
robustly active in all 3 (`dialect_match_rate` well above the 0.25
chance baseline for `n_dialects=4`, both species, every seed). But
`plasticity_mean` stayed within roughly one founder-std of the 0.1 starting
value in every seed, with no consistent direction across seeds, and
`plasticity_repro_spearman` (individual-level correlation between an
agent's plasticity and its own reproductive success) sat at essentially
zero in every seed, both species. No control-seed data was collected, so
this is descriptive only — not a Mann-Whitney-confirmed null — but the
pattern is the same "flat, no fitness correlation" shape seen in the 7
single-channel trait trials that preceded this one.

**Correction (2026-08-26):** a control seed 42 run does exist on disk
(`PPO_ECO_EVOLUTION_CULTURAL_PLASTICITY_NEUTRAL_CONTROL_SEED42_2026-08-04_13-17-06`,
367/1000 iterations, presumably interrupted when the effort was stopped) —
the "no control seed was run" framing above is not quite accurate, though it
doesn't change the substance: a single partial control seed still isn't a
Mann-Whitney-confirmed comparison.

**Addendum (2026-08-26): Hunt (2006) model-fit corroboration.** Checked with
`predpreygrass/evolutionary/model_selection.py`'s AICc model selection
(Stasis/URW/GRW) against every available seed's own `plasticity_mean`
trajectory — the two pilot seeds (1: 300 gens, 2: 150 gens), all three real
seeds (42/43/44: 1000 gens each), and the partial control seed above (42:
367 gens). **URW wins in every case**, `mstep` near zero (~1e-4 to 1e-5,
randomly signed) throughout — consistent with, and independent corroboration
of, the "flat, no fitness correlation" read above. Run via
`analyze_replication_seeds.py`'s Hunt-fit section.

## 4. Why this was stopped

Stopped by explicit user decision after reviewing the 3 real-seed results
above, before launching any control seed. Rationale: Trial 8 was designed
specifically to route around the failure mode diagnosed after Trial 7 —
add a second, frequency-dependent, coordination-game inheritance channel
instead of another single smooth scalar trait, per the Hinton & Nowlan
theoretical note in `predpreygrass/evolutionary/RESULTS.md`. The real-seed
data shows the same flat-drift, zero-fitness-correlation pattern as every
prior trial anyway, despite the design change — so continuing to the full
neutral-control replication (3 more ~7h runs, ~21h of compute) was judged
not worth it before seeing whether the mechanism-level fix actually
changes the outcome. The remaining unknown is technically still open (does
this flat-looking real-seed drift exceed an even-flatter control-noise
floor?), but the descriptive signal was judged too weak to justify the
compute for confirming what looks like an eighth null result.

**If a future session picks this up:** the cheapest next check is probably
not another full 6-seed replication. Consider either (a) a single
real-vs-control seed pair (not 3+3) as a fast, cheap directional check
before committing to the full replication again, or (b) treating this as
sufficient evidence (alongside Trials 1–7) that a single shared per-species
PPO policy on this environment may not produce a detectable Baldwin/Darwin
signal for *any* smooth or frequency-dependent trait design tried so far,
and pivoting the search itself — e.g. per-agent (not per-species-shared)
policies, or a trait with a harder, more obligate fitness gate in the
style of `eco_evolutionary_nuptial_gift`'s design intent (though that
module's own real-run results were similarly discouraging — see its
README).

**A third, literature-grounded candidate: environmental non-stationarity.**
This module's specific trait — a genetically-evolving *propensity to
culturally learn* (`plasticity`) — has a real theoretical precedent outside
this codebase: **Rogers' Paradox** (Rogers, 1988). Rogers modeled a gene
controlling social-vs-individual learning and found that at evolutionary
equilibrium in a static environment, the social-learning gene shows **no
fitness advantage** over individual learning — social learning only pays
off when copying a behavior tuned to a *different* environment than the
copier is currently in, and that advantage dilutes to zero as social
learners come to dominate the population. Follow-up work resolving the
paradox (Boyd & Richerson, 1995; Enquist, Eriksson & Ghirlanda, 2007) shows
the gene *does* acquire a detectable, directional selection signal once the
environment is made **changing/heterogeneous**, which makes individual
learning costlier and unreliable enough for conditional/conformist social
learning to become a real fitness advantage.

PredPreyGrass's environment is static (no seasonal or regime shift in
grass-growth, energy dynamics, etc.) — exactly Rogers' null condition, not
a trait-design flaw. If this line is revisited, adding some form of
environmental non-stationarity (e.g., a periodically-shifting grass-growth
rate or energy-cost regime) is a theoretically-motivated candidate for
actually producing a `plasticity` selection signal, rather than trying a
ninth static-environment trait variant.
