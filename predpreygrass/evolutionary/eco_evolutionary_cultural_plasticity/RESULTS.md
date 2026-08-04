# Training Analysis — eco_evolutionary_cultural_plasticity

**Status: pilot complete, full neutral-control replication in progress.**
See `README.md` for the trait design (dual inheritance: `plasticity`
genetic, `dialect` cultural) and `predpreygrass/evolutionary/RESULTS.md`'s
Trial 8 entry for the cross-module framing.

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

## 3. Full replication (in progress)

Launched via `run_replication_seeds.sh`: 3 real seeds (42/43/44) + 3
neutral-control seeds (42/43/44), 1000 iterations each, sequential (GPU
sharing risk — see `predpreygrass/non_evolutionary/reward_shaping/README.md`'s
"Concurrent vs. sequential training"). Compares real vs. control
`plasticity` drift via `analyze_replication_seeds.py` once enough seeds
finish — this is the actual criterion-3 (selection-driven genome drift)
test. Results will be added here once the replication completes.
