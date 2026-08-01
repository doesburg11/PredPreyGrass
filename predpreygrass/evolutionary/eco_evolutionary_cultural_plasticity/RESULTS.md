# Training Analysis — eco_evolutionary_cultural_plasticity

**Status: implementation + verification only. No PPO training pilot or
replication run has been launched yet.** See `README.md` for the trait
design (dual inheritance: `plasticity` genetic, `dialect` cultural) and
`predpreygrass/evolutionary/RESULTS.md`'s Trial 8 entry for the cross-module
framing and the current state of the follow-up decision on whether/when to
launch a real run.

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

## 2. Not yet done

- **Sustainability/coexistence pilot** under PPO training (criteria 1/2 of
  the project's Darwin/Baldwin goal). The satiation-throttle constants are
  ported unchanged from already-validated modules, so this is expected to
  transfer, but hasn't been confirmed for this specific environment variant
  (the two new observation channels and the coordination-bonus energy
  dynamics are new).
- **Real vs. neutral-control multi-seed replication** on `plasticity` drift
  (criterion 3 — the actual dual-inheritance selection test). Requires the
  pilot above first, per this project's own established discipline (every
  prior module paid for skipping straight to replication at least once).
- **Parameter tuning.** `n_dialects=4`, `culture_range=3`,
  `coordination_bonus_multiplier=1.5`, `plasticity_check_interval=5`, and
  the founder `plasticity_mean=0.1` are reasonable starting defaults, not
  validated against real training data.

## 3. Next step

Launch a short single-seed pilot (a few hundred iterations) to check
sustainability/coexistence before committing to a full replication —
same pilot-first discipline used by every prior module in this family.
