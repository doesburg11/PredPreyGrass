# Exact Reproduction Checklist for Leibo et al. (2019)

Goal: make this module a strict, auditable reproduction target instead of a Malthusian-inspired adaptation.

Status legend:
- DONE: implemented and validated in this repo.
- OPEN: not implemented yet.
- PARTIAL: implemented but not yet equivalent to paper protocol.

## 1) Ecological update loop

1. DONE: Per-island `phi` is computed and used to update species allocation `mu`.
2. DONE: `mu` is maintained per species over islands and used at reset allocation.
3. DONE: strict mode disables within-episode reproduction (`enable_within_episode_reproduction=False`).
4. PARTIAL: strict mode uses return-based `phi`, but protocol-level parity still needs exact ecological-step batching/aggregation verification against paper procedure.
5. DONE: add a deterministic regression test that checks a hand-computed `phi`/`mu` update for a fixed synthetic episode batch.
6. DONE: exact runs now use a deterministic reset sequence driven by `EXACT_SEED`.

Implementation anchors:
- `predpreygrass_rllib_env.py` (`_compute_phi_from_episode`, `_update_mu_from_phi`, reset allocation)
- `config/config_env.py` (strict/default knobs)

## 2) Learner stack parity

1. DONE: trainer uses APPO and `vtrace=True`.
2. DONE: module spec supports LSTM (`use_lstm`, `max_seq_len`, `lstm_cell_size`).
3. DONE: set and lock explicit V-trace clipping thresholds to 1.0 in trainer config.
4. DONE: KL-loss terms are disabled and locked in exact config.
5. DONE: create a dedicated "exact" training config so paper settings do not drift with exploratory defaults.
6. DONE: cite and lock article-level learner details where published: V-trace clipping 1.0, LSTM unroll 20, LSTM size 64, baseline scaling 0.5, discount 0.99, RMSProp epsilon 0.0001, RMSProp decay 0.99, and batch size 32 trajectories translated to 640 unroll timesteps.
7. PARTIAL: article reports log-uniform entropy and learning-rate ranges, not exact sampled values for each plotted run. The exact config freezes one documented in-range LR and the entropy-range midpoint.

Implementation anchors:
- `tune_ppo_malthusian_rl.py`
- `tune_appo_malthusian_exact.py`
- `config/config_appo_exact.py`
- `config/config_ppo_cpu.py`
- `config/config_ppo_gpu_default.py`
- `utils/networks.py`

## 3) Policy-sharing semantics

1. DONE: one policy per species role (`type_1_predator`, `type_2_predator`, `type_1_prey`, `type_2_prey`) mapped by ID.
2. PARTIAL: policy-sharing semantics are present, but exact species/task mapping to paper experiments still needs explicit experiment templates and locked seeds.
3. DONE: tests verify `policy_mapping_fn` (PPG) and `article_policy_mapping_fn` (article tasks) map all live agent IDs to species-level policies and raise on malformed IDs.

Implementation anchors:
- `tune_ppo_malthusian_rl.py` (`policy_mapping_fn`)

## 4) Environment and protocol parity

1. PARTIAL: hard-island structure exists and can represent archipelago dynamics.
2. DONE: define one canonical paper-protocol env preset in `config/config_paper_protocol.py`.
3. DONE: separate exact protocol config from research/adaptation config.
4. DONE: document every intentional deviation from paper assumptions in a single deviation table.
5. DONE: add runnable text-grounded Clamity and Allelopathy task reconstructions.
6. DONE: add named article-condition presets for Allelopathy heterogeneous dynamic, homogeneous dynamic, fixed-population-32, Clamity dynamic population, Clamity fixed-population-32, and Clamity single-agent baseline.
7. DONE: add solitary evaluation island support for Clamity-style reporting without feeding those solitary returns into the archipelago `phi -> mu` update.
8. DONE: agent facing state (0=N,1=E,2=S,3=W) maintained in both article task environments; 15×15 observation window rotates egocentrically to follow facing direction (Section 2.4).
9. PARTIAL: exact article environment parity remains blocked by unpublished constants in the paper.

Implementation anchors:
- `config/config_env.py`
- `README.md`

## 5) Evaluation parity

1. DONE: add reproduction evaluator for paper-like island metrics and plots.
2. DONE: add multi-seed run harness and summary table generator.
3. DONE: define mapped-protocol acceptance bands plus an explicit article-exact blocked status.
4. DONE: persist run metadata needed for auditability: seed, exact env/trainer configs, citation map, git commit, and config checksum.
5. DONE: evaluator reconstructs Clamity-style mean solitary return and keeps article-task run quality separate from article-exact acceptance.
6. DONE: add full article-condition matrix harness and condition-aware `condition_summary.csv`, `figure2_clamity_summary.csv`, `figure3_allelopathy_summary.csv`, `figure2_clamity_summary.png`, and `figure3_allelopathy_summary.png`.
7. DONE: acceptance report now includes `article_condition_coverage`, requiring every named article condition to have at least the configured minimum number of completed seeds.
8. DONE: run metadata now includes software environment snapshots, git dirty status, and verified config checksums; acceptance fails metadata-integrity checks if those are missing or invalid.

Suggested new files:
- `experiments/exact_protocol.yaml` (or python config file)
- `evaluate_exact_reproduction.py`
- `scripts/run_exact_reproduction_seeds.sh`

## 6) Required tests before claiming exact reproduction

1. DONE: unit test for `phi` computation in strict mode against hand-calculated values.
2. DONE: unit test for `mu` update with fixed logits/eta/entropy term.
3. DONE: integration smoke test that verifies APPO config invariants (`vtrace`, clip thresholds, recurrent settings).
4. DONE: same-seed determinism test verifies `ArticleAllelopathyEnv` produces identical phi/mu/switching-cost episode summaries across runs with the same seed.

## Immediate execution plan (concrete next tasks)

1. DONE: Add `config/config_appo_exact.py` and lock exact learner settings.
2. DONE: Add `tune_appo_malthusian_exact.py` that only reads exact config and rejects incompatible overrides.
3. DONE: Add tests for strict `phi`/`mu` math and APPO invariants.
4. DONE: Add `EXACT_DEVIATIONS.md` and reduce deviations one by one.
5. DONE: Add a multi-seed exact reproduction script and summary CSV.

## Current verdict

As of now: NOT an exact article reproduction.

Reason: core Malthusian mechanism, locked APPO/V-trace learner details, frozen mapped protocol, article-task reconstruction environments, named article-condition presets, Clamity solitary evaluation support, multi-seed and full-matrix harnesses, evaluator, condition/figure-family summaries and plots, matrix coverage checks, metadata-integrity checks, acceptance bands, and citation mapping are in place. The remaining blocker is literal environment parity: the paper does not publish several constants needed to exactly reproduce Clamity and Allelopathy Figure 2/Figure 3 runs. A related official DeepMind Melting Pot Allelopathic Harvest source exists, but it is a later non-identical substrate, not the 2019 two-shrub Malthusian RL task. The official Lab2D repository was also checked and contains the simulator platform, not the article task source. No public 2019 Clamity or two-shrub Allelopathy source/supplement was found in the checked refs.
