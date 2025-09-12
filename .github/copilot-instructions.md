## PredPreyGrass – AI Assistant Working Rules

Purpose: Multi-agent evolutionary RL (predators, prey, grass) using Ray RLlib PPO across versioned experiment directories (`v1_0`, `v2_0`, `v3_0`, `v3_1`). Focus: reproducible experiment scripts, configurable environment energy/reproduction dynamics, and hyperparameter search.

### Architecture & Key Paths
- Core env: `src/predpreygrass/rllib/v3_1/predpreygrass_rllib_env.py` (gridworld + energy + reproduction logic). Earlier versions kept for experiment lineage.
- Config layers: `config_env_*.py` (environment physics & evolutionary parameters), `config_ppo_*.py` (PPO training defaults for cpu/gpu/search/pbt variants). Selection sometimes conditional on CPU count.
- Networks / multi-agent module spec: `utils/networks.py` (`build_multi_module_spec`) builds RLlib `MultiAgentRLModuleSpec` from per-policy spaces auto-derived from a sample env.
- Training scripts:
  - Single-run PPO: `train_ppo_multiagentenv.py`
  - Hyperparameter search (grid / Optuna / ASHA): `tune_ppo_multiagentenv*.py` / `*_search*.py`
  - PBT experiments: `tune_ppo_predpreygrass_pbt_*.py`
  - Evaluation & Red Queen tests: e.g. `evaluate_red_queen_freeze_type_1_only.py`
- Version directories separate hypothesis phases; don’t “upgrade” old versions—add a new `vX_Y` folder.

### Environment Dynamics Highlights
- Energy & reproduction constraints introduced in later versions (see `v3_1/README.md`): capped gains, max energy storage, reproduction chance, cooldown, transfer efficiencies (`energy_transfer_efficiency`, `reproduction_energy_efficiency`).
- Observation spaces are per-agent; policies named by `policy_mapping_fn` pattern: `type_{species}_{role}` extracted from agent IDs.

### PPO Config Pattern (new API stack)
Typical build sequence (see `tune_ppo_multiagentenv_search_3.py`):
1. Derive spaces from a temporary env instance.
2. Build multi-module spec via `build_multi_module_spec`.
3. `PPOConfig().environment(...).framework('torch').multi_agent(...).training(...).rl_module(...).learners(...).env_runners(...).resources(...).callbacks(PredatorScore)`.
4. Copy config for search (`search_space = ppo_config.copy(copy_frozen=False)`) then override sampled hyperparams (e.g. `lr`, `num_epochs`).

### Custom Metrics & Stopping
- Callback `PredatorScore` injects `score_pred` (scaled predator return) and writes one-shot CSV (`predator_100_hits.csv`) when threshold reached.
- Composite stopper wrapper `ReasonedStopper` + specific stoppers (`TrialPlateauStopper`, `DropStopper`, `MaximumIterationStopper`); final reason persisted by `FinalMetricsLogger` to `predator_final.csv` immediately at trial end.

### Hyperparameter / Resource Strategy
- Resource auto-derivation: recent scripts infer CPUs via `os.cpu_count()` (override with `RAY_NUM_CPUS`) to size env runners & concurrency. Original 32‑CPU layout: 3 env runners ×2 CPUs + learner + driver (≈8 CPUs/trial).
- Search uses Optuna (`OptunaSearch`) + ASHA (`ASHAScheduler`) with metric `score_pred` (mode max). Trials: `num_epochs` (qRandInt) and `lr` (loguniform) typically varied.

### Logging & Artifacts
- Ray results root: `~/Dropbox/02_marl_results/predpreygrass_results/ray_results/` with experiment subfolder `PPO_<timestamp>`.
- Per-experiment JSON of used configs: `run_config.json` (env + PPO defaults snapshot).
- CSV summaries: `predator_100_hits.csv`, `predator_final.csv` (append-only, header auto-added).
- Live terminal logging pattern (documented in `RUN_LIVE_LOGS.md`): unbuffered Python + `tee` → `logs/last_run_tune.log`.

### Conventions & Gotchas
- Do NOT delete old version folders—add new; cross-version comparisons rely on stable historical code.
- Policy IDs derived from agent_id structure `type_<n>_<role>`; keep naming consistent in env or mapping will break existing modules.
- When adding new environment parameters: update both `config_env_train.py` and `config_env_eval.py`; keep README tables in sync.
- Multi-agent module spec: ensure new policies appear by creating at least one agent of that type in the sample env used to collect spaces.
- New Ray API warnings are expected; suppress only if migrating back intentionally.

### Adding a New Experiment Variant (Example Workflow)
1. Copy latest version folder to `v3_2/` (keep `__init__.py` and `utils/`).
2. Adjust or add `config_ppo_*.py` with new hyperparams.
3. Create `tune_ppo_multiagentenv_search_<tag>.py` cloning pattern from `*_search_3.py`.
4. If adding metrics: subclass `DefaultCallbacks`, emit scalar on `on_train_result`, reference via `.callbacks(...)`.
5. Run with: `python -u src/predpreygrass/rllib/v3_2/tune_ppo_multiagentenv_search_<tag>.py 2>&1 | tee logs/last_run_tune.log`.

### External Dependencies
- Ray RLlib/Tune (>=2.49.0 per badge), PyTorch backend, Optuna for search, Pygame (visualization), Conda environment pinned in `predpreygrass_env.yml` / `pyproject.toml` / `requirements.txt`.

### AI Assistant DO / AVOID
DO: Reuse existing config patterns; maintain CSV schema; respect version isolation; update README tables when changing energy/reproduction logic.
AVOID: Refactoring old version directories; introducing breaking agent_id formats; silently changing default hyperparameters without recording in new config file.

### Quick Reference Commands
Create env & install (editable): `pip install -e .`
Run evaluation (example): `python src/predpreygrass/rllib/v1_0/evaluate_ppo_from_checkpoint_debug.py`
Launch search (latest): `python -u src/predpreygrass/rllib/v3_1/tune_ppo_multiagentenv_search_3.py`.

---
Feedback: Indicate any missing workflow (tests? data export? checkpoint replay) and this guide will be updated.