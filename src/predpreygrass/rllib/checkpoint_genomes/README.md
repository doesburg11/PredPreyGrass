**Quickstart**
1. Edit `src/predpreygrass/rllib/checkpoint_genomes/genome_config.json`.
2. Run:
```bash
python src/predpreygrass/rllib/checkpoint_genomes/generation_loop.py
```

**Overview**
`checkpoint_genomes` is a working fork of the sexual reproduction environment that treats RLlib checkpoints as genomes. The core idea is:
- Nature: policy weights stored in `module_state.pkl` inside a checkpoint.
- Nurture: PPO training from those inherited weights.
- Selection: keep the genomes that yield better cooperation metrics.

This folder contains scripts to train, evaluate, breed, and run a full generation loop with minimal repeated typing.

**Concept: What Is a Genome Here**
Each RLlib checkpoint contains per-policy weights in:
`checkpoint_*/learner_group/learner/rl_module/<policy_id>/module_state.pkl`

The child genome is created by mixing two parents' `module_state.pkl` tensors (crossover) plus optional Gaussian noise (mutation). Training then continues from that mixed checkpoint.

**How Crossover + Mutation Works (Detailed)**
This is implemented in `src/predpreygrass/rllib/checkpoint_genomes/make_child_checkpoint.py` and used by `tune_ppo.py` when restoring.

Step-by-step:
1. Copy parent A checkpoint to the child output directory. This preserves metadata and file structure.
2. For each policy under `rl_module/`:
   - Load parent A `module_state.pkl`
   - Load parent B `module_state.pkl`
   - Verify keys match (same network architecture)
3. For each tensor in the state dict:
   - Crossover: `child = alpha * A + (1 - alpha) * B`
   - Mutation: add Gaussian noise with stddev `sigma`
4. Write the mixed tensors back into the child `module_state.pkl`.

When training starts, `tune_ppo.py` restores from the child checkpoint. PPO then continues learning from those mixed weights (nurture on top of inherited nature).

**How Mutation Is Added**
Mutation is applied after crossover in `src/predpreygrass/rllib/checkpoint_genomes/make_child_checkpoint.py`:
- For numpy arrays (weights), it adds Gaussian noise: `N(0, sigma)` element‑wise.
- For scalar numeric values, it adds a single Gaussian sample.
- If `sigma` is `0.0`, no mutation is applied.
- Non‑numeric entries are left unchanged (copied from parent A).

So for each weight tensor:
`child = alpha * A + (1 - alpha) * B + N(0, sigma)`

**Key Files**
- `src/predpreygrass/rllib/checkpoint_genomes/tune_ppo.py`
- `src/predpreygrass/rllib/checkpoint_genomes/run_tune_ppo.py`
- `src/predpreygrass/rllib/checkpoint_genomes/tune_ppo_config.json`
- `src/predpreygrass/rllib/checkpoint_genomes/make_child_checkpoint.py`
- `src/predpreygrass/rllib/checkpoint_genomes/run_make_child_checkpoint.py`
- `src/predpreygrass/rllib/checkpoint_genomes/make_child_checkpoint_config.json`
- `src/predpreygrass/rllib/checkpoint_genomes/generation_loop.py`
- `src/predpreygrass/rllib/checkpoint_genomes/genome_config.json`
- `src/predpreygrass/rllib/checkpoint_genomes/evaluate_ppo_from_checkpoint_multi_runs.py`

**Quick Start (No Arguments)**
1. Edit `src/predpreygrass/rllib/checkpoint_genomes/genome_config.json` to set the seed checkpoint and output folder.
2. Run:
```bash
python src/predpreygrass/rllib/checkpoint_genomes/generation_loop.py
```
3. Inspect results under the output root you configured.

**Config Files**
`src/predpreygrass/rllib/checkpoint_genomes/genome_config.json`
- `seed_checkpoint`: checkpoint directory used to seed generation 0.
- `out_root`: where all generations are written.
- `population_size`: genomes per generation.
- `generations`: number of generations to run.
- `elite`: number of top genomes copied unchanged to next generation.
- `alpha`: crossover mix weight (0.0 = all parent B, 1.0 = all parent A).
- `sigma`: mutation stddev applied to weights.
- `seed`: RNG seed for selection.
- `fitness_key`: dot path in `defection_metrics_aggregate.json` used for selection.
- `train_script`: training entrypoint, defaults to `tune_ppo.py`.
- `eval_script`: evaluation entrypoint, defaults to `evaluate_ppo_from_checkpoint_multi_runs.py`.
- `skip_train`: skip PPO training if true.
- `skip_eval`: skip evaluation if true.

`src/predpreygrass/rllib/checkpoint_genomes/tune_ppo_config.json`
- `restore_checkpoint`: checkpoint to resume from (or null to start fresh).
- `trained_example_dir`: optional output folder to keep results outside default ray_results path.

`src/predpreygrass/rllib/checkpoint_genomes/make_child_checkpoint_config.json`
- `parent_a`: parent A checkpoint directory.
- `parent_b`: parent B checkpoint directory.
- `out_dir`: child checkpoint directory.
- `alpha`, `sigma`, `seed`: crossover and mutation parameters.
- `policies`: optional list of policy ids to restrict mixing.
- `overwrite`: allow overwriting `out_dir`.

**Common Workflows (No Arguments)**
1. Make one child checkpoint:
```bash
python src/predpreygrass/rllib/checkpoint_genomes/run_make_child_checkpoint.py
```
2. Train from a checkpoint:
```bash
python src/predpreygrass/rllib/checkpoint_genomes/run_tune_ppo.py
```
3. Full evolutionary loop:
```bash
python src/predpreygrass/rllib/checkpoint_genomes/generation_loop.py
```

**Where Outputs Go**
- Generation outputs: `src/predpreygrass/rllib/checkpoint_genomes/genomes`
- Per-genome checkpoints: `gen_###/genome_###/checkpoint_000000`
- Evaluation summaries: `eval/runs/.../summary_data/defection_metrics_aggregate.json`

**Generation Loop Note**
The loop now evaluates the final generation as well. You should expect `fitness.json`
and `eval/` outputs for the last `gen_###` directory.

**Fitness Key Examples**
You can select any scalar metric from `defection_metrics_aggregate.json`. Common options:
- `capture_outcomes.coop_capture_rate`
- `capture_outcomes.solo_capture_rate`
- `join_defect.defect_decision_rate`
- `capture_failures.team_capture_failure_rate`

**Notes and Tips**
- If you want all results inside this module, set `trained_example_dir` in `tune_ppo_config.json`.
- If you edit `tune_ppo.py` to change its default ray_results path, do it in `src/predpreygrass/rllib/checkpoint_genomes/tune_ppo.py`.
- Mutation (`sigma`) should be small at first. Large values can destroy learned structure.
- Generation loops can be expensive. Start with small `population_size` and `generations`.

**Compute Expectations**
- PPO training dominates runtime. As a rough rule, one generation costs roughly `population_size` times one training run plus evaluation time.
- Evaluation (multi-run) is usually cheaper than training but still scales with `N_RUNS` in the eval script.
- Start with `population_size=4`, `generations=2`, and `N_RUNS=3` to validate the pipeline.

**Troubleshooting**
- Missing checkpoint errors usually mean the path in a config file is wrong.
- If evaluation writes to unexpected locations, set `trained_example_dir` or edit `genome_config.json`.
- If imports ever point to the old module, ensure you are running files under `src/predpreygrass/rllib/checkpoint_genomes`.
