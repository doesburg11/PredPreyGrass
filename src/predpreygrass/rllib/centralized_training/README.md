# PredPreyGrass — Centralized Training (single shared policy)

This folder contains a centralized training variant of PredPreyGrass using Ray RLlib’s new API stack and PPO. All agents (predators and prey) share a single policy (“shared_policy”), trained jointly on the same observations/action space.

## What’s here
- Environment: `predpreygrass_rllib_env.py` (gridworld with predators, prey, grass; walls/occlusion supported by configs)
- Training script: `tune_ppo_centralized_training.py` (PPO + RLlib Tune; new RLModule API)
- Evaluation helpers:
  - `evaluate_ppo_from_checkpoint_debug.py`
  - `evaluate_ppo_from_checkpoint_default.py`
  - `evaluate_ppo_from_checkpoint_multi_runs.py`
- Configs: `config/`
  - Env presets (e.g., `config_env_zigzag_walls.py`, `config_env_train.py`, `config_env_eval.py`, …)
  - PPO presets for CPU/GPU and PBT/search variants (e.g., `config_ppo_cpu.py`, `config_ppo_gpu_default.py`)
- Utilities: `utils/` (env spec, networks, renderers, callback)
- Tests: `test/` (sanity checks for walls/occlusion)

## How it works (design sketch)
- Centralized policy: `policy_mapping_fn` always returns `"shared_policy"` so every agent uses the same policy parameters.
- Spaces: A sample env instance is created, and the first agent’s observation/action spaces are used to define the shared policy’s module spec.
- Network: Uses `DefaultPPOTorchRLModule` with a simple Conv stack (filters: 16, 32, 64; 3×3; ReLU).
- RLlib (new API):
  - `PPOConfig().environment(...).framework('torch').multi_agent(...).training(...).rl_module(...).learners(...).env_runners(...).resources(...).callbacks(...)`
  - Checkpoints produced every 10 iterations (keep up to 100).
- Callback: `EpisodeReturn` emits episode returns (useful for quick sanity in TensorBoard).

## Running training
By default, the script writes results under `~/Dropbox/02_marl_results/predpreygrass_results/ray_results/` with a timestamped experiment folder (`PPO_CENTRALIZED_TRAINING_<timestamp>`). It also saves a `run_config.json` snapshot (env + PPO).

Train with PPO (single shared policy):
```bash
python src/predpreygrass/rllib/centralized_training/tune_ppo_centralized_training.py
```
Notes:
- The environment preset imported by default is `config/config_env_zigzag_walls.py` (see the import at the top of the script). Switch to another env preset (e.g., `config_env_train.py`) by editing that import.
- PPO preset selection is done in `get_config_ppo()` based on `os.cpu_count()`.
  - If `os.cpu_count() == 32`: uses `config_ppo_gpu_default.py`
  - If `os.cpu_count() == 8`: uses `config_ppo_cpu.py`
  - Otherwise: falls back to `config_ppo_cpu.py`
  If you want a specific preset, simply change the imports inside `get_config_ppo()`.

## Evaluating checkpoints
You can load a specific checkpoint and render/evaluate:
```bash
python src/predpreygrass/rllib/centralized_training/evaluate_ppo_from_checkpoint_debug.py
```
or the default variant:
```bash
python src/predpreygrass/rllib/centralized_training/evaluate_ppo_from_checkpoint_default.py
```
For sweeping multiple runs:
```bash
python src/predpreygrass/rllib/centralized_training/evaluate_ppo_from_checkpoint_multi_runs.py
```
Depending on the script, you may need to point it at the checkpoint folder under the experiment directory created by training.

## Key configuration knobs
- Env (see `config/`): grid size, walls/occlusion, vision radius, energy/reproduction knobs, episode length (`max_steps`).
- PPO (see `config_ppo_*.py`):
  - `train_batch_size_per_learner`, `minibatch_size`, `num_epochs`, `gamma`, `lr`, `lambda_`, `entropy_coeff`, `vf_loss_coeff`, `clip_param`, `kl_coeff`, `kl_target`.
  - Runners/resources: `num_env_runners`, `num_envs_per_env_runner`, `rollout_fragment_length`, `num_cpus_per_env_runner`, `num_learners`, `num_gpus_per_learner`, `num_cpus_for_main_process`.

## Artifacts and logging
- Ray/Tune results
- Saved config snapshot
- Checkpoints: Tune-managed; by default every 10 iterations and at the end.
- Callback: episode returns are logged for visualization.

## Tips & troubleshooting
- Observation shape mismatches when evaluating usually come from changed env presets. Ensure evaluation uses the same env settings as training.
- On smaller machines, start with CPU presets and/or reduce `num_env_runners`, `num_envs_per_env_runner`, and batch sizes.
- GPU usage is controlled by the selected PPO preset (`num_gpus_per_learner`). Ensure your CUDA environment is set up if you toggle this.
- If you want role-specific policies (multi-policy training), switch from centralized mapping (`"shared_policy"`) to distinct policy IDs per role and update the `policies` dict and module spec accordingly.

## Folder map (quick reference)
- `tune_ppo_centralized_training.py` — main centralized PPO trainer
- `predpreygrass_rllib_env.py` — environment implementation for this variant
- `config/` — env and PPO presets (CPU/GPU/PBT)
- `utils/` — helpers (networks, renderers, callback)
- `evaluate_ppo_*.py` — checkpoint loaders and evaluators
