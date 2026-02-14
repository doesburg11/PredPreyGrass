# Reproducibility for stag_hunt_forward_view

This module is set up to make **each Ray experiment folder self‑contained** so you can reproduce training and evaluation later, even if the live repo changes.

## What gets captured per experiment

Every training run creates a new experiment directory under:

```
.../ray_results/<EXPERIMENT_NAME>/
```

Inside it, you’ll find:

```
REPRODUCE_CODE/
  predpreygrass/rllib/<module_name>/   # snapshot of this module only
  assets/images/icons/                 # icon assets used by pygame renderers
  CONFIG/
    config_env.json
    config_ppo.json
    run_config.json
  pip_freeze_train.txt

checkpoint_000xyz/
  ...

run_config.json
```

If you run evaluations with `SAVE_EVAL_RESULTS=True`, each eval output folder also contains:

```
checkpoint_000xyz/eval_*/REPRODUCE_CODE/
  predpreygrass/rllib/<module_name>/
  assets/images/icons/
  CONFIG/
    config_env.json
    run_config.json
  pip_freeze_eval.txt
```

### What is included
- **Code snapshot** of only `predpreygrass/rllib/<module_name>` (no other modules).
- **Icons** from `assets/images/icons`.
- **Resolved config JSONs** for the run.
- **pip freeze** of the Python environment.

### What is excluded
- `ray_results/`, `trained_examples/`, `__pycache__/`, and `*.pyc`.
- External system dependencies (CUDA, OS libraries, GPU driver versions, etc.).

## How the snapshot is used

The training and evaluation scripts auto‑detect when they are run from a snapshot and add `REPRODUCE_CODE` to `sys.path`. This ensures imports resolve to the snapshot, not the live repo.

Icon loading also prefers `REPRODUCE_CODE/assets/images/icons` when present.

## Reproducing a training run

Run the training script from the snapshot copy:

```bash
python /path/to/EXPERIMENT/REPRODUCE_CODE/predpreygrass/rllib/stag_hunt_forward_view_nature_nurture/tune_ppo.py
```

To keep new outputs inside the same experiment folder, set:

```bash
export TRAINED_EXAMPLE_DIR=/path/to/EXPERIMENT
```

This will write new results to:

```
EXPERIMENT/ray_results/
```

## Reproducing an evaluation run

Run the eval script from the snapshot copy:

```bash
python /path/to/EXPERIMENT/REPRODUCE_CODE/predpreygrass/rllib/stag_hunt_forward_view_nature_nurture/evaluate_ppo_from_checkpoint_debug.py
```

or

```bash
python /path/to/EXPERIMENT/REPRODUCE_CODE/predpreygrass/rllib/stag_hunt_forward_view_nature_nurture/evaluate_ppo_from_checkpoint_multi_runs.py
```

If you want the eval to record outputs back into the experiment, set:

```bash
export TRAINED_EXAMPLE_DIR=/path/to/EXPERIMENT
```

## Notes on determinism

This setup captures the **exact code + configs + Python packages** used at the time of the run. Full determinism still depends on:
- GPU/CPU driver versions
- CUDA/cuDNN versions
- OS libraries
- Non‑deterministic ops

If you need maximum reproducibility, consider capturing system info alongside `pip_freeze` (Python version, CUDA version, etc.).
