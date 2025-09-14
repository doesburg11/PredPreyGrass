Structure of the repository (updated)

Top-level (selected):
├── assets/                                 # Images, gifs, icons used in docs & visualizations
├── predpreygrass_env.yml                   # Conda environment specification (pin core deps)
├── pyproject.toml                          # Build + dependency metadata
├── requirements.txt                        # Primary Python dependencies
├── README.md                               # Project overview & usage
├── clean_ray_gpu.sh                        # Helper script to clean Ray temp dirs (GPU runs)
├── output/                                 # Generated experiment outputs (benchmarks, variations)
└── src/
    └── predpreygrass/
        ├── README.txt                    # (This file) Detailed structure snapshot
        ├── pettingzoo/                     # Legacy PettingZoo AEC environment + SB3 training
        │   ├── envs/                       # AEC env variants (deprecated for current RLlib work)
        │   ├── eval/                       # Legacy evaluation scripts (PettingZoo path)
        │   └── train/                      # Legacy SB3 training utilities
        ├── rllib/                          # Current RLlib multi-agent ecosystem (versioned)
        │   ├── archive/                    # Older experimental or shelved scripts
        │   ├── _on_hold_/                  # Temporarily parked ideas / WIP
        │   ├── hyper_parameter_tuning/     # Shared tuning helpers
        │   ├── ppg_2_policies/             # 2-policy (predators vs prey) baseline experiment variant
        │   ├── ppg_4_policies/             # 4-policy mutating agents experiment variant
        │   ├── ppg_visibility/             # Experiment variant exploring vision / visibility dynamics
        │   └── readme.md                   # (If present) Legacy RLlib notes
        └── utils/
            └── renderer.py                 # Shared rendering logic (legacy path)

Versioning Philosophy:
Each environment change is captured in a new version directory (v1_0 → v3_1). Older versions are immutable for reproducibility and comparative experiments. New hypotheses = new folder; never retrofit older ones.

Key Concepts (current v3_1 focus):
* Multi-agent PPO (Ray RLlib) with separate predator/prey/grass roles via policy mapping.
* Energy dynamics: capped storage, transfer & reproduction efficiencies.
* Evolutionary loop: birth, aging, mutation (in mutating variants), death.
* Performance layers: baseline dict implementation + optional array-core vectorized path (movement, decay, engagements).
* Hyperparameter search: Optuna + ASHA + PBT scripts with custom metric (score_pred) and composite stoppers.

Generated Artifacts (under user home results path, not all tracked here):
* Ray Tune results: ~/Dropbox/02_marl_results/predpreygrass_results/ray_results/
* CSV metrics: predator_100_hits.csv, predator_final.csv (auto-appended per trial)
* Videos / gifs: created via renderer utilities (see assets/images/gifs/)

Maintenance Notes:
* When adding a new env parameter → update config_env_train.py & config_env_eval.py in the active version.
* Add a new version folder instead of modifying previous ones.
* Ensure new agent types appear in sample env used for module spec so their policy modules are registered.
* Keep README tables (v3_1/README.md) aligned with any energy/reproduction parameter changes.
* For large-scale runs, prefer specifying RAY_NUM_CPUS to control resource partitioning explicitly.

Legacy vs Current:
* PettingZoo + SB3 paths retained for historical comparison; current experiments center on RLlib (v3_* and ppg_* variants).
* Older rllib versions (v1_0, v2_0, v3_0) remain as frozen baselines for evolutionary or performance regression analysis.

Planned / In-Flight (as referenced during development):
* Further array-core expansion (spawn/reproduction path & autogrow capacity management)
* Enhanced benchmark script (dict vs array-core vs legacy) with standardized warm-ups
* Additional Red Queen scenarios (mutating predator-prey arms races)

