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
        ├── README.txt                      # (This file) Detailed structure snapshot
        ├── pettingzoo/                     # Legacy PettingZoo AEC environment + SB3 training
        │   ├── envs/                       # AEC env variants (deprecated for current RLlib work)
        │   ├── eval/                       # Legacy evaluation scripts (PettingZoo path)
        │   └── train/                      # Legacy SB3 training utilities
        ├── rllib/                          # Current RLlib multi-agent ecosystem (versioned)
        │   ├── _on_hold_/                  # Temporarily parked ideas / WIP
        │   ├── base_evironment/            # 2-policy (predators vs prey) baseline experiment variant
        │   ├── mutating_agents/            # 4-policy mutating agents experiment variant [generalization base_evironment]
        │   ├── walls_occlusion/            # added occluded vision by adding walls in the environment [generalization mutating_agents]
        │   ├── red_queen/                  # exploring the red queen effect in 2-policy baseline experiment
        │   ├── hyper_parameter_tuning/     # Shared tuning helpers
        │   └── readme.md                   # (If present) Legacy RLlib notes
        └── utils/
            └── renderer.py                 # Shared rendering logic (legacy path)


Key Concepts development:
* Multi-agent PPO (Ray RLlib) with separate predator/prey/grass roles via policy mapping.
* Energy dynamics: capped storage, transfer & reproduction efficiencies.
* Evolutionary loop: birth, aging, mutation (in mutating variants), death.
* Performance layers: baseline dict implementation + optional array-core vectorized path (movement, decay, engagements).
* Hyperparameter search: Optuna + ASHA + PBT scripts with custom metric (score_pred) and composite stoppers.

Maintenance Notes:
* When adding a new env parameter → update config_env_train.py & config_env_eval.py in the active version.
* Add a new version folder instead of modifying previous ones.
* Ensure new agent types appear in sample env used for module spec so their policy modules are registered.
* Keep README tables aligned with any energy/reproduction parameter changes.
* For large-scale runs, prefer specifying RAY_NUM_CPUS to control resource partitioning explicitly.

Legacy vs Current:
* PettingZoo + SB3 paths retained for historical comparison; current experiments center on RLlib (v3_* and ppg_* variants).
* Older rllib versions remain as frozen baselines for evolutionary or performance regression analysis.

Planned / In-Flight (as referenced during development):
* Further array-core expansion (spawn/reproduction path & autogrow capacity management)
* Enhanced benchmark script (dict vs array-core vs legacy) with standardized warm-ups
* Additional Red Queen scenarios (mutating predator-prey arms races)

