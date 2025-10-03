Structure of the repository (updated)

PredPreyGrass
├── assets/                                 # Images, gifs, icons used in docs & visualizations
├── predpreygrass_env.yml                   # Conda environment specification (pin core deps)
├── pyproject.toml                          # Build + dependency metadata
├── requirements.txt                        # Primary Python dependencies
├── README.md                               # Project overview & usage
└── src/
    └── predpreygrass/
        ├── README.txt                      # (This file) Detailed structure snapshot
        ├── pettingzoo/                     # Legacy PettingZoo AEC environment + SB3 training
        │   ├── envs/                       # AEC env variants (deprecated for current RLlib work)
        │   ├── eval/                       # Legacy evaluation scripts (PettingZoo path)
        │   └── train/                      # Legacy SB3 training utilities
        ├── rllib/                          # Current RLlib multi-agent ecosystem (variants)
        │   ├── _on_hold_/                  # Temporarily parked ideas / WIP
        │   ├── base_environment/           # 2-policy baseline (predators vs prey)
        │   ├── centralized_training/       # One-policy centralized PPO variant
        │   ├── hyper_parameter_tuning/     # Shared tuning helpers
        │   ├── kin_selection/              # Kin selection experiments
        │   ├── mutating_agents/            # 4-policy mutating agents variant
        │   ├── selfish_gene/               # Selfish gene experiment variant
        │   ├── red_queen/                  # Red Queen effect experiments
        │   └── walls_occlusion/            # Occluded vision via walls
        └── utils/
            └── renderer.py                 # Shared rendering logic (legacy path)


Key Concepts development (selected):
* Multi-agent PPO (Ray RLlib) with separate predator/prey/grass roles via policy mapping.
* Energy dynamics and evolutionary loop variants across experiment folders.
* Hyperparameter search using Optuna/ASHA/PBT and custom metrics.

Maintenance Notes (selected):
* Prefer adding new experiment folders over modifying older ones.
* Keep config and README tables aligned when changing environment dynamics.
* Use RAY_NUM_CPUS to control resource partitioning for large runs.

Planned / In-Flight (selected):
* Array-core expansion for performance-critical paths.
* Standardized benchmarks and additional Red Queen scenarios.

