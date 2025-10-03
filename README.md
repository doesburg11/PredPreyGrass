[![Python 3.11.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.49.0-blue)](https://docs.ray.io/en/latest/rllib/)


# Predator-Prey-Grass
## Evolution in a multi-agent reinforcement learning gridworld

This repo explores the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via reinforcement learning) in ecological systems. We combine **Multi-Agent Reinforcement Learning** (MARL) with **evolutionary dynamics** to explore emergent behaviors in a multi-agent dynamic ecosystem of Predators, Prey, and regenerating Grass. Agents differ by speed, vision, energy metabolism, and decision policies—offering ground for open-ended adaptation. At its core lies a gridworld simulation where agents are not just *trained*—they are *born*, *age*, *reproduce*, *die*, and even *mutate* in a continuously changing environment.

<p align="center">
    <b>Trained Predator-Prey-Grass walls-occlusion environment</b></p>
<p align="center">
    <img align="center" src="./assets/images/gifs/walls_occlusion.gif" width="600" height="500" />
</p>


### Environments:

* **Base environment**: The two-policy base environment. ([implementation](src/predpreygrass/rllib/base_environment), [results](https://humanbehaviorpatterns.org/pred-prey-grass/overview-ppg))

* **Mutating agents**: A four-policy extension of the base environment. ([implementation](src/predpreygrass/rllib/mutating_agents), [results](https://humanbehaviorpatterns.org/pred-prey-grass/marl-ppg/experiments/mutating-agents/))

* **Centralized training**: A single-policy variant of the base environment ([implementation](https://github.com/doesburg11/PredPreyGrass/tree/main/src/predpreygrass/rllib/centralized_training))

* **Walls occlusion**: An extension with walls and occluded vision ([implementation](src/predpreygrass/rllib/walls_occlusion))

* **Selfish gene**: An extension with lineage rewards ([implementation](https://github.com/doesburg11/PredPreyGrass/tree/main/src/predpreygrass/rllib/selfish_gene))

### Experiments:

* Testing the **Red Queen Hypothesis** in the co-evolutionary setting of (non-mutating) predators and prey ([implementation](https://github.com/doesburg11/PredPreyGrass/blob/main/src/predpreygrass/rllib/red_queen/evaluate_red_queen_freeze_type_1_only.py), [results](https://humanbehaviorpatterns.org/pred-prey-grass/red-queen/))

* Testing the **Red Queen Hypothesis** in the co-evolutionary setting of mutating predators and prey ([implementation](src/predpreygrass/rllib/mutating_agents), [results](https://humanbehaviorpatterns.org/pred-prey-grass/marl-ppg/configurations/mutating_agents/#co-evolution-and-the-red-queen-effect))

* Testing **cooperation** by adding lineage rewards instead of direct reproduction rewards ([results](https://github.com/doesburg11/PredPreyGrass/tree/main/src/predpreygrass/rllib/selfish_gene#predpreygrass--selfish-gene-the-search-for-emergent-cooperation))

### Hyperparameter tuning

* Hyperparameter tuning base environment - **Population-Based Training** ([Implementation](src/predpreygrass/rllib/hyper_parameter_tuning/tune_population_based_training.py))


## Installation of the repository

**Editor used:** Visual Studio Code 1.101.0 on Linux Mint 22.0 Cinnamon

1. Clone the repository:
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
   - Choose environment: Conda
   - Choose interpreter: Python 3.11.11 or higher
   - Open a new terminal
   - ```bash
     pip install -e .
     ```
3. Install the additional system dependency for Pygame visualization:
    -   ```bash
        conda install -y -c conda-forge gcc=14.2.0
        ```
## Quick start
Run the pre-trained policy in a Visual Studio Code terminal:

```bash
python ./src/predpreygrass/rllib/base_environment/evaluate_ppo_from_checkpoint_debug.py

```
Or a random policy:
```bash
python ./src/predpreygrass/rllib/base_environment/random_policy.py

```



## References

- [RLlib: Industry-Grade, Scalable Reinforcement Learning](https://docs.ray.io/en/master/rllib/index.html)
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)
