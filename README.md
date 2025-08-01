[![Python 3.11.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.48.0-blue)](https://docs.ray.io/en/latest/rllib/)


# Predator-Prey-Grass
## Evolution in a multi-agent reinforcement learning gridworld

This repo explores the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via reinforcement learning) in ecological systems. We combine **Multi-Agent Reinforcement Learning** (MARL) with **evolutionary dynamics** to explore emergent behaviors in a multi-agent dynamic ecosystem of Predators, Prey, and regenerating Grass. Agents differ by speed, vision, energy metabolism, and decision policies—offering ground for open-ended adaptation. At its core lies a gridworld simulation where agents are not just *trained*—they are *born*, *age*, *reproduce*, *die*, and even *mutate* in a continuously changing environment.

<p align="center">
    <b>The Predator-Prey-Grass base-environment</b></p>
<p align="center">
    <img align="center" src="./assets/images/gifs/rllib_pygame_1000.gif" width="600" height="500" />
</p>

## Features [base environment](/src/predpreygrass/rllib/v1_0/)

* At startup Predators, Prey and Grass are randomly positioned on the gridworld.

* Predators and Prey are independently (decentralized) trained via their respective [RLlib policy module](https://docs.ray.io/en/master/rllib/rl-modules.html).:

  * **Predators** (red)
  * **Prey** (blue)

* **Energy-Based Life Cycle**: Movement, hunting, and grazing consume energy—agents must act to balance survival, reproduction, and exploration.

  * Predators and Prey **learn movement strategies** based on their **partial observations**.
  * Both expend **energy** as they move around the grid and **replenish energy by eating**:

    * **Prey** eat **Grass** (green) by moving onto a grass-occupied cell.
    * **Predators** eat **Prey** by moving onto the same grid cell.

  * **Survival conditions**:

    * Both Predators and Prey must act to prevent starvation (when energy runs out).
    * Prey must act to prevent being eaten by a Predator

  * **Reproduction conditions**:

      * Both Predators and Prey reproduce **asexually** when their energy exceeds a threshold.
      * New agents are spawned near their parent.
- **Sparse rewards**: agents only receive a reward when reproducing in the base configuration. However, this can be expanded with other rewards in the [environment configuration](src/predpreygrass/rllib/v1_0/config_env.py). The sparse rewards configuration is to show that the ecological system is able to sustain with this minimalistic optimized incentive for both Predators and Prey.

* Grass gradually regenerates at the same spot after being eaten by Prey. Grass, as a non-learning agent, is being regarded by the model as part of the environment, not as an actor.

## Environments:

* [Base enviornment](src/predpreygrass/rllib/v1_0)

* [Mutating agents](src/predpreygrass/rllib/v2_0)

* Changing river (adding water resource;under development)


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
Run the pre trained model in a Visual Studio Code terminal:

```bash
python ./src/predpreygrass/rllib/v1_0/evaluate_ppo_from_checkpoint_debug.py

```



## References

- [RLlib: Industry-Grade, Scalable Reinforcement Learning](https://docs.ray.io/en/master/rllib/index.html)
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)
