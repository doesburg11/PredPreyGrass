[![Python 3.11.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.46.0-blue)](https://docs.ray.io/en/latest/rllib/)


# Predator-Prey-Grass
## Evolution in a multi-agent reinforcement learning gridworld

We combine **Multi-Agent Reinforcement Learning** (MARL) with **evolutionary dynamics** to explore the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via MARL algorithms). This repo explores emergent behaviors in a multi-agent dynamic ecosystem of Predators, Prey, and regenerating Grass. Agents differ by speed, vision, energy metabolism, and decision policies—offering ground for open-ended adaptation. At its core lies a gridworld simulation where agents are not just *trained*—they are *born*, *age*, *reproduce*, *die*, and even *mutate* in a continuously changing environment.


<div align="center">
  <video src="https://github.com/doesburg11/PredPreyGrass/issues/10#issue-3144686858" autoplay loop muted playsinline style="max-width: 100%; height: auto;">
  </video>
</div>


## Features

* At startup Predator, Prey and Grass are randomly positioned.

* Predators and Prey are independantly (decentralized) trained via their own RLlib policy module.:

  * **Predators** (red)
  * **Prey** (blue)

* Predator and Prey **learn movement strategies** based on their **partial observations**.
* Both expend **energy** as they move around the grid and **replenish energy by eating**:

  * **Prey** eat **Grass** (green).
  * **Predators** eat **Prey** by moving onto the same grid cell.

* **Predator survival conditions**:

  * Preventing starvation (when energy runs out).

* **Prey survival conditions**:

  * Preventing starvation.
  * Preventing being eaten by a Predator.

* **Reproduction conditions**:

  * Both Predators and Prey reproduce **asexually** when their energy exceeds a threshold.
  * New agents are spawned close to their parent.

* Grass agents gradually regenerate at the same spot after being eaten by Prey.

## Environments:

* [Base environnment](src/predpreygrass/rllib/v1_0)

* [Mutating agents](src/predpreygrass/rllib/v2_0)

* [Changing river]()


## Installation of the repository

**Editor used:** Visual Studio Code 1.100.3 on Linux Mint 22.0 Cinnamon

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

## References

- [RLlib: Industry-Grade, Scalable Reinforcement Learning](https://docs.ray.io/en/master/rllib/index.html)
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)


<p align="center">
    <b>The Predator-Prey-Grass base-environment</b></p>
<p align="center">
    <img align="center" src="./assets/images/gifs/rllib_pygame_1000.gif" width="600" height="500" />
</p>
