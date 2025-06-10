[![Python 3.11.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.46.0-blue)](https://docs.ray.io/en/latest/rllib/)


# Predator-Prey-Grass
### Evolution in a multi-agent reinforcement learning gridworld

We combine **Multi-Agent Reinforcement Learning** (MARL) with **evolutionary dynamics** to explore the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via MARL algorithms). This repo explores emergent behaviors in a multi-agent dynamic ecosystem of predators, prey, and regenerating grass. Agents differ by speed, vision, energy metabolism, and decision policies—offering ground for open-ended adaptation. At its core lies a gridworld simulation where agents are not just *trained*—they are *born*, *age*, *reproduce*, *die*, and even *mutate* in a continuously changing environment.

<p align="center">
    <b>The Predator-Prey-Grass base-environment</b></p>
<p align="center">
    <img align="center" src="./assets/images/gifs/rllib_pygame_1000.gif" width="600" height="500" />
</p>

### Features

* At startup Predator, Prey and Grass are randomly positioned.

* Predators and Prey are independantly (decentralized) trained via their own RLlib policy module.:

  * **Predators** (red)
  * **Prey** (blue)

* Predator and Prey **learn movement strategies** based on their **partial observations**.
* Both expend **energy** as they move around the grid and **replenish energy by eating**:

  * **Prey** eat **Grass** (green).
  * **Predators** eat **Prey** by moving onto the same grid cell.

* **Predator death conditions**:

  * Starvation (when energy runs out).
* **Prey death conditions**:

  * Starvation.
  * Being eaten by a Predator.
* **Reproduction**:

  * Both Predators and Prey reproduce **asexually** when their energy exceeds a threshold.
  * New agents are spawned close to their parent.

* Grass agents regenerate at the same spot after being eaten by Prey.

### Code base:
* Environment

* Configuration

* Training

* Evaluation



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
3. Install the additional system dependency:
    -   ```bash
        conda install -y -c conda-forge gcc=14.2.0
        ```
## Running examples
- [PettingZoo in combination with SB3](https://github.com/doesburg11/PredPreyGrass/tree/main/src/predpreygrass/pettingzoo#getting-started-with-the-pettingzoosb3-framework)


## References

- [RLlib: Industry-Grade, Scalable Reinforcement Learning](https://docs.ray.io/en/master/rllib/index.html)
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)
