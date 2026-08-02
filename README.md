[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.55.1-blue)](https://docs.ray.io/en/latest/rllib/)


# Predator-Prey-Grass
## Multi-Agent Deep Reinforcement Learning meets Darwinian and Baldwinian evolution

Legacy snapshot: the pre-cleanup research codebase is archived at [PredPreyGrassLegacy](https://github.com/doesburg11/PredPreyGrassLegacy).

This project explores whether cooperative behavior, coevolution, defection, and free-riding can emerge and stabilize in a spatial, resource-limited ecosystem, by combining within-lifetime multi-agent reinforcement learning with population-level ecological and evolutionary dynamics. It probes the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via reinforcement learning) — including a direct test of the **Baldwin effect**: whether genetic selection and learned behavior actually shape each other, not just coexist. We combine **Multi-Agent Deep Reinforcement Learning** (MADRL) with **evolutionary dynamics** across a multi-agent dynamic ecosystem of Predators, Prey, and regenerating Grass. Agents differ by speed, vision, energy metabolism, and decision policies—offering ground for open-ended adaptation. At its core lies a gridworld simulation where agents are not just *trained*—they are *born*, *age*, *reproduce*, *die*, and even *mutate* in a continuously changing environment.

<p align="center">
    <b>Fixed-trait game-theoretic hunting (non-evolutionary branch): coevolution, cooperation, defection and free-riding</b></p>
<p align="center">
    <img align="center" src="./assets/images/gifs/stag_hunt_defect.gif" width="600" height="500" />
</p>

## Environments

This repo splits into two structurally different families of experiment, matching the
`predpreygrass/evolutionary/` vs `predpreygrass/non_evolutionary/` directory split:

- **Evolutionary**: agents carry a heritable genome trait, passed parent → offspring
  with mutation. What gets selected is discovered, not designed.
- **Non-evolutionary**: every agent trait is fixed; only the RL policy adapts. What
  emerges is a behavioral equilibrium under a given incentive design, not a change in
  the population's genetics.

### Darwinian/Baldwinian evolutionary environments

These environments layer a genuine evolutionary algorithm — founder genome, mutation, inheritance — on top of shared-policy PPO. Learned behavior (Baldwinian) determines which trait values survive to reproduce, closing a genome → phenotype → learned behavior → fitness → genome-frequency loop across generations. See **[predpreygrass/evolutionary/README.md](predpreygrass/evolutionary)** for the shared goal, success criteria, and cross-module trial log — start there before any individual module below.

* **[Eco-evolutionary](predpreygrass/evolutionary/eco_evolutionary)**: baseline of the family. Evolves a `speed` trait that sets a movement-distance threshold (1 vs. 2 tiles per move).

* **[Eco-evolutionary cadence](predpreygrass/evolutionary/eco_evolutionary_cadence)**: evolves the same `speed` trait, expressed as a graded movement cooldown instead of a discrete distance threshold.

* **[Eco-evolutionary cooperation](predpreygrass/evolutionary/eco_evolutionary_cooperation)**: evolves a `cooperation_rate` trait — the fraction of an agent's net energy gain donated to nearby same-species agents, relying on spatial viscosity (offspring spawn near parents) for implicit kin selection.

* **[Eco-evolutionary investment](predpreygrass/evolutionary/eco_evolutionary_investment)**: evolves an `offspring_investment_fraction` trait — how much energy a parent hands each offspring at birth.

* **[Eco-evolutionary metabolic rate](predpreygrass/evolutionary/eco_evolutionary_metabolic_rate)**: evolves a `metabolic_rate` trait that symmetrically scales both energy gain and basal energy cost.

* **["Stag hunt" nature + nurture](predpreygrass/evolutionary/stag_hunt_forward_view_nature_nurture)**: a hybrid case — predators carry a heritable cooperation trait (nature) alongside the learned voluntary `join_hunt` action (nurture); team-capture success depends on both.

### Fixed-trait behavioral & game-theoretic environments

These environments hold every agent trait fixed and instead vary the interaction mechanics or reward shaping. Agents are still born, reproduce, and die, but nothing is inherited or mutated — only the RL policy adapts, converging on a behavioral equilibrium (cooperate, defect, share, reciprocate) under a given incentive design.

* **[Base environment](predpreygrass/non_evolutionary/base_environment)**: the two-policy base environment. Only reproduction rewards. ([results](https://humanbehaviorpatterns.org/pred-prey-grass/overview-ppg))

* **Reward shaping**: five sibling environments comparing sparse vs. dense reward design, plus two RLlib-compliance bug fixes along the way. See **[predpreygrass/non_evolutionary/project_reward_shaping/README.md](predpreygrass/non_evolutionary/project_reward_shaping)** for the shared methodology and full results log — start there before any individual module below.

  * **[Sparse rewards, RLlib-fixed](predpreygrass/non_evolutionary/project_reward_shaping/base_environment_sparse_rewards)**: bug-fixed copy of the base environment — same sparse, reproduction-only reward, plus two RLlib-compliance fixes. The fair baseline for every other variant below.

  * **[Dense rewards](predpreygrass/non_evolutionary/project_reward_shaping/base_environment_dense_rewards)**: replaces the sparse reward with a dense, per-step net energy-delta reward (decay + move + eat + reproduction cost), no reproduction bonus.

  * **[Dense rewards, additive](predpreygrass/non_evolutionary/project_reward_shaping/base_environment_dense_rewards_additive)**: same dense per-step reward, plus the sparse variant's `+10` reproduction bonus layered on top.

  * **[Sparse rewards + eating bonus](predpreygrass/non_evolutionary/project_reward_shaping/base_environment_sparse_rewards_plus_eating)**: still no continuous energy-delta signal — same clean, discrete-event reward style as the sparse baseline, plus a flat reward for the eating event itself (`+1` predator / `+0.1` prey, asymmetric).

  * **[Sparse rewards + kick-back bonus](predpreygrass/non_evolutionary/project_reward_shaping/base_environment_sparse_rewards_plus_kickback)**: keeps the `+10` reproduction reward, adds a second `+10` "kick-back" to a grandparent every time its own child reproduces. Training completed 2026-08-01 (1000/1000 iterations); see that folder's README for results.

* **Cooperation**: ten sibling environments testing how cooperation emerges under a fixed-trait RL policy — joint/team hunting, cooperate/defect dilemmas with free-riding, reputation-conditioned cooperation, reciprocity (direct and spatial/network), and kin-selection altruism. See **[predpreygrass/non_evolutionary/project_cooperation/README.md](predpreygrass/non_evolutionary/project_cooperation)** for the shared framing.

  * **["Stag hunt"](predpreygrass/non_evolutionary/project_cooperation/stag_hunt)**: cooperative and solo hunting with large (mammoths) and small (rabbits) prey. Hunting mammoths usually provides more energy but also needs cooperation of humans and therefore yields a more uncertain outcome.

  * **[Stag hunt with defection](predpreygrass/non_evolutionary/project_cooperation/stag_hunt_defection)**: humans can hunt solo for rabbits but mammoths usually cannot be killed alone, so they have to decide to cooperate at an energy cost or to defect at zero cost, giving opportunities for free-riding.

  * **[Stag hunt forward view](predpreygrass/non_evolutionary/project_cooperation/stag_hunt_forward_view)**: stag hunt defection with forward-shifted predator observations.

  * **[Stag hunt reputation](predpreygrass/non_evolutionary/project_cooperation/stag_hunt_reputation)**: adds a per-predator reputation signal (join/defect history) on top of forward-view stag hunt defection, to test conditional cooperation.

  * **[Mammoth hunting](predpreygrass/non_evolutionary/project_cooperation/mammoths)**: mammoths are only hunted down and eaten by humans in its Moore neighborhood if the cumulative energy of the surrounding humans is *strictly larger* than the mammoth's energy. On failure (if cumulative human energy is too low), humans optionally lose energy proportional to their share of the attacking group's energy (`energy_percentage_loss_per_failed_attacked_prey`). On success, prey energy is split among attackers (proportional by default, optional equal split via `team_capture_equal_split`). Only reproduction rewards.

  * **[Mammoths defection](predpreygrass/non_evolutionary/project_cooperation/mammoths_defection)**: adds a voluntary join/free-ride decision to mammoth hunting.

  * **[Shared prey](predpreygrass/non_evolutionary/project_cooperation/shared_prey)**: this environment is very similar in logic to `mammoth hunting`, but in this case the typical energy level of a prey is smaller than that of a predator. With `mammoth hunting` this is typically the other way around: prey possess more energy than predators. Only reproduction rewards.

  * **[Direct reciprocity](predpreygrass/non_evolutionary/project_cooperation/direct_reciprocity)**: every prey is solo-catchable; predators get a voluntary `share_food` action, testing whether costly food sharing emerges without any coordination necessity.

  * **[Network reciprocity](predpreygrass/non_evolutionary/project_cooperation/network_reciprocity)**: fixed cooperator/defector prey strategies (cooperators donate energy to adjacent prey), testing whether spatial clustering of cooperators lets them persist against defectors (Nowak & May 1992).

  * **[Lineage rewards](predpreygrass/non_evolutionary/project_cooperation/lineage_rewards)**: agents are rewarded for descendants surviving over time, with fertility-age caps that shift agents from reproducing to protecting offspring late in life.

* **[Walls occlusion](predpreygrass/non_evolutionary/walls_occlusion)**: an extension with walls and occluded vision. Only reproduction rewards.

* **[Drive-conditioned environment](predpreygrass/non_evolutionary/drive_conditioned_environment)**: starts as a copy of the base environment; work in progress toward drive-conditioned behavior.

* **[Stag hunt vectorized](predpreygrass/non_evolutionary/stag_hunt_vectorized)**: a performance refactor of stag hunt (vectorized hot paths); not a new behavioral variant.

* **[Red Queen](predpreygrass/non_evolutionary/red_queen)**: independently configurable competing prey types under a shared, non-mutating predator policy, testing coevolutionary arms-race dynamics between learned policies rather than genomes.



### Experiments:

* Testing the **Red Queen Hypothesis** in the co-evolutionary setting of (non-mutating) predators and prey ([implementation](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/non_evolutionary/red_queen/evaluate_red_queen_freeze_type_1_only.py), [results](https://humanbehaviorpatterns.org/pred-prey-grass/red-queen/))

* Testing the **Red Queen Hypothesis** in the co-evolutionary setting of mutating predators and prey. ([implementation](predpreygrass/mutating_agents), [results](predpreygrass/mutating_agents#co-evolution-and-the-red-queen-effect))


### Sparse rewards versus Dense rewards

Started from a simple question: the base environment's only nonzero reward anywhere is a flat
`+10` bonus on successful reproduction — every other hook is `0.0`. Does that sparsity hurt
training, and would a dense, per-step energy-delta reward fix it? Five trained environment
variants and a full investigation later, the answer reversed the question: **sparse reward wins
on every axis tested, and adding density hurts** — not because of the sparsity itself, but
because a continuous signal layered into the same reward channel as reproduction adds noise that
outweighs the benefit it was meant to provide. Along the way this also surfaced two
real RLlib-compliance bugs present in `base_environment` (dying agents' terminal transitions
silently dropped, and agent IDs recycled within an episode in a way that conflates unrelated
individuals' trajectories) — fixed first in the reward-shaping modules below, and since then also
in `base_environment` itself. Not yet verified or fixed elsewhere in the repo.

**Full writeup — motivation, methodology, the bug discovery, every module's results, and open
questions — lives in
[`predpreygrass/non_evolutionary/project_reward_shaping/README.md`](predpreygrass/non_evolutionary/project_reward_shaping).**

### Hyperparameter tuning

* Hyperparameter tuning base environment - **Population-Based Training** ([Implementation](predpreygrass/hyper_parameter_tuning/tune_population_based_training.py))


## Installation of the repository

**Editor used:** Visual Studio Code 1.107.0 on Linux Mint 22.0 Cinnamon

1. Clone the repository:
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
   - Choose environment: Conda
   - Choose interpreter: Python 3.11.13 or higher
   - Open a new terminal
   - ```bash
     pip install -e .
     ```
3. Install the additional system dependency for Pygame visualization:
    -   ```bash
        conda install -y -c conda-forge gcc=14.2.0
        ```
## Quick start
Run a random policy in a Visual Studio Code terminal:

```bash
python ./predpreygrass/non_evolutionary/base_environment/random_policy.py

```

Pretrained checkpoints and historical training outputs are preserved in the legacy archive rather than shipped in the active source tree.

## References

### Darwinian vs. Baldwinian evolution

Moved to **[predpreygrass/evolutionary/README.md](predpreygrass/evolutionary#theory-darwinian-vs-baldwinian-evolution)** — the goal statement, success criteria, and theory references for the `eco_evolutionary_*` environments all live there together now.

### General

- [RLlib: Industry-Grade, Scalable Reinforcement Learning](https://docs.ray.io/en/master/rllib/index.html)
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/marl-book.pdf)
