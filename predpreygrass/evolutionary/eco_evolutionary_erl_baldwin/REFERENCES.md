# References — Evolutionary Reinforcement Learning

Research background for this experiment's ERL / Baldwin-effect design. Compiled 2026-08-23.

## How this project relates to the field

Most of the "Evolutionary Reinforcement Learning" literature below uses evolution as a
**search/optimization technique**: a population-based, gradient-free (or gradient-assisted)
way to solve credit assignment, exploration, and hyperparameter brittleness faster or more
robustly than plain RL. The evolutionary process there is an engineering choice, not a claim
about biological plausibility — success is measured by benchmark score or sample efficiency.

This experiment sits in the much smaller **Baldwin effect** line of work instead: the question
is whether/how an agent's lifetime learning changes what evolution can find, as a hypothesis
about real evolutionary dynamics in a sustainable ecological loop — not a benchmark-solving
technique. Sections below are tagged accordingly:

- **[efficiency]** — evolution/population methods used to make RL solve tasks faster or more
  robustly; no claim of biological realism
- **[realism]** — work directly addressing the evolution↔learning interaction as biological
  hypothesis; the literature this experiment actually builds on

## Surveys (start here for orientation)

- [Evolutionary Reinforcement Learning: A Survey](https://arxiv.org/abs/2303.04150) (Bai, Cheng, Jin, 2023) — the standard reference survey for the field
- [Evolutionary Reinforcement Learning: A Systematic Review and Future Directions](https://arxiv.org/pdf/2402.13296) (2024)
- [Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey on Hybrid Algorithms](https://arxiv.org/abs/2401.11963) (2024)
- [Reinforcement Learning-assisted Evolutionary Algorithm: A Survey and Research Opportunities](https://arxiv.org/abs/2308.13420) (2023) — the inverse direction (RL improving EA)
- [Combining Evolution and Deep Reinforcement Learning for Policy Search: A Survey](https://arxiv.org/pdf/2203.14009) (2022)
- [Awesome-Evolutionary-Reinforcement-Learning](https://github.com/yeshenpy/Awesome-Evolutionary-Reinforcement-Learning) — actively maintained GitHub paper/code list

## Core hybrid ERL algorithms (EA population + gradient-based RL) [efficiency]

- [Evolution-Guided Policy Gradient in Reinforcement Learning (ERL)](https://arxiv.org/abs/1805.07917) (Khadka & Tumer, NeurIPS 2018) — the paper that coined "ERL"; population evolves policies, best is periodically injected with a DDPG-trained actor
- [Collaborative Evolutionary Reinforcement Learning (CERL)](https://arxiv.org/abs/1905.00976) (Khadka et al., ICML 2019) — portfolio of learners + shared replay buffer, solves Humanoid where individual TD3 fails
- [Proximal Distilled Evolutionary Reinforcement Learning (PDERL)](https://arxiv.org/abs/1906.09807) (Bodnar, Day, Liò, AAAI 2020) — learned, non-destructive variation operators to fix ERL's catastrophic-forgetting crossover/mutation ([code](https://github.com/crisbodnar/pderl))
- [CEM-RL: Combining Evolutionary and Gradient-based Methods for Policy Search](https://arxiv.org/abs/1810.01222) (Pourchot & Sigaud, 2019) — Cross-Entropy Method population + TD3
- [Evolutionary Reinforcement Learning for Sample-Efficient Multiagent Coordination (MERL)](http://proceedings.mlr.press/v119/majumdar20a.html) (Majumdar, Khadka et al., ICML 2020) — splits team-objective (EA) from agent-objective (gradient) training
- [GEP-PG: Decoupling Exploration and Exploitation in Deep RL Algorithms](https://arxiv.org/abs/1802.05054) (Colas et al., ICML 2018) — Goal Exploration Process seeds DDPG's replay buffer

## Pure evolutionary / neuroevolution approaches to RL [efficiency]

- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Salimans, Ho, Chen, Sutskever — OpenAI, 2017) — the paper that revived ES for deep RL at scale
- [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for RL](https://arxiv.org/abs/1712.06567) (Such, Madhavan, Conti, Lehman, Stanley, Clune — Uber AI, 2017)
- [Evolving Neural Networks through Augmenting Topologies (NEAT)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) (Stanley & Miikkulainen, 2002) — foundational topology-evolving neuroevolution, still widely used/extended
- [Evolvability ES: Scalable and Direct Optimization of Evolvability](https://arxiv.org/abs/1907.06077) (Gajewski, Clune, Stanley, Lehman — Uber AI, 2019)

## Exploration & diversity-driven methods [efficiency, partial biological inspiration]

- [Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf) (Lehman & Stanley, 2011) — foundational novelty search paper
- [Illuminating Search Spaces by Mapping Elites (MAP-Elites)](https://members.loria.fr/jbmouret/qd.html) (Mouret & Clune, 2015) — quality-diversity foundation
- [Policy Gradient Assisted MAP-Elites (PGA-ME)](https://dl.acm.org/doi/10.1145/3449639.3459304) (Nilsson & Cully, GECCO 2021) — TD3-style gradient operator inside MAP-Elites for scaling to large NN controllers ([code](https://github.com/ollebompa/PGA-MAP-Elites))
- [QDax: A Library for Quality-Diversity and Population-based Algorithms with Hardware Acceleration](https://arxiv.org/pdf/2308.03665) (2023) — JAX-accelerated QD/ERL toolkit

## Open-endedness [efficiency, instrumental use of a biological idea]

- [POET: Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions](https://arxiv.org/abs/1901.01753) (Wang, Lehman, Clune, Stanley — Uber AI, 2019)
- [Enhanced POET: Open-Ended RL through Unbounded Invention of Learning Challenges and their Solutions](https://arxiv.org/pdf/2003.08536) (Wang et al., ICML 2020)

## Population-based training (hyperparameters/policies co-evolved) [efficiency]

- [Population Based Training of Neural Networks (PBT)](https://arxiv.org/abs/1711.09846) (Jaderberg et al. — DeepMind, 2017)

## Baldwin effect (evolution ↔ learning interaction) [realism]

The literature this experiment actually builds on.

- [How Learning Can Guide Evolution](http://www.cogsci.ucsd.edu/~rik/courses/cogs184_w10/readings/HintonNowlan97.pdf) (Hinton & Nowlan, 1987) — the original Baldwin-effect simulation
- [Interactions between Learning and Evolution](https://openended.protolife.org/t/interactions-between-learning-and-evolution/338) (Ackley & Littman, Artificial Life II, 1991, pp. 487–509) — companion foundational study to Hinton & Nowlan; agents with both an evolved genome and a lifetime learning (RL) component
- [Myths and Legends of the Baldwin Effect](https://arxiv.org/pdf/cs/0212036) — critical review of common misconceptions
- [Evolving Self-taught Neural Networks: The Baldwin Effect and the Emergence of Intelligence](https://arxiv.org/abs/1906.08854)
- [Meta-Learning by the Baldwin Effect](https://arxiv.org/pdf/1806.07917) — connects Baldwin effect to modern meta-learning framing

## GPU-scale / modern infrastructure [efficiency]

- [EvoRL: A GPU-accelerated Framework for Evolutionary Reinforcement Learning](https://arxiv.org/html/2501.15129v2) (2025)
