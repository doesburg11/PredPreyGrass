# Eco-Evolutionary Trials

## Goal

Demonstrate a **sustainable Darwin/Baldwin evolutionary loop** in a predator-prey
coevolution simulation: genetic evolution (Darwinian) and within-lifetime RL learning
(Baldwinian) feeding back into each other, not just coexisting side by side.

Each module below layers a genuine evolutionary algorithm — founder genome, mutation,
inheritance — on top of shared-policy PPO. A scalar trait is passed from parent to
offspring with mutation at each reproduction event; PPO policy weights are never
inherited, only shared per species. Learned behavior (Baldwinian) determines which trait
values survive to reproduce, closing a genome → phenotype → learned behavior → fitness →
genome-frequency loop across generations.

**Success requires all three of these together, not just one:**

1. **Sustainability** — populations coexist without frequent mid-episode
   collapse/extinction; most episodes reach full length rather than crashing early.
2. **Coexistence** — stable predator-prey coexistence (neither species chronically
   crashing or eliminating the other).
3. **Darwin/Baldwin loop** — the evolving genome trait shows genuine,
   *selection-driven* drift, not just neutral genetic drift. A directional-looking
   trend on its own is not sufficient evidence — it must be checked against a
   neutral-drift control (mutation active, reproduction decoupled from genome) before
   being trusted as real selection. Ideally also shows the reverse leg (genome drift
   measurably changing the RL fitness landscape, not just RL learning driving genome
   drift).

Population-regulation mechanisms must be biologically realistic / emergent (individual
energy/starvation dynamics, Lotka-Volterra-style), not an artificial top-down
population-census rule — no individual agent can sense a population-wide ratio, so
reproduction caps keyed on one are rejected even when they produce better raw
sustainability numbers.

## Modules

* **[eco_evolutionary](eco_evolutionary)** — baseline of the family. Evolves a `speed`
  trait that sets a movement-distance threshold (1 vs. 2 tiles per move).
* **[eco_evolutionary_cadence](eco_evolutionary_cadence)** — evolves the same `speed`
  trait, expressed as a graded movement cooldown instead of a discrete distance
  threshold. *Rejected* — the cadence mechanic itself structurally prevents predators
  from sustaining a population.
* **[eco_evolutionary_metabolic_rate](eco_evolutionary_metabolic_rate)** — evolves a
  `metabolic_rate` trait that symmetrically scales both energy gain and basal energy
  cost. Sustainability/coexistence solved; selection-driven drift **null** after
  replication.
* **[eco_evolutionary_investment](eco_evolutionary_investment)** — evolves an
  `offspring_investment_fraction` trait — how much energy a parent hands each offspring
  at birth. Sustainability/coexistence solved; selection-driven drift **null** after
  replication.
* **[eco_evolutionary_cooperation](eco_evolutionary_cooperation)** — evolves a
  `cooperation_rate` trait — the fraction of an agent's net energy gain donated to
  nearby same-species agents, relying on spatial viscosity (offspring spawn near
  parents) for implicit kin selection. Pilot result: likely null, not yet confirmed
  (see its `RESULTS.md`).

See **[RESULTS.md](RESULTS.md)** for the full cross-module trial log — the sequence of
attempts, why each pivot happened, and the current state of the search.

## Theory: Darwinian vs. Baldwinian evolution

- Baldwin, J. M. (1896). [A New Factor in Evolution](https://www.jstor.org/stable/2453130). *The American Naturalist*, 30(354), 441–451. — the original statement of the effect: learned behavior can steer which genotypes are favored by selection, without the learned behavior itself being inherited.
- Simpson, G. G. (1953). [The Baldwin Effect](https://www.jstor.org/stable/2405746). *Evolution*, 7(2), 110–117. — clarifies the effect against Lamarckian misreadings and gives it its modern name.
- Hinton, G. E., & Nowlan, S. J. (1987). [How Learning Can Guide Evolution](https://doi.org/10.1007/BF01148891). *Complex Systems*, 1, 495–502. — the canonical computational demonstration: individual learning smooths a rugged fitness landscape, making an otherwise unlikely genotype reachable by evolutionary search. See `RESULTS.md`'s theoretical note for why this paper's own stated limitation may explain the null results above.
- Ackley, D., & Littman, M. (1991). [Interactions Between Learning and Evolution](https://www.researchgate.net/publication/2461712_Interactions_Between_Learning_and_Evolution). In *Artificial Life II*, 487–509. — evolving agents that also learn during their lifetime, closest in spirit to this repo's genome-plus-PPO setup.
