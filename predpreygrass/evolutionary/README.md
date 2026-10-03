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

* **[eco_evolutionary](https://github.com/doesburg11/PredPreyGrass-archive/tree/main/eco_evolutionary)**
  *(archived)* — baseline of the family; a `speed` trait that sets a movement-distance
  threshold (1 vs. 2 tiles per move). A real but messy, unconfirmed signal (tentative
  Red Queen-style alternation between predator/prey speed selection, not replicated).
  Its own writeup's recommended follow-up, `eco_evolutionary_cadence`, was rejected
  outright — the entire documented lineage from this module is now archived, and nothing
  still active in the main repo depends on it or clones it directly.
* **Moved to [PredPreyGrass-archive](https://github.com/doesburg11/PredPreyGrass-archive)** —
  `eco_evolutionary_cadence` (Trial 1, rejected — the movement-cadence mechanic itself
  structurally prevents a sustainable predator population), `eco_evolutionary_cooperation`
  (Trial 5, likely null, paused after Pilot 1), `eco_evolutionary_metabolic_code` (Trial 7,
  complete, null and reversed on the headline metric), `eco_evolutionary_metabolic_rate`
  (Trial 3, null after proper 3-seed replication — this is also where the project's
  drift-vs-control replication methodology was built), and
  `eco_evolutionary_investment` (Trial 6, `offspring_investment_fraction` — looked like the
  one real exception after R9's n=3 prey separation hit the statistical ceiling, p=0.050, but
  R10's extension to n=6 reversed it, p=0.120 — a small-sample artifact, not a real effect;
  confirmed null, closing the family with no surviving exception). All five reached a real,
  concluded null result with nothing to build on and were archived to keep this repo
  uncluttered. Full code, tests and commit history preserved there; see that repo's README
  for what each found.
* **Moved to [PredPreyGrass-archive](https://github.com/doesburg11/PredPreyGrass-archive)** —
  `eco_evolutionary_cultural_plasticity` (Trial 8, gene-culture coevolution/dual
  inheritance), `eco_evolutionary_cultural_plasticity_seasonal` (Trial 9, the same
  mechanism with a flipping target dialect, testing Rogers' Paradox), and
  `eco_evolutionary_nuptial_gift` (obligate male provisioning) all reached a real,
  concluded null result and were archived out of this repo to keep it uncluttered.
  Full code, tests and commit history preserved there; see that repo's README for
  what each found. `RESULTS.md` here still has the cross-module trial narrative.

* **[eco_evolutionary_erl_flagship](eco_evolutionary_erl_flagship)** — *closed; positive on
  reward design, null on evolution discovering it.* Ports the ERL architecture to the richer
  flagship ecology. A hand-designed cautious reward (`avoider`) beats a reckless one in direct
  competition (9/10) and beats the sparse fitness reward 27/30 (p=0.000008), but the module's
  own evolution doesn't reliably find it (reward weights indistinguishable from neutral
  drift). See its [README](eco_evolutionary_erl_flagship/README.md).
* **[eco_evolutionary_erl_coevolution](eco_evolutionary_erl_coevolution)** (Trial 14) —
  *closed, POSITIVE result; stays in this repo, not an archive candidate.* Forks
  `eco_evolutionary_erl_baldwin` step by step toward predator–prey coevolution
  (prey-regulated carnivores, evolving carnivore genomes, ERL carnivores). **Key message:
  it replicates "nature + nurture beats nature alone" robustly** (ERL vs. evolution alone
  with prey-regulated carnivores: 1/20 vs. 10/20 prey extinct, p=0.003). Against learning
  alone the advantage is conditional: prey survival in a harsh world, ecosystem stability
  in an easier one (carnivores die out with L prey, never with ERL prey: 16/229 vs.
  0/230). Also documents a predator-side fitness plateau (persistent search) and a
  heritability → spatial-refuge mechanism. See its
  [RESULTS.md](eco_evolutionary_erl_coevolution/RESULTS.md).

See **[RESULTS.md](RESULTS.md)** for the full cross-module trial log — the sequence of
attempts, why each pivot happened, and the current state of the search.

## Theory: Darwinian vs. Baldwinian evolution

- Baldwin, J. M. (1896). [A New Factor in Evolution](https://www.jstor.org/stable/2453130). *The American Naturalist*, 30(354), 441–451. — the original statement of the effect: learned behavior can steer which genotypes are favored by selection, without the learned behavior itself being inherited.
- Simpson, G. G. (1953). [The Baldwin Effect](https://www.jstor.org/stable/2405746). *Evolution*, 7(2), 110–117. — clarifies the effect against Lamarckian misreadings and gives it its modern name.
- Hinton, G. E., & Nowlan, S. J. (1987). [How Learning Can Guide Evolution](https://doi.org/10.1007/BF01148891). *Complex Systems*, 1, 495–502. — the canonical computational demonstration: individual learning smooths a rugged fitness landscape, making an otherwise unlikely genotype reachable by evolutionary search. See `RESULTS.md`'s theoretical note for why this paper's own stated limitation may explain the null results above.
- Ackley, D., & Littman, M. (1991). [Interactions Between Learning and Evolution](https://www.researchgate.net/publication/2461712_Interactions_Between_Learning_and_Evolution). In *Artificial Life II*, 487–509. — evolving agents that also learn during their lifetime, closest in spirit to this repo's genome-plus-PPO setup.
