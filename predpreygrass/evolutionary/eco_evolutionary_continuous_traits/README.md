# eco_evolutionary_continuous_traits

**Status (2026-10-24): pilot, v2 round, Baldwin-style branches and their
replication done** ([RESULTS.md](RESULTS.md)). With expensive capacity
(pilot) the traits evolved by selection into cheap bodies and the ecosystem
and the learning arms race collapsed. The two cheaper-capacity runs (v2: 2×
and 4× cheaper, also with fixed observation scales) were stable (coexistence
50 of 50); both species traded view distance for a wide field of view, and
speed rose and fell again; newer predators caught less, in CT-04 mostly
through their evolved bodies, in CT-05 slightly and through behaviour, while
the prey's gains came from behaviour. Branching CT-05 into learning and
frozen arms, predator speed rose only where predators kept learning; the
replication with 3 seeds per arm confirmed it (+0.18, 95% CI +0.14 to
+0.22, exact p = 0.005, same direction in all 12 runs, RESULTS.md section
5). Write-up: [Learning Steers Body Evolution](https://humanbehaviorpatterns.org/learning-selection-interaction/predpreygrass-continuous/evolving-bodies).

Heritable body traits in the continuous Predator-Prey-Grass world: does an
eco-evolutionary arms race appear when animals' bodies evolve while their
behaviour is learned?

[`base_environment_continuous`](../../non_evolutionary/continuous_space/base_environment_continuous/)
showed a learning arms race in that world: one shared PPO policy per
species, newer predators catching fixed prey better and newer prey evading
fixed predators better, in five independent runs, with and without
Aquarium's quirks (its RESULTS.md, sections 2 and 3). That is co-adaptation
by learning only: no genomes, no heritable variation, no selection. This
module, a copy of that one, adds evolution: every animal carries a genome
of body traits, and its offspring inherit it with mutation.

The project's goal is a sustainable, significant Darwin/Baldwin loop in
predator-prey coevolution. The closest earlier result, ERL coevolution
(Trial 14,
[`eco_evolutionary_erl_coevolution`](../eco_evolutionary_erl_coevolution/)),
evolved per-agent networks with a weak lifetime learner on a grid and
produced no arms race. Here the learner is the PPO policy that does produce
one, and evolution acts on bodies.

## Design

Full rules in [SPEC.md](SPEC.md), section 11.

- **Genes:** top speed, view distance and field of view, as log-ratios to
  the species' base values (×0.5 to ×2).
- **Cost:** a larger capacity raises the resting metabolic cost (speed
  1.0, vision 0.5, field of view 0.25 per unit of relative change); moving
  still costs per unit of speed. The coefficients are starting guesses.
- **Inheritance:** asexual, with mutation; each episode's founders are drawn
  from the survivors of the previous one, so the gene pool evolves across
  PPO episodes. Each simulation worker evolves its own population.
- **Learning:** one shared PPO policy per species, which sees each animal's
  own genes (3 extra inputs) and can learn to use the body it has.
- **World:** every quirk of SPEC.md section 9 fixed (the world of BE-04
  and BE-05).
- **Neutral control:** `trait_expression = False` carries the genes the same
  way without effect, measuring drift.

What would count as a result, in order:

1. **Selection, not drift:** within every run, selection differentials
   (covariance of each gene with relative offspring count, logged per
   episode) consistently away from zero in expressed runs and around zero
   in neutral runs, with trait trajectories that follow them. The neutral
   control alone is not enough: expressed traits change population sizes,
   so the two conditions drift differently.
2. **Coevolution:** a predator trait change followed by a matching prey
   change (or the reverse), e.g. a speed or vision race, in addition to the
   learning arms race.
3. **Baldwin interaction:** learning changing the course of evolution, e.g.
   traits evolving differently with a frozen policy than with a learning
   one.

**Units of replication.** Each simulation worker evolves its own gene pool,
but all workers train one shared policy per species, so a policy change
learned on one island changes fitness on every other: workers are not
independent replicates. Independent training runs are.

**Known limitation:** each episode restarts from 6 predators and 8 prey
drawn from the survivors of the last one, a tight bottleneck between
episodes that strengthens drift.

## Vision and energy: the biology behind the vision genes

The model charges every animal a resting energy cost that rises with its
view distance and field of view (SPEC.md, section 11). This section sets
out what is known about the energy cost of vision in real animals, and
where the model follows or departs from it (references at the end of this
README).

**Seeing is energetically expensive.**

- Laughlin, de Ruyter van Steveninck and Anderson (1998) measured the energy
  used by blowfly (*Calliphora vicina*) photoreceptors while they code
  light. Neural signalling costs several orders of magnitude more than the
  thermodynamic minimum: about 10^4 ATP molecules per bit at a chemical
  synapse and 10^6–10^7 ATP per bit for graded signals in a photoreceptor or
  interneuron. In noise-limited systems a low-capacity pathway carries
  information more cheaply.
- Niven, Anderson and Laughlin (2007) compared the same photoreceptors
  (R1–6) in four fly species. A fixed cost, about 20% of a photoreceptor's
  maximum energy use, is paid just to keep it ready (its resting
  potential, in darkness). The fixed cost, the total cost and the cost per
  bit all rise with a photoreceptor's maximum information rate:
  higher-performance photoreceptors (about 1,000 bits/s in the flesh fly
  against about 200 bits/s in *Drosophila melanogaster*) make information
  more expensive. The authors call this a law of diminishing returns that
  "severely penalises overcapacity".
- Niven and Laughlin (2008) review the evidence that energy limits are a
  selective pressure on the evolution of sensory systems, drawing on energy
  measurements in rabbit retina and fly photoreceptors. Neurons use much
  energy both at rest and when signalling, largely to pump ions back across
  their membranes.
- Moran, Softley and Warrant (2015) estimated the cost of the visual system
  (eyes and optic tectum) in the eyed surface form of the Mexican tetra
  (*Astyanax mexicanus*): about 15% of resting metabolism in a 1 g fish,
  falling to about 5% at 8.5 g as eyes and brain become relatively smaller
  during growth. The cave form, which lost its eyes, saves this energy, and
  the authors propose that the cost of vision contributed to eye loss in
  the dark.

**How this compares with the model.**

- *A cost paid whether or not vision is used.* Real eyes have a fixed
  running cost, as in the fly photoreceptors' cost in darkness. Charging
  vision through the resting cost, paid every step, follows this.
- *Better vision costs more.* In flies, higher performance makes each bit
  dearer and overcapacity is penalised; in the model, a longer view
  distance raises the resting cost. Both predict that vision shrinks when
  its benefit does not cover its cost, as in cave fish, and as view distance
  did in every expressed run here (RESULTS.md).
- *The size of the cost is not calibrated.* The fish's whole visual system
  costs 5–15% of resting metabolism. In the model the base vision is part
  of the species' resting cost, and only changes are charged: with the
  pilot's coefficient (0.5), doubling view distance raises resting cost by
  50%; with CT-04's (0.25) by 25%, CT-05's (0.125) by 12.5%. The
  coefficients were chosen as starting guesses, not derived from these
  measurements.

**Field of view is mostly a trade-off of geometry, not of energy.**

- In mammals, the orientation of the eye sockets (orbits) is closely tied
  to how much the two eyes' visual fields overlap (Heesy 2004). Across more
  than 331 mammal taxa, predators (faunivores) have more forward-facing,
  convergent orbits than other mammals, which Heesy interprets as binocular
  vision and depth perception for catching prey (Heesy, SICB meeting
  abstract). Forward-facing eyes give a large binocular overlap but less
  panoramic coverage.
- Across 214 land vertebrates, pupil shape follows ecological niche (Banks
  et al. 2015): vertical slit pupils are common in ambush predators, where
  the depth-of-field blur they create gives cues for judging distance,
  while prey with eyes on the sides of the head tend to have horizontal
  pupils that keep the view along the ground sharp and panoramic for
  spotting predators. (The authors note the explanation for horizontal pupils
  is less settled than for vertical ones.)
- So in nature a wide field of view is bought mainly by giving up binocular
  overlap and the depth judgement it supports, not by spending energy. The
  model charges energy for field of view and gives a narrow field no
  advantage of its own. That may be one reason field of view increased in
  nearly every expressed run, in predators as well as prey. A more
  realistic model would make a narrow, forward field pay off, for instance
  with more precise distance information or more observation slots in
  front, so that predators could evolve "hunting eyes" and prey
  "panoramic eyes".

## Files

- [`environment.py`](environment.py): the world with traits.
- [`callbacks.py`](callbacks.py): logs each species' mean and spread of
  every gene, cost multiplier, numbers and births at the end of every
  episode (TensorBoard tags `env_runners/traits/...`).
- [`train.py`](train.py): PPO training; prints the mean genes per species
  every iteration and saves every worker's gene pool with each checkpoint.
- [`checkpoints.py`](checkpoints.py): loads a checkpoint's policies and gene
  pool.
- [`analysis/pilot.py`](analysis/pilot.py): the pilot's figure and
  results JSON (CT-01 against CT-02).
- [`analysis/v2_costs.py`](analysis/v2_costs.py): the v2 round's figure and
  results JSON (CT-03 to CT-05), with bootstrap intervals for the
  tournaments.
- [`evolve_frozen.py`](evolve_frozen.py): evolution with both policies
  frozen (RLlib cannot train nothing), for the Baldwin-style branches.
- [`analysis/cross_policy_bodies.py`](analysis/cross_policy_bodies.py),
  [`analysis/baldwin_branches.py`](analysis/baldwin_branches.py): behaviour
  against bodies in the tournaments; the four branches.
- [`analysis/final_check.py`](analysis/final_check.py),
  [`tournament.py`](tournament.py): evaluation with each checkpoint's
  evolved gene pool.
- [`config/`](config/): settings (traits and quirk fixes in
  `config_env.py`).
- [`tests/`](tests/): traits (`test_traits.py`, including bit-for-bit
  equivalence with `base_environment_continuous` when traits are off),
  quirk fixes and the environment.

Unchanged copies from `base_environment_continuous`: `geometry.py`,
`grass.py`, `render.py`, `eval.py`, `assets/`.

## Usage

From the repository root:

```bash
python predpreygrass/evolutionary/eco_evolutionary_continuous_traits/train.py
python -m pytest predpreygrass/evolutionary/eco_evolutionary_continuous_traits/tests
```

Runs are saved in the module's `runs/` folder (gitignored).

## TODO: next experimental steps

- [x] **Replicate the Baldwin-style branches** (RESULTS.md section 5, done
  2026-10-24): 3 seeds × 4 arms plus a neutral arm. The pre-specified
  contrast was positive: runs with predator learning had a +0.18 larger
  predator-speed change than runs with frozen predators (95% CI +0.14 to
  +0.22, exact p = 0.005), in the same direction in all 12 runs.
- [ ] **Mechanism of the predator-speed effect**: does speed pay only once
  the policy adapts to it (e.g. score frozen vs learned policies on bodies
  of different speed), or do frozen and learning predators differ in
  demography? And does it hold from other branch points?

## References

- Banks, M. S., et al. (2015). "Why do animal eyes have pupils of different
  shapes?" *Science Advances*, e1500391.
  [PMC4643806](https://pmc.ncbi.nlm.nih.gov/articles/PMC4643806/)
- Heesy, C. P. (2004). "On the relationship between orbit orientation and
  binocular visual field overlap in mammals." *The Anatomical Record Part
  A* 281A, 1104–1110.
- Heesy, C. P. "Hunters and the hunted: ecological influences on mammalian
  orbit orientation." Society for Integrative and Comparative Biology,
  meeting abstract (year not given). [sicb.org](https://sicb.org/?p=33915)
- Laughlin, S. B., de Ruyter van Steveninck, R. R., & Anderson, J. C.
  (1998). "The metabolic cost of neural information." *Nature
  Neuroscience* 1(1), 36–41. [doi:10.1038/236](https://doi.org/10.1038/236)
- Moran, D., Softley, R., & Warrant, E. J. (2015). "The energetic cost of
  vision and the evolution of eyeless Mexican cavefish." *Science
  Advances* 1(8), e1500363.
  [doi:10.1126/sciadv.1500363](https://doi.org/10.1126/sciadv.1500363)
- Niven, J. E., Anderson, J. C., & Laughlin, S. B. (2007). "Fly
  photoreceptors demonstrate energy-information trade-offs in neural
  coding." *PLoS Biology* 5(4), e116.
  [PMC1828148](https://pmc.ncbi.nlm.nih.gov/articles/PMC1828148)
- Niven, J. E., & Laughlin, S. B. (2008). "Energy limitation as a selective
  pressure on the evolution of sensory systems." *Journal of Experimental
  Biology* 211, 1792–1804.
  [doi:10.1242/jeb.017574](https://doi.org/10.1242/jeb.017574)
