# eco_evolutionary_continuous_traits

## Key message (2026-10-24)

**Learning changes how bodies evolve.** From CT-05 at iteration 1,500,
predator speed rose in all 6 runs where the predators kept learning and
fell in all 6 where their behaviour was frozen (+0.18, 95% CI +0.14 to
+0.22, p = 0.005); from iteration 2,500 it rose more in all 6 learning runs
than in all 6 frozen ones (+0.19, p = 0.005).

- **Found:** a replicated Baldwin-style effect (criterion 3 below).
- **Not found:** a body arms race between predators and prey (criterion 2).
  Both species evolved the same way: shorter view distance, wider field of
  view.
- **Mechanism (2026-10-25):** extra speed pays about twice as much under
  the learned predator policies as under the frozen one, against the same
  prey (+0.46, 95% interval +0.24 to +0.68, all 6 runs; RESULTS.md
  section 6).
- **Frozen arms (2026-10-25):** with both policies frozen, faster
  predators' extra births are largely offset by lower survival of their
  lineages; net selection on speed is small and inconsistent, in line with
  the small fall there (RESULTS.md section 7).
- **Second branch point (2026-10-26):** from CT-05 at iteration 2,500 the
  same contrast is +0.19 (95% CI +0.09 to +0.28, p = 0.005), against +0.18
  from iteration 1,500 (RESULTS.md section 8).
- **Overall (2026-10-26):** learning robustly changes the direction in
  which predator speed evolves, most likely because learned behaviour lets
  faster predators turn extra births into survivors. Not shown: an arms
  race in bodies between the species, adaptation to speed specifically
  (the learned policies differ in everything they learned), a classic
  Baldwin effect (learning is shared per species), or generality beyond
  CT-05. For the project's goal this is the learning-to-body half of the
  loop, running one way; the coevolution half is missing (TODO: field-of-
  view trade-off, per-agent learning). Details: RESULTS.md, key message.
- **Still open:** whether the mechanism is adaptation to speed itself, why
  speed fell when only predators were frozen, and whether the effect holds
  beyond CT-05's trajectory (see TODO).

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

## About this module

Heritable body traits in the continuous Predator-Prey-Grass world: does an
eco-evolutionary arms race appear when animals' bodies evolve while their
behaviour is learned?

[`base_environment_continuous`](../../../non_evolutionary/continuous_space/base_environment_continuous/)
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
[`eco_evolutionary_erl_coevolution`](../../discrete_space/eco_evolutionary_erl_coevolution/)),
evolved per-agent networks with a weak lifetime learner on a grid and
produced no arms race. Here the learner is the PPO policy that does produce
one, and evolution acts on bodies.

## Origin and relation to ERL

The module was started on 2026-10-15 (commit 956e5d0) as the next step
toward the Darwin/Baldwin goal, from two earlier results. First,
`base_environment_continuous` had just shown a learning arms race in the
continuous world, but with learning only. Second, ERL coevolution (Trial
14) had evolving per-agent networks on a grid, but a lifetime learner too
weak to produce an arms race. Rather than port the ERL design to the
continuous world, the choice was to keep the strong PPO learner and let
evolution act on bodies. The module is a copy of
`base_environment_continuous` with genes added, every quirk of SPEC.md
section 9 fixed and a neutral control.

It is a sibling of the ERL line, not a continuous version of it:

| | ERL (Trials 11, 12, 14) | This module (Trial 15) |
| --- | --- | --- |
| World | discrete grid | continuous space |
| What the genes encode | behaviour (each agent's own network) | body: speed, view distance, field of view |
| Who learns | each agent, during its own life | one shared PPO policy per species |
| Learner | weak | strong (produces the learning arms race) |
| Parent module | `eco_evolutionary_erl_baldwin` → `eco_evolutionary_erl_coevolution` | `base_environment_continuous` |

Trials 1 to 10 also used one shared policy per species and gave null
results, because the genome could reach behaviour only indirectly. ERL
solved this by giving each agent its own genome-driven network. This module
solves it differently: the genes change the body itself, and the policy
sees each animal's genes and can learn to use the body it has. Each design
has what the other lacks. ERL has heritable behaviour and individual
lifetime learning, but no arms race. This module has the arms race and an
effect of learning on how genes evolve, but its learning is done by the
whole species, not by each animal during its life. Joining the two, with
per-agent learning in the continuous world, has not been built yet.

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
- [`analysis/baldwin_ecology.py`](analysis/baldwin_ecology.py): numbers
  and selection in the replication's arms (RESULTS.md section 5.1).
- [`mechanism_speed.py`](mechanism_speed.py),
  [`analysis/mechanism_speed.py`](analysis/mechanism_speed.py): the
  mechanism test, predators with fixed speed genes under frozen and learned
  policies (RESULTS.md section 6).
- [`analysis/frozen_trajectory.py`](analysis/frozen_trajectory.py),
  [`decompose_frozen.py`](decompose_frozen.py),
  [`analysis/decompose_frozen.py`](analysis/decompose_frozen.py): why speed
  fell in the frozen arms (RESULTS.md section 7).
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
python predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/train.py
python -m pytest predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/tests
```

Runs are saved in the module's `runs/` folder (gitignored).

## TODO: next experimental steps

- [x] **Replicate the Baldwin-style branches** (RESULTS.md section 5, done
  2026-10-24): 3 seeds × 4 arms plus a neutral arm. The pre-specified
  contrast was positive: runs with predator learning had a +0.18 larger
  predator-speed change than runs with frozen predators (95% CI +0.14 to
  +0.22, exact p = 0.005), in the same direction in all 12 runs.
- [x] **Mechanism of the predator-speed effect** (RESULTS.md section 5;
  done 2026-10-25, section 6).
  Does extra speed pay only once the policy has adapted to it? Score the
  frozen branch-point predator policy and the end-of-branch learned ones on
  predators whose speed gene is set to fixed values (for example −0.2, 0,
  +0.2, +0.4 around the branch point's +0.19), against the same prey, and
  compare offspring and survival, not only catches. If speed pays more under
  learned policies, adaptation of behaviour to bodies is the likely route;
  if not, check whether frozen and learning arms differ in predator and prey
  numbers enough to change selection on speed. Result: speed pays more under
  the learned policies (+0.46, all 6 runs, from the same starting prey), at
  this branch point; prey numbers within an episode still differ between
  predator policies.
- [x] **Why speed fell in the frozen arms** (raised in RESULTS.md section 6;
  done 2026-10-25 for the both-frozen arm, section 7). Speed
  still pays under the frozen policy in the one-episode test. Score the
  frozen policy against later frozen-arm gene pools, and split lineage
  fitness into births and survival, to see where its advantage goes.
  Result: later gene pools did not remove the advantage; in an evolving
  both-frozen population faster lineages' extra births are largely offset
  by lower survival, so net selection on speed is small and inconsistent.
- [ ] **Why speed fell when only predators were frozen** (RESULTS.md
  section 7). Its fall (−0.16) held in all three seeds and is not explained;
  there the prey kept learning. Re-run that arm with lineages tracked
  (`decompose_frozen.py` with the prey learning, or with each saved prey
  policy in turn).
- [x] **Other branch points** (done 2026-10-26 for CT-05 at iteration
  2,500, RESULTS.md section 8). Repeat the four arms (3 seeds each) from a
  different CT-05 iteration (for example 1,000 or 2,500) or from CT-04, to
  test whether the effect depends on the state the branches start from.
  Same queue (`launchers/queue_baldwin_replication.sh`, at most 3 RLlib
  runs on the GPU at once); about 20 hours for the RLlib arms. Result: the
  contrast replicated (+0.19, p = 0.005), from a state with much slower
  predators (+0.06 against +0.19). It took 34 hours (11:38 to 21:46 the
  next day) on 3 GPU slots.
- [ ] **Toward coevolution: a field-of-view trade-off.** Both species now
  evolve the same way (shorter view distance, wider field of view), so
  there is no body arms race. Give a narrow, forward field of view its own
  advantage (for example more precise distance information or more
  observation slots in front; see "Vision and energy" above), so that
  predators can evolve "hunting eyes" and prey "panoramic eyes", and test
  whether the two species' bodies then change in response to each other.
- [ ] **Toward a classic Baldwin effect: per-agent learning.** Learning here
  is one shared PPO policy per species. Combine this module with the ERL
  design (per-agent, genome-driven networks with lifetime learning; see
  "Origin and relation to ERL"), keeping a learner strong enough for the
  learning arms race. A much bigger build than the item above.
- [ ] **A branch point outside CT-05.** CT-04 (costs 0.5× the pilot's,
  against CT-05's 0.25×) would test whether the effect depends on the cost
  of capacity, not only on the starting state: `BLREP_BRANCH` only selects
  CT-05 iterations, so the launcher needs a run option first.

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
