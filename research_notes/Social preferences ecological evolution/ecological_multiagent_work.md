# Related work: social preferences in ecological multi-agent evolution

**Research date:** 2026-10-27  
**Target:** `predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference`

## Bottom line

I found a close published computational antecedent to the broad idea of
socially influenced behavior evolving in a spatial predator–prey world:
Huang et al.'s wolf–caribou pursuit model evolves caribou escape strategies
with genetic programming and reports that adding empathy improves collective
escape. Other work separately establishes individual-based predator–prey
movement coevolution, evolving predator–prey behavior and morphology,
ecological models in which social information changes prey population
dynamics, empirical predator effects on schooling prey, and experimental
evolution between predators and cooperative prey defenses.

I did **not** locate an exact published match that makes a scalar social
preference (for example, attraction to or avoidance of conspecifics) itself a
heritable, mutating trait under ecological selection in a continuous-space
predator–prey simulation, while also using multi-agent reinforcement learning
or evolutionary computation. This is a bounded literature search, not proof
that no such work exists. In particular, Huang et al. is a close exception to
any claim that social behavior or empathic cooperation has never been evolved
in predator–prey simulations; the distinction is what is directly selected.

Inspection of the target directory's README and implementation is important
for interpreting novelty: the current module is explicitly a frozen-policy
proof of concept. It adds a fixed action-logit bias toward or away from the
nearest visible predator conspecific, evaluates uniform and mixed non-heritable
preferences, and does not change policy training, reward, genes, or
inheritance. It is not yet a simulation of preference evolution
([module README](../../predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/README.md);
[social bias implementation](../../predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/social_bias.py);
[individual preference assignment](../../predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/individual_preferences.py)).

## Closest computational studies

### Empathic caribou escape in a spatial predator–prey world — closest social-behavior analogue

Fangwei Huang, Madoka Takahara, Ivan Tanev, and Katsunori Shimohara (2018),
“Emergence of Collective Escaping Strategies of Various Sized Teams of
Empathic Caribou Agents in the Wolf-caribou Predator-prey Problem,”
*IEEJ Transactions on Electronics, Information and Systems* 138(5):619–626.
[DOI: 10.1541/ieejeiss.138.619](https://doi.org/10.1541/ieejeiss.138.619);
[J-STAGE article and abstract](https://www.jstage.jst.go.jp/article/ieejeiss/138/5/138_619/_article/-char/en).

- The paper describes genetic programming to evolve escape strategies for
  teams of caribou pursued by one superior wolf in a simulated two-dimensional
  infinite toroidal world. Its abstract reports that incorporating empathy
  improved the efficiency and effectiveness of evolved escape behavior, and
  that the relationship between team size and escape efficiency was nonlinear.
- This is the closest found for the joint ingredients “spatial predator–prey,”
  evolving behavior, and prosocial/empathic coordination. It establishes that
  collective escape and empathy-inspired behavior in an ecological pursuit
  task are not, by themselves, novel.
- The abstract describes empathy as incorporated into the agents and genetic
  programming as evolving their escape strategies. It does not report that a
  heritable empathy/social-preference parameter itself mutates and is selected
  over generations. Nor is the method described as MARL/PPO.
- Huang's 2019 dissertation, *Evolving social behavior of caribou agents in
  wolf-caribou predator-prey pursuit problem*, is available from the
  [Doshisha University repository](https://doshisha.repo.nii.ac.jp/records/1533)
  and has
  [DOI 10.14988/di.2019.0000000577](https://doi.org/10.14988/di.2019.0000000577).
  It is useful for source-history/method detail; the 2018 journal article is
  the more concise peer-reviewed citation.

### Eco-evolutionary movement coevolution

Christoph Netz, Hanno Hildenbrandt, and Franz J. Weissing (2021), “Complex
eco-evolutionary dynamics induced by the coevolution of predator–prey
movement strategies,” *Evolutionary Ecology* 36(1):1–17.
[DOI: 10.1007/s10682-021-10140-x](https://doi.org/10.1007/s10682-021-10140-x).

- This individual-based model places predators and prey on a fine-grained
  resource landscape. Both make repeated movement decisions using heritable,
  evolvable movement strategies; the authors report that coevolution produces
  changing spatial patterns, including spiral waves and static spots.
- It is a close reference for coupling behavioral evolution to ecological
  population/spatial dynamics and for making selection operate on
  individual-level behavior. The abstract does not identify a social
  preference or cooperation trait, and it is not framed as a learned
  multi-agent policy.

### Predator–prey morphology and behavior coevolution

Randal S. Olson, Arend Hintze, Fred C. Dyer, Jason H. Moore, and Christoph
Adami (2016), “Exploring the coevolution of predator and prey morphology and
behavior,” in *Proceedings of the 2016 Conference on Artificial Life*.
[DOI: 10.7551/978-0-262-33936-0-ch045](https://doi.org/10.7551/978-0-262-33936-0-ch045).

- The agent-based evolutionary model coevolves predator and prey behavior and
  morphology. The authors report a coevolutionary cycle involving prey
  swarming behavior and predator visual systems, and discuss predator
  confusion as a potential mechanism.
- This establishes a computational precedent for evolving social aggregation
  in an ecology with predators. The reported evolving variables are behaviors
  and morphology; the paper does not describe an explicit heritable
  “attraction/avoidance preference” trait or MARL training.

### Individual-based evolution of an ecological simulation

Robin Gras, Didier Devaurs, Adrianna Wozniak, and Adam Aspinall (2009), “An
Individual-Based Evolving Predator-Prey Ecosystem Simulation Using a Fuzzy
Cognitive Map as the Behavior Model,” *Artificial Life* 15(4):423–463.
[DOI: 10.1162/artl.2009.gras.012](https://doi.org/10.1162/artl.2009.gras.012).

- Each individual has an evolving fuzzy cognitive map that evaluates
  ecological and internal-state inputs, including predator/prey distance,
  food, potential breeding partners, energy, fear, hunger, and curiosity, and
  selects actions such as evasion, eating, or breeding. The paper describes
  individual variation and species emergence.
- This is an established evolutionary-computation approach to individual
  behavior and ecological adaptation, but the abstract does not identify
  social preference evolution, multi-agent reinforcement learning, or the
  target's action-bias design.

### Social information in predator–prey ecology

Zoltán Tóth and Gabriella Kőmüves (2022), “Social information-mediated
population dynamics in non-grouping prey,” *Behavioral Ecology and
Sociobiology* 76 (article 87).
[DOI: 10.1007/s00265-022-03215-4](https://doi.org/10.1007/s00265-022-03215-4).

- This individual-based model allows predator-avoidance behavior to spread
  among randomly moving prey through nearby observers, without cohesive prey
  groups. The authors report that social-information use could increase
  equilibrium prey abundance in most examined scenarios, with detection
  probability and fitness costs among the model parameters.
- It is directly relevant to ecology-mediated effects of social information,
  but it models the spread/use of avoidance information rather than evolution
  of a continuous attraction/avoidance preference, and is not a MARL
  benchmark.

## Empirical ecological grounding

### Predation risk varies with prey position and alignment in schools

Jolle Wolter Jolles, Matthew M. G. Sosna, Geoffrey P. F. Mazué, Colin R.
Twomey, Joseph B. Bak-Coleman, Daniel I. Rubenstein, and Iain D. Couzin
(2022), “Both prey and predator features predict the individual predation
risk and survival of schooling prey,” *eLife* 11:e76344.
[DOI: 10.7554/eLife.76344](https://doi.org/10.7554/eLife.76344).

- High-resolution tracking of northern pike attacking golden shiners found
  that both predator and prey spatial/kinematic features predicted individual
  targeting and survival. Fish in central school positions but relatively
  far from, and less aligned with, neighbors were more likely to be targeted;
  the abstract reports that 70% of attacks were successful.
- This is empirical support that individual social configuration can alter
  predator risk. It does not measure preference inheritance or evolutionary
  change, so it is useful for grounding social-neighborhood metrics and
  ecological mechanisms rather than claiming preference evolution.

### Cooperative and individual prey defenses across experimental evolution

Magali de la Cruz Barrón, Ellen van Velzen, Uli Klümper, Markus Weitere,
Thomas Ulrich Berendonk, and David Kneis (2023), “Shifts from cooperative to
individual-based predation defense determine microbial predator-prey
dynamics,” *The ISME Journal* 17(5):775–785.
[DOI: 10.1038/s41396-023-01381-5](https://doi.org/10.1038/s41396-023-01381-5).

- The study followed bacterivorous flagellates and *Pseudomonas putida* in
  co-culture for five weeks (about 35 predator generations; 16 replicates).
  Its abstract reports a consistent succession from an effective cooperative
  defense based on toxic metabolites to other bacterial defenses.
- This is empirical experimental-evolution evidence that predator–prey
  ecological dynamics can involve transitions between cooperative and
  individual defense. It concerns microbial defense mechanisms, not
  individual movement preferences or MARL.

### Empirical spatial predator–prey selection

Abigail M. Weber, Nicole T. Gorman, Michael E. Egan, Michael W. Eichholz,
Peter E. Schlichting, Daniel J. Skinner, and Guillaume Bastille-Rousseau
(2026), “Unveiling the predator–prey spatial game using multiple habitat
selection functions,” *Ecosphere* 17(5).
[DOI: 10.1002/ecs2.70660](https://doi.org/10.1002/ecs2.70660).

- Using fawn kill locations and GPS locations for bobcats, coyotes, and female
  white-tailed deer, the authors develop a multi-perspective framework
  distinguishing landscape-level predation risk, prey-relative risk,
  predator success, and the landscape of fear. Their abstract reports that
  elevated risk was driven primarily by predator efficiency and sometimes by
  female deer use of risky fawn habitat.
- This helps ground the ecology as a coupled spatial decision problem rather
  than simply pairwise distance minimization. It is an empirical habitat
  selection study, not an evolutionary algorithm or MARL work.

## Social structure, information, and feedback framing

Maurício Cantor, Adriana A. Maldonado-Chaparro, Kristina B. Beck, Hanja B.
Brandl, Gerald G. Carter, Peng Ju He, Friederike Hillemann, James A.
Klarevas-Irby, Mina Ogino, Danai Papageorgiou, Lea Prox, and Damien Roger
Farine (2021), “The importance of individual-to-society feedbacks in animal
ecology and evolution,” *Journal of Animal Ecology* 90(1):27–44.
[DOI: 10.1111/1365-2656.13336](https://doi.org/10.1111/1365-2656.13336).

- This review describes a feedback loop: individual social decisions create
  social structure, which shapes exposure to resources, risks, predators,
  competitors, and cooperators, and in turn shapes selection on individual
  behavior. It notes that relatively few studies connect these levels.
- It supplies a strong conceptual basis for a model in which preference
  changes alter neighborhood structure and predation risk, which then
  differentially affects survival and reproduction. As a review, it is not
  evidence of a particular implemented model.

Liisa Hämäläinen, Hannah M. Rowland, Johanna Mappes, and Rose Thorogood
(2021), “Social information use by predators: expanding the information
ecology of prey defences,” *Oikos* 130(11):1867–1881.
[DOI: 10.1111/oik.08743](https://doi.org/10.1111/oik.08743).

- This review synthesizes evidence for social information among predators
  across encounter, detection, identification, approach, subjugation, and
  consumption, and discusses how this may change selection pressures on prey
  defenses.
- It is a useful reminder that a model's social interaction should not be
  restricted to prey sociality if the model's predator agents can also use
  social information. It is a review, not a computational implementation.

## MARL benchmarks and open implementations

### Aquarium — closest open continuous-space predator–prey MARL environment

Michael Kölle, Yannick Erpelding, Fabian Ritz, Thomy Phan, Steffen Illium,
and Claudia Linnhoff-Popien (2024), “Aquarium: A Comprehensive Framework for
Exploring Predator-Prey Dynamics Through Multi-Agent Reinforcement Learning
Algorithms,” in *Proceedings of the 16th International Conference on Agents
and Artificial Intelligence (ICAART), Volume 1*, pp. 59–70.
[DOI: 10.5220/0012382300003636](https://doi.org/10.5220/0012382300003636);
[open implementation: `michaelkoelle/marl-aquarium`](https://github.com/michaelkoelle/marl-aquarium).

- The paper and repository describe a PettingZoo-compatible predator–prey
  environment on a two-dimensional edge-wrapping plane, with physics-based
  movement and configurable observations, rewards, prey replication, and
  predator starvation. The paper's demonstration trains prey with PPO and
  reports better coordination/sample efficiency from parameter sharing than
  individual learning in its preliminary experiments.
- This is the closest open MARL environment found for the ecological and
  continuous-space parts of the target. It makes a strong implementation
  reference, but the published work does not introduce inherited social
  preferences or evolutionary adaptation of them.

### Melting Pot — social interaction/generalization benchmark

Joel Z. Leibo, Edgar A. Dueñez-Guzman, Alexander Vezhnevets, John P. Agapiou,
Peter Sunehag, Raphael Koster, Jayd Matyas, Charlie Beattie, Igor Mordatch,
and Thore Graepel (2021), “Scalable Evaluation of Multi-Agent Reinforcement
Learning with Melting Pot,” *Proceedings of the 38th International Conference
on Machine Learning*, PMLR 139:6187–6199.
[PMLR paper and citation](https://proceedings.mlr.press/v139/leibo21a.html);
[open implementation](https://github.com/google-deepmind/meltingpot).

- Melting Pot supplies social-dilemma and other social-interaction scenarios
  (including reciprocity, resource sharing, and task partitioning) and
  evaluates generalization to unfamiliar agents/situations. The repository
  currently describes more than 50 substrates and over 256 test scenarios.
- It is a useful complementary benchmark for testing social adaptation and
  generalization, but is not a continuous-space predator–prey evolutionary
  ecosystem.

### Multi-Agent Particle Environments — continuous particle-world MARL

Ryan Lowe et al. (2017; revised 2020), “Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments,” arXiv:1706.02275.
[Paper](https://arxiv.org/abs/1706.02275);
[original particle environment repository (archived)](https://github.com/openai/multiagent-particle-envs);
[maintained PettingZoo MPE implementation](https://pettingzoo.farama.org/environments/mpe/).

- The paper studies multi-agent actor-critic learning in cooperative and
  competitive tasks. The original implementation describes a particle world
  with simulated physics, continuous observations, and discrete actions; its
  repository now points users to PettingZoo for maintained environments.
- It is a useful MARL baseline for continuous spatial interaction and
  coordination, but it does not represent long-run ecological inheritance,
  reproduction, or preference evolution as described in the target research
  question.

## What is established vs. what could be novel

| Component | Status in prior work | Implication for the target |
|---|---|---|
| Spatial predator–prey interaction and ecologically meaningful movement | Established in individual-based eco-evolutionary models and empirical spatial-risk studies (Netz et al. 2021; Weber et al. 2026). | The spatial ecological setting itself is not novel. |
| Social aggregation/empathy affecting escape from predators | Established computationally in the Huang et al. caribou model; empirical studies also show that position/alignment within schools predicts attack risk (Jolles et al. 2022). | Novelty should not be claimed for social movement or predator-risk effects alone. |
| Cooperative defense evolving in predator–prey settings | Established in ecological experimental evolution and social-information models (de la Cruz Barrón et al. 2023; Tóth & Kőmüves 2022). | The contribution must specify what social trait is selected and how it changes over generations. |
| MARL in customizable continuous predator–prey ecology | Aquarium is a published, open PettingZoo environment with a PPO demonstration. | MARL predator–prey simulation is not novel by itself. |
| Social preferences as a heritable and evolving individual trait, coupled to ecological fitness and population feedback | No exact match located in this search. Closest computational studies evolve behavior, movement strategies, or defense, not clearly a separate inherited scalar preference trait; Aquarium is an RL environment rather than preference-evolution research. | This is the most defensible candidate novelty, if implemented and demonstrated with explicit inheritance, mutation/variation, selection, and ecological consequences. |

The phrase “social preference” needs operational definition. The existing
target module assigns attraction/avoidance values externally and applies them
as post-policy directional logit biases. In a future evolutionary extension,
the potential novelty would be that this independently represented parameter
is inherited (or otherwise evolves), rather than merely that the resulting
agents exhibit socially aligned motion. The existing README explicitly says
there are no social genes or preference inheritance yet.

## Research implications and suggested comparisons

1. **Position the question narrowly:** “Does selection on an inherited
   conspecific attraction/avoidance trait produce stable ecological and
   behavioral outcomes?” This distinguishes a trait evolving under selection
   from Huang et al.'s selection on escape strategy with empathy incorporated.
2. **Use Aquarium as a direct environment reference**, not as evidence that
   social-preference evolution has already been solved. Compare boundary
   handling, observations, action interface, prey reproduction, and ecological
   reward design before adopting components.
3. **Measure preference and behavior separately.** Track preference values,
   ancestry, births/deaths, reproductive success by preference, and
   observation-conditioned action alignment; separately track predator/prey
   abundance, survival, starvation, spatial association, and persistence.
   The ecological feedback framing is consistent with Cantor et al.'s
   individual-to-society loop.
4. **Separate selection on social preference from selection on policy.**
   A useful experimental contrast is fixed frozen policies with assigned
   preferences; preferences evolving while policy is fixed; then joint
   policy/preference adaptation. This helps identify which evolving component
   accounts for any observed change.
5. **Use independent evolutionary replicates.** Report genotype/trait
   distributions and lineage-level effects as well as population means;
   individual decisions within one trajectory are not independent evolutionary
   replicates. The current module's README already cautions that its small
   matched-seed fixed-policy runs do not establish fitness or stable groups.
6. **Keep empirical claims distinct from simulations.** The schooling-fish
   and microbial experimental-evolution papers give empirical grounding for
   predator risk and cooperative defense. They do not validate a synthetic
   attraction/avoidance gene or a learned policy.

## Search scope and limitations

Searches used OpenAlex scholarly metadata/abstracts, DOI resolver and
publisher/venue pages (including J-STAGE and PMLR), and GitHub repository
search/pages. Search concepts included spatial predator–prey social behavior,
evolving empathy/caribou, predator–prey movement coevolution, social
information and defense, ecological MARL environments, and evolving social
preferences. Citation details and claims above were checked against primary
publisher/venue pages where available and indexed metadata for abstracts or
bibliographic fields otherwise. Some publisher pages were blocked to direct
fetching; no claims depending on inaccessible full-text details are made.

An absence result is necessarily limited by search terms, index coverage, and
whether a paper describes preference as empathy, sociality, or a movement
strategy rather than “social preference.” The 2018 Huang et al. paper is
therefore especially important when framing any novelty statement. The
searched sources support “no exact match found here,” not “the first ever.”
