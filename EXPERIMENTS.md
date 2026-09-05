# Experiments and environments

Full catalogue of environments and experiments referenced from the [README](README.md#start-here).

This repo splits into two structurally different families of experiment, matching the
`predpreygrass/evolutionary/` vs `predpreygrass/non_evolutionary/` directory split:

- **Evolutionary**: agents carry a heritable genome trait, passed parent → offspring
  with mutation. What gets selected is discovered, not designed.
- **Non-evolutionary**: every agent trait is fixed; only the RL policy adapts. What
  emerges is a behavioral equilibrium under a given incentive design, not a change in
  the population's genetics.

## Darwinian/Baldwinian evolutionary environments

These environments layer a genuine evolutionary algorithm — founder genome, mutation, inheritance — on top of shared-policy PPO. Learned behavior (Baldwinian) determines which trait values survive to reproduce, closing a genome → phenotype → learned behavior → fitness → genome-frequency loop across generations. See **[predpreygrass/evolutionary/README.md](predpreygrass/evolutionary)** for the shared goal, success criteria, and cross-module trial log — start there before any individual module below.

* **[Eco-evolutionary](predpreygrass/evolutionary/eco_evolutionary)**: baseline of the family. Evolves a `speed` trait that sets a movement-distance threshold (1 vs. 2 tiles per move).

* **[Eco-evolutionary cadence](predpreygrass/evolutionary/eco_evolutionary_cadence)**: evolves the same `speed` trait, expressed as a graded movement cooldown instead of a discrete distance threshold.

* **[Eco-evolutionary cooperation](predpreygrass/evolutionary/eco_evolutionary_cooperation)**: evolves a `cooperation_rate` trait — the fraction of an agent's net energy gain donated to nearby same-species agents, relying on spatial viscosity (offspring spawn near parents) for implicit kin selection.

* **[Eco-evolutionary investment](predpreygrass/evolutionary/eco_evolutionary_investment)**: evolves an `offspring_investment_fraction` trait — how much energy a parent hands each offspring at birth.

* **[Eco-evolutionary metabolic rate](predpreygrass/evolutionary/eco_evolutionary_metabolic_rate)**: evolves a `metabolic_rate` trait that symmetrically scales both energy gain and basal energy cost.

* **["Stag hunt" nature + nurture](predpreygrass/evolutionary/stag_hunt_forward_view_nature_nurture)**: a hybrid case — predators carry a heritable cooperation trait (nature) alongside the learned voluntary `join_hunt` action (nurture); team-capture success depends on both.

* **[Eco-evolutionary metabolic code](predpreygrass/evolutionary/eco_evolutionary_metabolic_code)**: replaces the earlier single continuous-scalar traits with a combinatorial, needle-in-haystack metabolic code, testing whether selection can find a rare high-fitness combination that smooth-scalar traits couldn't drift toward.

* **[Eco-evolutionary metabolic rate — positive control](predpreygrass/evolutionary/eco_evolutionary_metabolic_rate_positive_control)**: a deliberate positive control — clones `eco_evolutionary_metabolic_rate` with a sharpened, super-linear fitness gradient, to check whether the pipeline can detect selection-driven drift at all when the advantage is overwhelming.

* **[Eco-evolutionary cultural plasticity](predpreygrass/evolutionary/eco_evolutionary_cultural_plasticity)**: gene-culture (dual-inheritance) coevolution — a heritable `plasticity` trait gates how readily an agent adopts a locally-shared, non-genetic `dialect`.

* **[Eco-evolutionary cultural plasticity, seasonal](predpreygrass/evolutionary/eco_evolutionary_cultural_plasticity_seasonal)**: the cultural-plasticity trait under a dialect that periodically flips target, testing whether an external, time-varying "correct answer" (rather than a self-referential local majority) gives `plasticity` something to win by tracking.

* **[ERL Baldwin](predpreygrass/evolutionary/eco_evolutionary_erl_baldwin)**: a structurally different architecture — each agent gets its own genome-conditioned policy network, rather than a single shared-policy scalar side-channel. The project's strongest confirmed result: ERL significantly outperforms the prior shared-policy trials (p < 0.00001).

* **[Nuptial-gift giving](predpreygrass/evolutionary/eco_evolutionary_nuptial_gift)**: sexed predators with obligate male provisioning — males hunt but never reproduce directly, females can never sustain themselves on grazing alone and depend on a male-to-female energy gift to reproduce.

## Fixed-trait behavioral & game-theoretic environments

These environments hold every agent trait fixed and instead vary the interaction mechanics or reward shaping. Agents are still born, reproduce, and die, but nothing is inherited or mutated — only the RL policy adapts, converging on a behavioral equilibrium (cooperate, defect, share, reciprocate) under a given incentive design.

* **[Base environment](predpreygrass/non_evolutionary/base_environment)**: the two-policy base environment. Only reproduction rewards. ([results](https://humanbehaviorpatterns.org/pred-prey-grass/overview-ppg))

* **[Base environment, seasonal](predpreygrass/non_evolutionary/base_environment_seasonal)**: same mechanics as the base environment, plus a seasonal grass-regrowth cycle (a square wave alternating "abundant" and "scarce" phases over the episode) instead of a flat regrowth rate.

* **Reward shaping**: five sibling environments comparing sparse vs. dense reward design — see the [README headline result](README.md#headline-result-sparse-rewards-beat-dense-rewards) for the summary finding. See **[predpreygrass/non_evolutionary/project_reward_shaping/README.md](predpreygrass/non_evolutionary/project_reward_shaping)** for the shared methodology and full results log — start there before any individual module below.

  * **[Sparse rewards](predpreygrass/non_evolutionary/project_reward_shaping/base_environment_sparse_rewards)**: same sparse, reproduction-only reward as the base environment. The fair baseline for every other variant below.

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

* **[Drive-conditioned environment](predpreygrass/non_evolutionary/drive_conditioned_environment)**: starts as a copy of the base environment, adding biologically-motivated internal-state signals (hunger, reproductive readiness, local threat/opportunity) as extra observation channels — reward and action space stay unchanged, only what the agent can see is enriched. See that folder's README for the full rationale and predicted effects.

* **[Red Queen](predpreygrass/non_evolutionary/red_queen)**: independently configurable competing prey types under a shared, non-mutating predator policy, testing coevolutionary arms-race dynamics between learned policies rather than genomes.

## Experiments

* Testing the **Red Queen Hypothesis** in the co-evolutionary setting of (non-mutating) predators and prey ([original implementation](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/non_evolutionary/red_queen/evaluate_red_queen_freeze_type_1_only.py), [results](https://humanbehaviorpatterns.org/pred-prey-grass/red-queen/)). A stronger multi-seed, multi-checkpoint-pair evaluation harness plus a training script (previously missing) were added later — see [`red_queen/README.md`](predpreygrass/non_evolutionary/red_queen) for the current methodology and what's still needed to produce a new result.

* Testing the **Red Queen Hypothesis** in the co-evolutionary setting of mutating predators and prey — this earlier implementation predates the evolutionary/non-evolutionary directory split and isn't in the active source tree; see the [legacy archive](https://github.com/doesburg11/PredPreyGrassLegacy) for that codebase.
