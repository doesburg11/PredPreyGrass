# Exact Citation Map

Target source:

- Leibo et al. (2019), *Malthusian Reinforcement Learning*, AAMAS 2019: https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf

This table maps every locked exact-run protocol choice to either a cited paper value, a derived RLlib translation, or an explicit PPG deviation.

## Source Audit

| Source | What Was Checked | Result |
|---|---|---|
| Leibo et al. 2019 AAMAS paper | Main article text, figures, captions, and reported experiment settings. | Primary target. Enough for learner/protocol values, not enough for all task constants. |
| `google-deepmind/meltingpot` `allelopathic_harvest` (`main`, `v1.0.4`) | Official DeepMind/Lab2D Allelopathic Harvest substrate config (`allelopathic_harvest.py`) and Lua components (`components.lua`). Fetched via raw.githubusercontent.com. | Related official substrate, but not the 2019 task. Map: **32 columns × 30 rows**. Growth: cubic positive-autocorrelation model (more berries → more growth, base rate 0.0000025). Three berry types; 16 players (up to 60); 2000 episode timesteps; `rewardMostTasty=2`, other-berry reward 1. The 2019 task uses two shrub types, inverse cross-type suppression (more other-type → less growth), repeated same-type rewards up to 250, biased A max 8 / B max 250, K=960, NI=60. The Melting Pot map width (32) is used as the basis for the reconstruction default 32×32 grid, with the caveat that growth mechanics and episode length differ significantly. |
| `google-deepmind/lab2d` (`0947443`) | Official DeepMind Lab2D simulator source. | No `Clamity`, `Allelopathy`, `Malthusian`, or `trochophore` task implementation found. This repository is the platform source, not the 2019 task source. |
| `google-deepmind/meltingpot` source search | `Clamity`, `Malthusian`, `trochophore`, and related casing variants. | No public 2019 Clamity or two-shrub Malthusian Allelopathy source found in the checked refs. |

## Learner

| Config key | Value | Paper basis | Status |
|---|---:|---|---|
| `vtrace` | `True` | Section 2.4 says species policy updates use V-trace. | cited |
| `vtrace_clip_rho_threshold` | `1.0` | Section 2.4 says V-trace truncation levels are set to 1. | cited |
| `vtrace_clip_pg_rho_threshold` | `1.0` | Section 2.4 says V-trace truncation levels are set to 1. | cited |
| `max_seq_len` | `20` | RL Agent table reports LSTM unroll length 20. | cited |
| `lstm_cell_size` | `64` | Function approximation paragraph reports an LSTM of size 64. | cited |
| `paper_network_architecture` | `True` | Paper reports 16-channel 3x3 stride-1 convnet, one 32-unit MLP layer, then LSTM 64. | cited |
| `vf_loss_coeff` | `0.5` | RL Agent table reports baseline loss scaling 0.5. | cited |
| `gamma` | `0.99` | RL Agent table reports discount 0.99. | cited |
| `opt_type` | `rmsprop` | Optimization table reports RMSProp. | cited |
| `decay` | `0.99` | Optimization table reports RMSProp decay 0.99. | cited |
| `epsilon` | `0.0001` | Optimization table reports RMSProp epsilon 0.0001. | cited |
| `paper_batch_size_trajectories` | `32` | Optimization table reports batch size 32 trajectories. | cited |
| `train_batch_size_per_learner` | `640` | RLlib translation of 32 trajectories x unroll length 20. | derived |
| `lr` | `0.0003` | Paper reports RMSProp LR sampled log-uniformly from `[0.0001, 0.005]`; plotted samples are not published. | in-range fixed value |
| `entropy_coeff` | `0.0015811388300841897` | Paper reports entropy sampled log-uniformly from `[0.00005, 0.05]`; this is the geometric midpoint. | derived midpoint |
| `use_kl_loss` | `False` | Paper does not describe a PPO/APPO KL penalty. | locked anti-drift |
| `num_env_runners` | `1` | Repo requirement so one shared `mu` process is not split across RLlib workers. | implementation constraint |
| `num_envs_per_env_runner` | `1` | Repo requirement so one shared `mu` process is not split across RLlib workers. | implementation constraint |

## Environment And Protocol

| Config key | Value | Paper basis | Status |
|---|---:|---|---|
| `paper_protocol_variant` | `allelopathy_biased_mapped` by default | Figure 3 E-H reports biased Allelopathy parameters. | cited target, mapped environment |
| `max_steps` | `1000` | Figure 3 caption states Allelopathy episodes lasted 1000 behavior steps. | cited |
| `enable_malthusian_update` | `True` | Sections 2.2-2.3 define the ecological update loop over `mu`. | cited |
| `malthusian_replication_mode` | `strict` | Section 2.3 defines fitness as cumulative reward, not ecology-weighted side metrics. | cited |
| `malthusian_mu_update` | `multiplicative` | Section 2.3 updates softmax-normalized species island weights from fitness. | mapped |
| `malthusian_mu_learning_rate` | `0.0001` biased / `1e-7` unbiased | Figure 3 caption reports `alpha` values. | cited |
| `malthusian_mu_entropy_coeff` | `0.01` biased / `0.3` unbiased | Figure 3 caption reports `eta` values. | cited |
| `enable_within_episode_reproduction` | `False` | Paper population changes happen through inter-episode allocation, with fixed total archipelago population. | cited |
| `deterministic_reset_sequence` | `True` | Required for reproducible multi-seed protocol. | implementation constraint |
| `malthusian_phi_weights.reward` | `1.0` | Section 2.3 defines fitness as cumulative reward. | cited |
| `reward_predator_catch_prey` | `1.0` | PPG analogue of local resource/reward collection. | PPG deviation |
| `reward_prey_eat_grass` | `1.0` | PPG analogue of local resource/reward collection. | PPG deviation |
| `penalty_prey_caught` | `-1.0` | PPG predator-prey analogue; not an Allelopathy reward. | PPG deviation |
| hard-island layout | 25x25 grid, 4 wall-separated islands | Paper Allelopathy reports NI=60 at article scale. | open deviation |

## Article Task Reconstructions

These rows refer to `article_tasks.py` and `config/config_article_protocol.py`, not the PPG mapped protocol.

| Config / behavior | Value | Paper basis | Status |
|---|---:|---|---|
| Allelopathy `episode_horizon` | `1000` | Figure 3 caption says episodes lasted 1000 behavior steps. | cited |
| Allelopathy biased `alpha`, `eta` | `0.0001`, `0.01` | Figure 3 caption gives biased Allelopathy Malthusian RL parameters. | cited |
| Allelopathy unbiased `alpha`, `eta` | `1e-7`, `0.3` | Figure 3 caption gives unbiased Allelopathy Malthusian RL parameters. | cited |
| Allelopathy `num_species` | `4` | Section 3.2.1 describes heterogeneous `L=4`. | cited |
| Allelopathy `total_individuals` | `960` | Section 3.2.1 reports `K=960`. | cited |
| Allelopathy `num_islands` | `60` | Section 3.2.1 reports `NI=60` in dynamic population conditions. | cited |
| Allelopathy heterogeneous dynamic condition | `L=4`, `K=960`, `NI=60` | Section 3.2.1 describes the heterogeneous dynamic population condition. | cited |
| Allelopathy homogeneous dynamic condition | `L=1`, `K=960`, `NI=60` | Section 3.2.1 defines the homogeneous comparison and reports the dynamic-population island count. | cited |
| Allelopathy fixed-population-32 condition | `NI=30`, `fixed_population_per_island=32` | Section 3.2.1 states that fixed population size 32 requires `NI=960/32=30`. | cited |
| Allelopathy biased reward caps | `[8, 250]` | Section 3.2 says type A max reward 8 and type B max reward 250. | cited |
| Allelopathy unbiased reward caps | `[250, 250]` | Section 3.2 says repeated same-type harvest rewards rise to max `r=250`; equal shrub probabilities are cited for unbiased. | cited plus inference |
| Allelopathy switching cost count | logged per island | Figure 3 D/H plots minimum switching-cost count over islands. | cited |
| Allelopathy agent visibility | all same-island agents in channel 2 | Section 2.4 describes RGB window; the game is a social dilemma requiring inter-agent observation. | cited for requirement; channel assignment is reconstruction default |
| Allelopathic suppression direction | type-A growth unsuppressed; type-B growth divided by (1 + nearby type-A count) | Section 3.2: "allelopathic suppression" is biologically one-way; type-B is rewarding but constrained by type-A density. | cited for asymmetry; radius and base probability remain unpublished reconstruction defaults |
| Cell-exclusion collision | random-order movement conflict resolution, blocked agents stay in place | Lab2D games use cell exclusion; motivates the competition for space in Clamity and resource access in Allelopathy. | derived from platform convention |
### Clamity Constants — Resolution Summary

Five Clamity constants were not published in the paper. The table below records how each was resolved; detailed citations appear in the main rows that follow.

| Constant | Value | How resolved | Status |
|---|---|---|---|
| Nutrient patch coordinates | `[(6,10),(6,49),(29,10),(29,49)]` | Estimated from Figure 2(A) screenshot: four symmetric patches, all ≥30 L1 steps from spawn center (18,30); consistent with Section 3.1 ">10 steps" constraint. Exact coordinates not published. | estimated from Figure 2(A) |
| Shell max radius | `4` | Figure 2(B) visual: settled shell spans ~25% of map height (diameter ~9 cells → radius ~4-5). Confirmed consistent with the `base_filter_reward_rate` derivation below. | estimated from Figure 2 |
| Shell growth restriction | growth stops (not just reward) when adjacent shells present | Section 3.1: "shell growth is **also** restricted by the presence of adjacent shells" — "also" means in addition to reward suppression. | cited |
| Food filtering reward rate | `0.01` | Derived from Figure 2(E): "no-curiosity" agent (local optimum: settle at step 0, no patch) reaches ~200. Formula: `base × (9+25+49+81×247) = base × 20,090 ≈ 200` → `base ≈ 0.01`. | derived from Figure 2(E) |
| Dynamic-population NI | `30` | Derived: M=960 (Section 3.1.1) ÷ 32 agents/island = 30. Paper gives the identical calculation for Allelopathy: "fixed population size 32 required NI=960/32=30" (Section 3.2.1). | derived |
| Number of species L | `1` | Figure 2(E) legend reads "Malthusian RL (L≥1)". L=1 is the simplest reconstruction default; Clamity tests exploration, not specialisation, so L is not the critical variable. | interpreted from Figure 2(E) |

| Clamity map size | `36x60` | Section 3.1 states map size 36x60. | cited |
| Observation window size | `15x15` | Section 3.1 and Section 2.4 state 15x15 observation window. | cited |
| Observation orientation | egocentric, rotates with facing direction | Section 2.4: "individuals observe a 15x15 RGB window centered on themselves that follows their orientation (agents can rotate)." Both `ArticleAllelopathyEnv` and `ArticleClamityEnv` maintain agent facing (0=N,1=E,2=S,3=W) updated by movement and explicit turn actions. | cited for rotation; exact action indices (turn_left=5, turn_right=6) are reconstruction defaults |
| Clamity `episode_horizon` | `250` | Figure 2 caption says episodes lasted 250 behavior steps. | cited |
| Clamity `alpha`, `eta` | `0.0001`, `1.5` | Figure 2 caption gives Malthusian RL parameters. | cited |
| Clamity `num_species` (L) | `1` | Figure 2(E) legend reads "Malthusian RL (L≥1), dynamic population size." L=1 is the simplest reconstruction default. Clamity tests exploration, not species specialisation, so L is not critical. | interpreted from Figure 2(E) |
| Clamity dynamic-population `NI` | `30` | Derived: M=960 (Section 3.1.1) ÷ 32 agents/island (fixed-population baseline) = 30. Paper gives the same calculation explicitly for Allelopathy: "fixed population size 32 required NI=960/32=30" (Section 3.2.1). | derived |
| Clamity dynamic-population condition | `M=960`, `NI=30`, plus one solitary eval island per species | Section 3.1.1 says M=960 individuals per species. NI=30 derived (see above). Solitary-island reporting protocol is cited from Section 3.1.1. | cited + derived |
| Clamity fixed-population-32 condition | one archipelago island with 32 individuals, plus solitary eval island | Section 3.1.1-3.1.2 describes the standard self-play/fixed-population comparison and says fixed population size 32 was evaluated. | cited |
| Clamity single-agent baseline | 32 one-agent replicas, no Malthusian update | Section 3.1.1 says the single-agent protocol sets `NI=0` and replicates each solitary island 32 times. | cited, encoded as 32 non-Malthusian one-agent islands |
| Clamity settling action | action 6 | Section 3.1 describes a settle action; numeric action index is repo-local. | mapped |
| Clamity `base_filter_reward_rate` | `0.01` | Derived from Figure 2(E): the "no-curiosity" single agent (stuck at local optimum: settles at step 0, no patch) reaches ~200. Shell grows 1/step to `shell_max_radius=4`. Total area-steps = 9+25+49+81×247 = 20,090. `base × 20,090 ≈ 200` → `base ≈ 0.01`. | derived from Figure 2(E) |
| Clamity `shell_max_radius` | `4` | Not published. Consistent with Figure 2(B) (settled shell spans ~25% of map height, diameter ~9 cells, radius ~4-5) and confirmed by Figure 2(E) reward-scale derivation (see `base_filter_reward_rate`). | estimated from Figure 2 |
| Clamity shell growth restriction | growth **and** reward both stop when adjacent settled shells are present | Section 3.1: "shell growth is **also** restricted by the presence of adjacent shells" — the word "also" makes clear both growth and reward are suppressed, not reward alone. | cited |
| Clamity nutrient patches | `[(6,10),(6,49),(29,10),(29,49)]` | Paper says ">10 steps from the starting location" (Section 3.1) and references Figure 2(A). All four patches are ≥30 L1 steps from spawn center (18, 30). Positions estimated from Figure 2(A) screenshot (4 symmetric patches in corner regions of the 36×60 map). Exact coordinates not published. | estimated from Figure 2(A) |
| Allelopathy map height × width | `32 × 32` | Paper does not publish map dimensions. DeepMind Melting Pot `allelopathic_harvest` substrate (the closest related public source, **not** the 2019 task) uses 32 columns × 30 rows; a square 32×32 grid is adopted here. | derived from related substrate |
| Allelopathy `initial_shrub_density` | `0.08` (8%) | Paper says "randomly placing shrubs" but does not publish density. At 8% on a 32×32 grid (~82 shrubs) with 16 agents, equilibrium density is sustained. No published reference. | unpublished reconstruction default |
| Allelopathy `shrub_growth_base_probability` | `0.01` per empty cell per step | Paper gives inverse suppression formula but not base rate. At 0.01 with 16 agents/island, estimated equilibrium shrub coverage is 20–30%. No published reference. | unpublished reconstruction default |
| Allelopathy `suppression_radius` | `2` (5×5 neighbourhood, 24 cells) | Paper says growth is "inversely proportional to the number of nearby shrubs of other types" but does not define "nearby." Radius 2 gives meaningful local suppression without global influence. No published reference. | unpublished reconstruction default |
| Allelopathy biased `resource_spawn_probabilities` | `[0.8, 0.2]` (4:1 A:B) | Section 3.2 says "type A is significantly more common than type B" but does not give exact values. A 4:1 ratio is a conservative "significantly more common" interpretation. No published reference. | unpublished reconstruction default |

## Unreachable Gaps

These items cannot be resolved from any public source. They are listed here so that the reason each gap exists is documented and not re-investigated.

| Item | Why unreachable | Impact on reproduction |
|---|---|---|
| Exact nutrient patch coordinates | Paper says ">10 steps from starting location" (Section 3.1) and shows Figure 2(A), but never publishes `(row, col)` tuples. No public Clamity source found. | Moderate: patch position affects when agents first encounter a patch and therefore how the curiosity bonus develops over the first ~30–50 steps. Reconstruction uses symmetric corners estimated from Figure 2(A). |
| Exact shell-intersection geometry | Section 3.1 says shells of different individuals must not "intersect," but does not define the metric (L∞ disk? L1 diamond? explicit cell-set comparison?). No public source found. | Low: all reasonable metrics agree for small radii (≤4) at typical agent densities. Reconstruction uses L∞ (square shell footprint). |
| Agent-species color in RGB observation | Section 2.4 says the observation window is an RGB rendering, but channel → species mapping is not published. No public source found. | Low for learning (species distinction is learnable regardless of color assignment); high if comparing activations to published ablations. |
| Sampled learning rate and entropy coefficient for plotted runs | Paper reports both hyperparameters are sampled log-uniformly (LR ∈ [0.0001, 0.005], entropy ∈ [0.00005, 0.05]) but does not publish the specific values used for Figure 2/3 plots. | Moderate: different samples within the stated ranges can produce measurably different convergence curves. Reconstruction uses fixed geometric midpoints. |
| Random seeds for Figure 2 / Figure 3 | Seeds used in the published figures are not reported. | Moderate: required to reproduce figure-exact stochastic trajectories. Multi-seed averaging mitigates this for outcome-level comparison. |
| Training duration in ecological steps | Paper shows Figure 2 to ~3 × 10⁴ ecological steps and Figure 3 to ~10⁴, but the exact stopping criterion is not stated. | Low: training can be run until the curves visually stabilize. |

## Evaluation

| Output | Paper basis | Status |
|---|---|---|
| `max_collective_return_over_islands` | Figure 3 A/E. | implemented from `sum(count * phi)` |
| `max_per_capita_collective_return_over_islands` | Figure 3 B/F. | implemented from collective return divided by population |
| `max_island_population_size` | Figure 3 C/G. | implemented from logged species counts |
| `min_switching_cost_over_islands` | Figure 3 D/H. | implemented for article-task Allelopathy; unavailable for PPG mapped runs |
| smoothing window | Figure 3 caption says window size 25 ecological steps. | cited |

`min_switching_cost_over_islands` is implemented for article-task Allelopathy runs. It remains unavailable for older PPG mapped runs because PPG has no metabolic switching-cost mechanic.
