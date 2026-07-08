# Exact Reproduction Deviations

Source target:

- Leibo et al. (2019), *Malthusian Reinforcement Learning*, AAMAS 2019: https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf

Purpose: keep a single audit table for all known differences between this module and a strict article reproduction. A run should not be described as an exact reproduction while any OPEN item below affects the claimed result.

Status legend:

- ALIGNED: implemented and checked against the intended exact protocol.
- LOCKED: implementation choice is frozen for exact runs, but still needs paper-level validation or broader outcome validation.
- OPEN: not implemented, not verified, or known to differ.

## Deviation Table

| Area | Current Code | Exact-Reproduction Status | Required Action |
|---|---|---|---|
| Population feedback mechanism | `phi` updates `mu`, and `mu` changes reset allocation by species/island. | ALIGNED at mechanism level | Keep regression tests green for `phi` and `mu`. |
| Strict fitness signal | Strict mode computes `phi` from per-agent cumulative return only. | ALIGNED | Paper Section 2.3 defines individual fitness as cumulative reward and island fitness as a species mean. |
| Allocation update | Exact protocol uses multiplicative/logit update with article alpha/eta values for the selected Allelopathy variant. | LOCKED | Current update is a practical RLlib/env translation of the paper's policy-gradient update over softmax weights. |
| Parallelism | `MuServer` Ray Actor (`utils/mu_server.py`) holds global mu and fires the ecological update once all NI island phi reports arrive. Set `ARTICLE_DISTRIBUTED_RUNNERS=NI` (e.g. 60) to run one env runner per island. Falls back to single-process mode (original behaviour) when the env var is unset. | ALIGNED | Section 2.4: "The island simulation and the species neural network updates were implemented as separate processes." Each RLlib env runner handles one island; V-trace batches from all runners are merged asynchronously by the learner, matching the paper's separate-process circular queue. The paper ran on multiple physical machines; this runs on one machine with multiple Ray workers. The concurrency model is equivalent; hardware scale is not — this is a resource difference, not an algorithmic one. |
| Learner algorithm | Exact trainer uses APPO with V-trace enabled and RMSProp optimizer settings from the article where published. | LOCKED | Learning rate and entropy coefficient remain frozen derived choices because the article reports ranges, not plotted run samples. |
| V-trace clipping | Exact config locks both V-trace thresholds to `1.0`. | ALIGNED | Paper Section 2.4 states V-trace truncation levels are set to 1. |
| KL loss | Exact config disables KL loss and sets `kl_coeff=0.0`. | LOCKED | Paper does not describe an APPO KL penalty; this is locked to avoid PPO-style drift. |
| Recurrent policy | Exact config enables LSTM with `max_seq_len=20`, `lstm_cell_size=64`, and paper-style 16-channel conv/32-MLP module config. | ALIGNED where published | Observation tensor and action semantics still come from PPG, not the paper games. |
| Observation orientation | Article task environments (`ArticleAllelopathyEnv`, `ArticleClamityEnv`) maintain agent facing direction (N/E/S/W) and rotate the 15×15 egocentric observation window to follow it. | ALIGNED | Section 2.4 states the observation window follows the agent's orientation. Turn actions (5=turn_left; 6=turn_right for Allelopathy, 6=settle for Clamity) and movement actions 1–4 update facing. |
| Agent visibility in Allelopathy | Channel 2 of the Allelopathy observation shows all agents on the same island (not just self at the center pixel). Clamity already showed all agents in channel 0. | ALIGNED | Section 2.4 describes an RGB observation window rendered from the environment; other agents must be visible for the game to function as a social dilemma. |
| Cell-exclusion collision | Both article task environments resolve movement conflicts in random agent order; an agent attempting to enter an occupied cell is blocked and stays in place (facing still updates). | ALIGNED | Lab2D cell exclusion is standard for all substrate games in the paper's series. |
| Allelopathic suppression direction | Type-A shrub growth uses the base growth probability with no suppression; type-B growth is divided by (1 + count of nearby type-A shrubs). | ALIGNED | "Allelopathy" is biologically one-directional (A suppresses B). The paper's game name and Section 3.2 motivate the asymmetry: type-B is rewarding but ecologically constrained by type-A. |
| Environment protocol | `config/config_paper_protocol.py` freezes the mapped PPG paper protocol, and `article_tasks.py` implements text-grounded Clamity/Allelopathy reconstructions. | PARTIAL | Not article-exact until unpublished task constants are recovered from source/supplement. |
| Island/task layout | Current mapped protocol is a 25x25 four-island wall cross (PPG mapped env). Article task envs use a 32×32 open grid per island. | OPEN | Paper Allelopathy uses NI=60 at article scale; the PPG layout is an adaptation. The 32×32 dimension is derived from the related Melting Pot substrate, not published in the 2019 paper. |
| Rewards and energy | Article Allelopathy reconstruction implements repeated-harvest reward caps and switching-cost counts; Clamity reconstruction implements settling, shell growth (restricted by adjacency), nutrient patches, and shell-health reward suppression. | PARTIAL | `base_filter_reward_rate=0.01` and `shell_max_radius=4` derived from Figure 2(E). Shell growth restriction by adjacent shells now cited (Section 3.1: "also restricted"). Nutrient patch coordinates remain estimated from Figure 2(A). Shell-intersection geometry (L∞ metric) is a reconstruction default. |
| Article condition presets | `config_article_protocol.py` defines named dynamic, homogeneous, fixed-population, single-agent, and solitary-evaluation conditions. | ALIGNED | Clamity dynamic `NI=30` is now derived (M=960 ÷ 32 = 30, same logic the paper gives for Allelopathy). All other cited conditions are locked. |
| Clamity solitary evaluation | Article-task envs can add one solitary eval island per species and log `malthusian/solitary_return/<species>` without including it in `phi -> mu`. | LOCKED | This matches the paper protocol structure, but exact outcome reproduction still needs original Clamity task constants. |
| Article matrix coverage | Acceptance reports include `article_condition_coverage` and require every named article condition to have enough seeds before matrix coverage passes. | LOCKED | Article-exact acceptance still remains false until original environment constants/source exist. |
| Run metadata integrity | Exact/article trainers persist git status, package versions, and config checksums; evaluator recomputes checksums and fails metadata-integrity checks if invalid. | LOCKED | Older runs without the new metadata schema must be rerun before they can pass the stricter acceptance gate. |
| Official related environment source | DeepMind Melting Pot includes `allelopathic_harvest`, audited at `main` and `v1.0.4`. | OPEN for exact 2019 reproduction | This is a later related substrate, not the 2019 two-shrub Allelopathy task: it uses 3 berry types, 16 players, 2000 timesteps, planting/zapping actions, reward-most-tasty 2, and other-berry reward 1. |
| Official simulator repository | DeepMind Lab2D was audited at commit `0947443`. | OPEN for exact 2019 reproduction | It contains the simulator platform, but no public Clamity or 2019 two-shrub Allelopathy task implementation was found. |
| Evaluation metrics | `evaluate_exact_reproduction.py` emits paper-like CSVs, plots, switching-cost plots, and acceptance reports. | LOCKED | Literal acceptance still blocked by unavailable original constants and seeds. |
| Multi-seed protocol | `scripts/run_exact_reproduction_seeds.py` runs exact protocol seeds and then evaluates them. | LOCKED | Use at least three completed seeds for mapped-protocol acceptance. |
| Reproducibility metadata | `run_config.json` saves env/trainer config, citation map, git commit, and config checksum. | LOCKED | Package version snapshot can still be added if needed for publication artifacting. |

## Current Claim Boundary

This module can currently claim:

- implemented core Malthusian population feedback,
- locked exact APPO/V-trace entrypoint,
- deterministic unit coverage for strict `phi` and multiplicative `mu`,
- APPO invariant coverage for exact mode,
- frozen mapped paper protocol,
- runnable text-grounded Clamity and Allelopathy reconstructions,
- orientation-following 15×15 egocentric observation windows with turn actions (Section 2.4),
- all same-island agents visible in the Allelopathy observation (social dilemma requirement),
- cell-exclusion collision detection with random-order conflict resolution in both article task environments,
- one-way allelopathic suppression (A suppresses B, not symmetric),
- policy-mapping drift tests for both PPG and article-task trainers,
- same-seed determinism test for article task environments,
- named article-condition presets and Clamity solitary evaluation metrics,
- distributed island training via `MuServer` Ray Actor (one runner per island, matching Section 2.4 multi-process architecture; activated with `ARTICLE_DISTRIBUTED_RUNNERS=NI`),
- article matrix coverage checks in acceptance reports,
- run metadata integrity checks with package snapshots and config checksum verification,
- Figure 2/Figure 3 family CSV and PNG summaries for condition-level reconstructed outcomes,
- paper-like evaluator, plots, multi-seed harness, and mapped-protocol acceptance bands.

It should not yet claim:

- exact reproduction of all article experiments,
- matching article figures,
- statistically reproduced article outcomes.

Blocking reason: the paper does not publish enough task constants to truthfully claim a literal Figure 2/Figure 3 reproduction. A related official DeepMind Melting Pot Allelopathic Harvest source exists, but it is a later, non-identical substrate and cannot be substituted for the 2019 article task. The official Lab2D repository contains the simulator platform, but not these 2019 tasks. No public 2019 Clamity or two-shrub Malthusian Allelopathy source/supplement was found in the checked refs. Missing constants are listed in `config/config_article_protocol.py` under `ARTICLE_EXACT_BLOCKERS`.
