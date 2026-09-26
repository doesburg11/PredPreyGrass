# predator_bands: results so far (2026-09-25 / 09-26)

A consolidated write-up of the `predator_bands` experiments. Details, exact settings and per-run tables are in [`README.md`](README.md); the
designs are in [`BANDS_DESIGN.md`](../predator_complementary_diet/BANDS_DESIGN.md) and [`MAMMOTH_DESIGN.md`](MAMMOTH_DESIGN.md). Everything
here is **one seed per configuration** unless stated, PPO training of 100-300 iterations, and descriptive: no causal claims.

## The question

Predators are the "human" analogue in this project. With sexual reproduction (the earlier module), can a **complementary diet** (a predator
needs both meat and fruit), **band living** (bands of families and singles that share food, marry between bands, avoid incest) and, later,
**group big game** give a viable, differentiated, socially structured population, where the structure is learned and not scripted?

What is built (all designed rules, none learned): a fruit store and a meat store per predator with death or no breeding if either runs out;
initial bands with within-band food sharing (meat and fruit rates set separately), kin exclusion and marriage; scripted prey (5x faster
training, since prey were ~79% of the learner's data); and default-off options tested for cohesion: a band compass in the observation, roaming
threats with group defense, distance-scaled sharing, and mammoths that only a band party can kill.

## 1. Viability: with matched controls, the diet requirement was sustained only when band sharing was wide enough

| Setting (seed 42) | Result |
|---|---|
| Pair-only design (`predator_complementary_diet`), learned prey: meat share 0.25, then 0.10 | Females extinct in ~100% of episodes at 0.25 and ~90-100% at 0.10; the failure moves from meat to fruit deficiency |
| Same design, scripted prey, meat share 0.10 / 0.15 / 0.20 (150 iterations) | Female extinction 93% / 99% / 94% in the last blocks; meat-deficiency deaths dominate |
| Bands, 5 bands, band sharing 0.3 flat, meat share 0.10, fruit regrowth 0.04 (100 iterations) | Extinction 92% early, 67% late; the no-sharing control ~100% |
| Same but fruit regrowth 0.08 (200 iterations) | Extinction 100% at iterations 25-75, 77-85% later; the no-sharing control 96-100% throughout |
| Bands, meat sharing 0.6 / fruit 0.3, meat share 0.10, fruit regrowth 0.08 (seeds 42 and 43) | **Viable and replicated**: extinction 2-12% late, episodes ~970-1000 steps, 6-9 females and ~28 males alive, ~35-40 births per episode |
| Same, `male_joins_female` marriage rule (seed 42) | Viable, more females (12-14) but only ~2.4 of 5 bands survive (the population merges) |

In these runs males caught ~135 prey (~400 energy) per episode and kept most of it, while females hunted little; with a 30% flat share females still
starved of meat, and sharing meat more widely than fruit (0.6 against 0.3) went with viability (the controls with less or no sharing were not viable). This is
an association within matched configurations (viability also changed with fruit regrowth and training length across some comparisons), not a proof that
sharing is necessary in general. Meat share 0.05 sustained females at 0.3 sharing, but that requirement is weak. The sharing rates are designed, so this
shows what the payoffs produce given sharing, not that sharing emerges.

## 2. What the trained populations do (5 bands, seeds 42 and 43, plus the marriage variant)

- **Sex differentiation in foraging emerges.** Males approach and take prey (approach bias +0.17 to +0.22; 16-20% of own foraging is prey); females approach
  fruit like males but barely approach prey (+0.02 to +0.04; 2.4-2.7% prey). Females get 75-80% of their meat from others (males ~22%).
- **Single females had similar observed means to paired ones** (73-79% of their meat from the band; life ~230-295 steps against ~246-295 for founding-couple
  females; pooled descriptive means without an inferential comparison, and many lives are cut off at the episode end). Females of all kinds live about half as
  long as males (~230-260 against ~470-510 steps).
- **How little meat a female took in and still lived:** the rules put the meat drain at 0.010 (idle) to 0.018 (moving) per step at meat share 0.10; the
  lowest observed net meat intake (own hunting + received - given, per step) among females that lived 800+ steps was ~0.016-0.017 (small samples: 11, 26,
  24 females). This is an observed minimum among selected survivors, not a physiological threshold: intake is not consumption, and the 2.5 starting store
  contributes.
- **No sign that bands cohere on their own.** In the first 100 steps the share of decisions with a band-mate in sharing range is 0.78 (male) and 0.85 (female)
  against 0.77 for random movers (a small edge for females only), and males' nearest band-mate is somewhat farther than a random mover's (~4.1 against ~3.5 cells in
  seed 42); both sexes move *away* from any nearby predator, same band or not (approach bias -0.05 to -0.09 in the 5-band runs, up to -0.13 in the 3-band runs). Scatter grows with time in a band (out of sharing range: founders 31-37%, born 5-19%, recent movers 8-12%, movers over 100
  steps 12-25%); marriage is not the main source (a marriage needs a mate within 3 cells).

## 3. Attempts to make bands stay together (3 bands unless noted; control = no threats, no mammoths, flat sharing within 5 cells, 150 iterations)

Cohesion measures (20 episodes, final checkpoint): share of observations with no band-mate within sharing range (founders / born / moved over 100 steps),
late nearest same-band distance and within-range share. Control: 36% / 21% / 28%; 4.5 cells (males), 4.0 (females); 0.72 / 0.81.

| Intervention | Viability | Cohesion vs control |
|---|---|---|
| Band compass (direction and distance to the nearest band-mate) | Viable; more females alive at the end of training episodes (5-8 against 3-4, training-block averages) | Not analysed alone; with mammoths and threats no gain (below) |
| 4 roaming threats, kill chance 0.5 | **Collapse**: episodes ~77-83 steps, ~16 kills per episode, 86-90% of kills of a predator with no defender | Not analysed |
| 1 threat at 0.5, or 2 threats at 0.25 | Collapse (episodes ~185 steps, ~14 kills) | Not analysed |
| Threats that rest after a kill (100 steps) or a failed attack (10), 1 threat, kill chance 0.3 | Viable (episodes ~915, extinction 23%, ~7 kills per episode, 78% lone) | One run looked better (born 13% out of range, late within-range 0.80 / 0.87) |
| Same with wider defense (radius 3, 2 defenders repel; threats driven off 10x as often) | Viable (extinction 38%) | Not better (born 17%; late 0.73 / 0.83): the earlier improvement was **not reproduced** |
| Same as the rest run plus compass | Viable (extinction 33%) | Not better (founders 43%, born 20%; late 0.71 / 0.82) |
| Sharing scaled down with distance (range 5) | **Worse viability**: extinction 85%, ~45 meat shared per episode | Worse (founders 44%, born 29%; nearest band-mate 5.7 / 5.0 cells) |
| Flat sharing within 3 cells | Worse viability (extinction 82%) | Not comparable ("range" changed); nearest band-mate 5.2 / 4.4 |
| Mammoths (2 x 20 energy), solo hunts 2% / deadly | Viable; **no joint hunt learned** (solo attempts ~8 per episode, ~3 deaths) | Not analysed |
| Mammoths with a smoother curve (solo 10%) | Viable; mostly **solo kills** (energy 100-124 per episode), no joint hunt | Not analysed |
| Mammoths, solo useless and harmless (party of 1: 0), compass off | Viable; pairs ~3 attempts per episode, flat | Not better (founders 40%; 4.8 / 3.8; 0.69 / 0.82) |
| Same with compass on (300 iterations) | Viable; **pair-sized hunts increase** (9 attempts, 3.6 kills per episode; mammoth energy 4.6 to 80) | Not better (38% / 21% / 28%; 4.6 / 4.0; 0.70 / 0.79) |
| 4 mammoths of 40 energy, compass on | Viable; pair hunts keep growing (20 attempts, 6.4 kills; ~290 energy per episode, about as much as band sharing) | Not better (42% / 20% / 30%; 4.8 / 3.9; 0.69 / 0.81) |

## 4. What can and cannot be concluded

- **Supported (within one seed per run):** a complementary diet is not sustainable for a pair-only design at any meat share tried, and is sustainable once meat is
  shared widely within bands; a sex differentiation in foraging and a heavy reliance of females on shared meat emerge; a coordinated pair hunt of big
  game increase when solo attempts are pointless and the band compass is on (the data do not show whether this is intentional coordination or attacking when a
  band-mate happens to be adjacent).
- **Not found:** a learned pull toward band-mates. Across the compass, threats (several strengths), wider defense, costlier distance and mammoths of both sizes,
  none clearly improved a cohesion measure over the control (distance-scaled sharing made it worse). The one apparent improvement (threats that rest, first run) did not replicate in the
  two follow-up runs.
- **Limits.** One seed per configuration and 100-300 training iterations, so small effects are not detectable and variation between runs of the same setting is
  not measured. The control is a 150-iteration run while some tests trained for 300 (not perfectly matched). Threat and mammoth parameters were guesses and
  were adjusted after seeing results. Rollout measures mix action preference with the states the policy creates and with selection (predators that stray may die
  before they are measured). Sharing, marriage, kin exclusion, threats and mammoths are designed rules, so results describe what these payoffs produce, not
  that the behaviors would emerge without them. Deaths carry no direct penalty in the reward, so avoidance must be learned from survival alone.
- **Untested:** memory or communication in the policy, several seeds, longer training (500+ iterations), requiring parties of three or more, and defining band
  membership by proximity (see `BANDS_DESIGN.md`, section 9).

## Sources of the numbers

Per-run settings and the behaviour and scatter tables are in `README.md`; the rollout analyses are logged in `~/simulation_results/band_behavior/*.log`
(`behavior_*`, `scatter_*`, `female_meat_*`). Training figures quoted for viability, threat kills, lone-kill shares, repelled counts, shared meat and mammoth
attempts and kills are block averages of the last logged block of the TensorBoard `ecology/*` scalars of the runs named `PPO_PREDATOR_BANDS_*`
(`CALIB_*`, `COMPASS3_*`, `THREATS4*`, `THREATCAL*`, `THREATREST_*`, `SHARE_*`, `MAMMOTH*`) under `~/simulation_results/ray_results/`; those scalars are not
reproduced elsewhere in the repository. A Codex fact check of a first draft of this file found several unsupported or overstated statements (the pair-only
sweep, a merged sharing row, causal wording, the cohesion wording); they are corrected here.
