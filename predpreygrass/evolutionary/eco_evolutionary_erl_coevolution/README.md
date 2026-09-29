# ERL Coevolution — building gradually on the ERL Baldwin result

Fork of [`eco_evolutionary_erl_baldwin`](../eco_evolutionary_erl_baldwin/README.md)
(Trial 11: ERL beats E/L/F/B, p<0.00001, n=100/condition, its RESULTS.md §9).
That module is left untouched. The goal here is to move from its setup, where
only the prey adapts and the carnivores are a fixed hazard, toward real
predator-prey **coevolution**. Each step makes one change and has its own
pass/fail question:

| step | change | question |
|---|---|---|
| 0 | none: reproduce §9 exactly | is this fork a faithful baseline? |
| 1 | carnivore numbers regulated by prey supply (no immigration); behavior still hard-coded | can both species coexist? |
| 2 | carnivores get a genome (evolution, no learning) | does predator behavior evolve by selection, not drift? |
| 3 | carnivores get full ERL; 2×2 prey {E, ERL} × carnivore {E, ERL} | does learning still help prey when predators adapt too? |
| 4 | time-shifted contests between checkpoints (past/future opponents) | is there an arms race? |

Steps 2–4 are not built yet.

## Files

- `config.py`: `config_step0` (§9's exact config, commit 47534a0) and
  `config_step1` = step 0 + `STEP1_OVERRIDES`. `PRESETS` maps names to both.
- `world.py`: erl_baldwin's world with the C/K/S dead-end conditions removed,
  plus the step-1 knobs (`carnivore_spawn_interval = 0`,
  `carnivore_immigration_until`) and death-cause counters. Nothing added draws
  random numbers.
- `genome.py`: the §9-era genome (commit 47534a0), **not** erl_baldwin's HEAD.
  See "Step 0" below for why.
- `networks.py`, `metrics.py`, `checkpoint.py`, `visualize.py`: copied unchanged
  apart from import paths.
- `study.py`: parallel batch runner (`run`) and analysis (`analyze`). Results go
  to one JSON line per run, and reruns skip finished jobs.
  `analyze --compare-study` checks every run against §9's logs, run for run.
- `run_simulation.py`: single inspectable run with checkpoint/resume,
  progress + lineage CSVs, and optional rendering.

## Step 0: faithful baseline

erl_baldwin's current HEAD **does not** reproduce §9 run-for-run: 0/20
early-extinct seeds matched their §9 extinction step. The cause is that the
kin-selection and alarm-call commits (2026-08-24/25) added two extra random
draws per founder, mutation and crossover (`kinship_sensitivity`,
`alarm_call_propensity`). Those draws are consumed even under ERL/E/L/F/B,
so the RNG stream shifts for every strategy. The dynamics are unchanged, but
HEAD has never been re-validated at §9's scale. Checking out 47534a0 in a
temporary worktree reproduces 20/20 exactly.

This fork therefore uses the 47534a0 genome, and with `config_step0` it
reproduces §9 exactly:
- `tests/test_step0_reproduces_study.py` pins one seed per strategy to its
  §9 extinction step.
- The longer-horizon check is `study.py run --preset step0 --seeds 1-20
  --steps 50000` followed by `analyze --compare-study`.

**Result (2026-09-28): 100/100 runs match §9 exactly** (20 seeds × 5
strategies, 50k-step budget). The ranking also reproduces at this smaller
scale. ERL reached the budget in 80% of runs, E 55%, B 40%, L 35%, F 5%. ERL
beats L (p=0.008), F (p=2e-6) and B (p=0.005), all Mann-Whitney. ERL vs. E is
p=0.1 at n=20 and 50k steps; §9 separated them at n=100 and 1M steps.

## Step 1: carnivores regulated by prey

In step 0, carnivore numbers are set by immigration (one every 200 steps).
Their own reproduction threshold (18) sits close to their energy cap (20), so
their births are rare by design. The erl_baldwin longitudinal run ended at 878
agents vs. 5 carnivores. Step 1 removes immigration and makes carnivore
reproduction reachable, so carnivore numbers track prey through individual
energy budgets only. Carnivore behavior stays the fixed hand-coded rule.

Calibration so far (ERL only), with results in
`~/simulation_results/erl_results/coevo_step1_*`:

- **Sweep 1: no immigration, reproduction threshold/cost 12/6, 14/7, 16/8**
  (6 seeds each, 30k budget). At 14/7 and 16/8, carnivores go extinct in
  every run within ~600–3,000 steps, boom then bust. Naive, random-init
  agents die in large numbers from walking into walls, and the corpses fuel
  a carnivore boom. Carnivores starve once agents become competent. **At
  12/6, 3 of 6 seeds showed sustained predator-prey cycles.** Seed 2 had both
  species alive for the full 30k steps, with agents 150–450, carnivores
  10–65, and 4,881 carnivore births. Seeds 3 and 4 coexisted for 13.8k and
  9.1k steps before carnivores died out at a cycle low. **At 16/8, seed 2
  also coexisted for the full 30k steps** (481 agents, 32 carnivores at the
  end). A cost of 8 equals the newborn's 8 energy, so that birth is already
  energy-neutral. The survivor is the same seed in both settings, though, so
  this may be a favorable seed rather than a general result.
- **Caveat on 12/6:** in the inherited mechanics a newborn carnivore always
  gets `initial_energy_carnivore` (8), whatever the parent paid. At cost 6,
  every birth creates 2 units of energy from nothing. That is an artificial
  energy source, so `carnivore_energy_conserving_birth` (child gets exactly
  the parent's cost) now exists. Sweep 4 re-tests fast reproduction with it.
- **Sweep 2: slower carnivore life history** (basal cost 0.04/0.02 ×
  threshold 14/18): no run survives past ~3,000 steps, and about half end
  with carnivores driving naive agents extinct. Slowing carnivores down does
  not help; the recovery speed of a fast-reproducing predator seems to be
  what carries it through prey lows.
- **Sweep 3: 20k-step warm-up with step-0 immigration, then immigration off**
  (`carnivore_immigration_until`). With threshold 14, agents go extinct
  during the warm-up itself, because breeding carnivores plus immigration is
  too much predation for naive prey.
- **Sweep 4: energy-conserving births** (10 seeds per setting, 30k budget):
  12/6, 12/8 and 10/5 each coexist in only 0–1 of 10 seeds. Across all
  no-immigration runs, 68/82 lose their carnivores before step 3,000 (the
  opening boom-bust on naive prey). Of the 14 that get past 3,000, 9 still
  lose them between 6k and 16k steps at a cycle low, when only ~10–20
  carnivores are alive (demographic-stochasticity extinction).
- **Sweep 3 result: 20k warm-up with §9's carnivore reproduction parameters
  (threshold 18, cost 10: energy-lossy births) but *10* initial carnivores
  (inherited from the old step-1 preset, not §9's 5 -- see the correction
  below), then immigration off. 6/6 seeds coexisted for the full 40k steps after immigration stopped**
  (60k budget). Carnivore births were 7,500–8,900 per run against 100
  immigrants, so carnivore numbers were regulated by prey. Seed 5 ended with
  a single carnivore, a near-extinction. With threshold 14 (cost 7), the
  same warm-up gives 3/6 at the full budget and 2/6 agent extinctions during
  the warm-up. With basal cost 0.04 added, 0/6.

**`config_step1` is now this setting** (decided 2026-09-28): step 0
unchanged, but immigration stops after step 20,000
(`carnivore_immigration_until=20000`), and the run ends if carnivores then
die out. Rationale: an established prey population meets a carnivore
population that must live off it alone. Starting both species cold fails in
~90% of runs. It is still only 6 seeds and ERL only, so the confirming
comparison (20 seeds × 5 strategies × 60k steps) is next.

### Confirming comparison (2026-09-28): coexistence is NOT robust yet; ERL's prey advantage holds

`config_step1` (commit ed7aedb): 20 seeds × 5 strategies × 60k steps, results in
`~/simulation_results/erl_results/coevo_step1_compare`.

**Correction first.** Sweep 3's 6/6 does not carry over to this preset. Sweep 3
also inherited `n_initial_carnivores=10` from the old step-1 preset, while
`config_step1` uses §9's 5. Both runs replay exactly, so this is deterministic,
not noise. A 5-carnivore difference at step 0 changes the whole trajectory.
On seeds 1–6, 3/6 now coexist.

**Coexistence after immigration stops (step 20k):**

| strategy | prey alive at 20k | then coexist to 60k | carnivores die out | prey die out |
|---|---|---|---|---|
| ERL | 17/20 | 8 | 9 | 3 (all before 20k) |
| E | 11/20 | 3 | 8 | 9 |
| L | 7/20 | 0 | 7 | 13 |
| F | 1/20 | 0 | 1 | 19 |
| B | 9/20 | 2 | 7 | 11 |

Even for ERL, only about half the populations that reach the switch keep
their carnivores. 5 of ERL's 9 carnivore extinctions come within ~1,500 steps
of the switch (seeds 1, 8, 10, 16, 20), and several of those had only 1–9
carnivores left at step 20k. Stopping immigration abruptly often lands on a
cycle low. Step 1's pass criterion, robust coexistence, is **not met**.

**Prey survival: the §9 ranking holds with prey-regulated carnivores.** Agent
extinctions out of 20 were ERL 3, E 9, L 13, F 19, B 11. ERL beats L
(Fisher p=0.003), F (p=4e-7) and B (p=0.02). ERL vs. E is p=0.08 at n=20,
the same as step 0 at this sample size (p=0.1); §9 needed n=100 and 1M
steps to separate them. The coexistence-time Mann-Whitney tests say the same
(ERL vs. E/L/F/B: p=0.08/0.004/5e-6/0.009).

Death causes are unchanged from step 0. Predation is only ~12–15% of agent
deaths; ~85% are wounds, mainly from walking into walls. Carnivores are a
weak selective force on the prey, which matters for the arms-race steps.

### Bigger world (2026-09-28): coexistence becomes robust

Same `config_step1` (20k warm-up, then no immigration, 60k budget), ERL only,
20 seeds, on a 150×150 grid with **every density held fixed**. Per-cell
rates scale on their own; absolute counts are scaled by the area ratio 2.25:
`n_initial_agents` 135, `n_initial_carnivores` 11, `min_plants` 112,
`min_trees` 225, `max_population_cap` 4500, `carnivore_spawn_interval` 89 (same
immigrants per area). Results in `~/simulation_results/erl_results/coevo_step1_g150`.

| | 150×150 | 100×100 (same seeds) |
|---|---|---|
| prey alive at the switch (20k) | 19/20 | 17/20 |
| ...of which keep carnivores to 60k | **19/19** | 8/17 |
| lowest carnivore count after the switch, coexisting runs | **18–54** | 2–14 |
| carnivores at 60k | 52–238 | ~20–60 |
| predation share of agent deaths | 20% | 13% |

The one prey extinction (seed 20, step 1,233) happened during the warm-up.
Mechanism: in 100×100, carnivores repeatedly dip to a handful of individuals
at cycle lows and die out by chance (demographic stochasticity). In 150×150
the low point never goes below 18, and prey never below ~320. Carnivores also
become a stronger selective force (20% of agent deaths vs. 13%).

**`config_step1` now uses this 150×150 world** (decided 2026-09-28), via
`config.scale_world_area()`. It replays the validated runs exactly. The
5-strategy comparison at 150×150 is next.
Runtime is ~45 min per surviving 60k-step run (~22 steps/sec).

### 5-strategy comparison at 150×150 (2026-09-29): step 1 passes; ERL beats E/F/B, not L

`config_step1` (commit 3bea7de): 20 seeds × 5 strategies × 60k steps, results in
`~/simulation_results/erl_results/coevo_step1_compare150`. ERL's runs are
identical to the validated 150×150 runs.

| strategy | prey extinct (of 20) | alive at the switch | coexist to 60k | carnivores died out |
|---|---|---|---|---|
| ERL | 1 | 19 | 19 | 0 |
| E | 10 | 10 | 10 | 0 |
| L | 3 | 17 | 15 | 2 |
| F | 12 | 8 | 7 | 1 |
| B | 11 | 9 | 9 | 0 |

**Coexistence: step 1's criterion is met across strategies.** Of the 63 prey
populations alive when immigration stopped, 60 kept their carnivores to 60k.

**Prey survival:** ERL beats E (Fisher p=0.003), F (p=0.0004) and B
(p=0.001). ERL vs. L is **not** significant (p=0.6; the coexistence-time
Mann-Whitney gives p=0.11). Compared with 100×100 (same seeds), the decisive
comparison moved. There, ERL vs. E was p=0.08 and ERL vs. L p=0.003. L's prey
extinctions dropped from 13/20 to 3/20.

Candidate explanation for L, **not tested**: L clones genomes, but clonal
lineages still compete, so L is selection among founders plus learning.
With 135 founders instead of 60, the pool more often contains a good innate
evaluation function. If that's right, the L result reflects founder-pool size
as much as the new ecology. The bigger world changes the founder count and
the carnivore regime at the same time, so these two effects are confounded
here. A direct test would be L on 150×150 with 60 founders, or on 100×100
with 135 founders.

Predation share of agent deaths is 17–20% for every strategy (13% on
100×100).

### Founder-pool test (2026-09-29): the bigger world, not founder count, is what helps L

ERL and L at 150×150 with **60 founders** (`--set n_initial_agents=60`), 20 seeds
each, 60k steps; results in `~/simulation_results/erl_results/coevo_step1_founders60`.

| 150×150 world | ERL prey extinct | L prey extinct | ERL vs L, prey extinction (Fisher) | ERL vs L, coexistence time (Mann-Whitney) |
|---|---|---|---|---|
| 135 founders | 1/20 | 3/20 | p=0.6 | p=0.11 |
| 60 founders | 1/20 | 5/20 | p=0.18 | p=0.013 |
| *100×100, 60 founders* | *3/20* | *13/20* | *p=0.003* | |

With 60 founders, L gets only slightly worse (3→5 prey extinct, 135 vs. 60
not significant, p=0.69), and ERL doesn't change. With the *same* 60 founders,
L goes from 13/20 extinct on 100×100 to 5/20 on 150×150. So the
founder-pool hypothesis is largely rejected: most of L's improvement comes
with the bigger world itself. Lower founder density at 150×150 would, if
anything, work against L. Which aspect of the bigger world helps L (larger
populations, more room to escape, a different carnivore regime) is not
separated here. ERL's edge over L shows on coexistence time at 60 founders
(p=0.013; L populations also lose their carnivores 3 times vs. ERL's 0). On
prey extinction alone, n=20 doesn't separate ERL from L in the bigger world.

### ERL vs. L at n=60 (2026-09-29): same prey survival, but only L loses its carnivores

Seeds 21–60 added for ERL and L under the same tag as the 150×150 comparison
(`config_step1`, 135 founders, 60k steps), so there are now 60 seeds each.

| measure | ERL (n=60) | L (n=60) | Fisher p |
|---|---|---|---|
| prey extinct | 5 | 8 | 0.56 |
| prey alive at the switch (20k) | 55 | 52 | |
| carnivores die out after the switch | **0 of 55** | **7 of 52** | **0.005** |
| any collapse (either species) | 5 | 15 | 0.026 |

Coexistence time (Mann-Whitney) gives p=0.028. The first 20 and the added 40
seeds point the same way (carnivores lost: 0 vs. 2, then 0 vs. 5).

**Settled: on prey extinction, ERL and L are not distinguishable in the bigger
world** (5/60 vs. 8/60). The real difference is ecological. With L prey, the
carnivore population dies out after immigration stops in 7 of 52 runs,
between steps 24.9k and 59.0k. With ERL prey this never happens (0 of 55).
Why is **not** established. One candidate is that L's clonal prey populations
push carnivores into deeper cycle lows (for example through more synchronized
behavior, or by eating the corpses carnivores depend on). Another is that ERL
prey support larger carnivore populations; mean carnivores were 118.5 for
ERL vs. 91.8 for L. Checking carnivore minima and corpse use per strategy
would distinguish these. For the project goal (sustainable coevolution),
ERL gives the more stable predator-prey system, not better-surviving prey.

## Step 2: carnivores carry a genome (evolution, no learning)

Design (decided 2026-09-29), config `step2` / `step2_neutral` = `step1` +
`carnivore_mode`:

- **Genome to behavior:** the same single-layer action network as the prey
  over 10 carnivore inputs: prey signal N/S/E/W (exactly what the hand-coded
  rule sees), adjacent cell blocked N/S/E/W, energy, health. Founder weights
  are **seeded to approximate the hand-coded rule** (prey signal *i* → action
  *i*, weight +10; blocked *i* → action *i*, weight −10), plus N(0, 1) per
  founder. Founders start competent, since a predator arriving in a
  territory isn't naive. Seeded founders make ~14 kills per 1,000
  carnivore-steps early on, vs. ~16 for the rule. The per-founder variation
  leaves room for selection.
- **Reproduction:** sexual like the prey. Crossover with the nearest carnivore
  within `mate_search_radius` (else a copy), then mutation (rate 0.05,
  std 0.2).
- **Neutral-marker control (`step2_neutral`):** identical inheritance code,
  so same parents, same random draws and same offspring credit, but the genome
  is **not expressed**. Every carnivore acts with the canonical seed network.
  Genome change is then pure drift under the same demography. A first
  version that drew donor genomes from random living carnivores was not
  neutral (Codex review), because better-surviving genomes stay in that pool
  longer.
- Carnivore lineage (one row per carnivore, survivors censored, all 44 action
  weights) goes to `carnivore_lineage/`. Cumulative kills, carnivore-steps and
  the mean `carn_pursuit` / `carn_avoid` traits go to the time series.

**Pass/fail criteria, fixed before running** (ERL prey, `step2` vs.
`step2_neutral`, same seeds):

1. **Coexistence:** among runs with prey alive at the switch (20k), ≥90%
   keep their carnivores to the budget (step 1: 60/63).
2. **Hunting improves by selection:** the change in kill rate (kills per 1,000
   carnivore-steps, last 10k steps vs. first 5k) is larger under `step2`
   than under `step2_neutral` (Mann-Whitney across seeds). The neutral
   change absorbs prey adaptation and demography.
3. **Selection, not drift, in the genome:** per-generation `pursuit` /
   `avoid` trait trajectories. `step2` seeds agree on the direction of
   change and prefer directional (GRW) models in the Hunt test.
   `step2_neutral` seeds show no consistent direction.

### Step 2 pilot (2026-09-29): coexistence holds; hunting "improvement" is recovery; no directional genome change

ERL prey, `step2` vs. `step2_neutral`, 20 seeds each, 60k steps (commit
4a70edf). Results in `~/simulation_results/erl_results/coevo_step2_pilot`, from
`analyze_step2.py`:

| criterion | result |
|---|---|
| 1. Coexistence | PASS: 20/20 (neutral 19/19) |
| 2. Kill-rate change, step2 vs. neutral | PASS as written: +2.08 vs. +0.69 per 1,000 carnivore-steps, p=1e-7. **Invalid comparison, see below.** |
| 3. Direction of `pursuit` / `avoid` change | FAIL: positive in 9/20 and 6/20 seeds (sign p=0.82 / 0.12), like neutral (11/19, 7/19) |

**Criterion 2's pass is a flaw in the control design.** Absolute kill rates
(kills per 1,000 carnivore-steps):

| | first 5k | 20–30k | 50–60k |
|---|---|---|---|
| step2 | 14.7 | 16.6 | 16.7 |
| step2_neutral | 16.4 | 17.1 | 17.1 |

The neutral control expresses the noise-free seed network, so it starts ahead
of the noisy step-2 founders. Evolving carnivores recover to about the seed
network's level but never exceed it. The larger "improvement" is recovery
from a founder handicap the control never had. Recovery itself implies
selection, but it is catch-up, not evolution beyond the hand-coded rule. A
fair control needs matched starting competence. For example, each carnivore
could express seed + fresh, **non-heritable** N(0, 1) noise, giving the same
phenotypic variation with no response to selection possible.

Exploratory checks (not pre-registered):
- Genomes drift *away* from the seed in both conditions (mean distance
  6.6 → 7.2 step2, 6.6 → 7.3 neutral; difference p=0.36), so no sign of
  purifying selection toward the seed.
- No single one of the 44 weights/biases changes direction consistently
  across step-2 seeds after Bonferroni correction.
- Performance recovers while weights keep drifting. That fits many equally
  good weight settings (selection fixes the performance-relevant
  combinations, drift moves the rest), but this is not tested.

### Carnivore headroom probe (2026-09-29): behavior barely changes per-capita fitness

Fixed, non-evolving carnivore variants against ERL prey in the step-1 world,
8 seeds × 30k steps each (commit e2e179c). Kill rate is measured over
5k–30k steps, and only for runs where prey survived. Results in
`~/simulation_results/erl_results/coevo_headroom`.

| carnivore behavior | kills / 1,000 carnivore-steps | births / 1,000 carnivore-steps | mean carnivores | mean prey | prey extinct |
|---|---|---|---|---|---|
| seed network (weight 10) | 16.9 | 3.61 | 88 | 520 | 1/8 |
| hand-coded rule | 16.3 | 3.50 | 111 | 737 | 0/8 |
| sharp (weight 30) | 16.8 | 4.00 | 87 | 385 | 3/8 |
| soft (weight 5) | 17.1 | 4.02 | 118 | 466 | 3/8 |
| no obstacle avoidance | 16.6 | 3.85 | 149 | 602 | 2/8 |
| rule, ignores sheltered prey | **15.5** | **3.13** | 99 | 598 | 0/8 |

- **Per-capita carnivore fitness is nearly flat across quite different
  behaviors.** Kill rates span 15.5–17.1 (±5%). Births per carnivore-step
  for every variant are indistinguishable from the seed network (p ≥ 0.44),
  except "ignore sheltered prey", which is *worse* (p=0.0002).
- The differences show up at the **population level** instead: carnivore
  numbers (87–149), prey numbers (385–737) and prey extinctions (0–3/8). This
  is the expected ecological feedback. Better hunting means more carnivores
  and fewer prey, and per-capita success evens out (numerical response).
- The one extra piece of information tested (sheltered prey are
  unattackable) does not help; ignoring them hurts. Waiting near sheltered
  prey apparently pays.
- **Caveat:** each run here is monomorphic, with all carnivores using the
  same behavior. Selection acts on *differences within* a population, which
  is invasion fitness. A variant that looks equal when everyone uses it can
  still beat the resident as a rare mutant, or lose to it. So these results
  show that the carnivore fitness landscape is flat at the population level,
  consistent with step 2's drift-like genome change. They do not directly
  measure within-population selection.

### Matched control (2026-09-29): kill rate is not a valid selection measure; an unexplained opening difference

`step2_nonheritable` (commit 821fd8c), 20 seeds, 60k steps, compared with the
step-2 pilot on the same seeds:

| | step2 | step2_nonheritable | step2_neutral |
|---|---|---|---|
| prey extinct | 0/20 | **8/20** (all at steps 586–2,349) | 1/20 |
| kills / 1,000 carnivore-steps, first 5k | 14.7 | 16.2 | 16.4 |
| same, 50–60k | 16.7 | 17.2 | 17.1 |

- The control was designed to match step 2's starting competence, but it
  doesn't in practice. Founders are identical, yet the conditions diverge
  within the first few hundred steps. Over steps 0–500, step 2 kills 19–27 per
  1,000 carnivore-steps and the control 16–21. Per-capita kill rate tracks
  the ecological state (prey density in the opening boom) more than hunting
  skill; the headroom probe points the same way. **Criterion 2 therefore
  cannot isolate selection with either control and is dropped as a selection
  test.** Within-population selection is measured by the competition test
  (`mixed` mode) instead.
- **Unexplained:** the control loses its prey in the opening in 8/20 runs,
  step 2 in 0/20. A direct check found no bug. Newborns express equivalent
  networks in both modes at step 500 (mean pursuit 10.00 vs. 9.98, avoid
  −9.97 vs. −10.02, same bias and off-diagonal size). Under step 2, expressed
  behavior is less varied (per-carnivore pursuit spread 0.29 vs. 0.49 after
  ~4 generations): a few lineages quickly dominate. Whether lower predator
  diversity explains the milder opening is untested.

### Competition test (2026-09-29): real within-population selection, but the seed sits on a peak

`mixed` mode (commit f838ea3). Resident seed network vs. one mutant type,
50/50 start, type inherited. ERL prey, step-1 world, 10 seeds × 40k steps.
Results in `~/simulation_results/erl_results/coevo_invasion`.

Pre-registered analysis (`analyze_invasion.py`, after the switch, 20k→40k):
nothing significant. Birth ratios mutant/resident are 0.98–1.01 (all
p ≥ 0.15). The "no avoidance" mutant was already nearly gone at the switch.

Exploratory, whole trajectory from the 50/50 start (runs that reached the
switch):

| mutant | mutant frequency at 500 → 5k → 20k | warm-up birth ratio (Wilcoxon) | frequency at 20k vs. identical |
|---|---|---|---|
| identical (neutral) | 0.51 → 0.53 → 0.52 | 1.000 (p=0.55) | — |
| no obstacle avoidance | 0.40 → 0.16 → 0.09 (0.00 at 40k) | 0.926 (p=0.02) | p=0.0006 |
| sharp (30) | 0.62 → 0.48 → 0.61 | 0.987 (p=1) | p=0.42 |
| soft (5) | 0.44 → 0.51 → 0.38 | 0.981 (p=0.055) | p=0.28 |

- **Within-population selection is real.** Carnivores without obstacle
  avoidance are driven out of mixed populations, although in the monomorphic
  headroom probe a population made up entirely of them did as well per
  capita. Selection acts on relative differences, which the probe could not
  measure.
- **Nothing tested beats the seed network.** Sharper and softer pursuit are
  neutral within noise; the only clear effect is against a worse variant.
- This fits the step-2 pilot. Selection purges bad variants, the rest drifts,
  and performance recovers to about the seed's level but not beyond. With
  the current 10 inputs and a single-layer network, the seed appears to sit
  on or near a fitness peak, which leaves evolution no uphill to climb.
  This is a candidate explanation, not a proof: only three directions in
  weight space were tested.
- Method note: the pre-registered window (after the switch) missed the
  selection, which acted mostly during the warm-up when carnivores breed
  fastest.

### Rich-input competition test (2026-09-29): richer senses give no headroom

`mixed` mode with `carnivore_obs=rich` (commit 85b7f19). The rich seed resident
(all prey channels +10, identical behavior to the basic seed) competes with one
informed mutant, 50/50 start, 10 seeds × 40k. Primary readout fixed before the
run (commit c66aab0): frequency at the switch vs. identical, Bonferroni over 4
variants (p < 0.0125). Results in `~/simulation_results/erl_results/coevo_rich_invasion`.

| mutant | frequency 500 → 20k | warm-up birth ratio (Wilcoxon) | frequency vs. identical |
|---|---|---|---|
| identical | 0.51 → 0.52 | 1.000 (p=0.55) | — |
| prefers live prey (living 15) | 0.48 → 0.63 | 1.003 (p=0.73) | p=0.28 |
| prefers carcasses (corpse 15) | 0.45 → 0.45 | 0.975 (p=0.016) | p=0.80 |
| avoids carcasses (corpse 5) | 0.44 → 0.36 | 0.974 (p=0.078) | p=0.65 |
| ignores sheltered prey (sheltered 0) | 0.54 → 0.45 | 0.988 (p=0.016) | p=1.0 |

- **No informed variant beats the resident.** Deviations are neutral or tend
  to be slightly deleterious (carcass preference and ignoring sheltered prey:
  birth ratio < 1 at p≈0.016 each, not significant after correction).
- The identical-mutant runs are identical to the basic-input identical runs,
  confirming the rich seed's behavioral equivalence end to end.
- Together with the basic competition test: in every direction tested so far,
  the seed network is at or near a within-population fitness peak. Richer
  *senses* alone don't open headroom for a single-layer network. What the
  architecture cannot express (anything state-dependent, e.g. "hunt only when
  hungry", since energy/health cannot change the chosen direction) is untested.

### State-dependent competition test (2026-09-29): one candidate (persistent search), and the test's limits

`mixed` mode, rich inputs, hand-coded state-dependent mutants vs. the rich seed
resident (commit edc2df4), 10 seeds × 40k. Primary readout and threshold
(p < 0.05/3) fixed before the run. Results in
`~/simulation_results/erl_results/coevo_state_invasion`.

| mutant | freq 500 → 20k (pre-registered, vs. identical) | warm-up birth ratio | freq at 500 vs. identical (exploratory) | prey extinct |
|---|---|---|---|---|
| identical | 0.51 → 0.52 | 1.000 | — | 2/10 |
| persistent search | 0.79 → 0.79 (p=0.22, n=5) | 0.977 (p=0.06) | **0.75 vs. 0.51, p=0.004** | 5/10 (p=0.35) |
| sated scavenger | 0.46 → 0.58 (p=0.69) | 0.957 (p=0.016) | 0.45, p=0.52 | 3/10 |
| wounded scavenger | 0.55 → 0.34 (p=0.32) | 0.961 (p=0.004) | 0.55, p=0.21 | 1/10 |

- **Pre-registered: no mutant beats the resident.** The wounded scavenger is
  worse: its birth ratio after the switch is also 0.944 (p=0.004), and it
  falls to 0.06 by 40k. The sated scavenger tends to be worse.
- **Exploratory: persistent search pulls ahead early** (0.75 at step 500 vs.
  0.51, p=0.004), while carnivores are numerous (~120–380) and drift is weak.
  In all 5 runs that reached 40k it went to fixation, but see the next point.
  It also tended to exterminate prey more often (5/10 vs. 2/10, n.s.).
- **The competition test is low-powered in this setup.** Around step 1,000
  carnivores crash to 1–5 individuals in most runs (the opening boom-bust),
  and warm-up immigration (types alternating 50/50) refills them. So the
  frequency at the switch is largely reset. After the switch, populations of
  tens drift hard: the *identical* mutant went to fixation in 5/8 runs and
  to 0–8% in two. Only large effects (like "no avoidance") survive this.
  Persistent search's 5/5 fixation is therefore not distinguishable from
  drift.
- Persistence is memory-based, so a single-layer network over the current
  inputs cannot express it. It could approximately if the previous move
  were an input (weight previous-action_i → action_i, dominated by the ±10
  pursuit/avoid weights whenever prey or walls are seen). That would need no
  hidden layer.
- `analyze_invasion.py` fix: a type that dies out (no births after the switch)
  now yields an undefined (NaN) birth ratio instead of a crash.

### Confirmatory competition test (2026-09-29): persistent search beats the seed carnivore

Delayed assignment (commit 317e6aa). Carnivores are all residents through the
20k warm-up. After the last immigrant, every other living carnivore by id
becomes a persistent-search mutant (exact 50/50), then 30k steps of
competition without immigration. 20 seeds per tag, rich inputs, ERL prey.
Pre-registered primary (`analyze_competition.py`, written before the run):
per-seed net growth rate difference (births − deaths per 1,000
carnivore-steps, mutant − resident) vs. the identical control. Results in
`~/simulation_results/erl_results/coevo_competition`.

| mutant | net growth difference (median) | vs. 0 | end frequency | took over / lost |
|---|---|---|---|---|
| identical | +0.04 | p=0.52 | 0.60 | 7 / 4 |
| **persistent search** | **+0.44** | p=6e-5 | **1.00** | **13 / 0** |

**Primary: persistent search vs. identical, p=1.6e-5, mutant better.** It took
over in all 13 runs that reached 50k. n=15 per tag had both types at the split;
5 seeds lost their prey during the (identical) warm-up in both tags.

- This is the first behavior found that beats the seed carnivore within a
  population: real headroom for predator evolution. It is memory-based
  (repeat the previous move when no prey is visible). A single-layer network
  over the current inputs cannot express it, but it could approximately if
  the previous move were an input.
- **Watch for coexistence:** in 2 persist runs the carnivore population died
  out after the split (identical: 0). The opening test also had more prey
  extinctions with persist (5/10 vs. 2/10, n.s.). A more efficient predator
  may over-exploit its prey. Too few cases to conclude, but relevant for
  step 2 with evolvable persistence.

### Step 2b: can evolution discover persistence? (pre-registered 2026-09-29, before running)

`carnivore_obs = "rich_memory"`: the rich inputs plus a one-hot of the
carnivore's previous move (22 inputs). The seed weight for previous move
i → action i is **0**, so founders behave exactly like the rich seed and
persistence is not built in. Founder noise (N(0, 1)) and mutation give that
weight variation. The hand-coded persist rule corresponds to a weight of about
+3.3. `step2` vs. `step2_neutral` (neutral marker), ERL prey, 20 seeds each,
60k steps, `analyze_step2.py --layout rich_memory`.

Pass/fail, fixed now:
1. **Coexistence:** ≥90% of runs with prey alive at the switch keep their
   carnivores to 60k.
2. **Persistence evolves by selection:** the `persist` trait's net change
   (last vs. first generation mean) is positive in significantly more `step2`
   seeds than chance (sign test p < 0.05), and not in `step2_neutral`.
   Supporting: `step2`'s persist net change is larger than neutral's
   (Mann-Whitney), and GRW is preferred more often.

Expected risk: ~130 carnivore generations may move the trait only part of the
way toward +3.3. The pass rule is about direction, not magnitude.

### Step 2b result (2026-09-30): persistence does not evolve from zero

Commit 963c3bf, results in `~/simulation_results/erl_results/coevo_step2b`.

| criterion | result |
|---|---|
| 1. Coexistence | PASS: 20/20 (neutral 18/18) |
| 2. Persistence evolves by selection | **FAIL**: persist net change positive in 10/20 seeds (sign p=1), neutral 9/18; median change 0.25, far from the rule's ~+3.3 |

Pursuit and avoid again show no consistent direction. The kill-rate numbers are
reported only (dropped as a selection test).

Exploratory, from `carnivore_lineage/` (carnivores born after the switch, dead,
~14,000 per seed): the standardized selection gradient of offspring count on
the persist trait is essentially zero. The median is −0.001, positive in 10/20
seeds, the same as the neutral marker (7/18, difference p=0.46). Standing
variation in the trait is small (within-population SD 0.24).

Interpretation (hedged): the competition test shows that full persistence
(~+3.3) clearly beats the seed. Small persistence weights barely change
behavior and carry no detectable advantage. The fitness landscape along this
trait looks flat near 0 and rises only at large values, a plateau that small
mutations (std 0.2) cannot feel. This is the needle-in-a-haystack situation of
Hinton & Nowlan (1987), where learning is supposed to help evolution (the
Baldwin effect). Not tested: whether selection acts once variation reaches the
advantageous region (e.g. wider founder variation on the persist weights), and
whether learning carnivores find persistence within a lifetime.

