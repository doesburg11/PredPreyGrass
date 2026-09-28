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
- **Sweep 3 result: 20k warm-up with §9's own carnivore parameters
  (threshold 18, cost 10: energy-lossy births), then immigration off.
  6/6 seeds coexisted for the full 40k steps after immigration stopped**
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

